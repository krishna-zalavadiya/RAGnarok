"""
scoring/llm_reranker.py — Lightweight local LLM reranker using Qwen2.5-1.5B-Instruct (Q4 GGUF).

Scores the top-N candidates from RRF on a structured prompt.
No network calls during ranking. Model loads from local cache.

Usage:
    reranker = LLMReranker()
    pool = reranker.score_pool(rrf_pool, jd_intent, candidate_store, top_n=300)
    # Each result in pool now has .llm_score (float 0.0–1.0)

Tune via config.py:
    LLM_MODEL_PATH, LLM_TOP_N, LLM_BLEND_FACTOR, LLM_N_THREADS, LLM_N_CTX
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional
from pipeline.schemas import JDIntent, CandidateFeatureVector
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container — attach llm_score to whatever object your RRF returns
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMScoredResult:
    """
    Wraps an existing RRF result and adds an llm_score field.
    If your RRFResult already has extra fields, just add llm_score directly
    to that dataclass instead of using this wrapper.
    """
    candidate_id: str
    rrf_score: float
    llm_score: float = 0.5       # neutral default if LLM skipped
    paths_present: list = None

    def __post_init__(self):
        if self.paths_present is None:
            self.paths_present = []

# Prompt
def build_jd_summary(jd: JDIntent) -> str:

    try:
        req_skills = ", ".join(jd.required_skills)
        nice_to_have_skills = ", ".join(jd.nice_to_have_skills)
    except Exception:
        req_skills = "embeddings, retrieval, ranking, Python"
        nice_to_have_skills = "None"

    try:
        yoe = (
            f"Required Experience: {jd.yoe_min}-{jd.yoe_max} years. "
            f"Ideal Experience: {jd.yoe_ideal_min}-{jd.yoe_ideal_max} years."
        )
    except Exception:
        yoe = "Required Experience: 5-9 years."

    try:
        location = ", ".join(jd.preferred_locations)

        if jd.relocation_accepted is False:
            relocation_info = "Relocation is not accepted."
        elif jd.relocation_accepted is True:
            relocation_info = "Relocation is accepted."
        else:
            relocation_info = "Relocation preference not specified."

    except Exception:
        location = "Pune, Noida"
        relocation_info = "Relocation preference not specified."

    return (
        f"Role: Senior AI Engineer\n"
        f"{yoe}\n"
        f"Preferred Locations: {location}\n"
        f"{relocation_info}\n"
        f"Required Skills: {req_skills}\n"
        f"Nice-to-have Skills: {nice_to_have_skills}\n"
        f"Candidates with consulting-only backgrounds should receive lower scores.\n"
        f"Candidates with product-company experience should receive higher scores."
    )

# Cabdidate Summary
def build_candidate_summary(cfv: CandidateFeatureVector) -> str:
    try:
        skills = []
        for s in sorted(
            cfv.skills,
            key=lambda x: (x.assessment_score, x.endorsements),
            reverse=True
        )[:]:
            skill_str = f"{s.name_raw} ({s.proficiency}"
            if s.assessment_score >= 0:
                skill_str += f", score={s.assessment_score:.0f}"
            skill_str += ")"
            skills.append(skill_str)

        skills_str = ", ".join(skills)
    except Exception:
        skills_str = "Unknown"

    title = cfv.current_title or "Unknown"
    company = cfv.current_company or "Unknown"
    yoe = f"{cfv.years_of_experience:.1f} years"

    try:
        roles = []
        for role in cfv.career_history[:]:
            roles.append(
                f"{role.title} at {role.company} ({role.duration_months} months)"
            )
        career_summary = "; ".join(roles)
    except Exception:
        career_summary = "Unknown"

    try:
        education_summary = ", ".join(
            f"{e.degree} in {e.field_of_study} from {e.institution}"
            for e in cfv.education[:2]
        )
    except Exception:
        education_summary = "Unknown"

    try:
        signals = cfv.signals

        github = "Yes" if signals.has_github else "No"
        open_to_work = "Yes" if signals.open_to_work_flag else "No"

        activity = f"{signals.days_since_active} days ago"

        notice_period = f"{signals.notice_period_days} days"

    except Exception:
        github = "Unknown"
        open_to_work = "Unknown"
        activity = "Unknown"
        notice_period = "Unknown"

    product_exp = "Yes" if cfv.has_product_co_experience else "No"
    consulting_only = "Yes" if cfv.is_consulting_only else "No"

    return (
        f"Experience: {yoe}\n"
        f"Current Role: {title} at {company}\n"
        f"Location: {cfv.location}\n"
        f"Product Company Experience: {product_exp}\n"
        f"Consulting Only Background: {consulting_only}\n"
        f"Open To Work: {open_to_work}\n"
        f"Last Active: {activity}\n"
        f"Notice Period: {notice_period}\n"
        f"GitHub Linked: {github}\n"
        f"Top Skills: {skills_str}\n"
        f"Recent Career History: {career_summary}\n"
        f"Education: {education_summary}\n"
        f"Headline: {cfv.headline}\n"
        f"Summary: {cfv.summary[:300]}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template — single structured scoring prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert technical recruiter evaluating candidates for a Senior AI Engineer role at an early-stage Series A startup.

Important principles:

- Prioritize evidence over keywords.
- Prefer production ML systems experience.
- Prefer retrieval, ranking, recommendation and search systems.
- Prefer product-company experience.
- Prefer ownership and execution.
- Active candidates are better.
- Consulting-only careers are negative signals.
- Pure research backgrounds are negative signals.
- Do not overvalue trendy frameworks.

Scoring:

9-10 = Exceptional fit
7-9 = Strong fit
5-7 = Moderate fit
3-5 = Weak fit
0-3 = Poor fit

Return ONLY a number and reason why not full score or why such a low score
"""

SCORING_PROMPT_TEMPLATE = """\
JOB DESCRIPTION

{jd_summary}

CANDIDATE

{candidate_summary}

Score:"""


# ─────────────────────────────────────────────────────────────────────────────
# Main reranker class
# ─────────────────────────────────────────────────────────────────────────────

class LLMReranker:

    @staticmethod
    def download_model(repo_id: str, filename: str, local_dir: str):
        from huggingface_hub import hf_hub_download
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Downloading model {filename} from {repo_id} to {local_dir}...")
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

    def __init__(
        self,
        model_path: str,
        n_threads: int = 8,
        n_ctx: int = 2048,
        verbose: bool = False,
    ):

        self._model_path = model_path
        self._n_threads = n_threads
        self._n_ctx = n_ctx
        self._verbose = verbose

        self._llm = None

    def _load(self):

        if self._llm is not None:
            return

        from llama_cpp import Llama

        logger.info("Loading LLM...")

        t0 = time.perf_counter()

        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=self._verbose,
        )

        logger.info(
            "LLM loaded in %.2fs",
            time.perf_counter() - t0
        )

    def score_one(
        self,
        jd_summary: str,
        candidate_summary: str
    ) -> float:

        self._load()

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content":SCORING_PROMPT_TEMPLATE.format(
                    jd_summary = jd_summary,
                    candidate_summary = candidate_summary
                )
            }
        ]

        try:

            out = self._llm.create_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=3
            )

            raw = (
                out["choices"][0]
                ["message"]["content"]
                .strip()
            )

            score = float(raw)

            score = max(
                0.0,
                min(10.0, score)
            )

            return score / 10.0

        except Exception as e:

            logger.debug(
                "Failed to parse score (%s)",
                e
            )

            return 0.5

    def score_candidates(
        self,
        candidates: list[CandidateFeatureVector],
        jd: JDIntent,
        max_workers: int = 8
    ) -> list[tuple[str, float]]:

        self._load()

        t0 = time.perf_counter()

        logger.info(
            "Preparing summaries for %d candidates...",
            len(candidates)
        )

        jd_summary = build_jd_summary(jd)

        with ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:

            candidate_summaries = list(
                executor.map(
                    build_candidate_summary,
                    candidates
                )
            )

        logger.info(
            "Starting LLM scoring..."
        )

        results = []

        scored = 0

        for cfv, summary in zip(
            candidates,
            candidate_summaries
        ):

            score = self.score_one(
                jd_summary,
                summary
            )

            results.append(
                (
                    cfv.candidate_id,
                    score
                )
            )

            scored += 1

            if scored % 10 == 0:

                logger.info(
                    "Scored %d/%d candidates",
                    scored,
                    len(candidates)
                )

        elapsed = (
            time.perf_counter() - t0
        )

        logger.info(
            "Finished scoring %d candidates in %.1fs",
            len(candidates),
            elapsed
        )

        results.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return results