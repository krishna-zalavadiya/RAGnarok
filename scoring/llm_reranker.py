from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from pipeline.schemas import JDIntent, CandidateFeatureVector, TrustVerdict, ComponentScores

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

# Maximum character length for individual signal values in the LLM prompt.
_SIGNAL_VAL_MAX: int = 120

# Number of signals to include in the prompt block.
_ADVOCATE_SIGNALS_IN_PROMPT: int = 3   # top-3 advocate (HIGH first)
_SKEPTIC_SIGNALS_IN_PROMPT: int = 2    # top-2 skeptic (HIGH first)


def _smart_truncate(value: str, max_len: int) -> str:
    
    if len(value) <= max_len:
        return value
    sliced = value[:max_len].rstrip()
    last_comma = sliced.rfind(", ")
    if last_comma >= max_len // 2:
        return sliced[:last_comma] + "…"
    return sliced + "…"


def _build_signal_block(trust: TrustVerdict) -> str:
    lines: list[str] = []

    # ── STRENGTHS block ───────────────────────────────────────────────────────
    adv_signals = trust.advocate_signals[:_ADVOCATE_SIGNALS_IN_PROMPT]
    if adv_signals:
        lines.append("STRENGTHS")
        for sig in adv_signals:
            val = _smart_truncate(sig.value, _SIGNAL_VAL_MAX)
            tier_tag = f"[{sig.confidence:<3}]"
            lines.append(f"  {tier_tag} {sig.label}: {val}")

    # ── RISKS block ───────────────────────────────────────────────────────────
    skep_signals = trust.skeptic_signals[:_SKEPTIC_SIGNALS_IN_PROMPT]
    if skep_signals:
        lines.append("RISKS")
        for sig in skep_signals:
            val = _smart_truncate(sig.value, _SIGNAL_VAL_MAX)
            sev_tag = sig.severity.replace("MODERATE", "MOD ")
            lines.append(f"  [{sev_tag:<3}] {sig.label}: {val}")

    # ── KEY CONDITION (top falsifiability condition) ───────────────────────────
    if trust.falsifiability:
        cond = trust.falsifiability[0]
        for prefix in (
            "This ranking holds UNLESS ",
            "This ranking becomes MORE ROBUST if ",
            "This ranking is critically weakened by ",
            "This ranking is notably affected if ",
            "This ranking is marginally affected by ",
            "This ranking's confidence is reduced by ",
        ):
            if cond.startswith(prefix):
                cond = cond[len(prefix):].strip()
                break
        if len(cond) > 90:
            cond = cond[:87].rstrip() + "…"
        lines.append(f"KEY CONDITION: {cond}")

    return "\n".join(lines) if lines else "(no signals)"


def _tier_label(rank: int) -> str:
    """Map rank to a tier adjective the LLM can use for tone."""
    if rank <= 5:
        return "top-5 — exceptional fit"
    if rank <= 15:
        return "top-15 — strong fit"
    if rank <= 40:
        return "mid-tier — solid but with gaps"
    if rank <= 70:
        return "lower-mid — notable weaknesses"
    return "bottom-tier — poor fit"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt constants
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are a technical recruiter writing a 2-sentence candidate brief. "
    "Use ONLY the facts listed below — no invented claims. "
    "Sentence 1: state the single most important fact (strength or risk) "
    "specific to THIS candidate — name the exact skill, company, or number. "
    "Do NOT open with 'The strongest signal that explains the position'. "
    "Start directly with the fact (e.g. '82% skill match across FAISS...' or "
    "'Inactive 120 days — availability unclear...'). "
    "Sentence 2: give the key counterpoint or the KEY CONDITION that would change this assessment. "
    "Be concise. No filler phrases." \
    "Both sentence combined must be less than 80 words (60 preferred)"
)

_USER_TEMPLATE = """\
Job: Senior AI Eng | 5-9yr | product-co | retrieval/ranking skills
Rank #{rank} of 100 | {tier} | {verdict} ({confidence}% confidence)
Candidate: {yoe}yr exp | composite score {composite:.3f}
Facts:
{signal_block}
Brief:"""


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class LLMReranker:
    """Local GGUF LLM used exclusively for post-ranking justification."""

    @staticmethod
    def download_model(repo_id: str, filename: str, local_dir: str) -> None:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading %s from %s to %s …", filename, repo_id, local_dir)
        hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)

    def __init__(
        self,
        model_path: str,
        n_threads: int = 4,
        n_ctx: int = 512,
        verbose: bool = False,
        max_workers: int = 4,
    ) -> None:
        self._model_path = model_path
        self._n_threads = n_threads
        self._n_ctx = n_ctx
        self._verbose = verbose
        self._max_workers = max_workers
        self._llm = None
        self._lock = threading.Lock()  # llama_cpp.Llama is NOT thread-safe

    def preload(self) -> None:
        """
        Eagerly load the GGUF model into memory.

        Call this once at pipeline startup (e.g. in runner.py __init__) so the
        model is warm before any candidates arrive.  Calling it again is a
        no-op.
        """
        self._load()

    def _load(self) -> None:
        if self._llm is not None:
            return
        # pyrefly: ignore [missing-import]
        from llama_cpp import Llama
        logger.info("Loading LLM for justification …")
        t0 = time.perf_counter()
        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=self._n_ctx,
            n_threads=self._n_threads,
            verbose=self._verbose,
            use_mlock=False,
        )
        logger.info("LLM loaded in %.2fs", time.perf_counter() - t0)

    # ── Core inference ────────────────────────────────────────────────────────

    def _justify_one(
        self,
        rank: int,
        trust: TrustVerdict,
        fallback: str,
        candidate: Optional[CandidateFeatureVector] = None,
        composite_score: float = 0.0,
    ) -> str:
        signal_block = _build_signal_block(trust)
        tier = _tier_label(rank)
        yoe = f"{candidate.years_of_experience:.1f}" if candidate else "?"

        prompt = _USER_TEMPLATE.format(
            rank=rank,
            tier=tier,
            verdict=trust.verdict,
            confidence=int(round(trust.confidence_pct)),
            yoe=yoe,
            composite=composite_score,
            signal_block=signal_block,
        )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            with self._lock:  # serialize: llama_cpp is not thread-safe
                out = self._llm.create_chat_completion(
                    messages=messages,
                    temperature=0.4,
                    max_tokens=100,
                    stop=["\n\n", "Sentence 3", "3.", "\nJob:"],
                )
            text = out["choices"][0]["message"]["content"].strip()

            _BANNED_OPENERS = (
                "Write",
                "You are",
                "Job:",
                "The strongest signal that explains the position",
                "The single most important fact specific to this candidate is",
                "The key strength is the candidate",
            )
            if len(text) < 30 or any(text.startswith(p) for p in _BANNED_OPENERS):
                logger.debug("LLM output rejected (too short or generic echo): %r", text[:80])
                return fallback

            return text[:320]

        except Exception as exc:
            logger.debug("LLM justify_one failed: %s", exc)
            return fallback

    # ── Batch justification (public API) ──────────────────────────────────────

    def justify_candidates(
        self,
        candidates: list[CandidateFeatureVector],
        jd: JDIntent,
        ranks: dict[str, int],
        trust_verdicts: Optional[dict[str, TrustVerdict]] = None,
        fallbacks: Optional[dict[str, str]] = None,
        top_n: Optional[int] = None,
        composite_scores: Optional[dict[str, float]] = None,
    ) -> dict[str, str]:
        
        self._load()

        trust_verdicts = trust_verdicts or {}
        fallbacks = fallbacks or {}

        _composite: dict[str, float] = composite_scores or {}
        if composite_scores is not None and len(composite_scores) == 0:
            logger.warning(
                "LLM: composite_scores was passed as an empty dict; "
                "all prompt headers will show composite=0.000"
            )

        results: dict[str, str] = {}
        t0 = time.perf_counter()

        # ── Partition candidates into LLM vs rule-based buckets ───────────────
        llm_batch: list[tuple[int, CandidateFeatureVector, str, TrustVerdict, float]] = []
        for i, cfv in enumerate(candidates, start=1):
            cid = cfv.candidate_id
            rank = ranks.get(cid, i)

            raw_fallback = fallbacks.get(cid)
            if raw_fallback is None:
                logger.warning(
                    "LLM: no fallback string for candidate %s (rank %d) — "
                    "using generic placeholder; check that the trust layer "
                    "produced a complete fallbacks dict.",
                    cid,
                    rank,
                )
                raw_fallback = f"Ranked based on composite score (rank {rank})."

            # Candidates beyond top_n skip LLM inference entirely.
            if top_n is not None and rank > top_n:
                results[cid] = raw_fallback
                continue

            trust = trust_verdicts.get(cid)
            if trust is None:
                logger.debug("LLM: no trust verdict for %s, using fallback", cid)
                results[cid] = raw_fallback
                continue

            composite = _composite.get(cid, 0.0)
            llm_batch.append((rank, cfv, raw_fallback, trust, composite))

        logger.info(
            "LLM: generating signal-grounded justifications for %d candidates "
            "(%d skipped via rule-based fallback) …",
            len(llm_batch),
            len(results),
        )

        if not llm_batch:
            return results

        llm_count = 0
        skipped_count = len(candidates) - len(llm_batch)

        def _run(args: tuple) -> tuple[str, str]:
            rank, cfv, fallback, trust, composite = args
            return cfv.candidate_id, self._justify_one(
                rank=rank,
                trust=trust,
                fallback=fallback,
                candidate=cfv,
                composite_score=composite,
            )

        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            futures = {pool.submit(_run, args): args for args in llm_batch}
            for future in as_completed(futures):
                try:
                    cid, justification = future.result()
                    results[cid] = justification
                    llm_count += 1

                    if llm_count % 10 == 0:
                        elapsed = time.perf_counter() - t0
                        rate = llm_count / elapsed if elapsed > 0 else 1.0
                        remaining = max(0, len(llm_batch) - llm_count)
                        eta = remaining / rate if rate > 0 else 0.0
                        logger.info(
                            "LLM: %d/%d done | %d skipped (rule-based) | "
                            "%.1fs elapsed | ETA %.0fs",
                            llm_count,
                            len(llm_batch),
                            skipped_count,
                            elapsed,
                            eta,
                        )
                except Exception as exc:
                    # Surface unexpected future errors — don't silently swallow.
                    args = futures[future]
                    cid = args[1].candidate_id
                    fallback = args[2]
                    logger.warning("LLM: future failed for %s: %s — using fallback", cid, exc)
                    results[cid] = fallback
                    llm_count += 1

        elapsed = time.perf_counter() - t0
        logger.info(
            "LLM: justified %d candidates via LLM, %d via rule-based, "
            "in %.1fs (%.2f s/LLM-candidate)",
            llm_count,
            skipped_count,
            elapsed,
            elapsed / max(1, llm_count),
        )
        return results