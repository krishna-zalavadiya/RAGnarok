from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import config
from indexing.bm25_builder import ONTOLOGY
from pipeline.schemas import CandidateFeatureVector, JDIntent, SkillRecord

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CAPABILITY TREE
# Each cluster has a weight (must sum to 1.0 across clusters).
# Each capability inside a cluster has a weight (must sum to 1.0
# within the cluster).  Skills are lowercase — matched against
# candidate skill.name (already lowercased in CandidateFeatureVector).
# ─────────────────────────────────────────────────────────────

_CAPABILITY_TREE: dict = {

    "retrieval_systems": {
        "weight": 0.30,
        "capabilities": {
            "embeddings": {
                "weight": 0.30,
                "skills": frozenset({
                    "sentence transformers", "sentence-transformers",
                    "embeddings", "embedding", "text embeddings",
                    "transformers", "bert", "sbert", "sentence-bert",
                    "bge", "e5", "openai embeddings",
                }),
            },
            "retrieval": {
                "weight": 0.25,
                "skills": frozenset({
                    "retrieval", "dense retrieval", "sparse retrieval",
                    "semantic search", "rag", "retrieval augmented generation",
                    "information retrieval", "hybrid retrieval", "hybrid search",
                }),
            },
            "vector_db": {
                "weight": 0.25,
                "skills": frozenset({
                    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
                    "vector database", "vector search", "vector db",
                    "ann", "approximate nearest neighbours",
                    "approximate nearest neighbors",
                }),
            },
            "search_engine": {
                "weight": 0.20,
                "skills": frozenset({
                    "elasticsearch", "opensearch", "lucene",
                    "bm25", "hybrid search", "rrf",
                    "keyword search", "full text search",
                }),
            },
        },
    },

    "ranking_eval": {
        "weight": 0.30,
        "capabilities": {
            "metrics": {
                "weight": 0.40,
                "skills": frozenset({
                    "ndcg", "mrr", "map",
                    "mean reciprocal rank", "mean average precision",
                    "ranking evaluation", "offline evaluation",
                    "a/b testing", "ab testing", "online evaluation",
                }),
            },
            "ranking_models": {
                "weight": 0.40,
                "skills": frozenset({
                    "ranking", "reranking", "re-ranking",
                    "cross-encoder", "cross encoder",
                    "bi-encoder", "bi encoder",
                    "learning to rank", "ltr",
                    "xgboost", "lightgbm", "catboost",
                    "recommendation systems", "recommender systems",
                }),
            },
            "ml_infra": {
                "weight": 0.20,
                "skills": frozenset({
                    "mlflow", "wandb", "weights and biases",
                    "feature store", "model serving", "model deployment",
                    "inference optimization",
                }),
            },
        },
    },

    "python_engineering": {
        "weight": 0.20,
        "capabilities": {
            "python": {
                "weight": 0.60,
                "skills": frozenset({
                    "python", "python3", "python 3",
                }),
            },
            "serving": {
                "weight": 0.20,
                "skills": frozenset({
                    "fastapi", "pydantic", "flask", "django",
                    "rest api", "api development",
                }),
            },
            "orchestration": {
                "weight": 0.20,
                "skills": frozenset({
                    "airflow", "prefect", "dagster",
                    "kubernetes", "docker",
                }),
            },
        },
    },

    "production_ml": {
        "weight": 0.15,
        "capabilities": {
            "ml_systems": {
                "weight": 0.50,
                "skills": frozenset({
                    "machine learning", "ml", "applied ml",
                    "deep learning", "neural networks",
                    "pytorch", "tensorflow", "jax",
                    "scikit-learn", "sklearn",
                }),
            },
            "data_engineering": {
                "weight": 0.50,
                "skills": frozenset({
                    "spark", "kafka", "flink",
                    "data pipelines", "etl",
                    "sql", "pandas", "numpy",
                    "data engineering",
                }),
            },
        },
    },

    "llm_systems": {
        "weight": 0.05,
        "capabilities": {
            "fine_tuning": {
                "weight": 0.50,
                "skills": frozenset({
                    "fine-tuning", "fine tuning", "finetuning",
                    "lora", "qlora", "peft",
                    "llm fine-tuning", "instruction tuning",
                }),
            },
            "llm_engineering": {
                "weight": 0.50,
                "skills": frozenset({
                    "llm", "large language models", "gpt",
                    "langchain", "llamaindex", "llama index",
                    "prompt engineering", "openai",
                    "llama", "mistral", "gemini",
                }),
            },
        },
    },
}

assert abs(
    sum(c["weight"] for c in _CAPABILITY_TREE.values()) - 1.0
) < 1e-6, "Cluster weights must sum to 1.0"

for _cname, _cluster in _CAPABILITY_TREE.items():
    assert abs(
        sum(cap["weight"] for cap in _cluster["capabilities"].values()) - 1.0
    ) < 1e-6, f"Capability weights in '{_cname}' must sum to 1.0"


# ─────────────────────────────────────────────────────────────
# PROFICIENCY / TRUST / BOOST
# ─────────────────────────────────────────────────────────────

_PROF_REQUIRED: dict[str, float] = dict(config.PROFICIENCY_MULTIPLIERS)
_PROF_NICE: dict[str, float]     = {k: v * 0.70 for k, v in config.PROFICIENCY_MULTIPLIERS.items()}
_PROF_DEFAULT: float             = config.PROFICIENCY_MULTIPLIERS.get("beginner", 0.40)

_DURATION_TRUST_MIN: float    = config.DURATION_TRUST_MIN
_DURATION_TRUST_MAX_MONTHS: int = config.DURATION_TRUST_MAX_MONTHS

_ENDORSEMENT_CAP: int  = config.ENDORSEMENT_BOOST_CAP
_ENDORSEMENT_MAX: float = config.ENDORSEMENT_BOOST_MAX

_ASSESSMENT_THRESHOLD: float = config.ASSESSMENT_SCORE_THRESHOLD
_ASSESSMENT_WEIGHT: float    = config.ASSESSMENT_SCORE_WEIGHT

_DISQUALIFIER_HARD_PENALTY: float = getattr(config, "DISQUALIFIER_HARD_PENALTY", 0.25)
_DISQUALIFIER_SOFT_PENALTY: float = getattr(config, "DISQUALIFIER_SOFT_PENALTY", 0.70)
# Hard disq only fires for expert/advanced: intermediate is incidental exposure,
# not a primary domain indicator. Previously included "intermediate" which was
# too aggressive (zeroed candidates with only a passing familiarity with CV/speech).
_DISQUALIFIER_HARD_PROFICIENCY: frozenset[str] = frozenset(("expert", "advanced"))

# Normalised final-score weights (keep ratio 2:1 from config, but normalise to [0,1])
_REQ_W: float  = config.REQUIRED_SKILL_WEIGHT / (
    config.REQUIRED_SKILL_WEIGHT + config.NICE_TO_HAVE_SKILL_WEIGHT
)
_NICE_W: float = config.NICE_TO_HAVE_SKILL_WEIGHT / (
    config.REQUIRED_SKILL_WEIGHT + config.NICE_TO_HAVE_SKILL_WEIGHT
)


# ─────────────────────────────────────────────────────────────
# DATACLASSES
# ─────────────────────────────────────────────────────────────

@dataclass(slots=True)
class ClusterScore:
    cluster_name:   str
    score:          float
    matched_skills: list[str]
    coverage_pct:   float


@dataclass(slots=True)
class SkillMatchResult:
    candidate_id:          str
    skill_match_score:     float
    required_score:        float
    nice_to_have_score:    float
    matched_required:      list[str]
    matched_nice_to_have:  list[str]
    matched_disqualifiers: list[str]
    hard_disqualifier:     bool
    soft_disqualifier:     bool
    cluster_scores:        list[ClusterScore]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _duration_trust(duration_months: int) -> float:
    if duration_months <= 0:
        return _DURATION_TRUST_MIN
    if duration_months >= _DURATION_TRUST_MAX_MONTHS:
        return 1.0
    frac = duration_months / _DURATION_TRUST_MAX_MONTHS
    return _DURATION_TRUST_MIN + frac * (1.0 - _DURATION_TRUST_MIN)


def _endorsement_boost(endorsements: int) -> float:
    if endorsements <= 0:
        return 0.0
    clamped = min(endorsements, _ENDORSEMENT_CAP)
    return _ENDORSEMENT_MAX * math.log1p(clamped) / math.log1p(_ENDORSEMENT_CAP)


def _effective_proficiency(skill: SkillRecord, prof_multiplier: float) -> float:
    if skill.assessment_score >= _ASSESSMENT_THRESHOLD:
        assessment_norm = min(skill.assessment_score / 100.0, 1.0)
        return (
            (1.0 - _ASSESSMENT_WEIGHT) * prof_multiplier
            + _ASSESSMENT_WEIGHT * assessment_norm
        )
    return prof_multiplier


def _build_lookup(
    skills: list[SkillRecord],
) -> tuple[dict[str, SkillRecord], dict[str, SkillRecord]]:
    """
    Build direct and synonym lookup maps from candidate's skill list.
    Direct map: canonical skill name → SkillRecord (full credit).
    Synonym map: ontology synonym → SkillRecord (partial credit via ONTOLOGY_PARTIAL_CREDIT).
    """
    direct_map:  dict[str, SkillRecord] = {}
    synonym_map: dict[str, SkillRecord] = {}

    for skill in skills:
        name_lower = skill.name.lower().strip()
        direct_map[name_lower] = skill
        for synonym in ONTOLOGY.get(name_lower, []):
            if synonym not in direct_map:
                synonym_map.setdefault(synonym, skill)

    return direct_map, synonym_map


def _score_skill_against_lookup(
    skill_name: str,
    direct_map: dict[str, SkillRecord],
    synonym_map: dict[str, SkillRecord],
    prof_table: dict[str, float],
    endorse_factor: float = 1.0,
) -> tuple[float, str | None]:
    """
    Score a single required/nice-to-have skill name against the candidate lookup.

    Returns (score, matched_skill_name_raw) where score is the proficiency ×
    trust + endorsement value, and matched_skill_name_raw is None if no match.
    """
    if skill_name in direct_map:
        skill_rec = direct_map[skill_name]
        credit = 1.0
    elif skill_name in synonym_map:
        skill_rec = synonym_map[skill_name]
        credit = config.ONTOLOGY_PARTIAL_CREDIT
    else:
        return 0.0, None

    prof   = skill_rec.proficiency.lower() if skill_rec.proficiency else ""
    prof_m = _effective_proficiency(skill_rec, prof_table.get(prof, _PROF_DEFAULT))
    trust  = _duration_trust(skill_rec.duration_months)
    endorse = _endorsement_boost(skill_rec.endorsements) * endorse_factor

    return credit * (prof_m * trust + endorse), skill_rec.name_raw


# ─────────────────────────────────────────────────────────────
# SCORER
# ─────────────────────────────────────────────────────────────

class SkillMatchScorer:
    """
    Hierarchical capability-cluster scorer.

    The tree is: clusters → capabilities → skill sets.
    Each capability score = best single-skill match within that capability
    (a candidate who knows FAISS gets full vector_db capability credit —
    they don't need to list Pinecone AND Weaviate AND Qdrant too).
    Capability scores are weighted within their cluster; cluster scores
    are weighted across clusters to produce required_score.
    """

    def __init__(self) -> None:
        # Pre-expand ontology for every skill set at construction time.
        self._expanded_tree: dict = {}
        for cluster_name, cluster in _CAPABILITY_TREE.items():
            capability_map = {}
            for cap_name, cap in cluster["capabilities"].items():
                capability_map[cap_name] = {
                    "weight": cap["weight"],
                    "expanded_skills": self._expand_skills(list(cap["skills"])),
                }
            self._expanded_tree[cluster_name] = {
                "weight": cluster["weight"],
                "capabilities": capability_map,
            }

    # ── Public API ────────────────────────────────────────────

    def score(
        self,
        candidate: CandidateFeatureVector,
        jd: JDIntent,
    ) -> SkillMatchResult:

        direct_map, synonym_map = _build_lookup(candidate.skills)

        cluster_scores, required_score = self._score_clusters(direct_map, synonym_map)
        nice_score, matched_nice       = self._score_nice_to_have(jd, direct_map, synonym_map)
        hard_disq, soft_disq, matched_disq = self._check_disqualifiers(jd, direct_map, synonym_map)

        # Normalised weighted combination
        raw = _REQ_W * required_score + _NICE_W * nice_score

        if hard_disq:
            raw *= _DISQUALIFIER_HARD_PENALTY
        elif soft_disq:
            raw *= _DISQUALIFIER_SOFT_PENALTY

        matched_required = sorted({
            skill
            for cs in cluster_scores
            for skill in cs.matched_skills
        })

        return SkillMatchResult(
            candidate_id=candidate.candidate_id,
            skill_match_score=round(float(max(0.0, min(1.0, raw))), 6),
            required_score=round(required_score, 6),
            nice_to_have_score=round(nice_score, 6),
            matched_required=matched_required,
            matched_nice_to_have=matched_nice,
            matched_disqualifiers=matched_disq,
            hard_disqualifier=hard_disq,
            soft_disqualifier=soft_disq,
            cluster_scores=cluster_scores,
        )

    def score_all(
        self,
        candidates: list[CandidateFeatureVector],
        jd: JDIntent,
    ) -> dict[str, SkillMatchResult]:
        """
        Score all candidates. Returns dict[candidate_id → SkillMatchResult]
        for O(1) lookup in CompositeScorer.rank().
        """
        t0 = time.perf_counter()
        results = {c.candidate_id: self.score(c, jd) for c in candidates}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "SkillMatchScorer: scored %d candidates in %.1f ms "
            "(hard_disq=%d, soft_disq=%d, mean=%.3f).",
            len(results),
            elapsed_ms,
            sum(1 for r in results.values() if r.hard_disqualifier),
            sum(1 for r in results.values() if r.soft_disqualifier),
            sum(r.skill_match_score for r in results.values()) / len(results) if results else 0.0,
        )
        return results

    # ── Cluster / capability scoring ──────────────────────────

    def _score_capability(
        self,
        expanded_skills: set[str],
        direct_map: dict[str, SkillRecord],
        synonym_map: dict[str, SkillRecord],
    ) -> tuple[float, list[str]]:
        """
        Score one capability (e.g. "vector_db").

        Strategy: take the BEST single matching skill within this capability.
        Rationale: a candidate who knows FAISS has the vector-db capability —
        they don't need to also know Pinecone to prove it. We reward breadth
        at the cluster level (multiple capabilities covered), not within a
        capability.

        Returns (score in [0,1], list of matched skill name_raws).
        """
        best_score: float         = 0.0
        matched:    list[str]     = []

        for skill_name in expanded_skills:
            raw, matched_skill = _score_skill_against_lookup(
                skill_name, direct_map, synonym_map, _PROF_REQUIRED,
            )
            if raw > 0 and matched_skill:
                matched.append(matched_skill)
                if raw > best_score:
                    best_score = raw

        # best_score is in [0, prof_max × trust_max + endorse_max]
        # Normalise to [0, 1] using the theoretical max.
        max_possible = (
            max(_PROF_REQUIRED.values()) * 1.0  # trust = 1.0 at cap
            + _ENDORSEMENT_MAX
        )
        normalised = best_score / max_possible if max_possible > 0 else 0.0

        return float(min(1.0, normalised)), sorted(set(matched))

    def _score_clusters(
        self,
        direct_map: dict[str, SkillRecord],
        synonym_map: dict[str, SkillRecord],
    ) -> tuple[list[ClusterScore], float]:
        """
        Score all clusters and return (cluster_score_list, overall_required_score).
        overall_required_score is the weighted sum of cluster scores, in [0, 1].
        """
        cluster_scores: list[ClusterScore] = []
        total_score = 0.0

        for cluster_name, cluster in self._expanded_tree.items():
            capabilities     = cluster["capabilities"]
            cluster_score    = 0.0
            matched_cluster: list[str] = []
            covered_caps     = 0

            for cap_name, cap in capabilities.items():
                cap_score, matched = self._score_capability(
                    cap["expanded_skills"], direct_map, synonym_map,
                )
                cluster_score += cap_score * cap["weight"]
                matched_cluster.extend(matched)
                if cap_score > 0:
                    covered_caps += 1

            total_score += cluster_score * cluster["weight"]

            cluster_scores.append(ClusterScore(
                cluster_name=cluster_name,
                score=round(cluster_score, 6),
                matched_skills=sorted(set(matched_cluster)),
                coverage_pct=round(
                    covered_caps / len(capabilities) if capabilities else 0.0,
                    4,
                ),
            ))

        return cluster_scores, float(min(1.0, total_score))

    def _score_nice_to_have(
        self,
        jd: JDIntent,
        direct_map: dict[str, SkillRecord],
        synonym_map: dict[str, SkillRecord],
    ) -> tuple[float, list[str]]:
        if not jd.nice_to_have_skills:
            return 0.0, []

        total_score = 0.0
        matched:    list[str] = []
        seen:       set[str]  = set()

        for skill_name in jd.nice_to_have_skills:
            norm = skill_name.lower().strip()
            if norm in seen:
                continue
            seen.add(norm)

            raw, matched_skill = _score_skill_against_lookup(
                norm, direct_map, synonym_map, _PROF_NICE, endorse_factor=0.5,
            )
            total_score += raw
            if matched_skill:
                matched.append(matched_skill)

        max_per_skill = max(_PROF_NICE.values()) * 1.0 + _ENDORSEMENT_MAX * 0.5
        max_possible  = len(seen) * max_per_skill
        score         = total_score / max_possible if max_possible > 0 else 0.0

        return float(min(1.0, score)), sorted(set(matched))

    def _check_disqualifiers(
        self,
        jd: JDIntent,
        direct_map: dict[str, SkillRecord],
        synonym_map: dict[str, SkillRecord],
    ) -> tuple[bool, bool, list[str]]:
        if not jd.disqualifier_skills:
            return False, False, []

        disq_expanded: frozenset[str] = self._expand_skills_frozen(jd.disqualifier_skills)

        hard    = False
        soft    = False
        matched: list[str] = []

        # Check direct matches first, then synonyms
        for skill_name in disq_expanded:
            skill_rec = direct_map.get(skill_name) or synonym_map.get(skill_name)
            if skill_rec is None:
                continue
            if skill_rec.name_raw not in matched:
                matched.append(skill_rec.name_raw)
            prof = skill_rec.proficiency.lower() if skill_rec.proficiency else ""
            if prof in _DISQUALIFIER_HARD_PROFICIENCY:
                hard = True
            else:
                soft = True

        return hard, soft, matched

    # ── Ontology expansion ────────────────────────────────────

    def _expand_skills(self, skills: list[str]) -> set[str]:
        """DFS expansion through ONTOLOGY graph. Returns a flat set of all aliases."""
        expanded: set[str] = set()
        stack = [s.lower().strip() for s in skills if s.strip()]
        while stack:
            skill = stack.pop()
            if skill in expanded:
                continue
            expanded.add(skill)
            for synonym in ONTOLOGY.get(skill, []):
                norm = synonym.lower().strip()
                if norm not in expanded:
                    stack.append(norm)
        return expanded

    def _expand_skills_frozen(self, skills: list[str]) -> frozenset[str]:
        return frozenset(self._expand_skills(skills))

    def __repr__(self) -> str:
        return f"SkillMatchScorer({len(self._expanded_tree)} clusters)"


# ─────────────────────────────────────────────────────────────
# MODULE SINGLETON
# ─────────────────────────────────────────────────────────────

_SCORER = SkillMatchScorer()


def score_skill_match(
    candidate: CandidateFeatureVector,
    jd: JDIntent,
) -> SkillMatchResult:
    return _SCORER.score(candidate, jd)
