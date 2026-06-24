from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Validate scoring weights sum to 1.0 at import time.
# If someone edits config.py and breaks the constraint, this fires immediately.
_weight_sum = config.WEIGHT_SKILL + config.WEIGHT_CAREER + config.WEIGHT_BEHAVIORAL+config.WEIGHT_TRAJECTORY
if abs(_weight_sum - 1.0) > config._WEIGHT_SUM_TOLERANCE:
    raise ValueError(
        f"Scoring weights in config.py must sum to 1.0, got {_weight_sum:.8f}. "
        f"WEIGHT_SKILL={config.WEIGHT_SKILL}, WEIGHT_CAREER={config.WEIGHT_CAREER}, "
        f"WEIGHT_BEHAVIORAL={config.WEIGHT_BEHAVIORAL}"
    )

_bw_sum = sum(config.BEHAVIORAL_WEIGHTS.values())
if abs(_bw_sum - 1.0) > config._WEIGHT_SUM_TOLERANCE:
    raise ValueError(
        f"BEHAVIORAL_WEIGHTS in config.py must sum to 1.0, got {_bw_sum:.8f}."
    )

_CANDIDATE_ID_RE = re.compile(config.SUBMISSION_CANDIDATE_ID_PATTERN)


# ─────────────────────────────────────────────────────────────────────────────
# SKILL RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkillRecord:
    """
    Normalised skill entry from a candidate's skills[] array.

    Constructed by pipeline/candidate_parser.py from raw JSON.
    Consumed by scoring/skill_match.py and ontology/graph_traversal.py.
    """

    name: str                       # Lowercase-normalised skill name
    name_raw: str                   # Original casing from profile
    proficiency: str                # "beginner" | "intermediate" | "advanced" | "expert"
    endorsements: int               # Raw endorsement count (≥0)
    duration_months: int            # Months of use (0 = not specified)
    assessment_score: float         # 0–100 from redrob platform; -1.0 if not taken

    def __post_init__(self) -> None:
        if self.proficiency not in config.PROFICIENCY_MULTIPLIERS:
            raise ValueError(
                f"Invalid proficiency '{self.proficiency}' for skill '{self.name}'. "
                f"Must be one of: {list(config.PROFICIENCY_MULTIPLIERS.keys())}"
            )
        if self.endorsements < 0:
            raise ValueError(f"endorsements must be ≥0, got {self.endorsements}")
        if self.duration_months < 0:
            raise ValueError(f"duration_months must be ≥0, got {self.duration_months}")


# ─────────────────────────────────────────────────────────────────────────────
# CAREER ENTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CareerEntry:
    """
    Single job in a candidate's career_history[] array.

    Constructed by pipeline/candidate_parser.py.
    Consumed by scoring/career_quality.py, indexing/trajectory_builder.py,
    and indexing/honeypot_registry.py.
    """

    company: str                    # Company name (original casing)
    company_lower: str              # Lowercase for consulting-firm lookup
    title: str                      # Job title
    start_date: date                # Parsed from ISO string
    end_date: Optional[date]        # None if is_current=True
    duration_months: int            # From raw field (may differ from date diff)
    is_current: bool
    industry: str                   # Industry label from raw data
    industry_lower: str             # Lowercase for product-industry lookup
    company_size: str               # e.g. "1001-5000"
    description: str                # Free-text role description


# ─────────────────────────────────────────────────────────────────────────────
# EDUCATION ENTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EducationEntry:
    """
    Single education record from a candidate's education[] array.

    Used by indexing/honeypot_registry.py (overlapping date detection)
    and scoring/career_quality.py (institution tier).
    """

    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str]            # GPA/percentage string or None
    tier: str                       # "tier_1" | "tier_2" | "tier_3" | "tier_4" | "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# REDROB BEHAVIORAL SIGNALS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RedrobSignals:
    """
    All 23 behavioral signals from the redrob_signals object.

    Constructed by pipeline/candidate_parser.py from raw JSON.
    Consumed by scoring/behavioral.py, indexing/feature_store.py,
    and retrieval/signal_path.py.

    Special values:
      github_activity_score = -1.0  →  no GitHub linked
      offer_acceptance_rate = -1.0  →  no offer history
    """

    profile_completeness_score: float   # 0–100
    signup_date: date
    last_active_date: date
    open_to_work_flag: bool
    profile_views_received_30d: int     # ≥0
    applications_submitted_30d: int     # ≥0
    recruiter_response_rate: float      # 0.0–1.0
    avg_response_time_hours: float      # ≥0
    skill_assessment_scores: dict[str, float]   # skill_name → 0–100
    connection_count: int               # ≥0
    endorsements_received: int          # ≥0
    notice_period_days: int             # 0–180
    expected_salary_min_lpa: float      # INR lakhs per annum
    expected_salary_max_lpa: float      # INR lakhs per annum
    preferred_work_mode: str            # "remote"|"hybrid"|"onsite"|"flexible"
    willing_to_relocate: bool
    github_activity_score: float        # 0–100 or -1.0
    search_appearance_30d: int          # ≥0
    saved_by_recruiters_30d: int        # ≥0
    interview_completion_rate: float    # 0.0–1.0
    offer_acceptance_rate: float        # 0.0–1.0 or -1.0
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool

    @property
    def days_since_active(self) -> int:
        """Days elapsed since last_active_date. Used for recency decay."""
        today = date.today()
        delta = today - self.last_active_date
        return max(0, delta.days)

    @property
    def has_github(self) -> bool:
        """True if GitHub is linked (score != -1)."""
        return self.github_activity_score >= 0.0

    @property
    def has_offer_history(self) -> bool:
        """True if candidate has prior offer history (rate != -1)."""
        return self.offer_acceptance_rate >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CANDIDATE FEATURE VECTOR  ← THE CORE CONTRACT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateFeatureVector:
    """
    Fully parsed and normalised representation of one candidate.

    This is the central data contract. Built by pipeline/candidate_parser.py.
    All downstream components (indexers, scorers, retrievers) consume this.

    Dev A: retrieval paths and trust layer read from this.
    Dev B: indexers and scorers write against this interface.

    NEVER add raw JSON fields here. All fields must be normalised.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    candidate_id: str               # "CAND_XXXXXXX" — validated on construction

    # ── Profile ───────────────────────────────────────────────────────────────
    headline: str                   # One-line headline
    summary: str                    # Multi-sentence summary
    location: str                   # Original location string
    location_lower: str             # Lowercase for location matching
    country: str
    years_of_experience: float      # From profile field
    current_title: str
    current_title_lower: str        # Lowercase for title matching
    current_company: str
    current_company_lower: str      # Lowercase for consulting-firm lookup
    current_company_size: str       # e.g. "1001-5000"
    current_industry: str
    current_industry_lower: str     # Lowercase for product-industry lookup

    # ── Structured sub-records ────────────────────────────────────────────────
    skills: list[SkillRecord]
    career_history: list[CareerEntry]
    education: list[EducationEntry]
    signals: RedrobSignals

    # ── Derived flags (set by candidate_parser.py) ────────────────────────────
    is_consulting_only: bool        # True if ALL companies are consulting firms
    has_product_co_experience: bool # True if ANY company is a product company
    total_career_months: int        # Sum of duration_months across all roles
    skill_names_lower: frozenset[str]  # Fast O(1) skill lookup set

    # ── Text corpus for embedding ─────────────────────────────────────────────
    # Concatenated text used for FAISS embedding and BM25 indexing.
    # Built by candidate_parser.py: headline + summary + titles + descriptions.
    embedding_text: str

    # ── Honeypot flag (set by indexing/honeypot_registry.py) ─────────────────
    # False by default; set to True during pre-computation if rules fire.
    is_honeypot: bool = False

    def __post_init__(self) -> None:
        if not _CANDIDATE_ID_RE.match(self.candidate_id):
            raise ValueError(
                f"Invalid candidate_id format: '{self.candidate_id}'. "
                f"Expected CAND_XXXXXXX (7 digits)."
            )
        if self.years_of_experience < 0 or self.years_of_experience > 50:
            raise ValueError(
                f"years_of_experience out of range: {self.years_of_experience}"
            )

    def has_skill(self, skill_name_lower: str) -> bool:
        """O(1) check for skill membership (normalised lowercase)."""
        return skill_name_lower in self.skill_names_lower

    def get_skill(self, skill_name_lower: str) -> Optional[SkillRecord]:
        """Return SkillRecord for a skill name, or None if not present."""
        for s in self.skills:
            if s.name == skill_name_lower:
                return s
        return None


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL RESULT (output of each retrieval path)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    One candidate retrieved by a single retrieval path.

    Produced by retrieval/semantic_path.py, keyword_path.py,
    ontology_path.py, trajectory_path.py, signal_path.py.

    Consumed by retrieval/rrf_fusion.py.
    """

    candidate_id: str
    path_score: float       # Raw score from this path (not normalised)
    path_name: str          # "semantic" | "keyword" | "ontology" | "trajectory" | "signal"
    rank_in_path: int       # 1-indexed rank within this path's results


# ─────────────────────────────────────────────────────────────────────────────
# RRF FUSED RESULT (output of RRF fusion)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RRFResult:
    """
    Candidate after Reciprocal Rank Fusion across all 5 paths.

    Produced by retrieval/rrf_fusion.py.
    Consumed by scoring/honeypot_filter.py → scoring/cross_encoder.py.
    """

    candidate_id: str
    rrf_score: float            # Σ 1/(k + rank) across all paths (with bonuses)
    paths_present: list[str]    # Which paths retrieved this candidate
    cross_encoder_score: float = 0.0  # Set by cross_encoder.py after rerank


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT SCORES (output of scoring layer)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComponentScores:
    """
    Decomposed scoring breakdown for one candidate.

    Produced by scoring/skill_match.py, career_quality.py, behavioral.py.
    Consumed by scoring/composite.py and trust/advocate.py + trust/skeptic.py.

    All component scores are in [0.0, 1.0] before composite fusion.
    """

    candidate_id: str

    # ── Primary components ────────────────────────────────────────────────────
    skill_score: float          # 0–1, weighted skill coverage
    career_score: float         # 0–1, career quality (product-co, YOE, trajectory)
    behavioral_score: float     # 0–1, engagement and availability multiplier

    # ── Sub-scores for transparency (used by trust layer + UI) ────────────────
    required_skill_coverage: float    # Fraction of required skills covered
    nice_to_have_coverage: float      # Fraction of NTH skills covered
    ontology_skills_matched: list[str]  # Skills matched via ontology (not direct)
    yoe_score: float              # 0–1 for years-of-experience band fit
    trajectory_velocity: float    # promotions/yr normalised to 0–1
    product_co_flag: bool         # True if any product-company in history
    consulting_only_flag: bool    # True if all consulting
    location_bonus: float         # Additive location/relocation bonus applied
    recency_score: float          # 0–1 recency decay score
    notice_period_score: float    # 0–1 notice period fitness

    # ── Uncertainty modifier ──────────────────────────────────────────────────
    uncertainty_penalty: float    # 0.7–1.0, multiplied into final score
    signal_count: int             # Number of non-empty signal types


# ─────────────────────────────────────────────────────────────────────────────
# TRUST VERDICT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdvocateSignal:
    """One piece of positive evidence from the Advocate agent."""
    label: str              # Human-readable signal description
    confidence: str         # "HIGH" | "MEDIUM" | "LOW"
    value: str              # Specific value or fact from profile


@dataclass
class SkepticSignal:
    """One risk flag from the Skeptic agent."""
    label: str              # Human-readable risk description
    severity: str           # "HIGH" | "MODERATE" | "LOW"
    value: str              # Specific value or fact causing the concern


@dataclass
class TrustVerdict:
    """
    Full adversarial trust analysis for one candidate.

    Produced by trust/verdict.py (combining advocate.py + skeptic.py).
    Consumed by trust/reasoning_generator.py and ui/components/candidate_card.py.
    """

    candidate_id: str
    advocate_signals: list[AdvocateSignal]
    skeptic_signals: list[SkepticSignal]

    # Verdict classification
    verdict: str            # "ROBUST" | "CONTESTED" | "FRAGILE"
    flip_risk: str          # "LOW" | "MEDIUM" | "HIGH"
    confidence_pct: float   # 0–100, overall ranking confidence

    # Falsifiability conditions (2 conditions that would change the rank)
    falsifiability: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# RANKED RESULT  ← FINAL PIPELINE OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankedCandidate:
    """
    Final ranked output for one candidate. Directly maps to one CSV row.

    Produced by pipeline/runner.py after all stages complete.
    Written to submission CSV by rank.py.

    candidate_id  → CSV column: candidate_id
    rank          → CSV column: rank  (1–100)
    final_score   → CSV column: score (non-increasing)
    reasoning     → CSV column: reasoning (1–2 sentences)
    """

    candidate_id: str
    rank: int                   # 1–100, assigned after final sort
    final_score: float          # 0.0–1.0, composite score
    reasoning: str              # 1–2 sentence explanation, no hallucination

    # Rich data retained for debugging and UI (not in CSV).
    components: Optional[ComponentScores] = None
    trust: Optional[TrustVerdict] = None
    feature_vector: Optional[CandidateFeatureVector] = None

    def __post_init__(self) -> None:
        if not (1 <= self.rank <= 100):
            raise ValueError(f"rank must be 1–100, got {self.rank}")
        if not (0.0 <= self.final_score <= 1.0):
            raise ValueError(
                f"final_score must be in [0,1], got {self.final_score} "
                f"for candidate {self.candidate_id}"
            )
        if not self.reasoning.strip():
            raise ValueError(
                f"reasoning must not be empty for candidate {self.candidate_id}"
            )

    def to_csv_row(self) -> dict[str, str]:
        """Return a dict matching the submission CSV header order."""
        return {
            "candidate_id": self.candidate_id,
            "rank": str(self.rank),
            "score": f"{self.final_score:.6f}",
            "reasoning": self.reasoning,
        }


# ─────────────────────────────────────────────────────────────────────────────
# JD INTENT  ← OUTPUT OF JD PARSER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JDIntent:
    """
    Structured intent extracted from the job description.

    Produced by pipeline/jd_parser.py.
    Consumed by all retrieval paths, scoring/skill_match.py,
    scoring/career_quality.py, and ontology/query_expander.py.
    """

    # ── Skill tiers ───────────────────────────────────────────────────────────
    required_skills: list[str]          # Must-have skills (lowercase normalised)
    nice_to_have_skills: list[str]      # Preferred but not required (lowercase)
    disqualifier_skills: list[str]      # Skills suggesting wrong domain (CV, speech)
    expanded_required: list[str]        # required + ontology synonyms (for BM25)

    # ── Experience ────────────────────────────────────────────────────────────
    yoe_min: float              # 5.0 for this JD
    yoe_max: float              # 9.0 for this JD (soft cap)
    yoe_ideal_min: float        # 5.0
    yoe_ideal_max: float        # 9.0

    # ── Location ──────────────────────────────────────────────────────────────
    preferred_locations: list[str]      # Lowercase city names
    relocation_accepted: bool

    # ── Disqualifiers ─────────────────────────────────────────────────────────
    disqualify_consulting_only: bool    # True for this JD
    disqualify_no_production: bool      # True for this JD

    # ── Dense representation ──────────────────────────────────────────────────
    # JD text encoded by jd_parser.py for FAISS query.
    embedding: Optional[list[float]] = None   # 384-dim MiniLM vector

    # ── Raw text for cross-encoder input ─────────────────────────────────────
    raw_text: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def validate_candidate_id(candidate_id: str) -> bool:
    """Return True if candidate_id matches CAND_XXXXXXX pattern."""
    return bool(_CANDIDATE_ID_RE.match(candidate_id))


def validate_ranked_list(ranked: list[RankedCandidate]) -> list[str]:
    """
    Validate a list of RankedCandidate objects against submission rules.

    Returns a list of error strings (empty = valid).
    Mirrors the logic in validate_submission.py so we can catch errors
    before writing the CSV.
    """
    errors: list[str] = []

    if len(ranked) != config.SUBMISSION_EXPECTED_ROWS:
        errors.append(
            f"Expected {config.SUBMISSION_EXPECTED_ROWS} ranked candidates, "
            f"got {len(ranked)}"
        )

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()

    for rc in ranked:
        # Duplicate ID check
        if rc.candidate_id in seen_ids:
            errors.append(f"Duplicate candidate_id: {rc.candidate_id}")
        seen_ids.add(rc.candidate_id)

        # Duplicate rank check
        if rc.rank in seen_ranks:
            errors.append(f"Duplicate rank: {rc.rank}")
        seen_ranks.add(rc.rank)

        # ID format
        if not validate_candidate_id(rc.candidate_id):
            errors.append(f"Invalid candidate_id format: {rc.candidate_id}")

    # Score monotonicity check
    sorted_ranked = sorted(ranked, key=lambda r: r.rank)
    for i in range(len(sorted_ranked) - 1):
        r1, r2 = sorted_ranked[i], sorted_ranked[i + 1]
        if r1.final_score < r2.final_score:
            errors.append(
                f"Non-monotonic scores: rank {r1.rank} score={r1.final_score:.6f} "
                f"< rank {r2.rank} score={r2.final_score:.6f}"
            )

    # Missing ranks check
    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)}")

    return errors


logger.debug("pipeline/schemas.py loaded — weight invariants verified.")