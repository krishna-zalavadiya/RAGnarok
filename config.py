from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PROJECT PATHS
# ─────────────────────────────────────────────────────────────────────────────

# Root of the repository — everything is relative to this.
PROJECT_ROOT: Path = Path(__file__).parent.resolve()

# ── Data inputs ──────────────────────────────────────────────────────────────
DATA_DIR: Path = PROJECT_ROOT / "data"
CANDIDATES_JSONL_GZ: Path = DATA_DIR / "candidates.jsonl.gz"
CANDIDATES_JSONL: Path = DATA_DIR / "candidates.jsonl"
SAMPLE_CANDIDATES_JSON: Path = DATA_DIR / "sample_candidates.json"
JD_PATH: Path = PROJECT_ROOT / "job_description.md"

# ── Precomputed index artifacts (built by precompute.py, loaded by rank.py) ──
INDEXES_DIR: Path = DATA_DIR / "indexes"
FAISS_INDEX_PATH: Path = INDEXES_DIR / "faiss.index"
FAISS_ID_MAP_PATH: Path = INDEXES_DIR / "candidate_ids.npy"
BM25_INDEX_PATH: Path = INDEXES_DIR / "bm25.pkl"
FEATURE_STORE_PATH: Path = INDEXES_DIR / "features.npy"
FEATURE_IDS_PATH: Path = INDEXES_DIR / "feature_ids.npy"
TRAJECTORY_PATH: Path = INDEXES_DIR / "trajectory.npy"
TRAJECTORY_IDS_PATH: Path = INDEXES_DIR / "trajectory_ids.npy"
HONEYPOT_SET_PATH: Path = INDEXES_DIR / "honeypots.pkl"

# ── Ontology ──────────────────────────────────────────────────────────────────
ONTOLOGY_DIR: Path = PROJECT_ROOT / "ontology"
SKILL_MAP_PATH: Path = ONTOLOGY_DIR / "skill_map.json"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
DEFAULT_SUBMISSION_PATH: Path = OUTPUT_DIR / "submission.csv"


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Bi-encoder for dense embeddings (FAISS index + query encoding).
# all-MiniLM-L6-v2: 22MB, 384-dim, ~80ms per batch on CPU.
# Downloaded by sentence-transformers on first use; cached in HF cache dir.
BI_ENCODER_MODEL: str = "all-MiniLM-L6-v2"

# Cross-encoder for pairwise JD × candidate reranking (top-50 only).
# ms-marco-MiniLM-L-6-v2: ~80MB, calibrated relevance scores.
# Only runs on top-50 post-RRF + post-honeypot-filter. ~4s total on CPU.
CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Embedding dimension produced by BI_ENCODER_MODEL.
EMBEDDING_DIM: int = 384

# FAISS index type. IVF256 gives fast ANN search on 100K vectors.
# nlist=256 means 256 Voronoi cells; nprobe=32 at query time.
FAISS_NLIST: int = 256
FAISS_NPROBE: int = 32

# Batch size for encoding candidates during pre-computation.
# Tune down if RAM is tight during pre-compute (outside 5-min window).
EMBEDDING_BATCH_SIZE: int = 512


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# Number of candidates retrieved per path before RRF fusion.
# Increased so RRF pool has enough diversity to fill 100 final slots.
SEMANTIC_PATH_TOP_K: int = 350       # Path 1: FAISS dense similarity
KEYWORD_PATH_TOP_K: int = 350        # Path 2: BM25 + ontology expansion
ONTOLOGY_PATH_TOP_K: int = 280       # Path 3: skill graph traversal (Tier-5 rescue)
TRAJECTORY_PATH_TOP_K: int = 220     # Path 4: career pattern match
SIGNAL_PATH_TOP_K: int = 200         # Path 5: behavioral engagement

# After RRF fusion, keep this many candidates before honeypot filter.
# Increased from 150 → 200 to ensure the pipeline has enough candidates
# to fill 100 slots after honeypots, disqualifiers, and CE reranking.
RRF_POOL_SIZE: int = 800

# After honeypot filter, feed this many to the cross-encoder.
# Must be >= SUBMISSION_TOP_K so the composite scorer can rank all 100.
CROSS_ENCODER_TOP_K: int = 300
CE_SCORE_TOP_K: int = 300
LLM_WEIGHT: float = 0.3

# LLM justification: only generate LLM briefs for this many top candidates.
# Candidates ranked below this threshold keep the rule-based reasoning_generator
# output (which is already signal-grounded and high quality).
# 60 covers the full ELITE+STRONG+MID tiers; WEAK tier (61-100) uses rule-based.
# Lower this value (e.g. 30) to trade justification coverage for faster runtime.
LLM_RERANKER_TOP_N: int = 50

# Final submission size (spec requirement).
SUBMISSION_TOP_K: int = 100

# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────
# RRF score = Σ 1 / (k + rank_in_path). k=120 is optimized for 5 deep paths.
RRF_K: int = 70

# Bonus multipliers for paths that rescue candidates missed by dense/keyword.
# Path 3 (ontology graph) and Path 5 (signal) get a boost so Tier-5 candidates
# don't get buried by candidates who just have more text.
RRF_ONTOLOGY_PATH_BONUS: float = 1.3   # Path 3 score multiplied by this
RRF_SIGNAL_PATH_BONUS: float = 1.1     # Path 5 score multiplied by this

# Cross-encoder blend factor used in scoring/composite.py.
# final = (1 - CE_BLEND_FACTOR) × weighted_sum + CE_BLEND_FACTOR × ce_score
# Reduced from 0.30 → 0.15: the MS-MARCO cross-encoder is calibrated for web
# passage retrieval, not HR matching. Its sigmoid output clusters ~0.4–0.6,
# compressing scores. Lower blend preserves the config-weight driven ranking
# while still letting CE break ties in the top-50.
CE_BLEND_FACTOR: float = 0.15


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE SCORING WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
# Must sum to 1.0. Validated in pipeline/schemas.py at import time.
#
# Rationale from JD analysis:
#   - Skill (0.40): Primary signal. JD lists exact required skills.
#   - Career (0.35): JD heavily penalises consulting-only backgrounds,
#     title-chasers, and non-product-company experience.
#   - Behavioral (0.25): JD explicitly says to down-weight unavailable
#     candidates. Platform engagement = actual availability.

WEIGHT_SKILL: float = 0.40
WEIGHT_CAREER: float = 0.30
WEIGHT_BEHAVIORAL: float = 0.20
WEIGHT_TRAJECTORY: float = 0.10

# Guard: checked in schemas.py. If weights drift during calibration, this catches it.
_WEIGHT_SUM_TOLERANCE: float = 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# SKILL SCORING PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Required skills count twice as much as nice-to-have skills.
REQUIRED_SKILL_WEIGHT: float = 2.0
NICE_TO_HAVE_SKILL_WEIGHT: float = 1.0

# Proficiency level multipliers applied to the skill coverage score.
# "beginner" listed skill contributes less than "expert".
PROFICIENCY_MULTIPLIERS: dict[str, float] = {
    "beginner":     0.40,
    "intermediate": 0.65,
    "advanced":     0.85,
    "expert":       1.00,
}

# duration_months trust factor: skills with 0 duration_months get penalised.
# Scores are linearly interpolated: 0 months → MIN, ≥ MAX_MONTHS → 1.0.
DURATION_TRUST_MIN: float = 0.50      # score floor for skills with 0 months
DURATION_TRUST_MAX_MONTHS: int = 24   # months at which trust factor = 1.0

# Endorsements boost: log-scaled, capped at this endorsement count.
# Prevents a candidate with 1000 endorsements on one skill dominating.
ENDORSEMENT_BOOST_CAP: int = 50
ENDORSEMENT_BOOST_MAX: float = 0.10   # maximum additive boost from endorsements

# Partial credit for adjacent skills via ontology. A skill that is a
# synonym/co-skill of a required skill earns this fraction of full credit.
ONTOLOGY_PARTIAL_CREDIT: float = 0.60

# If a candidate has completed a Redrob skill assessment, their score
# modulates the proficiency multiplier. Below this score → use raw proficiency.
ASSESSMENT_SCORE_THRESHOLD: float = 40.0
ASSESSMENT_SCORE_WEIGHT: float = 0.20  # blend weight: 0.20×assessment + 0.80×proficiency


# ─────────────────────────────────────────────────────────────────────────────
# CAREER QUALITY PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# JD explicitly says 5–9 years is the target band.
YOE_BAND_MIN: float = 4.0    # soft lower bound (4+ considered, 5+ ideal)
YOE_BAND_IDEAL_MIN: float = 5.0
YOE_BAND_IDEAL_MAX: float = 9.0
YOE_BAND_MAX: float = 12.0   # soft upper bound (>12 not disqualified but lower score)

# Consulting-only penalty. If ALL career history is at these firms, apply penalty.
# Per JD: "People who have only worked at consulting firms ... we won't move forward."
CONSULTING_FIRMS: frozenset[str] = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mindtree", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree", "persistent systems", "coforge",
})
CONSULTING_ONLY_PENALTY: float = 0.35   # multiply career score by this if all consulting

# Product-company bonus. These industries indicate product-first companies.
PRODUCT_INDUSTRIES: frozenset[str] = frozenset({
    "software", "fintech", "food delivery", "e-commerce", "edtech",
    "healthtech", "saas", "ai/ml", "transportation", "gaming",
    "media", "marketplace", "travel tech",
})
PRODUCT_CO_BONUS: float = 1.20   # multiply base career score by this

# Trajectory velocity: promotions per year is the primary signal.
# Min/max used for percentile normalisation across the pool.
TRAJECTORY_PROMOTIONS_PER_YEAR_FLOOR: float = 0.0
TRAJECTORY_PROMOTIONS_PER_YEAR_CAP: float = 1.5

# Location bonus. Candidates in these cities get a small additive boost
# because the JD says Pune/Noida preferred; Delhi NCR/Hyderabad/Mumbai welcome.
PREFERRED_LOCATIONS: dict[str, float] = {
    "noida":     0.08,
    "pune":      0.08,
    "delhi":     0.05,
    "gurgaon":   0.05,
    "hyderabad": 0.04,
    "mumbai":    0.04,
    "bangalore": 0.03,
    "bengaluru": 0.03,
    "chennai":   0.02,
}

# Willing-to-relocate bonus (added on top of location bonus if applicable).
RELOCATION_BONUS: float = 0.03


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIORAL SIGNAL PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

# Recency decay: last_active_date score = exp(-RECENCY_LAMBDA × days_since_active).
# λ = 0.005 → 14 days ago ≈ 0.93, 90 days ≈ 0.64, 180 days ≈ 0.41, 365 days ≈ 0.16.
RECENCY_LAMBDA: float = 0.005

# Notice period scoring thresholds (in days).
NOTICE_PERIOD_IDEAL_MAX: int = 30    # ≤30 days → full score (JD says "love sub-30")
NOTICE_PERIOD_ACCEPTABLE_MAX: int = 60  # ≤60 days → moderate score
NOTICE_PERIOD_MAX: int = 90          # >90 days → significant penalty

# Sub-score weights within the behavioral multiplier (must sum to 1.0).
BEHAVIORAL_WEIGHTS: dict[str, float] = {
    "recency":              0.25,   # last_active_date recency
    "response_rate":        0.20,   # recruiter_response_rate
    "open_to_work":         0.15,   # open_to_work_flag
    "notice_period":        0.15,   # notice_period_days
    "github_activity":      0.10,   # github_activity_score
    "profile_completeness": 0.10,   # profile_completeness_score
    "interview_completion": 0.05,   # interview_completion_rate
}

# github_activity_score is -1 when not linked. Treat -1 as neutral (0.5).
GITHUB_NOT_LINKED_DEFAULT: float = 0.5

# offer_acceptance_rate is -1 when no prior offers. Treat -1 as neutral (0.5).
OFFER_ACCEPTANCE_NO_HISTORY_DEFAULT: float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# UNCERTAINTY PENALTY (SPARSE PROFILES)
# ─────────────────────────────────────────────────────────────────────────────
# Candidates with very few signals get a confidence penalty on their final score.
# Prevents hallucinated-looking profiles from scoring high when data is thin.

# Number of distinct non-empty signal types required for full confidence.
MIN_SIGNAL_TYPES_FOR_FULL_CONFIDENCE: int = 5

# Multiplier applied when signal count < MIN_SIGNAL_TYPES_FOR_FULL_CONFIDENCE.
# Linear interpolation: 0 signals → PENALTY_FLOOR, ≥ MIN → 1.0.
UNCERTAINTY_PENALTY_FLOOR: float = 0.70


# ─────────────────────────────────────────────────────────────────────────────
# HONEYPOT DETECTION RULES
# ─────────────────────────────────────────────────────────────────────────────
# Pre-compute flags these candidates offline. Zero cost at ranking time.

# Rule 2: Expert proficiency on a skill with 0 duration_months.
# Schema allows duration_months to be 0 — honeypots abuse this.
HONEYPOT_EXPERT_MIN_DURATION_MONTHS: int = 1  # expert needs at least 1 month

# Rule 3: Salary range anomaly — min > max.
# e.g. expected_salary_range_inr_lpa: {min: 30, max: 5} is impossible.
# We use a strict check: any min > max is flagged.

# Rule 4: Profile completeness score < threshold with suspiciously many skills.
# Honeypots stuff the skills section but leave the profile empty.
HONEYPOT_COMPLETENESS_THRESHOLD: float = 35.0
HONEYPOT_SKILLS_STUFFING_COUNT: int = 15

# Rule 5: Years of experience wildly inconsistent with career history span.
# If sum(duration_months) / 12 differs from years_of_experience by > threshold.
HONEYPOT_YOE_DISCREPANCY_YEARS: float = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# TRUST LAYER THRESHOLDS (Advocate / Skeptic / Verdict)
# ─────────────────────────────────────────────────────────────────────────────

# Advocate: minimum score to tag a signal as HIGH confidence.
ADVOCATE_HIGH_CONFIDENCE_THRESHOLD: float = 0.75
ADVOCATE_MEDIUM_CONFIDENCE_THRESHOLD: float = 0.50

# Skeptic: risk flag severity thresholds.
SKEPTIC_HIGH_RISK_INACTIVITY_DAYS: int = 90     # last_active > 90 days → HIGH risk
SKEPTIC_HIGH_RISK_RESPONSE_RATE: float = 0.15   # recruiter_response_rate < 0.15 → HIGH
SKEPTIC_MODERATE_NOTICE_DAYS: int = 60          # notice_period > 60 → MODERATE risk

# Verdict classification: how many HIGH risks → FRAGILE vs CONTESTED vs ROBUST.
VERDICT_FRAGILE_HIGH_RISK_COUNT: int = 2    # ≥2 HIGH risks → FRAGILE
VERDICT_CONTESTED_HIGH_RISK_COUNT: int = 1  # 1 HIGH risk → CONTESTED
# 0 HIGH risks and any signal mix → ROBUST


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONSTRAINTS (for validation and benchmarking)
# ─────────────────────────────────────────────────────────────────────────────

# Hard wall-clock limit for the ranking step (seconds). Used by benchmark script.
RANKING_WALL_CLOCK_LIMIT_SECONDS: int = 300   # 5 minutes

# Soft per-stage budget targets (seconds). Informational only — logged by runner.
STAGE_BUDGETS: dict[str, float] = {
    "load_indexes":       10.0,
    "encode_jd":           0.5,
    "semantic_path":       1.0,
    "keyword_path":        2.0,
    "ontology_path":       1.0,
    "trajectory_path":     1.0,
    "signal_path":         0.5,
    "rrf_fusion":          0.2,
    "honeypot_filter":     0.1,
    "cross_encoder":       6.0,
    "composite_scoring":   1.0,
    "trust_layer":         2.0,
    "csv_write":           0.5,
}

# Maximum RAM usage target during ranking step (bytes). Informational.
RAM_LIMIT_BYTES: int = 16 * 1024 ** 3   # 16 GB


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

LOG_LEVEL: str = "INFO"   # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# ─────────────────────────────────────────────────────────────────────────────
# SUBMISSION SPEC CONSTANTS (mirrors validate_submission.py — do not change)
# ─────────────────────────────────────────────────────────────────────────────

SUBMISSION_REQUIRED_HEADER: list[str] = ["candidate_id", "rank", "score", "reasoning"]
SUBMISSION_CANDIDATE_ID_PATTERN: str = r"^CAND_[0-9]{7}$"
SUBMISSION_EXPECTED_ROWS: int = 100
SUBMISSION_RANK_MIN: int = 1
SUBMISSION_RANK_MAX: int = 100


# ─────────────────────────────────────────────────────────────────────────────
# LLM Configuration
# ─────────────────────────────────────────────────────────────────────────────


# Path to the downloaded GGUF model file.
# Run precompute.py once to download it; rank.py loads from here (no network).
LLM_MODEL_PATH: str = str(Path("models/qwen2.5-1.5b-instruct-q4_k_m.gguf"))
 
# How many RRF survivors to pass through the LLM scorer.
# These are already filtered by honeypot + RRF — LLM is a final quality gate.
# At ~200-300ms/candidate on 8 cores: 300 ≈ 75s, well within 5-min budget.
LLM_TOP_N: int = 300
 
# CPU threads for llama.cpp inference.
# Set to your machine's physical core count for best throughput.
LLM_N_THREADS: int = 8
 
# Context window. Keep small (512-1024) — our prompts are ~150 tokens.
# Larger context = more RAM + slower inference.
LLM_N_CTX: int = 512
 
# Whether to run the LLM reranker. Set False to skip entirely (e.g. for tests).
LLM_RERANKER_ENABLED: bool = True
 
# HuggingFace repo and filename used by LLMReranker.download_model()
# Called only from precompute.py — never from rank.py
LLM_HF_REPO_ID: str   = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
LLM_HF_FILENAME: str  = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_MODEL_DIR: str    = "models/"


REQUIRED_SKILL_COVERAGE_THRESHOLD=0.30
REQUIRED_SKILL_COVERAGE_MAX_SCORE=0.45