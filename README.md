# RAGnarok 
### Intelligent Candidate Ranking System · Redrob Hackathon

> **A five-path hybrid retrieval + cross-encoder reranking pipeline that ranks 100,000 candidates against a job description in under 60 seconds on CPU — with no network, no GPU, and an adversarial trust layer that generates honest, falsifiable reasoning for every ranked candidate.**

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Key Design Decisions](#key-design-decisions)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Quickstart](#quickstart)
- [Two-Phase Execution](#two-phase-execution)
- [Pipeline Stages](#pipeline-stages)
- [Scoring Model](#scoring-model)
- [Honeypot Detection](#honeypot-detection)
- [Trust Layer & Reasoning](#trust-layer--reasoning)
- [Runtime Budget](#runtime-budget)
- [Configuration Reference](#configuration-reference)
- [Running Tests](#running-tests)
- [Submission Checklist](#submission-checklist)
- [What's Next](#whats-next)

---

## Architecture Overview

RAGnarok is built around one core insight from the job description:

> *"A Tier 5 candidate may not use the words 'RAG' or 'Pinecone' in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit."*

A keyword search or a single dense retrieval path will miss these candidates. RAGnarok runs **five independent retrieval paths** — semantic, keyword, ontology graph, trajectory, and behavioral signals — fuses them with Reciprocal Rank Fusion, then reranks the survivors with a cross-encoder before applying a composite scorer and an adversarial trust layer.

```
candidates.jsonl.gz  ──┐
                        ├──▶  [PRE-COMPUTATION — offline]
job_description.md   ──┘       Candidate Parser · JD Parser · Ontology Builder
                                FAISS Index · BM25 Index · Feature Store
                                Trajectory DB · Honeypot Registry
                                           │
                                           ▼
                              Honeypot Candidate Remover
                                           |
                                           ▼
                        [RANKING WINDOW — ≤ 5 min · CPU · no network]
                         ┌─────────────────────────────────────────┐
                         │  Path 1: Semantic   (FAISS · top-350)   │
                         │  Path 2: Keyword    (BM25  · top-350)   │
                         │  Path 3: Ontology   (graph · top-280)   │
                         │  Path 4: Trajectory (career· top-220)   │
                         │  Path 5: Signal     (behav.· top-200)   │
                         └──────────────┬──────────────────────────┘
                                        │ RRF Fusion (top-800)
                                        ▼
                            Cross-Encoder Rerank
                                        │
                                        ▼
                         Composite Scorer (0.40 · 0.30 · 0.20 · 0.10)
                                        │
                                        ▼
                         Adversarial Trust Layer
                         (Advocate · Skeptic · Verdict)
                                        │
                                        ▼
                     LLM Reasoning for LLM_TOP_N Candidates
                                        │
                                        ▼
                              submission.csv  (top-100)
```

---

## Key Design Decisions

### Five retrieval paths — not one

Single-path systems fail on sparse profiles. Each path rescues a different class of candidate:

| Path | What it finds | Why it matters |
|------|--------------|----------------|
| **Semantic (FAISS)** | Dense vector similarity via `all-MiniLM-L6-v2` | Catches rich profiles with contextual skill descriptions |
| **Keyword (BM25 + ontology)** | Exact + synonym token match | Catches sparse profiles and non-standard skill names |
| **Ontology graph** | Domain-transfer traversal (e.g. RecSys → IR) | Rescues Tier-5 candidates who built equivalent systems under different names |
| **Trajectory** | IC-riser patterns, product-co flag, YOE band | Filters consulting-only careers and title-chasers without surface-level keyword matching |
| **Signal** | `open_to_work`, `last_active`, `recruiter_response_rate` | Eliminates unreachable "perfect-on-paper" candidates early |

### Honeypots are filtered before the expensive cross-encoder

The cross-encoder runs ~80ms per candidate. Pre-filtering impossible profiles with O(1) registry lookups means we never waste that budget on clearly fake entries. Honeypot rate target: **< 5% in top-100**.

### No LLM API calls — ever

All reasoning is template-based + Localy Light weight LLM based and derived from actual profile fields. No hallucination is possible because every sentence references a field that was read from the candidate record. This satisfies the **no-network** constraint and the spec's Stage 4 hallucination check.

### Adversarial reasoning (Advocate + Skeptic + Verdict)

Rather than generating uniform praise for every candidate, the trust layer runs an advocate scan (positive signals) and a skeptic scan (concerns) independently, then synthesises a verdict (`ROBUST` / `CONTESTED` / `FRAGILE`). The reasoning column reflects both sides honestly — which is exactly what Stage 4 manual review rewards. This is then given to the LLM for proper validation and reason for LLM_TOP_N candidates.

---

## Project Structure

```
.
├── config.py                   # All constants — never hardcode elsewhere
├── rank.py                     # CLI entry point → produces submission.csv
├── precompute.py               # index builder helper
├── precompute_llm.py           # Offline LLM downaloader (run once)
├── build_indexes.py            # Index builder helper
├── job_description.md          # The target JD
├── parsed_job_description.json # Cached JD parse output
├── submission_metadata.yaml    # Hackathon submission metadata
├── requirements.txt            # Pinned dependencies
│
├── pipeline/
│   ├── candidate_parser.py     # Raw JSON → CandidateFeatureVector
│   ├── jd_parser.py            # JD markdown → JDIntent
│   ├── runner.py               # Pipeline orchestrator (all 10 stages)
│   └── schemas.py              # Pydantic dataclasses for all pipeline objects
│
├── indexing/
│   ├── faiss_builder.py        # Builds / loads the FAISS IVF256 index
│   ├── bm25_builder.py         # Builds / loads the BM25 inverted index
│   ├── feature_store.py        # 23 behavioural signal vectors
│   ├── trajectory_builder.py   # Career pattern index (promotions, IC-riser)
│   └── honeypot_registry.py    # 5-rule offline honeypot detector
│
├── ontology/
│   ├── skill_map.json          # Synonym + implication map (PyTorch↔TensorFlow etc.)
│   ├── query_expander.py       # Expands BM25 queries via skill_map
│   └── graph_traversal.py      # Walks adjacency graph for domain-transfer candidates
│
├── retrieval/
│   ├── semantic_path.py        # Path 1 — FAISS cosine search
│   ├── keyword_path.py         # Path 2 — BM25 + ontology expansion
│   ├── ontology_path.py        # Path 3 — graph traversal
│   ├── trajectory_path.py      # Path 4 — career pattern match
│   ├── signal_path.py          # Path 5 — behavioural engagement
│   └── rrf_fusion.py           # Reciprocal Rank Fusion across all paths
│
├── scoring/
│   ├── skill_match.py          # Skill coverage · proficiency · duration trust
│   ├── career_quality.py       # Product-co · YOE band · trajectory velocity
│   ├── behavioral.py           # Recency decay · response rate · notice period
│   ├── composite.py            # 0.40/0.35/0.25 weighted combination
│   ├── cross_encoder.py        # ms-marco-MiniLM-L-6-v2 pairwise reranker
│   ├── honeypot_filter.py      # Runtime honeypot removal (pre-CE)
|   ├── trajectory.py           # Runtime career trajectory analysis
|   └── llm_reranker.py         # Runtime LLM Reranking top candidates
│
├── trust/
│   ├── advocate.py             # Scans for positive signals → HIGH/MEDIUM/LOW
│   ├── skeptic.py              # Scans for risk flags → HIGH/MODERATE/LOW
│   ├── verdict.py              # Synthesises ROBUST / CONTESTED / FRAGILE
│   └── reasoning_generator.py  # Template-based, fact-grounded reasoning strings
│
├── scripts/
│   └── validate_output.py      # Local submission validator
│
├── tests/
│   ├── conftest.py
│   ├── test_retrieval.py
│   └── test_scoring.py
│
└── data/
    ├── candidates.jsonl          # Full 100K candidate pool
    └── indexes/                  # Built by precompute.py
        ├── faiss.index
        ├── bm25.pkl
        ├── candidate_ids.npy
        ├── features_ids.npy
        ├── features.npy
        ├── trajectory.npy
        ├── trajectory_ids.npy
        └── honeypots.pkl
```

---

## Setup & Installation

**Requirements:** Python 3.10+, CPU only, ≤ 16 GB RAM(16 GB RAM preferred).

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/ragnarok.git
cd ragnarok
pip install -r requirements.txt
```

### 2. Download language models (once, requires network)

```bash
# spaCy model for JD parsing
python -m spacy download en_core_web_sm

# Sentence-transformers models (bi-encoder + cross-encoder)
python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('all-MiniLM-L6-v2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
"

python precompute_llm_addition.py
```
Models are cached in `~/.cache/huggingface/` after first download. During ranking (no-network window), they load from cache.


### 3. Place your data

```bash
# Full dataset (from hackathon bundle)
cp /path/to/candidates.jsonl data/candidates.jsonl

# Job description is already in the repo
ls job_description.md
```

### 4. Mount the precomputed indeses from google drive or precompute them locally.

Google-Drive Link : [https://drive.google.com/drive/folders/1TwEprdrDkvJUufSJXA8ZnPxSTYVYQUDd?usp=drive_link](https://drive.google.com/drive/folders/1TwEprdrDkvJUufSJXA8ZnPxSTYVYQUDd?usp=drive_link)
Place them in the ```/data/indexes``` folder.

OR

Precompute them locally
 
```bash
python -m build_indexes
```

---

## Quickstart

**One command to produce `output/submission.csv`** (after pre-computation):

```bash
python rank.py
```

With explicit paths:

```bash
python rank.py \
  --input data/candidates.jsonl \
  --output output/submission.csv \
  --top-k 100
```



Validating Output:

```bash
python scripts/validate_output.py output/submission.csv
```

---

## Two-Phase Execution

RAGnarok is explicitly designed as a two-phase system. The 5-minute ranking constraint applies **only to Phase 2**.

### Phase 1 — Pre-computation (run once, offline)

Builds all indexes from the raw candidate pool. This step can take longer than 5 minutes and requires network access to download models on first run.

```bash
python precompute.py --candidates data/candidates.jsonl
```

What this builds (saved to `data/indexes/`):

| Artifact | Size | Description |
|----------|------|-------------|
| `faiss.index` | ~150 MB | IVF256 dense vector index over all profiles |
| `candidate_ids.npy` | ~3 MB | Candidate ID ↔ FAISS row mapping |
| `bm25.pkl` | ~30 MB | BM25 inverted index with ontology expansion |
| `features.npy` | ~50 MB | 23 behavioural signal vectors |
| `trajectory.npy` | ~20 MB | Career pattern index |
| `honeypots.pkl` | < 1 MB | Pre-flagged honeypot candidate set |

### Phase 2 — Ranking (≤ 5 minutes, no network)

```bash
python rank.py --input data/candidates.jsonl --output output/submission.csv
```

All indexes load from disk. No API calls. Runs fully offline.

---

## Pipeline Stages

The `PipelineRunner` in `pipeline/runner.py` executes these stages in order:

| # | Stage | Module | Output |
|---|-------|--------|--------|
| 1 | **Honeypot filter** | `indexing/honeypot_registry.py` | Flags impossible profiles in-place |
| 2 | **Load indexes** | `indexing/faiss_builder.py` etc. | FAISS, BM25, feature store in memory |
| 3 | **Five retrieval paths** | `retrieval/` | ~200 raw candidate results across all paths |
| 4 | **RRF fusion** | `retrieval/rrf_fusion.py` | Deduplicated top-150 pool |
| 5 | **Cross-encoder rerank** | `scoring/cross_encoder.py` | Top-120 with CE scores blended in |
| 5b | **LLM rerank** | `scoring/llm_reranker.py` | Top-300 scored by Qwen2.5-1.5B; llm_score attached |
| 6 | **Composite scoring** | `scoring/composite.py` | 0.60·weighted + 0.25·CE + 0.15·LLM |
| 7 | **Trust layer** | `trust/verdict.py` | `ROBUST` / `CONTESTED` / `FRAGILE` verdicts |
| 8 | **Reasoning generation** | `trust/reasoning_generator.py` | 1–2 sentence per-candidate reasoning |
| 9 | **Assemble output** | `pipeline/runner.py` | `RankedCandidate` list |
| 10 | **Write + validate CSV** | `rank.py` | `submission.csv` |

---

## Scoring Model

### Composite score formula

```
final_score = 0.40 × skill_match
            + 0.30 × career_quality
            + 0.20 × behavioral
            + 0.10 × trajectory
            × uncertainty_penalty        (0.70–1.00 for sparse profiles)
```

The cross-encoder score is blended in at 30% weight inside `scoring/composite.py`:

```
blended = 0.70 × weighted_sum + 0.30 × cross_encoder_score
```

### Skill Match Score (`weight: 0.40`)

- Required skills count **2×**, nice-to-have skills count **1×**
- Proficiency multipliers: `beginner 0.40 · intermediate 0.65 · advanced 0.85 · expert 1.00`
- Duration trust factor: skills with 0 months get a 0.50 floor; full trust at 24+ months
- Ontology partial credit: adjacent skills earn **0.60×** of full credit
- Endorsement boost: log-scaled, capped at 50 endorsements, max additive boost 0.10

### Career Quality Score (`weight: 0.30`)

- Product-company bonus: **1.20×** for software, fintech, SaaS, food-tech, AI/ML, etc.
- Consulting-only penalty: **0.30×** if entire career is at TCS/Wipro/Infosys/Accenture et al.
- YOE band: ideal 5–9 years; scores taper outside `[4, 12]`
- Trajectory velocity: promotions/year, percentile-normalised across pool
- Location bonus: Noida/Pune `+0.08`, Delhi/Gurgaon `+0.05`, Hyderabad/Mumbai `+0.04`, Bangalore `+0.03`

### Behavioral Score (`weight: 0.20`)

| Signal | Weight |
|--------|--------|
| `last_active_date` recency (exponential decay λ=0.005) | 0.25 |
| `recruiter_response_rate` | 0.20 |
| `open_to_work_flag` | 0.15 |
| `notice_period_days` (ideal ≤ 30) | 0.15 |
| `github_activity_score` | 0.10 |
| `profile_completeness_score` | 0.10 |
| `interview_completion_rate` | 0.05 |

### Trajectory Score (`weight: 0.10`)
---

## Honeypot Detection

Five rules are evaluated offline during pre-computation and cached in `data/indexes/honeypots.pkl`. At ranking time, flagged candidates are removed before the cross-encoder in O(1) lookup.

| Rule | Signal |
|------|--------|
| **Experience vs. founding date** | `years_at_company > (now - company_founded)` |
| **Expert skill with zero duration** | `proficiency = "expert"` AND `duration_months < 1` |
| **Salary range impossibility** | `expected_salary_min > expected_salary_max` |
| **Profile stuffing** | `completeness_score < 35` AND `skill_count > 15` |
| **YOE discrepancy** | `|sum(duration_months)/12 − years_of_experience| > 4 years` |

Target: honeypot rate < 5% in top-100 (well below the spec's 10% disqualification threshold).

---

## Trust Layer & Reasoning

Every candidate in the top-100 gets an `Advocate → Skeptic → Verdict → Reasoning` pass.

### Advocate scan
Checks for: required skill coverage, product-company trajectory, assessment scores > 70, recent activity. Tags each signal `HIGH / MEDIUM / LOW` confidence.

### Skeptic scan
Checks for: missing required skills, consulting-only history, `last_active > 90 days`, `notice_period > 60 days`, low recruiter response rate. Tags each `HIGH / MODERATE / LOW` risk.

### Verdict synthesis

| Verdict | Condition |
|---------|-----------|
| `ROBUST` | 0 HIGH risks |
| `CONTESTED` | 1 HIGH risk |
| `FRAGILE` | ≥ 2 HIGH risks |

### Reasoning output format

```
"7 years applied ML at product companies (Swiggy, Razorpay); strong FAISS + embedding retrieval match;
 notice period 45 days is above ideal. Verdict: CONTESTED — would flip to FRAGILE if response
 rate confirmed < 15%."
```

Every sentence references a field that was read from the profile. No hallucination is structurally possible.

---

## Runtime Budget

Measured on an 8-core CPU machine with 16 GB RAM. All times are for the ranking phase only (indexes pre-built).

| Stage | Target | Notes |
|-------|--------|-------|
| Load FAISS index | ~2s | IVF256, ~150 MB |
| Load BM25 + feature store | ~6s | Pickle + numpy |
| Encode JD query | ~0.1s | 1 sentence → 384-dim vector |
| Five retrieval paths | ~1.5s | All paths combined |
| RRF fusion | ~0.05s | Pure Python |
| Honeypot filter | ~0.01s | O(1) set lookup |
| Cross-encoder (120 candidates) | ~10s | ~80ms × 120 |
| Composite scoring | ~0.5s | Vectorised numpy |
| Trust layer (top-100) | ~1s | Rule-based, no model |
| LLM Reasoning | ~200s | From Trust Layer |
| CSV write + validate | ~0.1s | |
| **Total estimate** | **~230s** | **Well within 5-min budget** |

---

## Configuration Reference

All tunable parameters live in `config.py`. Never hardcode values in other modules.

```python
# Retrieval pool sizes
SEMANTIC_PATH_TOP_K   = 350
KEYWORD_PATH_TOP_K    = 350 
ONTOLOGY_PATH_TOP_K   = 280
TRAJECTORY_PATH_TOP_K = 220
SIGNAL_PATH_TOP_K     = 200 
RRF_POOL_SIZE         = 800
CROSS_ENCODER_TOP_K   = 300

# Scoring weights (must sum to 1.0)
WEIGHT_SKILL          = 0.40
WEIGHT_CAREER         = 0.30
WEIGHT_BEHAVIORAL     = 0.20
WEIGHT_TRAJECTORY     = 0.10

# Cross-encoder blend
CE_BLEND_FACTOR       = 0.30

# RRF bonuses for rescue paths
RRF_ONTOLOGY_PATH_BONUS = 1.3
RRF_SIGNAL_PATH_BONUS   = 1.1

# Consulting-only hard penalty
CONSULTING_ONLY_PENALTY = 0.35

# Sparse profile uncertainty floor
UNCERTAINTY_PENALTY_FLOOR = 0.70
```

See `config.py` for the full reference including all thresholds, firm lists, location bonuses, and behavioural sub-weights.

---

## Running Tests

```bash
# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=term-missing

# Specific test files
pytest tests/test_honeypot.py -v
pytest tests/test_scoring.py -v
pytest tests/test_e2e.py -v
```

---

## Submission Checklist

Before uploading to the portal, verify each item:

- [ ] `output/submission.csv` has exactly 100 data rows (plus header)
- [ ] Ranks are integers 1–100, each appearing exactly once
- [ ] Scores are monotonically non-increasing with rank
- [ ] All `candidate_id` values match `CAND_XXXXXXX` format and exist in `candidates.jsonl`
- [ ] No duplicate `candidate_id` entries
- [ ] Reasoning column is non-empty for all 100 rows
- [ ] Each reasoning references specific, verifiable facts from the profile
- [ ] Honeypot rate in top-100 verified < 10%
- [ ] `python rank.py` runs end-to-end on a clean machine within 5 minutes
- [ ] `submission_metadata.yaml` is filled in at repo root
- [ ] Sandbox / demo link is live and accepts a small candidate sample

Run the local validator:

```bash
python scripts/validate_output.py output/submission.csv
```

---

## LLM Reranker

RAGnarok includes a lightweight local LLM scoring stage (`scoring/llm_reranker.py`) that runs after RRF fusion on the top-300 candidates. It uses `Qwen2.5-1.5B-Instruct` in 4-bit quantised GGUF format via `llama-cpp-python` — no GPU, no network, ~1 GB RAM.

### How it fits in the pipeline

```
RRF pool (top-150)
       │
       ▼
Cross-Encoder (top-120)       ~10s
       │
       ▼
LLM Reranker  (top-300)       ~75s   ← new stage
       │
       ▼
Composite Scorer
```


### Key files

| File | Purpose |
|------|---------|
| `scoring/llm_reranker.py` | `LLMReranker` class — load, score_one, score_pool |
| `config_llm_additions.py` | Constants to paste into `config.py` |
| `runner_llm_patch.py` | Exact diff showing where to insert LLM block in `runner.py` |
| `precompute_llm_addition.py` | `download_llm_model()` to add to `precompute.py` |
| `test_llm_reranker.py` | Standalone smoke test with mock schemas |

### Tuning

All LLM parameters live in `config.py`:

```python
LLM_MODEL_PATH       = "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_TOP_N            = 50     # candidates to score
LLM_N_THREADS        = 8       # match your CPU core count
LLM_N_CTX            = 512     # keep small — prompts are ~150 tokens
LLM_BLEND_FACTOR     = 0.15    # weight in composite
LLM_RERANKER_ENABLED = True    # set False to skip (e.g. in unit tests)
```

### RAM budget with LLM added

| Component | RAM |
|-----------|-----|
| Qwen2.5-1.5B Q4 GGUF | ~1.0 GB |
| FAISS + BM25 + indexes | ~250 MB |
| Candidate objects (100K) | ~800 MB |
| PyTorch + sentence-transformers | ~500 MB |
| OS + Python baseline | ~300 MB |
| **Total peak** | **~2.9 GB** |

Well within the 16 GB constraint.

---

## Dependencies

Key packages (see `requirements.txt` for pinned versions):

| Package | Version | Role |
|---------|---------|------|
| `sentence-transformers` | 3.4.1 | Bi-encoder + cross-encoder |
| `torch` | 2.5.1 | CPU-only backend |
| `faiss-cpu` | 1.8.0 | Dense vector index |
| `rank-bm25` | 0.2.2 | Sparse keyword retrieval |
| `spacy` | 3.7.4 | JD parsing |
| `numpy` | 1.26.4 | Feature arrays |
| `pandas` | 2.2.3 | CSV output |
| `fastapi` | 0.115.14 | Optional sandbox API |

---

*Built for the Redrob Intelligent Candidate Discovery & Ranking Hackathon.*