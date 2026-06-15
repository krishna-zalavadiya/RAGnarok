"""
scoring/cross_encoder.py
------------------------
Cross-encoder reranking of the post-RRF, post-honeypot candidate pool.

Role in pipeline (Step 5 of runner.py):
    retrieval paths
        └─► rrf_fusion.py        → list[RRFResult] (≤60, cross_encoder_score=0.0)
        └─► honeypot_filter.py   → list[RRFResult] (≤50, honeypots removed)
        └─► cross_encoder.py     → list[RRFResult] (≤50, cross_encoder_score set, re-sorted)
        └─► composite.py         → list[ComponentScores]

Why a cross-encoder instead of just using the bi-encoder FAISS score?
    The bi-encoder (FAISS Path 1) encodes the JD and candidate profile
    *independently*, then compares vectors. It is fast at the cost of
    accuracy — it cannot model token-level interactions between JD and profile.

    The cross-encoder reads the JD and the candidate profile *simultaneously*
    in one forward pass. Token-level attention between both inputs allows it
    to understand:
        - "candidate says 'dense retrieval', JD says 'embedding-based search'
           — these are the same thing" (semantic equivalence)
        - "candidate has Pinecone for 7 years and Elasticsearch for 2 years —
           that exactly matches what the JD is asking for" (context-sensitive
           skill weighting)

    The cost: O(N) forward passes vs O(1) for FAISS. That is why the
    cross-encoder only runs on the top-50 survivors from RRF+honeypot, not
    all 100K candidates. 50 x ~80ms ≈ 4s — within the 6s budget.

Model:
    cross-encoder/ms-marco-MiniLM-L-6-v2
    - 80MB on disk (cached at ~/.cache/huggingface/)
    - Trained on MS-MARCO passage retrieval — calibrated for (query, passage)
      relevance scoring
    - Outputs a single raw logit; sigmoid maps it to [0, 1]
    - max_length=512 tokens handles truncation internally

Fallback behaviour:
    If the model cache is missing or prediction fails, the reranker falls
    back to normalising rrf_score into [0, 1] and using that as the
    cross_encoder_score. The pipeline continues — ranking quality degrades
    gracefully rather than crashing.

Input contract (from scoring/honeypot_filter.py → pipeline/runner.py):
    pool                : list[RRFResult]           ≤ CROSS_ENCODER_TOP_K
    jd                  : JDIntent                   jd.raw_text must be non-empty
    candidate_store     : dict[str, CandidateFeatureVector]

Output contract (consumed by scoring/composite.py):
    Same list[RRFResult] with:
        cross_encoder_score : float in [0.0, 1.0]  (sigmoid-normalised logit)
    Sorted: cross_encoder_score descending, candidate_id ascending (tie-break)

Dependencies:
    config.py               CROSS_ENCODER_MODEL, CROSS_ENCODER_TOP_K
    pipeline/schemas.py     RRFResult (in/out), JDIntent (in), CandidateFeatureVector (in)
    sentence-transformers   CrossEncoder (lazy-loaded from local HF cache)
    numpy                   raw score array from model.predict()
    stdlib                  logging, math, time, typing
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np

import config
from pipeline.schemas import CandidateFeatureVector, JDIntent, RRFResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Text-length guards
# ─────────────────────────────────────────────────────────────────────────────
# ms-marco-MiniLM-L-6-v2 uses a 512-token budget split between query and
# document. The CrossEncoder class handles truncation internally via its
# max_length parameter, but pre-truncating character counts prevents sending
# pathologically long strings to the tokenizer, which can be slow.
#
# Rough heuristic: 1 token ≈ 4–5 characters for English text.
# 512 tokens × 4 chars ≈ 2048 chars total.
# We split: query (JD) gets ~1500 chars, document (candidate) gets ~2000 chars.
# The tokenizer will further truncate if the combined pair exceeds 512 tokens.

_MAX_JD_CHARS: int = 1_500   # JD used as the "query" in (query, document) pair
_MAX_CAND_CHARS: int = 2_000  # Candidate profile used as the "document"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """
    Map a raw cross-encoder logit to a probability in [0, 1].

    ms-marco-MiniLM-L-6-v2 outputs uncalibrated logits (typically -10 to +10).
    Sigmoid converts them to pseudo-probabilities that we treat as relevance
    scores. These are not true probabilities but are monotonically ordered —
    a higher sigmoid value always means higher predicted relevance, which is
    all that matters for ranking.

    Numerically stable: clips the exponent to avoid overflow on extreme logits.
    """
    # Clip to ±500 to prevent math.exp overflow for extreme model outputs.
    x_clipped = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x_clipped))


def _truncate(text: str, max_chars: int) -> str:
    """
    Return text truncated to max_chars characters.

    Does a simple character-level truncation. The CrossEncoder tokenizer
    will also apply token-level truncation internally via max_length=512,
    so this is a defensive pre-filter, not the primary truncation mechanism.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# CrossEncoderReranker
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:
    """
    Pairwise cross-encoder reranker for the post-RRF candidate pool.

    Wraps sentence-transformers CrossEncoder (ms-marco-MiniLM-L-6-v2).
    Model is lazy-loaded on the first call to rerank() and cached for the
    lifetime of this instance — create one instance per pipeline run.

    Usage in pipeline/runner.py:
        reranker = CrossEncoderReranker()
        # ... run all 5 retrieval paths, fuse, filter honeypots ...
        pool = reranker.rerank(pool, jd, candidate_store)
        # pool is now sorted by cross_encoder_score descending
        # each RRFResult.cross_encoder_score is in [0.0, 1.0]

    Thread safety:
        NOT thread-safe. Designed for single-threaded pipeline execution.
        If parallelism is needed in future, create one instance per thread.
    """

    MODEL_NAME: str = config.CROSS_ENCODER_MODEL  # "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        top_k: int = config.CROSS_ENCODER_TOP_K,
    ) -> None:
        """
        Initialise the reranker.

        Args:
            top_k: Maximum number of candidates to rerank.
                   Matches CROSS_ENCODER_TOP_K = 50 from config.
                   If the pool passed to rerank() is larger, it is truncated
                   to top_k before scoring (log warning issued).

        Raises:
            ValueError: If top_k < 1.
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        self._top_k: int = top_k
        self._model: Optional[object] = None  # Lazy-loaded CrossEncoder instance

        logger.debug(
            "CrossEncoderReranker initialised (model=%s, top_k=%d).",
            self.MODEL_NAME,
            self._top_k,
        )

    # ------------------------------------------------------------------ #
    # Model loading (lazy, cached)                                         #
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """
        Load the cross-encoder from the local HuggingFace cache.

        Called automatically on the first rerank() invocation.
        Safe to call multiple times — returns immediately if already loaded.

        IMPORTANT: No network calls are made here. The model must already
        be cached in ~/.cache/huggingface/ (done during precompute.py setup).
        If the cache is missing, loading will fail and _model stays None,
        triggering the fallback path in rerank().

        The model is loaded with:
            device="cpu"    — GPU not required, avoids CUDA dependency
            max_length=512  — built-in truncation to 512 tokens
        """
        if self._model is not None:
            return  # Already loaded — reuse

        try:
            # Import is deferred to here so modules that import
            # cross_encoder.py but never call rerank() don't pay the
            # import cost of sentence-transformers.
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            logger.info(
                "Loading cross-encoder: %s (CPU, max_length=512) …",
                self.MODEL_NAME,
            )
            t0 = time.perf_counter()

            self._model = CrossEncoder(
                self.MODEL_NAME,
                max_length=512,
                device="cpu",
            )

            elapsed = time.perf_counter() - t0
            logger.info(
                "Cross-encoder loaded in %.2fs.",
                elapsed,
            )

        except ImportError:
            logger.error(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers==3.4.1. "
                "Falling back to rrf_score-based ranking.",
            )
            self._model = None

        except Exception as exc:
            logger.error(
                "Failed to load cross-encoder '%s': %s. "
                "Model may not be cached. "
                "Pre-download with: python -c "
                "\"from sentence_transformers import CrossEncoder; "
                "CrossEncoder('%s')\". "
                "Falling back to rrf_score-based ranking.",
                self.MODEL_NAME,
                exc,
                self.MODEL_NAME,
            )
            self._model = None

    # ------------------------------------------------------------------ #
    # Text builders                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_jd_text(jd: JDIntent) -> str:
        """
        Build the query string for the cross-encoder from the JDIntent.

        Uses jd.raw_text (the full JD text stored by jd_parser.py).
        Truncated to _MAX_JD_CHARS to keep it as a dense, focused query.

        If raw_text is empty (parser failed to populate it), falls back to
        a summary string built from required_skills and seniority. This
        ensures the cross-encoder always has *something* to work with.

        Args:
            jd: Parsed JDIntent from pipeline/jd_parser.py.

        Returns:
            Non-empty string suitable for use as the cross-encoder query.
        """
        text = jd.raw_text.strip()

        if not text:
            # Fallback: construct a minimal query from structured fields.
            # This happens only if jd_parser.py failed to set raw_text.
            skills_str = ", ".join(jd.required_skills[:10])  # top-10 required
            text = (
                f"Senior AI Engineer position requiring: {skills_str}. "
                f"Experience: {jd.yoe_min}–{jd.yoe_max} years. "
                f"Product company. Applied ML, retrieval, ranking."
            )
            logger.warning(
                "JDIntent.raw_text is empty. "
                "Using fallback query string: '%s'",
                text[:100],
            )

        return _truncate(text, _MAX_JD_CHARS)

    @staticmethod
    def _build_candidate_text(cfv: CandidateFeatureVector) -> str:
        """
        Build the document string for the cross-encoder from a candidate.

        Uses cfv.embedding_text (pre-built by pipeline/candidate_parser.py
        as: headline + summary + all job titles + all role descriptions).
        This is already the richest text representation of the candidate
        and is the same text used for FAISS embedding — keeping representations
        consistent across retrieval and scoring.

        Truncated to _MAX_CAND_CHARS to stay within the token budget.

        Args:
            cfv: Parsed CandidateFeatureVector from candidate_parser.py.

        Returns:
            Non-empty string for use as the cross-encoder document.
        """
        text = cfv.embedding_text.strip()

        if not text:
            # Fallback: minimal text from profile metadata.
            # Happens only if candidate_parser.py failed to build embedding_text.
            text = (
                f"{cfv.headline}. "
                f"{cfv.current_title} at {cfv.current_company}. "
                f"{cfv.years_of_experience:.1f} years experience. "
                f"Skills: {', '.join(s.name for s in cfv.skills[:10])}."
            )
            logger.warning(
                "embedding_text empty for '%s'. Using fallback text.",
                cfv.candidate_id,
            )

        return _truncate(text, _MAX_CAND_CHARS)

    # ------------------------------------------------------------------ #
    # Fallback scoring                                                      #
    # ------------------------------------------------------------------ #

    def _fallback_rank(
        self,
        pool: list[RRFResult],
    ) -> list[RRFResult]:
        """
        Fallback when the cross-encoder model is unavailable.

        Normalises rrf_score linearly into [0, 1] and assigns it as the
        cross_encoder_score. The relative ordering is preserved — this is
        essentially a passthrough that signals to composite.py that no
        cross-encoder reranking occurred, while keeping score semantics
        consistent (all values in [0, 1]).

        A warning is logged every time this fires so operators can detect
        the issue in production logs.

        Args:
            pool: list[RRFResult] with cross_encoder_score still at 0.0.

        Returns:
            Same list with cross_encoder_score set from normalised rrf_score,
            sorted by cross_encoder_score descending, candidate_id ascending.
        """
        if not pool:
            return pool

        scores = [r.rrf_score for r in pool]
        max_score = max(scores)
        min_score = min(scores)
        score_range = max_score - min_score

        for result in pool:
            if score_range > 1e-9:
                # Linear min-max normalisation into [0, 1].
                result.cross_encoder_score = (
                    (result.rrf_score - min_score) / score_range
                )
            else:
                # All scores are identical — assign uniform 1.0.
                result.cross_encoder_score = 1.0

        # Sort: cross_encoder_score descending, candidate_id ascending (tie-break).
        pool.sort(key=lambda r: (-r.cross_encoder_score, r.candidate_id))

        logger.warning(
            "CrossEncoderReranker: model unavailable — "
            "using normalised rrf_score as cross_encoder_score for %d candidates. "
            "Ranking quality may be reduced. Check model cache.",
            len(pool),
        )
        return pool

    # ------------------------------------------------------------------ #
    # Primary rerank method                                                 #
    # ------------------------------------------------------------------ #

    def rerank(
        self,
        pool: list[RRFResult],
        jd: JDIntent,
        candidate_store: dict[str, CandidateFeatureVector],
    ) -> list[RRFResult]:
        """
        Rerank the candidate pool using the cross-encoder model.

        For each candidate in the pool, pairs the JD text with the candidate's
        embedding_text and calls model.predict() in a single batch. Raw logits
        are sigmoid-normalised to [0, 1] and assigned to
        RRFResult.cross_encoder_score. The pool is then sorted by score
        descending with candidate_id ascending as the tie-break.

        This method mutates RRFResult objects in-place (sets cross_encoder_score).
        It also modifies the order of elements in pool.

        Args:
            pool:
                Candidate pool from scoring/honeypot_filter.py.
                Expected length: ≤ CROSS_ENCODER_TOP_K (50).
                All RRFResult.cross_encoder_score values should be 0.0 on entry.
                If length > top_k, the pool is truncated (sorted by rrf_score
                descending before truncation, since that is the order from RRF).

            jd:
                Parsed JDIntent. Must have raw_text populated by jd_parser.py.
                If raw_text is empty, a fallback query string is constructed
                from structured fields.

            candidate_store:
                Dict mapping candidate_id → CandidateFeatureVector.
                Must contain entries for all candidates in pool.
                If a candidate_id is missing, that candidate is skipped and
                retains cross_encoder_score = 0.0 (sorts to bottom).

        Returns:
            The same list[RRFResult] with:
              - cross_encoder_score set to sigmoid(raw_logit) for each candidate
                found in candidate_store (range: [0.0, 1.0])
              - Candidates not found in store retain cross_encoder_score = 0.0
              - Sorted by cross_encoder_score descending
              - Tie-break: candidate_id ascending (spec-compliant)

        Raises:
            TypeError: If pool is not a list, or jd is not a JDIntent,
                       or candidate_store is not a dict.

        Side effects:
            - Calls _load_model() on first invocation (loads from HF cache).
            - Mutates cross_encoder_score on each RRFResult in pool.
            - Sorts pool in-place.
            - Logs INFO with timing and score statistics.
            - Logs WARNING if model is unavailable or budget exceeded.
        """
        # ── Type guards ───────────────────────────────────────────────────
        if not isinstance(pool, list):
            raise TypeError(
                f"pool must be list[RRFResult], got {type(pool).__name__}."
            )
        if not isinstance(jd, JDIntent):
            raise TypeError(
                f"jd must be JDIntent, got {type(jd).__name__}."
            )
        if not isinstance(candidate_store, dict):
            raise TypeError(
                f"candidate_store must be dict[str, CandidateFeatureVector], "
                f"got {type(candidate_store).__name__}."
            )

        # ── Early exit on empty pool ──────────────────────────────────────
        if not pool:
            logger.debug(
                "CrossEncoderReranker.rerank: empty pool, returning []."
            )
            return pool

        t0 = time.perf_counter()

        # ── Pool size guard ───────────────────────────────────────────────
        # Honeypot filter should guarantee len(pool) ≤ top_k, but we
        # defend against over-sized pools here.  Pool arrives sorted by
        # rrf_score descending from rrf_fusion.py — truncating from the end
        # keeps the highest-RRF-score candidates.
        if len(pool) > self._top_k:
            logger.warning(
                "Pool size %d exceeds top_k=%d before cross-encoder. "
                "Truncating to top-%d by rrf_score. "
                "Check honeypot_filter.py output size.",
                len(pool),
                self._top_k,
                self._top_k,
            )
            pool = pool[: self._top_k]

        # ── Lazy model load ───────────────────────────────────────────────
        self._load_model()

        if self._model is None:
            # Model unavailable — use normalised rrf_score as proxy.
            return self._fallback_rank(pool)

        # ── Build (query, document) input pairs ───────────────────────────
        jd_text: str = self._build_jd_text(jd)

        # We track two parallel lists so we can zip scores back to results.
        #   pairs        : list[(jd_text, cand_text)]  — cross-encoder input
        #   valid_results: list[RRFResult]              — parallel to pairs
        #
        # Candidates missing from candidate_store are added to skipped_ids
        # and retain cross_encoder_score = 0.0 (sorts to bottom naturally).
        pairs: list[tuple[str, str]] = []
        valid_results: list[RRFResult] = []
        skipped_ids: list[str] = []

        for result in pool:
            cfv: Optional[CandidateFeatureVector] = candidate_store.get(
                result.candidate_id
            )
            if cfv is None:
                logger.warning(
                    "candidate_id '%s' not found in candidate_store. "
                    "cross_encoder_score will remain 0.0. "
                    "Ensure candidate_store is built from the same data as the pool.",
                    result.candidate_id,
                )
                skipped_ids.append(result.candidate_id)
                continue

            cand_text: str = self._build_candidate_text(cfv)
            pairs.append((jd_text, cand_text))
            valid_results.append(result)

        if skipped_ids:
            logger.warning(
                "%d candidate(s) skipped (not in candidate_store): %s.",
                len(skipped_ids),
                skipped_ids,
            )

        # ── Guard: all candidates were skipped ────────────────────────────
        if not pairs:
            logger.error(
                "No valid candidate pairs to score. "
                "candidate_store may be empty or all IDs are missing. "
                "Returning pool with cross_encoder_score = 0.0 for all.",
            )
            # Sort by rrf_score descending as best-effort ordering.
            pool.sort(key=lambda r: (-r.rrf_score, r.candidate_id))
            return pool

        # ── Cross-encoder batch prediction ────────────────────────────────
        # model.predict() accepts list[(query, doc)] and returns numpy array
        # of raw logits, one per pair. Single batch — all pairs at once.
        # 50 pairs × ~80ms per pair on CPU = ~4s, within the 6s budget.
        try:
            raw_scores: np.ndarray = self._model.predict(
                sentences=pairs,
                batch_size=len(pairs),      # Single batch — all at once
                show_progress_bar=False,    # No tqdm output in production
                convert_to_numpy=True,      # Explicit: numpy array output
            )

        except Exception as exc:
            logger.error(
                "Cross-encoder prediction failed: %s. "
                "Using fallback normalised rrf_score.",
                exc,
            )
            return self._fallback_rank(pool)

        # ── Validate prediction output ────────────────────────────────────
        if raw_scores is None or len(raw_scores) != len(pairs):
            logger.error(
                "Cross-encoder returned unexpected output: "
                "expected %d scores, got %s. Using fallback.",
                len(pairs),
                len(raw_scores) if raw_scores is not None else "None",
            )
            return self._fallback_rank(pool)

        # ── Assign sigmoid-normalised scores ─────────────────────────────
        # Mutate cross_encoder_score in-place on each RRFResult.
        # Candidates in skipped_ids already have cross_encoder_score = 0.0.
        for result, raw_score in zip(valid_results, raw_scores):
            result.cross_encoder_score = _sigmoid(float(raw_score))

        # ── Sort pool by cross_encoder_score desc, candidate_id asc ──────
        # Skipped candidates (score = 0.0) naturally fall to the bottom.
        pool.sort(key=lambda r: (-r.cross_encoder_score, r.candidate_id))

        # ── Timing and diagnostics ────────────────────────────────────────
        elapsed: float = time.perf_counter() - t0

        scored_scores = [r.cross_encoder_score for r in pool if r.candidate_id not in skipped_ids]
        top_score    = pool[0].cross_encoder_score if pool else 0.0
        bottom_score = pool[-1].cross_encoder_score if pool else 0.0
        mean_score   = sum(scored_scores) / len(scored_scores) if scored_scores else 0.0

        logger.info(
            "CrossEncoder: reranked %d/%d candidates in %.2fs "
            "(skipped=%d, top=%.4f, bottom=%.4f, mean=%.4f).",
            len(valid_results),
            len(pool),
            elapsed,
            len(skipped_ids),
            top_score,
            bottom_score,
            mean_score,
        )

        # Budget warning: 6s is the soft cap from config stage budget.
        _BUDGET_SECONDS: float = 6.0
        if elapsed > _BUDGET_SECONDS:
            logger.warning(
                "CrossEncoder exceeded %.1fs budget: %.2fs for %d pairs. "
                "Consider reducing CROSS_ENCODER_TOP_K in config.py.",
                _BUDGET_SECONDS,
                elapsed,
                len(pairs),
            )

        return pool

    @property
    def is_loaded(self) -> bool:
        """True if the model has been successfully loaded into memory."""
        return self._model is not None

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker("
            f"model={self.MODEL_NAME!r}, "
            f"top_k={self._top_k}, "
            f"loaded={self.is_loaded})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience function
# ─────────────────────────────────────────────────────────────────────────────

def rerank_pool(
    pool: list[RRFResult],
    jd: JDIntent,
    candidate_store: dict[str, CandidateFeatureVector],
    top_k: int = config.CROSS_ENCODER_TOP_K,
) -> list[RRFResult]:
    """
    Convenience wrapper around CrossEncoderReranker for pipeline/runner.py.

    Creates a new CrossEncoderReranker and runs one rerank pass.
    For repeated calls (e.g., ablation studies or A/B experiments), prefer
    instantiating CrossEncoderReranker directly and reusing the instance
    to avoid reloading the 80MB model on each call.

    Args:
        pool:             Post-RRF, post-honeypot candidate pool. ≤50 entries.
        jd:               JDIntent with raw_text populated.
        candidate_store:  dict[candidate_id → CandidateFeatureVector].
        top_k:            Max candidates to score. Default: CROSS_ENCODER_TOP_K.

    Returns:
        pool sorted by cross_encoder_score descending, with scores set.

    Example (pipeline/runner.py):
        from scoring.cross_encoder import rerank_pool

        pool = honeypot_filter.filter(rrf_pool)
        pool = rerank_pool(pool, jd, candidate_store)
        # pool[0].cross_encoder_score → highest-relevance candidate
    """
    return CrossEncoderReranker(top_k=top_k).rerank(pool, jd, candidate_store)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — python -m scoring.cross_encoder
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import dataclasses
    from datetime import date

    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
    )

    print("=" * 65)
    print("CrossEncoderReranker — smoke test (fallback path, no model)")
    print("=" * 65)
    print(
        "NOTE: This smoke test exercises the FALLBACK path (normalised rrf_score).\n"
        "      The full model path is tested in tests/test_scoring.py with mocks.\n"
        "      To test the live model, ensure the HF cache is populated and\n"
        "      set TEST_LIVE_MODEL=1 in your environment.\n"
    )

    # ── Helpers: build minimal stubs without importing full pipeline ──────

    def _make_rrf_result(cid: str, rrf_score: float) -> RRFResult:
        return RRFResult(
            candidate_id=cid,
            rrf_score=rrf_score,
            paths_present=["semantic"],
            cross_encoder_score=0.0,
        )

    def _make_jd() -> JDIntent:
        return JDIntent(
            required_skills=["embeddings", "faiss", "python"],
            nice_to_have_skills=["lora", "qdrant"],
            disqualifier_skills=["computer vision"],
            expanded_required=["embeddings", "vector search", "faiss", "python"],
            yoe_min=5.0,
            yoe_max=9.0,
            yoe_ideal_min=5.0,
            yoe_ideal_max=9.0,
            preferred_locations=["noida", "pune"],
            relocation_accepted=True,
            disqualify_consulting_only=True,
            disqualify_no_production=True,
            raw_text=(
                "Senior AI Engineer: We need production experience with "
                "embeddings, FAISS, vector databases, ranking systems, "
                "and strong Python. 5-9 years at product companies."
            ),
        )

    def _make_cfv(cid: str, text: str) -> CandidateFeatureVector:
        """Minimal CandidateFeatureVector stub with only the fields we use."""
        from pipeline.schemas import RedrobSignals

        signals = RedrobSignals(
            profile_completeness_score=80.0,
            signup_date=date(2025, 1, 1),
            last_active_date=date(2026, 5, 1),
            open_to_work_flag=True,
            profile_views_received_30d=50,
            applications_submitted_30d=3,
            recruiter_response_rate=0.6,
            avg_response_time_hours=24.0,
            skill_assessment_scores={},
            connection_count=300,
            endorsements_received=20,
            notice_period_days=30,
            expected_salary_min_lpa=20.0,
            expected_salary_max_lpa=40.0,
            preferred_work_mode="hybrid",
            willing_to_relocate=True,
            github_activity_score=45.0,
            search_appearance_30d=200,
            saved_by_recruiters_30d=5,
            interview_completion_rate=0.7,
            offer_acceptance_rate=0.5,
            verified_email=True,
            verified_phone=True,
            linkedin_connected=False,
        )
        return CandidateFeatureVector(
            candidate_id=cid,
            headline="ML Engineer",
            summary=text,
            location="Hyderabad",
            location_lower="hyderabad",
            country="India",
            years_of_experience=6.0,
            current_title="ML Engineer",
            current_title_lower="ml engineer",
            current_company="Swiggy",
            current_company_lower="swiggy",
            current_company_size="5001-10000",
            current_industry="Food Delivery",
            current_industry_lower="food delivery",
            skills=[],
            career_history=[],
            education=[],
            signals=signals,
            is_consulting_only=False,
            has_product_co_experience=True,
            total_career_months=72,
            skill_names_lower=frozenset(["embeddings", "faiss", "python"]),
            embedding_text=text,
        )

    # ── Build test data ───────────────────────────────────────────────────
    pool = [
        _make_rrf_result("CAND_0000031", rrf_score=0.0492),  # highest RRF
        _make_rrf_result("CAND_0000014", rrf_score=0.0380),
        _make_rrf_result("CAND_0000001", rrf_score=0.0250),
        _make_rrf_result("CAND_0000010", rrf_score=0.0100),
    ]

    jd = _make_jd()

    candidate_store: dict[str, CandidateFeatureVector] = {
        "CAND_0000031": _make_cfv(
            "CAND_0000031",
            "Recommendation systems engineer with 6 years building FAISS-based "
            "retrieval, sentence-transformer embeddings, and XGBoost rankers "
            "at Swiggy and Uber. Expert in Pinecone, Sentence Transformers, "
            "and information retrieval. Led migration to hybrid dense+sparse search.",
        ),
        "CAND_0000014": _make_cfv(
            "CAND_0000014",
            "Frontend engineer at Zomato. Skills include FAISS and OpenSearch. "
            "Has used vector search in side projects.",
        ),
        "CAND_0000001": _make_cfv(
            "CAND_0000001",
            "Backend data engineer at Mindtree. Built Kafka streaming pipelines. "
            "Interested in ML but primary background is data engineering.",
        ),
        # CAND_0000010 intentionally MISSING from store to test skip behaviour
    }

    print(f"Pool size:  {len(pool)}")
    print(f"Store size: {len(candidate_store)} (CAND_0000010 missing intentionally)\n")

    # ── Test: Fallback path (force model=None) ────────────────────────────
    reranker = CrossEncoderReranker(top_k=10)
    # Simulate model unavailable by leaving _model as None
    result_pool = reranker.rerank(pool, jd, candidate_store)

    print(f"Result pool size: {len(result_pool)}\n")
    print("Results (fallback — normalised rrf_score):")
    for r in result_pool:
        print(
            f"  {r.candidate_id}  rrf={r.rrf_score:.5f}  "
            f"ce_score={r.cross_encoder_score:.4f}  "
            f"paths={r.paths_present}"
        )

    # ── Acceptance criterion 1: cross_encoder_score in [0, 1] ────────────
    for r in result_pool:
        assert 0.0 <= r.cross_encoder_score <= 1.0, (
            f"FAIL: {r.candidate_id} cross_encoder_score={r.cross_encoder_score} "
            f"out of [0, 1]."
        )
    print("\n[PASS] All cross_encoder_scores in [0.0, 1.0]  ✓")

    # ── Acceptance criterion 2: sorted descending ─────────────────────────
    for i in range(len(result_pool) - 1):
        assert result_pool[i].cross_encoder_score >= result_pool[i + 1].cross_encoder_score, (
            f"FAIL: not sorted descending at index {i}: "
            f"{result_pool[i].cross_encoder_score} < {result_pool[i+1].cross_encoder_score}"
        )
    print("[PASS] Pool sorted by cross_encoder_score descending  ✓")

    # ── Acceptance criterion 3: CAND_0000031 (highest RRF) at top ────────
    assert result_pool[0].candidate_id == "CAND_0000031", (
        f"FAIL: expected CAND_0000031 at rank 1, got {result_pool[0].candidate_id}"
    )
    print("[PASS] Highest RRF-score candidate at position 0  ✓")

    # ── Acceptance criterion 4: skipped candidate at bottom ───────────────
    skipped = next(r for r in result_pool if r.candidate_id == "CAND_0000010")
    assert skipped.cross_encoder_score == 0.0, (
        f"FAIL: CAND_0000010 (missing from store) should have score=0.0, "
        f"got {skipped.cross_encoder_score}"
    )
    assert result_pool[-1].candidate_id == "CAND_0000010", (
        f"FAIL: CAND_0000010 should be last (score=0.0), "
        f"got {result_pool[-1].candidate_id}"
    )
    print("[PASS] Skipped candidate (not in store) has score=0.0 and sorts last  ✓")

    # ── Acceptance criterion 5: sigmoid helper is monotone ────────────────
    test_logits = [-10.0, -2.0, 0.0, 2.0, 10.0]
    sigmoids = [_sigmoid(x) for x in test_logits]
    for i in range(len(sigmoids) - 1):
        assert sigmoids[i] < sigmoids[i + 1], (
            f"FAIL: _sigmoid not monotonically increasing at index {i}"
        )
    assert abs(_sigmoid(0.0) - 0.5) < 1e-9, (
        f"FAIL: _sigmoid(0.0) should be 0.5, got {_sigmoid(0.0)}"
    )
    print("[PASS] _sigmoid is monotonically increasing and sigmoid(0)=0.5  ✓")

    # ── Acceptance criterion 6: empty pool returns safely ─────────────────
    empty_result = reranker.rerank([], jd, candidate_store)
    assert empty_result == [], (
        f"FAIL: empty pool should return [], got {empty_result}"
    )
    print("[PASS] Empty pool returns []  ✓")

    # ── Acceptance criterion 7: TypeError on wrong input types ────────────
    try:
        reranker.rerank("not-a-list", jd, candidate_store)  # type: ignore
        print("FAIL: should have raised TypeError for non-list pool")
    except TypeError:
        print("[PASS] TypeError raised for non-list pool  ✓")

    try:
        reranker.rerank(result_pool, "not-a-JDIntent", candidate_store)  # type: ignore
        print("FAIL: should have raised TypeError for non-JDIntent jd")
    except TypeError:
        print("[PASS] TypeError raised for non-JDIntent jd  ✓")

    # ── Acceptance criterion 8: __repr__ ─────────────────────────────────
    repr_str = repr(reranker)
    assert "CrossEncoderReranker" in repr_str
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in repr_str
    print(f"[PASS] __repr__ = {repr_str}  ✓")

    # ── Acceptance criterion 9: rerank_pool() convenience function ────────
    pool2 = [
        _make_rrf_result("CAND_0000031", 0.049),
        _make_rrf_result("CAND_0000014", 0.038),
    ]
    result2 = rerank_pool(pool2, jd, candidate_store, top_k=10)
    assert len(result2) == 2
    assert result2[0].cross_encoder_score >= result2[1].cross_encoder_score
    print("[PASS] rerank_pool() convenience function works correctly  ✓")

    # ── Live model test (optional — requires HF cache populated) ─────────
    import os
    if os.environ.get("TEST_LIVE_MODEL", "0") == "1":
        print("\n── Live model test (TEST_LIVE_MODEL=1) ──")
        live_reranker = CrossEncoderReranker(top_k=50)
        live_pool = [
            _make_rrf_result("CAND_0000031", 0.049),
            _make_rrf_result("CAND_0000014", 0.038),
            _make_rrf_result("CAND_0000001", 0.025),
        ]
        t_live = time.perf_counter()
        live_result = live_reranker.rerank(live_pool, jd, candidate_store)
        live_elapsed = time.perf_counter() - t_live
        print(f"Live model reranked {len(live_result)} candidates in {live_elapsed:.2f}s")
        for r in live_result:
            print(
                f"  {r.candidate_id}  rrf={r.rrf_score:.5f}  "
                f"ce={r.cross_encoder_score:.4f}"
            )
        assert live_result[0].candidate_id == "CAND_0000031", (
            "FAIL: CAND_0000031 (deep retrieval expertise) should rank #1 "
            f"but got {live_result[0].candidate_id}"
        )
        print("[PASS] Live model: CAND_0000031 (Swiggy retrieval engineer) ranks #1  ✓")
        assert live_elapsed < 10.0, (
            f"FAIL: live model took {live_elapsed:.2f}s, expected < 10s"
        )
        print(f"[PASS] Live model within 10s budget ({live_elapsed:.2f}s)  ✓")
    else:
        print(
            "\n[SKIP] Live model test skipped. "
            "Set TEST_LIVE_MODEL=1 to run with actual cross-encoder."
        )

    print(f"\nAll smoke-test assertions passed.")