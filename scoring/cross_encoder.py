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
_MAX_JD_CHARS: int = 1_500   # JD used as the "query" in (query, document) pair
_MAX_CAND_CHARS: int = 2_000  # Candidate profile used as the "document"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    # Clip to ±500 to prevent math.exp overflow for extreme model outputs.
    x_clipped = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x_clipped))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ─────────────────────────────────────────────────────────────────────────────
# CrossEncoderReranker
# ─────────────────────────────────────────────────────────────────────────────

class CrossEncoderReranker:

    MODEL_NAME: str = config.CROSS_ENCODER_MODEL  # "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        top_k: int = config.CROSS_ENCODER_TOP_K,
    ) -> None:
        
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
        
        if self._model is not None:
            return 

        try:
            from sentence_transformers import CrossEncoder

            logger.info(
                "Loading cross-encoder: %s (CPU, max_length=512) …",
                self.MODEL_NAME,
            )
            t0 = time.perf_counter()

            self._model = CrossEncoder(
                self.MODEL_NAME,
                max_length=512,
                device="cpu",
                local_files_only=True,
            )

            elapsed = time.perf_counter() - t0
            logger.info(
                "Cross-encoder loaded in %.2fs (Local Only Mode).",
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
                "Model may not be cached locally. "
                "Falling back to rrf_score-based ranking.",
                self.MODEL_NAME,
                exc,
            )
            self._model = None

    # Text builders
    @staticmethod
    def _build_jd_text(jd: JDIntent) -> str:
        
        text = jd.raw_text.strip()

        if not text:
            
            skills_str = ", ".join(jd.required_skills[:10])
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

    # Fallback scoring
    def _fallback_rank(
        self,
        pool: list[RRFResult],
    ) -> list[RRFResult]:
        
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

    # Primary rerank method 
    def rerank(
        self,
        pool: list[RRFResult],
        jd: JDIntent,
        candidate_store: dict[str, CandidateFeatureVector],
    ) -> list[RRFResult]:
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
        try:
            import torch

            torch.set_num_threads(3)

            n_batches = math.ceil(len(pairs) / 16)
            logger.info(
                "CrossEncoder: scoring %d pairs in %d batches (batch_size=16) …",
                len(pairs), n_batches,
            )

            raw_scores: np.ndarray = self._model.predict(
                sentences=pairs,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
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
        for result, raw_score in zip(valid_results, raw_scores):
            result.cross_encoder_score = _sigmoid(float(raw_score))

        # ── Sort pool by cross_encoder_score desc, candidate_id asc ──────
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
        return self._model is not None

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker("
            f"model={self.MODEL_NAME!r}, "
            f"top_k={self._top_k}, "
            f"loaded={self.is_loaded})"
        )

# Module-level convenience function
def rerank_pool(
    pool: list[RRFResult],
    jd: JDIntent,
    candidate_store: dict[str, CandidateFeatureVector],
    top_k: int = config.CROSS_ENCODER_TOP_K,
) -> list[RRFResult]:
    return CrossEncoderReranker(top_k=top_k).rerank(pool, jd, candidate_store)
