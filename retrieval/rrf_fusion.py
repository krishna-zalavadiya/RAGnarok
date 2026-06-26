from __future__ import annotations

import logging
import time
from typing import Optional

import config
from pipeline.schemas import RetrievalResult, RRFResult

logger = logging.getLogger(__name__)

# Path bonus lookup — built once at import time from config constant
# Maps each retrieval path's PATH_NAME → RRF score multiplier.
# Paths not present in this dict default to 1.0.
_PATH_BONUS: dict[str, float] = {
    "semantic":   1.0,
    "keyword":    1.0,
    "ontology":   config.RRF_ONTOLOGY_PATH_BONUS,   # 1.3
    "trajectory": 1.0,
    "signal":     config.RRF_SIGNAL_PATH_BONUS,     # 1.1
}


# RRFFusion
class RRFFusion:
    def __init__(
        self,
        rrf_k:     int = config.RRF_K,
        pool_size: int = config.RRF_POOL_SIZE,
    ) -> None:
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {rrf_k}.")
        if pool_size < 1:
            raise ValueError(f"pool_size must be >= 1, got {pool_size}.")

        self._rrf_k:     int = rrf_k
        self._pool_size: int = pool_size

        logger.debug(
            "RRFFusion initialised (rrf_k=%d, pool_size=%d, bonuses=%s).",
            self._rrf_k,
            self._pool_size,
            {k: v for k, v in _PATH_BONUS.items() if v != 1.0},
        )

    # Primary fuse method 
    def fuse(
        self,
        path_results: dict[str, list[RetrievalResult]],
    ) -> list[RRFResult]:
        if not isinstance(path_results, dict):
            raise TypeError(
                f"path_results must be dict[str, list[RetrievalResult]], "
                f"got {type(path_results).__name__}."
            )

        if not path_results:
            logger.warning("RRFFusion.fuse: empty path_results. Returning [].")
            return []

        t0 = time.perf_counter()

        # ── Accumulate RRF contributions ──────────────────────────────────
        # scores[cid]        = accumulated RRF score
        # path_hits[cid]     = set of paths where this candidate appeared
        scores:    dict[str, float]     = {}
        path_hits: dict[str, set[str]]  = {}

        total_input   = 0   # total result objects processed
        skipped_dupes = 0   # within-path duplicates skipped

        for path_name, results in path_results.items():
            if not results:
                continue

            bonus: float = _PATH_BONUS.get(path_name, 1.0)
            if path_name not in _PATH_BONUS:
                logger.warning(
                    "Unknown path name '%s' in RRF fusion. "
                    "Using bonus=1.0. Add it to _PATH_BONUS if intentional.",
                    path_name,
                )

            # Guard: deduplicate within a single path.
            # Each candidate should appear at most once per path, but
            # defensive coding here avoids double-counting bugs.
            seen_in_path: set[str] = set()

            for result in results:
                total_input += 1
                cid = result.candidate_id

                if cid in seen_in_path:
                    skipped_dupes += 1
                    logger.debug(
                        "Within-path duplicate: '%s' in path '%s' — skipped.",
                        cid, path_name,
                    )
                    continue
                seen_in_path.add(cid)

                contribution: float = (
                    bonus / (self._rrf_k + result.rank_in_path)
                )

                if cid not in scores:
                    scores[cid]    = 0.0
                    path_hits[cid] = set()

                scores[cid]    += contribution
                path_hits[cid].add(path_name)

        # ── Sort and truncate ─────────────────────────────────────────────
        # Primary:   rrf_score descending
        # Secondary: candidate_id ascending (spec-compliant tie-break,
        #            matches submission CSV sort requirement)
        sorted_ids: list[str] = sorted(
            scores,
            key=lambda cid: (-scores[cid], cid),
        )

        pool: list[RRFResult] = [
            RRFResult(
                candidate_id=cid,
                rrf_score=round(scores[cid], 8),
                paths_present=sorted(path_hits[cid]),
                cross_encoder_score=0.0,
            )
            for cid in sorted_ids[: self._pool_size]
        ]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # ── Diagnostics ───────────────────────────────────────────────────
        n_unique      = len(scores)
        n_active_paths = sum(1 for r in path_results.values() if r)
        n_multi_path  = sum(1 for ph in path_hits.values() if len(ph) > 1)
        n_single_path = n_unique - n_multi_path

        logger.info(
            "RRFFusion: %d paths × ~%d results → "
            "%d unique candidates (%d multi-path, %d single-path) → "
            "top-%d pool  (%.1f ms)",
            n_active_paths,
            total_input // max(1, n_active_paths),
            n_unique,
            n_multi_path,
            n_single_path,
            len(pool),
            elapsed_ms,
        )
        if skipped_dupes:
            logger.warning(
                "RRFFusion: skipped %d within-path duplicate entries.",
                skipped_dupes,
            )
        if elapsed_ms > 200.0:
            logger.warning(
                "RRFFusion took %.1f ms — expected < 200 ms. "
                "Check for unusually large path result sets.",
                elapsed_ms,
            )

        return pool

    # Introspection helpers 
    def score_breakdown(
        self,
        candidate_id: str,
        path_results: dict[str, list[RetrievalResult]],
    ) -> dict[str, float]:
        breakdown: dict[str, float] = {}

        for path_name, results in path_results.items():
            bonus = _PATH_BONUS.get(path_name, 1.0)
            for result in results:
                if result.candidate_id == candidate_id:
                    breakdown[path_name] = (
                        bonus / (self._rrf_k + result.rank_in_path)
                    )
                    break
            else:
                breakdown[path_name] = 0.0

        breakdown["total"] = sum(v for k, v in breakdown.items() if k != "total")
        return breakdown

    def __repr__(self) -> str:
        return (
            f"RRFFusion(rrf_k={self._rrf_k}, pool_size={self._pool_size})"
        )


# Module-level convenience — named-parameter interface for runner.py
def fuse_results(
    semantic:   Optional[list[RetrievalResult]] = None,
    keyword:    Optional[list[RetrievalResult]] = None,
    ontology:   Optional[list[RetrievalResult]] = None,
    trajectory: Optional[list[RetrievalResult]] = None,
    signal:     Optional[list[RetrievalResult]] = None,
    rrf_k:      int = config.RRF_K,
    pool_size:  int = config.RRF_POOL_SIZE,
) -> list[RRFResult]:
    path_results: dict[str, list[RetrievalResult]] = {}

    if semantic:   path_results["semantic"]   = semantic
    if keyword:    path_results["keyword"]    = keyword
    if ontology:   path_results["ontology"]   = ontology
    if trajectory: path_results["trajectory"] = trajectory
    if signal:     path_results["signal"]     = signal

    return RRFFusion(rrf_k=rrf_k, pool_size=pool_size).fuse(path_results)
