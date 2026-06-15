"""
retrieval/rrf_fusion.py
------------------------
Reciprocal Rank Fusion (RRF) across all five retrieval paths.

Core insight:
    Each retrieval path surfaces different candidates based on different
    signals. RRF merges them using the principle:

        "A candidate ranked highly in ANY path is probably relevant."

    The formula is:
        score(c) = Σ_path  bonus(path) / (RRF_K + rank_in_path(c))

    where RRF_K = 60 (standard smoothing constant).

    A candidate ranked #1 in Path 3 only (ontology):
        1.3 / (60 + 1) ≈ 0.02131

    A candidate ranked #25 in both Paths 1 and 2 (no overlap bonus):
        1/(85) + 1/(85) ≈ 0.02353

    Both survive comfortably in the top-60 pool — the ontology-only
    candidate is not buried just because semantic/keyword missed it.

Path bonus multipliers:
    "semantic":   1.0   — dense bi-encoder retrieval
    "keyword":    1.0   — BM25 + ontology-expanded sparse retrieval
    "ontology":   1.3   — domain-transfer Tier-5 rescue (needs extra lift)
    "trajectory": 1.0   — career-pattern match
    "signal":     1.1   — behavioral engagement (available candidates)

    Bonuses defined in config.RRF_ONTOLOGY_PATH_BONUS and
    config.RRF_SIGNAL_PATH_BONUS. All other paths use 1.0.

Deduplication:
    Cross-path: each candidate's contributions from all paths are summed
    into one RRF score — no duplicates in output.
    Within-path: if a path somehow returns the same candidate twice,
    only the first (best-ranked) occurrence counts.

Output:
    Top config.RRF_POOL_SIZE (60) RRFResult objects, sorted by rrf_score
    descending. Tie-break: candidate_id ascending (spec-compliant).
    cross_encoder_score is left at 0.0 — populated later by cross_encoder.py.

Consumed by:
    scoring/honeypot_filter.py   (removes flagged candidates pre-rerank)
    scoring/cross_encoder.py     (pairwise JD × candidate reranking)
    pipeline/runner.py

Dependencies:
    config.py           RRF_K, RRF_POOL_SIZE, path bonus constants
    pipeline/schemas.py RetrievalResult (input), RRFResult (output)
    stdlib              logging, time, typing — zero I/O, zero ML
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config
from pipeline.schemas import RetrievalResult, RRFResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Path bonus lookup — built once at import time from config constants
# ─────────────────────────────────────────────────────────────────────────────

# Maps each retrieval path's PATH_NAME → RRF score multiplier.
# Paths not present in this dict default to 1.0.
_PATH_BONUS: dict[str, float] = {
    "semantic":   1.0,
    "keyword":    1.0,
    "ontology":   config.RRF_ONTOLOGY_PATH_BONUS,   # 1.3
    "trajectory": 1.0,
    "signal":     config.RRF_SIGNAL_PATH_BONUS,     # 1.1
}


# ─────────────────────────────────────────────────────────────────────────────
# RRFFusion
# ─────────────────────────────────────────────────────────────────────────────

class RRFFusion:
    """
    Merges results from up to 5 retrieval paths using Reciprocal Rank Fusion.

    Usage in pipeline/runner.py:
        fusion = RRFFusion()

        pool: list[RRFResult] = fusion.fuse({
            "semantic":   semantic_results,
            "keyword":    keyword_results,
            "ontology":   ontology_results,
            "trajectory": trajectory_results,
            "signal":     signal_results,
        })
        # pool has ≤ 60 deduplicated RRFResult objects, sorted by rrf_score desc
    """

    def __init__(
        self,
        rrf_k:     int = config.RRF_K,
        pool_size: int = config.RRF_POOL_SIZE,
    ) -> None:
        """
        Args:
            rrf_k:     Smoothing constant in the denominator (default 60).
                       Higher values reduce the impact of rank differences.
            pool_size: Maximum candidates to return after fusion
                       (default config.RRF_POOL_SIZE = 60).
        """
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

    # ------------------------------------------------------------------ #
    # Primary fuse method                                                  #
    # ------------------------------------------------------------------ #

    def fuse(
        self,
        path_results: dict[str, list[RetrievalResult]],
    ) -> list[RRFResult]:
        """
        Merge retrieval results from multiple paths into a single ranked pool.

        Args:
            path_results: Dict mapping path name → list of RetrievalResult.
                          Keys must match PATH_NAME constants on each path class:
                          "semantic", "keyword", "ontology", "trajectory", "signal".
                          Missing or empty paths are silently skipped — you do
                          NOT need all 5 paths to be present.

        Returns:
            list[RRFResult] of length ≤ pool_size, sorted by rrf_score
            descending with candidate_id ascending as a tie-breaker.

            rrf_score        = Σ_path  bonus(path) / (rrf_k + rank_in_path)
            paths_present    = sorted list of path names where candidate appeared
            cross_encoder_score = 0.0  (populated by scoring/cross_encoder.py)

        Raises:
            TypeError:  path_results is not a dict.
        """
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

    # ------------------------------------------------------------------ #
    # Introspection helpers                                                #
    # ------------------------------------------------------------------ #

    def score_breakdown(
        self,
        candidate_id: str,
        path_results: dict[str, list[RetrievalResult]],
    ) -> dict[str, float]:
        """
        Return per-path RRF contribution for a single candidate.

        Useful for debugging why a candidate did or did not make the pool.

        Args:
            candidate_id: CAND_XXXXXXX string.
            path_results: Same dict passed to fuse().

        Returns:
            Dict mapping path_name → RRF contribution for this candidate.
            0.0 for paths where the candidate did not appear.
            "total" key contains the sum.
        """
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


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience — named-parameter interface for runner.py
# ─────────────────────────────────────────────────────────────────────────────

def fuse_results(
    semantic:   Optional[list[RetrievalResult]] = None,
    keyword:    Optional[list[RetrievalResult]] = None,
    ontology:   Optional[list[RetrievalResult]] = None,
    trajectory: Optional[list[RetrievalResult]] = None,
    signal:     Optional[list[RetrievalResult]] = None,
    rrf_k:      int = config.RRF_K,
    pool_size:  int = config.RRF_POOL_SIZE,
) -> list[RRFResult]:
    """
    Named-parameter convenience wrapper for pipeline/runner.py.

    Each path is optional — pass only the paths that ran successfully.
    Absent paths (None or empty list) are silently skipped.

    Args:
        semantic:   Results from retrieval/semantic_path.py.
        keyword:    Results from retrieval/keyword_path.py.
        ontology:   Results from retrieval/ontology_path.py.
        trajectory: Results from retrieval/trajectory_path.py.
        signal:     Results from retrieval/signal_path.py.
        rrf_k:      RRF smoothing constant (default config.RRF_K = 60).
        pool_size:  Output pool size (default config.RRF_POOL_SIZE = 60).

    Returns:
        list[RRFResult] of length ≤ pool_size, sorted by rrf_score desc.
    """
    path_results: dict[str, list[RetrievalResult]] = {}

    if semantic:   path_results["semantic"]   = semantic
    if keyword:    path_results["keyword"]    = keyword
    if ontology:   path_results["ontology"]   = ontology
    if trajectory: path_results["trajectory"] = trajectory
    if signal:     path_results["signal"]     = signal

    return RRFFusion(rrf_k=rrf_k, pool_size=pool_size).fuse(path_results)


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test — python -m retrieval.rrf_fusion
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import dataclasses

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    print("=" * 65)
    print("RRFFusion — smoke test")
    print("=" * 65)

    # ── Helper: build RetrievalResult list for a path ─────────────────────
    def make_results(
        path_name: str,
        id_rank_pairs: list[tuple[str, int]],
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                candidate_id=cid,
                path_score=1.0 / rank,
                path_name=path_name,
                rank_in_path=rank,
            )
            for cid, rank in id_rank_pairs
        ]

    # ── Scenario ──────────────────────────────────────────────────────────
    # CAND_A  — appears in semantic (rank 1) + keyword (rank 2): strong multi-path
    # CAND_B  — appears ONLY in ontology (rank 1): Tier-5 rescue target
    # CAND_C  — appears in semantic (rank 25) + keyword (rank 25): weaker multi-path
    # CAND_D  — appears only in signal (rank 1): highly engaged, unknown skills
    # CAND_E  — appears only in trajectory (rank 1): IC-riser, not in skill paths
    # CAND_F  — appears in semantic only (rank 1): strong dense match

    semantic_res = make_results("semantic", [
        ("CAND_A", 1), ("CAND_C", 25), ("CAND_F", 1),
        ("CAND_G", 3), ("CAND_H", 5),
    ])
    keyword_res = make_results("keyword", [
        ("CAND_A", 2), ("CAND_C", 25), ("CAND_G", 1),
    ])
    ontology_res = make_results("ontology", [
        ("CAND_B", 1), ("CAND_I", 3), ("CAND_J", 5),
    ])
    trajectory_res = make_results("trajectory", [
        ("CAND_E", 1), ("CAND_K", 2),
    ])
    signal_res = make_results("signal", [
        ("CAND_D", 1), ("CAND_A", 5), ("CAND_L", 3),
    ])

    fusion = RRFFusion()
    print(f"\nFusion config: {fusion}\n")

    t0 = time.perf_counter()
    pool = fusion.fuse({
        "semantic":   semantic_res,
        "keyword":    keyword_res,
        "ontology":   ontology_res,
        "trajectory": trajectory_res,
        "signal":     signal_res,
    })
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"Pool size: {len(pool)} candidates  ({elapsed_ms:.2f} ms)\n")
    pool_ids = [r.candidate_id for r in pool]

    for r in pool:
        print(
            f"  {r.candidate_id:<16}  rrf={r.rrf_score:.5f}"
            f"  paths={r.paths_present}"
        )

    # ── Acceptance criterion 1: CAND_B (path-3-only) in top-60 ───────────
    assert "CAND_B" in pool_ids, (
        f"FAIL: CAND_B (ontology path only, rank 1) not in pool. "
        f"Pool IDs: {pool_ids}"
    )
    print("\n[PASS] Ontology-only candidate (CAND_B) survives in pool  ✓")

    # ── Acceptance criterion 2: CAND_A (multi-path) scores highest ────────
    cand_a = next(r for r in pool if r.candidate_id == "CAND_A")
    cand_b = next(r for r in pool if r.candidate_id == "CAND_B")
    # CAND_A: 1/(61) + 1/(62) + 1.1×1/(65) ≈ 0.0164 + 0.0161 + 0.0169 ≈ 0.0494
    # CAND_B: 1.3×1/(61) ≈ 0.0213
    assert cand_a.rrf_score > cand_b.rrf_score, (
        f"FAIL: multi-path CAND_A ({cand_a.rrf_score:.5f}) should beat "
        f"ontology-only CAND_B ({cand_b.rrf_score:.5f})"
    )
    print(
        f"[PASS] Multi-path CAND_A ({cand_a.rrf_score:.5f}) > "
        f"ontology-only CAND_B ({cand_b.rrf_score:.5f})  ✓"
    )

    # ── Acceptance criterion 3: scores sorted descending ─────────────────
    for i in range(len(pool) - 1):
        assert pool[i].rrf_score >= pool[i + 1].rrf_score, (
            f"FAIL: not sorted descending at index {i}: "
            f"{pool[i].rrf_score} < {pool[i+1].rrf_score}"
        )
    print("[PASS] Pool sorted by rrf_score descending  ✓")

    # ── Acceptance criterion 4: no duplicate candidate_ids ────────────────
    assert len(pool_ids) == len(set(pool_ids)), (
        f"FAIL: duplicate candidate_ids in pool: "
        f"{[cid for cid in pool_ids if pool_ids.count(cid) > 1]}"
    )
    print("[PASS] No duplicate candidate_ids in pool  ✓")

    # ── Acceptance criterion 5: paths_present is correct ─────────────────
    # CAND_A appeared in semantic, keyword, signal
    cand_a_paths = sorted(cand_a.paths_present)
    assert "semantic" in cand_a_paths and "keyword" in cand_a_paths, (
        f"FAIL: CAND_A paths_present = {cand_a_paths}, "
        "expected semantic + keyword"
    )
    # CAND_B appeared only in ontology
    assert cand_b.paths_present == ["ontology"], (
        f"FAIL: CAND_B paths_present = {cand_b.paths_present}, expected ['ontology']"
    )
    print("[PASS] paths_present correctly reflects which paths retrieved each candidate  ✓")

    # ── Acceptance criterion 6: manual score verification (CAND_B) ────────
    # CAND_B: ontology path, rank 1, bonus 1.3
    expected_b = 1.3 / (config.RRF_K + 1)
    assert abs(cand_b.rrf_score - expected_b) < 1e-7, (
        f"FAIL: CAND_B expected {expected_b:.8f}, got {cand_b.rrf_score:.8f}"
    )
    print(
        f"[PASS] CAND_B score = {cand_b.rrf_score:.5f} "
        f"(manual: 1.3/(60+1) = {expected_b:.5f})  ✓"
    )

    # ── Acceptance criterion 7: within-path dedup works ───────────────────
    # If same candidate appears twice in a path, only count once
    dup_results = make_results("semantic", [
        ("CAND_DUP", 1),
        ("CAND_DUP", 2),  # same ID, should be skipped
        ("CAND_OTHER", 3),
    ])
    pool_dup = fusion.fuse({"semantic": dup_results})
    dup_score = next(r.rrf_score for r in pool_dup if r.candidate_id == "CAND_DUP")
    expected_dup = 1.0 / (config.RRF_K + 1)   # only rank 1 counts
    assert abs(dup_score - expected_dup) < 1e-7, (
        f"FAIL: within-path dup counted twice. Got {dup_score:.6f}, "
        f"expected {expected_dup:.6f}"
    )
    print("[PASS] Within-path duplicate correctly counted only once  ✓")

    # ── Acceptance criterion 8: tie-break by candidate_id ascending ───────
    # Build two candidates with identical scores
    tie_results = {
        "semantic": [
            RetrievalResult("CAND_ZZZ", 0.5, "semantic", 5),
            RetrievalResult("CAND_AAA", 0.5, "semantic", 5),
        ]
    }
    tie_pool = fusion.fuse(tie_results)
    tie_ids  = [r.candidate_id for r in tie_pool]
    assert tie_ids[0] == "CAND_AAA" and tie_ids[1] == "CAND_ZZZ", (
        f"FAIL: tie-break should be ascending by candidate_id, got {tie_ids}"
    )
    print("[PASS] Tie-break is candidate_id ascending  ✓")

    # ── Acceptance criterion 9: cross_encoder_score initialised to 0.0 ───
    assert all(r.cross_encoder_score == 0.0 for r in pool)
    print("[PASS] cross_encoder_score = 0.0 on all results  ✓")

    # ── Acceptance criterion 10: fuse_results() convenience function ──────
    pool2 = fuse_results(
        semantic=semantic_res, keyword=keyword_res,
        ontology=ontology_res, trajectory=trajectory_res,
        signal=signal_res,
    )
    assert [r.candidate_id for r in pool2] == [r.candidate_id for r in pool], (
        "FAIL: fuse_results() output differs from RRFFusion.fuse()"
    )
    print("[PASS] fuse_results() produces same output as RRFFusion.fuse()  ✓")

    # ── Acceptance criterion 11: score_breakdown() ────────────────────────
    breakdown = fusion.score_breakdown(
        "CAND_A",
        {"semantic": semantic_res, "keyword": keyword_res,
         "ontology": ontology_res, "trajectory": trajectory_res,
         "signal": signal_res},
    )
    assert abs(breakdown["total"] - cand_a.rrf_score) < 1e-7, (
        f"FAIL: breakdown total {breakdown['total']:.8f} != "
        f"rrf_score {cand_a.rrf_score:.8f}"
    )
    assert breakdown["ontology"] == 0.0   # CAND_A not in ontology
    print(f"[PASS] score_breakdown() correct: {breakdown}  ✓")

    print(f"\nAll smoke-test assertions passed.  (elapsed: {elapsed_ms:.2f} ms)")