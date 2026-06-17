"""
retrieval/signal_path.py
-------------------------
Retrieval Path 5: Behavioral engagement scoring (NEW — not in original Image 1).

Why this path exists:
    The JD is explicit: "a perfect-on-paper candidate who hasn't logged in
    for 6 months and has a 5% recruiter response rate is, for hiring purposes,
    not actually available. Down-weight them appropriately."

    Paths 1-4 score *who can do the job*. Path 5 scores *who is reachable
    right now* — candidates who are actively looking, respond to recruiters,
    have a short notice period, and maintain an active professional presence.

    These candidates appear in the top-15 of Path 5 and receive a 1.1×
    bonus in RRF fusion (config.RRF_SIGNAL_PATH_BONUS), giving them a
    modest lift over equally-skilled but unreachable candidates.

Key design: NO JDIntent parameter.
    Behavioral engagement is JD-agnostic — the same candidate is equally
    reachable regardless of whether we are hiring a Senior AI Engineer or
    a Product Manager. retrieve() takes only top_k.

Signals scored (7 pre-normalised columns from feature_store.npy):
    Col 0  recency_score          exp(-λ × days_since_last_active)
    Col 1  response_rate          recruiter_response_rate            [0,1]
    Col 2  open_to_work           1.0 if open_to_work_flag, else 0.0
    Col 3  notice_period_score    1.0 for ≤30d → 0.0 for >90d
    Col 4  github_activity        github_activity_score / 100;
                                  −1 (not linked) → GITHUB_NOT_LINKED_DEFAULT
    Col 5  profile_completeness   profile_completeness_score / 100
    Col 6  interview_completion   interview_completion_rate          [0,1]

Scoring formula (single vectorised dot-product):
    score = features[:, :7]  @  weight_vector

    where weight_vector = [BEHAVIORAL_WEIGHTS[k] for k in ordered keys]
                        = [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05]
                          (recency, response_rate, open_to_work,
                           notice_period, github_activity,
                           profile_completeness, interview_completion)

    All inputs are pre-normalised to [0,1] by feature_store.py, so
    score ∈ [0, 1] and runs in < 50 ms for 100 K candidates.

DEV B interface contract (indexing/feature_store.py must produce):
    features.npy      shape (N, 7+), dtype float32
                      First 7 columns must be in the order above.
                      Additional columns (e.g. raw signals, skill vectors)
                      are ignored by this path — do not reorder the first 7.
    feature_ids.npy   shape (N,), dtype object
                      CAND_XXXXXXX strings, same row order as features.npy

    Pre-computation responsibilities of feature_store.py:
      - Apply recency decay:  recency = exp(-RECENCY_LAMBDA × days)
      - Replace github −1:    github  = GITHUB_NOT_LINKED_DEFAULT (0.5)
      - Normalise notice:     0-30d→1.0, 31-60d→linear 1.0→0.5,
                              61-90d→linear 0.5→0.1, >90d→0.0
      - Divide completeness:  completeness_score / 100.0

Consumed by:
    retrieval/rrf_fusion.py     (RRF_SIGNAL_PATH_BONUS = 1.1 applied here)
    pipeline/runner.py          (Path 5 of the ranking pipeline)

Dependencies:
    config.py           FEATURE_STORE_PATH, FEATURE_IDS_PATH,
                        SIGNAL_PATH_TOP_K, BEHAVIORAL_WEIGHTS,
                        GITHUB_NOT_LINKED_DEFAULT
    pipeline/schemas.py RetrievalResult  (JDIntent NOT used)
    numpy               vectorised dot-product scoring
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

import config
from pipeline.schemas import RetrievalResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Feature column indices — must match the order in feature_store.py
# ─────────────────────────────────────────────────────────────────────────────

FEAT_COL_RECENCY:              int = 0
FEAT_COL_RESPONSE_RATE:        int = 1
FEAT_COL_OPEN_TO_WORK:         int = 2
FEAT_COL_NOTICE_PERIOD:        int = 3
FEAT_COL_GITHUB_ACTIVITY:      int = 4
FEAT_COL_PROFILE_COMPLETENESS: int = 5
FEAT_COL_INTERVIEW_COMPLETION: int = 6
N_SIGNAL_COLS:                 int = 7

# Ordered list of BEHAVIORAL_WEIGHTS keys matching column order above.
# This list drives _build_weight_vector() — never re-order it.
_WEIGHT_KEYS: list[str] = [
    "recency",
    "response_rate",
    "open_to_work",
    "notice_period",
    "github_activity",
    "profile_completeness",
    "interview_completion",
]

# Regex for candidate ID validation
_CAND_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


# ─────────────────────────────────────────────────────────────────────────────
# SignalPath
# ─────────────────────────────────────────────────────────────────────────────

class SignalPath:
    """
    Behavioral engagement retrieval path (Path 5 of 5).

    Surfaces candidates who are actively engaged with the platform right now:
    recently active, open to work, short notice period, responsive to
    recruiters, with an active GitHub and complete profile.

    Typical production usage:
        # Load once in runner.py at startup
        path = SignalPath.from_disk()

        # Call once per ranking run — no JDIntent needed
        results = path.retrieve(top_k=15)

    Unit-test usage (no real index needed):
        data = np.array([
            [0.95, 0.9, 1.0, 1.0, 0.7, 0.9, 0.85],  # highly engaged
            [0.10, 0.1, 0.0, 0.0, 0.5, 0.4, 0.50],  # inactive
        ], dtype=np.float32)
        ids = np.array(["CAND_0000031", "CAND_0000002"], dtype=object)
        path = SignalPath(feature_data=data, candidate_ids=ids)
        results = path.retrieve(top_k=15)
    """

    PATH_NAME: str = "signal"

    def __init__(
        self,
        feature_data:  Optional[np.ndarray] = None,
        candidate_ids: Optional[np.ndarray] = None,
        index_path:    Optional[Path] = None,
        id_map_path:   Optional[Path] = None,
    ) -> None:
        """
        Args:
            feature_data:   Pre-loaded numpy array shape (N, 7+), dtype float32.
                            If supplied, index_path is ignored.
            candidate_ids:  1-D numpy string array length N, aligned with
                            rows of feature_data.
            index_path:     Path to features.npy.
                            Defaults to config.FEATURE_STORE_PATH.
            id_map_path:    Path to feature_ids.npy.
                            Defaults to config.FEATURE_IDS_PATH.

        Raises:
            ValueError: feature_data supplied without candidate_ids, or
                        shape/alignment mismatch.
        """
        self._index_path:  Path = index_path  or config.FEATURE_STORE_PATH
        self._id_map_path: Path = id_map_path or config.FEATURE_IDS_PATH

        self._data:    Optional[np.ndarray] = None
        self._ids:     Optional[np.ndarray] = None
        self._weights: Optional[np.ndarray] = None   # built once on first use
        self._loaded:  bool = False

        if feature_data is not None:
            if candidate_ids is None:
                raise ValueError(
                    "candidate_ids must be provided when feature_data is given."
                )
            self._data = np.asarray(feature_data, dtype=np.float32)
            self._ids  = np.asarray(candidate_ids, dtype=object)
            self._validate_loaded_data()
            self._weights = self._build_weight_vector()
            self._loaded = True
            logger.debug(
                "SignalPath initialised with pre-loaded data (N=%d).",
                len(self._ids),
            )

    # ------------------------------------------------------------------ #
    # Factory — production path                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_disk(
        cls,
        index_path:  Optional[Path] = None,
        id_map_path: Optional[Path] = None,
    ) -> "SignalPath":
        """
        Load feature store from .npy files and return a ready instance.

        Call once in pipeline/runner.py at startup; reuse across retrieve()
        calls.

        Args:
            index_path:  Override for config.FEATURE_STORE_PATH.
            id_map_path: Override for config.FEATURE_IDS_PATH.

        Returns:
            Fully loaded SignalPath instance.

        Raises:
            FileNotFoundError: features.npy or feature_ids.npy not found.
            ValueError:        Shape mismatch, alignment error, or wrong
                               number of columns.
        """
        instance = cls(index_path=index_path, id_map_path=id_map_path)
        instance._ensure_loaded()
        return instance

    # ------------------------------------------------------------------ #
    # Primary retrieve method                                              #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        top_k: int = config.SIGNAL_PATH_TOP_K,
    ) -> list[RetrievalResult]:
        """
        Score all candidates by behavioral engagement and return top-K.

        No JDIntent required — engagement signals are JD-agnostic.

        Scoring is a single vectorised matrix-vector dot product:
            scores = feature_matrix[:, :7] @ weight_vector
        Running time: < 50 ms for 100 K candidates on a single CPU core.

        Args:
            top_k: Maximum candidates to return.
                   Defaults to config.SIGNAL_PATH_TOP_K (15).

        Returns:
            list[RetrievalResult] sorted by engagement score descending,
            length ≤ top_k. Candidates with score ≤ 0.0 excluded.

            path_name    = "signal"
            path_score   ∈ (0.0, 1.0]
            rank_in_path = 1-indexed position within this path

        Raises:
            ValueError: top_k < 1.
        """
        self._ensure_loaded()

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        if self._data is None or len(self._data) == 0:
            logger.warning("SignalPath: no data loaded. Returning [].")
            return []

        t0 = time.perf_counter()

        scores: np.ndarray = self._compute_scores()

        # Partial sort O(N log K) → exact sort of top-K slice
        effective_k  = min(top_k, len(scores))
        top_indices  = np.argpartition(scores, -effective_k)[-effective_k:]
        top_indices  = top_indices[np.argsort(scores[top_indices])[::-1]]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        results: list[RetrievalResult] = []
        rank = 0
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0.0:
                continue
            rank += 1
            results.append(
                RetrievalResult(
                    candidate_id=str(self._ids[idx]),
                    path_score=round(score, 6),
                    path_name=self.PATH_NAME,
                    rank_in_path=rank,
                )
            )
            if len(results) >= top_k:
                break

        logger.info(
            "SignalPath.retrieve: %d/%d candidates scored > 0, "
            "returning %d (top_k=%d, %.1f ms)",
            int(np.sum(scores > 0.0)),
            len(scores),
            len(results),
            top_k,
            elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------ #
    # Vectorised scoring                                                   #
    # ------------------------------------------------------------------ #

    def _compute_scores(self) -> np.ndarray:
        """
        Compute engagement scores for all N candidates in one dot product.

        Uses only the first N_SIGNAL_COLS (7) columns of the feature matrix.
        Any additional columns stored in features.npy (raw signals, derived
        features, etc.) are ignored here.

        Returns:
            numpy array shape (N,), dtype float32, values in [0.0, 1.0].
        """
        # Slice to the 7 engagement columns; clip to guard against any
        # out-of-range values written by feature_store.py
        feature_slice: np.ndarray = np.clip(
            self._data[:, :N_SIGNAL_COLS], 0.0, 1.0
        )
        scores: np.ndarray = feature_slice @ self._weights
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Weight vector construction                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_weight_vector() -> np.ndarray:
        """
        Build the weight vector from config.BEHAVIORAL_WEIGHTS in column order.

        The order is defined by _WEIGHT_KEYS, which mirrors the column order
        in features.npy. Both must be kept in sync with feature_store.py.

        Returns:
            numpy array shape (7,), dtype float32, values sum to 1.0.

        Raises:
            KeyError: A required weight key is missing from BEHAVIORAL_WEIGHTS.
            ValueError: Weights do not sum to 1.0 (catches config drift).
        """
        weights: list[float] = []
        for key in _WEIGHT_KEYS:
            if key not in config.BEHAVIORAL_WEIGHTS:
                raise KeyError(
                    f"BEHAVIORAL_WEIGHTS in config.py is missing key '{key}'. "
                    f"Expected keys: {_WEIGHT_KEYS}"
                )
            weights.append(float(config.BEHAVIORAL_WEIGHTS[key]))

        weight_arr = np.array(weights, dtype=np.float32)

        weight_sum = float(weight_arr.sum())
        if abs(weight_sum - 1.0) > 1e-4:
            raise ValueError(
                f"BEHAVIORAL_WEIGHTS in config.py sum to {weight_sum:.6f}, "
                "expected 1.0. Check config.BEHAVIORAL_WEIGHTS."
            )
        return weight_arr

    # ------------------------------------------------------------------ #
    # Loading helpers                                                      #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Load feature arrays from disk if not already in memory."""
        if self._loaded:
            return
        self._data    = self._load_feature_data(self._index_path)
        self._ids     = self._load_id_map(self._id_map_path)
        self._validate_loaded_data()
        self._weights = self._build_weight_vector()
        self._loaded  = True
        logger.info(
            "SignalPath loaded from disk: N=%d candidates, %d signal cols.",
            len(self._ids),
            N_SIGNAL_COLS,
        )

    @staticmethod
    def _load_feature_data(path: Path) -> np.ndarray:
        """
        Load features.npy — must have shape (N, 7+), dtype float32.

        Using allow_pickle=False because the file contains only float data.

        Raises:
            FileNotFoundError: File not found at path.
            ValueError:        Wrong shape (< 7 columns) or bad dtype.
            RuntimeError:      numpy I/O error.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Feature store not found: '{path}'. "
                "Run precompute.py (indexing/feature_store.py) first, "
                "or verify config.FEATURE_STORE_PATH."
            )
        try:
            arr: np.ndarray = np.load(str(path), allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load feature store from '{path}': {exc}. "
                "Delete the file and re-run precompute.py."
            ) from exc

        if arr.ndim != 2:
            raise ValueError(
                f"features.npy must be 2-D, got ndim={arr.ndim}."
            )
        if arr.shape[1] < N_SIGNAL_COLS:
            raise ValueError(
                f"features.npy has only {arr.shape[1]} columns, "
                f"but SignalPath requires at least {N_SIGNAL_COLS}. "
                "Verify indexing/feature_store.py column order."
            )
        logger.debug(
            "Feature data loaded: shape=%s, dtype=%s.",
            arr.shape, arr.dtype,
        )
        return arr.astype(np.float32)

    @staticmethod
    def _load_id_map(path: Path) -> np.ndarray:
        """
        Load feature_ids.npy — 1-D string array of CAND_XXXXXXX values.

        Raises:
            FileNotFoundError: File not found at path.
            ValueError:        Not 1-D or ID format incorrect.
            RuntimeError:      numpy I/O error.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"Feature ID map not found: '{path}'. "
                "Run precompute.py or verify config.FEATURE_IDS_PATH."
            )
        try:
            arr: np.ndarray = np.load(str(path), allow_pickle=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load feature IDs from '{path}': {exc}."
            ) from exc

        if arr.ndim != 1:
            raise ValueError(
                f"feature_ids.npy must be 1-D, got shape {arr.shape}."
            )
        for entry in arr[:5]:
            if not _CAND_ID_RE.match(str(entry)):
                raise ValueError(
                    f"Unexpected candidate_id format in feature_ids.npy: "
                    f"'{entry}'. Expected CAND_XXXXXXX."
                )
        logger.debug("Feature IDs loaded: %d entries.", len(arr))
        return arr

    def _validate_loaded_data(self) -> None:
        """Verify row count alignment between data and ID arrays."""
        if self._data is None or self._ids is None:
            return
        if len(self._data) != len(self._ids):
            raise ValueError(
                f"features.npy has {len(self._data)} rows but "
                f"feature_ids.npy has {len(self._ids)} entries. "
                "Re-run precompute.py to rebuild aligned indexes."
            )

    # ------------------------------------------------------------------ #
    # Introspection helpers                                                #
    # ------------------------------------------------------------------ #

    def explain_candidate(
        self,
        candidate_idx: int,
    ) -> dict[str, float]:
        """
        Return per-signal breakdown for a single candidate row.

        Useful for the trust layer and UI score breakdown.

        Args:
            candidate_idx: Row index into the feature matrix (0-based).

        Returns:
            Dict mapping signal name → weighted contribution to final score.
            Keys match _WEIGHT_KEYS. Sum of values = overall score.

        Raises:
            IndexError: candidate_idx out of range.
            RuntimeError: Called before index is loaded.
        """
        if not self._loaded or self._data is None:
            raise RuntimeError("SignalPath not loaded. Call from_disk() first.")
        if candidate_idx < 0 or candidate_idx >= len(self._data):
            raise IndexError(
                f"candidate_idx {candidate_idx} out of range "
                f"[0, {len(self._data)})."
            )
        row = np.clip(self._data[candidate_idx, :N_SIGNAL_COLS], 0.0, 1.0)
        return {
            key: float(row[i] * self._weights[i])
            for i, key in enumerate(_WEIGHT_KEYS)
        }

    @property
    def loaded(self) -> bool:
        """True if feature data is ready for scoring."""
        return self._loaded

    @property
    def n_candidates(self) -> int:
        """Number of candidates in the feature store (0 if not loaded)."""
        return int(len(self._ids)) if self._loaded and self._ids is not None else 0

    def __repr__(self) -> str:
        status = (
            f"n_candidates={self.n_candidates}, n_cols={N_SIGNAL_COLS}"
            if self._loaded else "not loaded"
        )
        return f"SignalPath({status})"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_signal(
    top_k: int = config.SIGNAL_PATH_TOP_K,
    index_path:  Optional[Path] = None,
    id_map_path: Optional[Path] = None,
) -> list[RetrievalResult]:
    """
    One-shot convenience: load feature store and retrieve top-K candidates.

    Creates a new SignalPath on each call (disk I/O).
    For repeated calls use SignalPath.from_disk() once and reuse.
    """
    path = SignalPath.from_disk(index_path=index_path, id_map_path=id_map_path)
    return path.retrieve(top_k=top_k)

