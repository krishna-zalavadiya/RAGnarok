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

# Feature column indices — must match the order in feature_store.py
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


# SignalPath
class SignalPath:
    PATH_NAME: str = "signal"

    def __init__(
        self,
        feature_data:  Optional[np.ndarray] = None,
        candidate_ids: Optional[np.ndarray] = None,
        index_path:    Optional[Path] = None,
        id_map_path:   Optional[Path] = None,
    ) -> None:
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

    # Factory — production path 
    @classmethod
    def from_disk(
        cls,
        index_path:  Optional[Path] = None,
        id_map_path: Optional[Path] = None,
    ) -> "SignalPath":
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

    # Vectorised scoring 
    def _compute_scores(self) -> np.ndarray:
        # Slice to the 7 engagement columns; clip to guard against any
        # out-of-range values written by feature_store.py
        feature_slice: np.ndarray = np.clip(
            self._data[:, :N_SIGNAL_COLS], 0.0, 1.0
        )
        scores: np.ndarray = feature_slice @ self._weights
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    # Weight vector construction  
    @staticmethod
    def _build_weight_vector() -> np.ndarray:
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

    # Loading helpers
    def _ensure_loaded(self) -> None:
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
        if self._data is None or self._ids is None:
            return
        if len(self._data) != len(self._ids):
            raise ValueError(
                f"features.npy has {len(self._data)} rows but "
                f"feature_ids.npy has {len(self._ids)} entries. "
                "Re-run precompute.py to rebuild aligned indexes."
            )

    # Introspection helpers 
    def explain_candidate(
        self,
        candidate_idx: int,
    ) -> dict[str, float]:
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
        return self._loaded

    @property
    def n_candidates(self) -> int:
        return int(len(self._ids)) if self._loaded and self._ids is not None else 0

    def __repr__(self) -> str:
        status = (
            f"n_candidates={self.n_candidates}, n_cols={N_SIGNAL_COLS}"
            if self._loaded else "not loaded"
        )
        return f"SignalPath({status})"


# Module-level convenience
def retrieve_signal(
    top_k: int = config.SIGNAL_PATH_TOP_K,
    index_path:  Optional[Path] = None,
    id_map_path: Optional[Path] = None,
) -> list[RetrievalResult]:
    path = SignalPath.from_disk(index_path=index_path, id_map_path=id_map_path)
    return path.retrieve(top_k=top_k)
