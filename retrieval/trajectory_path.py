from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

import config
from pipeline.schemas import JDIntent, RetrievalResult

logger = logging.getLogger(__name__)

# Column index constants — shared with indexing/trajectory_builder.py
COL_PROMOTIONS_PER_YEAR: int = 0   # float  >= 0.0
COL_YOE:                 int = 1   # float  years_of_experience
COL_HAS_PRODUCT_CO:      int = 2   # float  0.0 or 1.0
COL_IS_IC_RISER:         int = 3   # float  0.0 or 1.0
COL_CONSULTING_ONLY:     int = 4   # float  0.0 or 1.0
N_TRAJECTORY_COLS:       int = 5

# IC-riser score bonus (additive to PRODUCT_CO_BONUS from config)
_IC_RISER_BONUS: float = 1.10

# Regex for validating candidate ID format
_CAND_ID_RE = re.compile(r"^CAND_[0-9]{7}$")


# TrajectoryPath
class TrajectoryPath:

    PATH_NAME: str = "trajectory"

    def __init__(
        self,
        trajectory_data: Optional[np.ndarray] = None,
        candidate_ids: Optional[np.ndarray] = None,
        index_path: Optional[Path] = None,
        id_map_path: Optional[Path] = None,
    ) -> None:
        self._index_path:  Path = index_path  or config.TRAJECTORY_PATH
        self._id_map_path: Path = id_map_path or config.TRAJECTORY_IDS_PATH

        self._data: Optional[np.ndarray] = None
        self._ids:  Optional[np.ndarray] = None
        self._loaded: bool = False

        if trajectory_data is not None:
            if candidate_ids is None:
                raise ValueError(
                    "candidate_ids must be provided when "
                    "trajectory_data is supplied."
                )
            self._data = np.asarray(trajectory_data, dtype=np.float32)
            self._ids  = np.asarray(candidate_ids,  dtype=object)
            self._validate_loaded_data()
            self._loaded = True
            logger.debug(
                "TrajectoryPath initialised with pre-loaded data "
                "(N=%d candidates).",
                len(self._ids),
            )

    # Factory — production path 
    @classmethod
    def from_disk(
        cls,
        index_path:  Optional[Path] = None,
        id_map_path: Optional[Path] = None,
    ) -> "TrajectoryPath":
        instance = cls(index_path=index_path, id_map_path=id_map_path)
        instance._ensure_loaded()
        return instance

    # Primary retrieve method
    def retrieve(
        self,
        jd_intent: JDIntent,
        top_k: int = config.TRAJECTORY_PATH_TOP_K,
    ) -> list[RetrievalResult]:
        self._ensure_loaded()

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        if self._data is None or len(self._data) == 0:
            logger.warning("TrajectoryPath: no data loaded. Returning [].")
            return []

        t0 = time.perf_counter()

        # Vectorised scoring over entire candidate pool
        scores: np.ndarray = self._compute_scores(jd_intent)

        # Partial sort — O(N log K) rather than O(N log N)
        effective_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -effective_k)[-effective_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Build results — exclude zero-score candidates
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
                    path_score=score,
                    path_name=self.PATH_NAME,
                    rank_in_path=rank,
                )
            )
            if len(results) >= top_k:
                break

        logger.info(
            "TrajectoryPath.retrieve: %d/%d candidates scored > 0, "
            "returning %d (top_k=%d, %.1f ms)",
            int(np.sum(scores > 0.0)),
            len(scores),
            len(results),
            top_k,
            elapsed_ms,
        )
        return results

    # Vectorised scoring 
    def _compute_scores(self, jd_intent: JDIntent) -> np.ndarray:
        data = self._data  # shape (N, 5)

        promo_per_yr:   np.ndarray = data[:, COL_PROMOTIONS_PER_YEAR]
        yoe:            np.ndarray = data[:, COL_YOE]
        has_product_co: np.ndarray = data[:, COL_HAS_PRODUCT_CO] > 0.5
        is_ic_riser:    np.ndarray = data[:, COL_IS_IC_RISER]    > 0.5
        consulting_only:np.ndarray = data[:, COL_CONSULTING_ONLY] > 0.5

        # ── 1. YOE band score [0, 1] ──────────────────────────────────────
        yoe_scores = _yoe_band_score(
            yoe,
            ideal_min = jd_intent.yoe_ideal_min,
            ideal_max = jd_intent.yoe_ideal_max,
            soft_min  = jd_intent.yoe_min,
            soft_max  = jd_intent.yoe_max,
        )

        # ── 2. Velocity score [0, 1] ──────────────────────────────────────
        cap = max(config.TRAJECTORY_PROMOTIONS_PER_YEAR_CAP, 1e-9)
        velocity_scores = np.clip(promo_per_yr / cap, 0.0, 1.0)

        # ── 3. Weighted base score ────────────────────────────────────────
        # yoe_scores acts as a gate: if 0.0 (outside soft bounds) the
        # entire base collapses to 0.0, excluding the candidate cleanly.
        base: np.ndarray = yoe_scores * (0.60 + 0.40 * velocity_scores)

        # ── 4. Multiplicative modifiers ───────────────────────────────────
        # Product-company bonus
        base = np.where(
            has_product_co,
            np.minimum(1.0, base * config.PRODUCT_CO_BONUS),
            base,
        )
        # IC-riser bonus
        base = np.where(
            is_ic_riser,
            np.minimum(1.0, base * _IC_RISER_BONUS),
            base,
        )
        # Consulting-only penalty
        # Applied regardless of jd_intent.disqualify_consulting_only —
        # a penalty is always warranted; the disqualify flag is a hard
        # exclusion used by the candidate parser, not this path.
        base = np.where(
            consulting_only,
            base * config.CONSULTING_ONLY_PENALTY,
            base,
        )

        return np.clip(base, 0.0, 1.0).astype(np.float32)

    # Loading helpers 
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._data = self._load_trajectory_data(self._index_path)
        self._ids  = self._load_id_map(self._id_map_path)
        self._validate_loaded_data()
        self._loaded = True
        logger.info(
            "TrajectoryPath loaded from disk: N=%d candidates.",
            len(self._ids),
        )

    @staticmethod
    def _load_trajectory_data(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"Trajectory data not found: '{path}'. "
                "Run precompute.py (indexing/trajectory_builder.py) first, "
                "or verify config.TRAJECTORY_PATH."
            )
        try:
            arr: np.ndarray = np.load(str(path), allow_pickle=False)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load trajectory data from '{path}': {exc}. "
                "Delete the file and re-run precompute.py."
            ) from exc

        if arr.ndim != 2 or arr.shape[1] != N_TRAJECTORY_COLS:
            raise ValueError(
                f"trajectory.npy must have shape (N, {N_TRAJECTORY_COLS}), "
                f"got {arr.shape}. "
                "Verify indexing/trajectory_builder.py column order matches "
                "COL_* constants in retrieval/trajectory_path.py."
            )
        logger.debug(
            "Trajectory data loaded: shape=%s, dtype=%s.",
            arr.shape,
            arr.dtype,
        )
        return arr.astype(np.float32)

    @staticmethod
    def _load_id_map(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"Trajectory ID map not found: '{path}'. "
                "Run precompute.py or verify config.TRAJECTORY_IDS_PATH."
            )
        try:
            arr: np.ndarray = np.load(str(path), allow_pickle=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load trajectory IDs from '{path}': {exc}."
            ) from exc

        if arr.ndim != 1:
            raise ValueError(
                f"trajectory_ids.npy must be 1-D, got shape {arr.shape}."
            )

        # Spot-check format on first 5 entries
        for entry in arr[:5]:
            if not _CAND_ID_RE.match(str(entry)):
                raise ValueError(
                    f"Unexpected candidate_id format in trajectory_ids.npy: "
                    f"'{entry}'. Expected CAND_XXXXXXX (7 digits)."
                )
        logger.debug("Trajectory IDs loaded: %d entries.", len(arr))
        return arr

    def _validate_loaded_data(self) -> None:
        if self._data is None or self._ids is None:
            return
        if len(self._data) != len(self._ids):
            raise ValueError(
                f"trajectory.npy has {len(self._data)} rows but "
                f"trajectory_ids.npy has {len(self._ids)} entries. "
                "Re-run precompute.py to rebuild aligned indexes."
            )

    # Properties 
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def n_candidates(self) -> int:
        return int(len(self._ids)) if self._loaded and self._ids is not None else 0

    def __repr__(self) -> str:
        status = (
            f"n_candidates={self.n_candidates}"
            if self._loaded else "not loaded"
        )
        return f"TrajectoryPath({status})"


# Vectorised YOE band scoring — module-level for reuse by scoring/trajectory.py
def _yoe_band_score(
    yoe: np.ndarray,
    ideal_min: float,
    ideal_max: float,
    soft_min: float,
    soft_max: float,
) -> np.ndarray:
    scores = np.zeros_like(yoe, dtype=np.float32)

    # Guard against degenerate band (all zeros)
    if soft_min >= soft_max:
        return scores

    # ── Ideal band: score = 1.0 ────────────────────────────────────────
    in_ideal = (yoe >= ideal_min) & (yoe <= ideal_max)
    scores = np.where(in_ideal, 1.0, scores)

    # ── Below ideal: linear ramp from 0.4 (at soft_min) to 1.0 (at ideal_min)
    denom_low = float(max(1e-9, ideal_min - soft_min))
    below_ideal = (yoe >= soft_min) & (yoe < ideal_min)
    t_low = (yoe - soft_min) / denom_low
    score_low = 0.4 + 0.6 * t_low
    scores = np.where(below_ideal, score_low, scores)

    # ── Above ideal: linear ramp from 1.0 (at ideal_max) to 0.4 (at soft_max)
    denom_high = float(max(1e-9, soft_max - ideal_max))
    above_ideal = (yoe > ideal_max) & (yoe <= soft_max)
    t_high = (yoe - ideal_max) / denom_high
    score_high = 1.0 - 0.6 * t_high
    scores = np.where(above_ideal, score_high, scores)

    return np.clip(scores, 0.0, 1.0).astype(np.float32)


# Module-level convenience
def retrieve_trajectory(
    jd_intent: JDIntent,
    top_k: int = config.TRAJECTORY_PATH_TOP_K,
    index_path:  Optional[Path] = None,
    id_map_path: Optional[Path] = None,
) -> list[RetrievalResult]:
    path = TrajectoryPath.from_disk(
        index_path=index_path, id_map_path=id_map_path
    )
    return path.retrieve(jd_intent, top_k=top_k)

