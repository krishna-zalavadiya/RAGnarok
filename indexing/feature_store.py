from __future__ import annotations

import math
import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np

import config
from pipeline.schemas import CandidateFeatureVector
from indexing.trajectory_builder import TrajectoryAnalyzer

logger = logging.getLogger(__name__)

FEATURE_DIM = 30

# Work mode encoding — built once at module level
_WORK_MODE_MAP = {"onsite": 0.0, "hybrid": 0.5, "remote": 1.0}


class FeatureStore:
    """
    Pre-scores every candidate into a fixed 30-dim float32 vector.

    Usage:
        fs = FeatureStore()
        matrix = fs.build(candidates, save=True)   # → [N × 30]

        fs = FeatureStore()
        fs.load()
        vec = fs.get(candidate_id)                 # → np.ndarray [30]
    """

    def __init__(
        self,
        feature_path: Path = config.FEATURE_STORE_PATH,
        ids_path: Path = config.FEATURE_IDS_PATH,
    ) -> None:
        self.feature_path = feature_path
        self.ids_path = ids_path
        self._matrix: Optional[np.ndarray] = None   # [N × 30]
        self._id_to_idx: Optional[dict[str, int]] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        candidates: list[CandidateFeatureVector],
        save: bool = True,
        today: Optional[date] = None,
    ) -> np.ndarray:
        """
        Encode all candidates into the feature matrix.

        Args:
            candidates: parsed CandidateFeatureVector list (honeypot flags must
                        already be set — run HoneypotFilter.run_honeypot_filters first)
            save:       persist matrix + id list to disk
            today:      override 'today' for testing (default: date.today())

        Returns:
            np.ndarray of shape [N × 30], dtype float32, all values in [0, 1]
        """
        if not candidates:
            raise ValueError("candidates list is empty.")

        _today = today or date.today()
        trajectory_analyzer = TrajectoryAnalyzer()

        rows = [self._to_vector(c, _today, trajectory_analyzer) for c in candidates]
        matrix = np.stack(rows).astype(np.float32)

        assert matrix.shape == (len(candidates), FEATURE_DIM), (
            f"Expected [{len(candidates)} × {FEATURE_DIM}], got {matrix.shape}"
        )
        assert float(matrix.min()) >= 0.0 and float(matrix.max()) <= 1.0, (
            f"Values out of [0,1]: min={matrix.min():.4f} max={matrix.max():.4f}"
        )

        self._matrix = matrix
        self._id_to_idx = {c.candidate_id: i for i, c in enumerate(candidates)}

        if save:
            self._save(matrix, [c.candidate_id for c in candidates])

        logger.info(
            "FeatureStore built: shape=%s  min=%.3f  max=%.3f  open_to_work_sum=%d",
            matrix.shape,
            float(matrix.min()),
            float(matrix.max()),
            int(matrix[:, 2].sum()),
        )
        return matrix

    def load(self) -> None:
        """Load pre-built matrix and id map from disk."""
        if not self.feature_path.exists():
            raise FileNotFoundError(
                f"Feature store not found at '{self.feature_path}'. Run .build() first."
            )
        self._matrix = np.load(str(self.feature_path))
        ids: list[str] = np.load(str(self.ids_path), allow_pickle=True).tolist()
        self._id_to_idx = {cid: i for i, cid in enumerate(ids)}
        logger.info("FeatureStore loaded: shape=%s", self._matrix.shape)

    def get(self, candidate_id: str) -> np.ndarray:
        """Return the 30-dim feature vector for a single candidate."""
        self._require_loaded()
        idx = self._id_to_idx.get(candidate_id)
        if idx is None:
            raise KeyError(f"candidate_id '{candidate_id}' not in feature store.")
        return self._matrix[idx]

    def get_batch(self, candidate_ids: list[str]) -> np.ndarray:
        """Return [K × 30] matrix for an ordered list of candidate_ids."""
        self._require_loaded()
        # Single dict lookup per id + one numpy fancy-index — avoids K individual get() calls
        # each of which re-ran _require_loaded() and a redundant is_loaded check
        id_to_idx = self._id_to_idx
        indices = [id_to_idx[cid] for cid in candidate_ids]
        return self._matrix[indices]

    @property
    def is_loaded(self) -> bool:
        return self._matrix is not None

    # ── Core vector builder ───────────────────────────────────────────────────

    @staticmethod
    def _to_vector(c: CandidateFeatureVector, today: date, trajectory_analyzer: TrajectoryAnalyzer) -> np.ndarray:
        """
        Map one CandidateFeatureVector → float32 ndarray of shape [30].
        Every dimension is independently clipped to [0, 1].
        """
        s = c.signals
        traj = trajectory_analyzer.build_feature_vector(c)

        # [0] Recency: exponential decay on days since last active
        days_inactive = (today - s.last_active_date).days
        recency = math.exp(-config.RECENCY_LAMBDA * max(days_inactive, 0))

        # [1] Recruiter response rate
        response_rate = float(s.recruiter_response_rate)

        # [2] Open to work
        open_to_work = float(s.open_to_work_flag)

        # [3] Notice period score (tiered linear decay via config thresholds)
        nd = s.notice_period_days
        if nd <= config.NOTICE_PERIOD_IDEAL_MAX:
            notice_score = 1.0
        elif nd <= config.NOTICE_PERIOD_ACCEPTABLE_MAX:
            notice_score = 1.0 - 0.5 * (
                (nd - config.NOTICE_PERIOD_IDEAL_MAX)
                / (config.NOTICE_PERIOD_ACCEPTABLE_MAX - config.NOTICE_PERIOD_IDEAL_MAX)
            )
        elif nd <= config.NOTICE_PERIOD_MAX:
            notice_score = 0.5 - 0.3 * (
                (nd - config.NOTICE_PERIOD_ACCEPTABLE_MAX)
                / (config.NOTICE_PERIOD_MAX - config.NOTICE_PERIOD_ACCEPTABLE_MAX)
            )
        else:
            notice_score = 0.1

        # [4] GitHub activity (−1 when not linked → neutral default)
        github = (
            config.GITHUB_NOT_LINKED_DEFAULT
            if s.github_activity_score < 0
            else float(s.github_activity_score) / 100.0
        )

        # [5] Profile completeness
        completeness = float(s.profile_completeness_score) / 100.0

        # [6] Interview completion rate
        interview = float(s.interview_completion_rate)

        # [7] Offer acceptance rate (−1 = no history → neutral)
        offer = (
            config.OFFER_ACCEPTANCE_NO_HISTORY_DEFAULT
            if s.offer_acceptance_rate < 0
            else float(s.offer_acceptance_rate)
        )

        # [8–13] Platform engagement signals
        views_30d    = min(s.profile_views_received_30d / 100.0, 1.0)
        apps_30d     = min(s.applications_submitted_30d / 20.0, 1.0)
        search_30d   = min(s.search_appearance_30d / 200.0, 1.0)
        saved_30d    = min(s.saved_by_recruiters_30d / 20.0, 1.0)
        connections  = min(s.connection_count / 500.0, 1.0)
        endorsements = min(s.endorsements_received / config.ENDORSEMENT_BOOST_CAP, 1.0)

        # [14–16] Verification signals
        verified_email = float(s.verified_email)
        verified_phone = float(s.verified_phone)
        linkedin       = float(s.linkedin_connected)

        # [17] Willing to relocate
        relocate = float(s.willing_to_relocate)

        # [18] Notice period raw (inverted: lower days = higher score)
        notice_raw = 1.0 - min(s.notice_period_days / 90.0, 1.0)

        # [19] Preferred work mode
        work_mode = _WORK_MODE_MAP.get(s.preferred_work_mode, 0.5)

        # [20–21] Salary range (normalised to 80 LPA ceiling)
        salary_min = min(s.expected_salary_min_lpa / 80.0, 1.0)
        salary_max = min(s.expected_salary_max_lpa / 80.0, 1.0)

        # [22] Avg response time (inverted: faster = better; cap 72h)
        response_time = 1.0 - min(s.avg_response_time_hours / 72.0, 1.0)

        # [23–26] Career quality from TrajectoryAnalyzer
        yoe_score          = float(traj["yoe_score"])
        product_experience = float(traj["product_experience"])
        stability_score    = min(float(traj["avg_tenure"]) / 3.0, 1.0)
        job_hopper_stable  = 1.0 - float(traj["job_hopper"])

        # [27] Skill assessment: mean of all non-(-1) scores
        valid_scores = [v for v in s.skill_assessment_scores.values() if v >= 0]
        skill_assessment = (
            (sum(valid_scores) / len(valid_scores)) / 100.0
            if valid_scores
            else 0.5
        )

        # [28–29] Flag dimensions
        is_honeypot   = float(getattr(c, "is_honeypot", False))
        is_consulting = float(c.is_consulting_only)

        features = [
            recency,            # 0
            response_rate,      # 1
            open_to_work,       # 2
            notice_score,       # 3
            github,             # 4
            completeness,       # 5
            interview,          # 6
            offer,              # 7
            views_30d,          # 8
            apps_30d,           # 9
            search_30d,         # 10
            saved_30d,          # 11
            connections,        # 12
            endorsements,       # 13
            verified_email,     # 14
            verified_phone,     # 15
            linkedin,           # 16
            relocate,           # 17
            notice_raw,         # 18
            work_mode,          # 19
            salary_min,         # 20
            salary_max,         # 21
            response_time,      # 22
            yoe_score,          # 23
            product_experience, # 24
            stability_score,    # 25
            job_hopper_stable,  # 26
            skill_assessment,   # 27
            is_honeypot,        # 28
            is_consulting,      # 29
        ]

        vec = np.clip(np.array(features, dtype=np.float32), 0.0, 1.0)
        assert vec.shape == (FEATURE_DIM,)
        return vec

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, matrix: np.ndarray, ids: list[str]) -> None:
        self.feature_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.feature_path), matrix)
        np.save(str(self.ids_path), np.array(ids))
        logger.info(
            "Saved feature matrix → %s  |  ids → %s",
            self.feature_path, self.ids_path,
        )

    # ── Guards ────────────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                "FeatureStore not loaded. Call .build() or .load() first."
            )

    def __repr__(self) -> str:
        status = f"shape={self._matrix.shape}" if self.is_loaded else "not loaded"
        return f"FeatureStore({status})"