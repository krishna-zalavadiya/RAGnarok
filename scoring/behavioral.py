"""
scoring/behavioral.py — Behavioral signal scoring.

Turns the 23 `redrob_signals` fields on a CandidateFeatureVector into the
behavioral-facing fields of ComponentScores:

    behavioral_score      0-1, weighted per config.BEHAVIORAL_WEIGHTS
    recency_score         0-1, exp(-RECENCY_LAMBDA * days_since_active)
    notice_period_score   0-1, tiered notice-period fitness
    uncertainty_penalty   0.7-1.0, profile-sparsity confidence multiplier
    signal_count          0-9, how many "extra" signal types are populated

Sub-score formulas are kept identical to indexing/feature_store.py
(FeatureStore._to_vector dims [0], [3], [4], [5], [6], [1], [2]) so the
feature matrix and the composite-scoring breakdown never disagree about
what "recency" or "notice period" means for a given candidate.

Usage:
    scorer = BehavioralScorer()
    result = scorer.score(candidate)            # -> BehavioralResult
    results = scorer.score_all(candidates)      # -> dict[candidate_id, BehavioralResult]

Consumed by scoring/composite.py to populate ComponentScores.{behavioral_score,
recency_score, notice_period_score, uncertainty_penalty, signal_count}.

Dependencies:
  - config.py            (weights, thresholds, defaults)
  - pipeline/schemas.py   (CandidateFeatureVector, RedrobSignals)

No I/O. No network. Stateless — a single BehavioralScorer instance is safe
to share across threads / reuse across the whole candidate pool.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Optional

import config
from pipeline.schemas import CandidateFeatureVector, RedrobSignals

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Signal-presence checks — drive the uncertainty penalty
# ─────────────────────────────────────────────────────────────────────────────
# Each predicate answers "does this candidate actually have data for this
# signal type, or is it sitting at its empty/unknown default?". The count of
# True predicates is `signal_count`. A candidate with a thin profile (few
# True values) gets a lower `uncertainty_penalty`, which composite.py
# multiplies into the final score to avoid over-trusting sparse profiles.
_SIGNAL_PRESENCE_CHECKS: dict[str, Callable[[RedrobSignals], bool]] = {
    "profile_views_30d":      lambda s: s.profile_views_received_30d > 0,
    "applications_30d":       lambda s: s.applications_submitted_30d > 0,
    "search_appearance_30d":  lambda s: s.search_appearance_30d > 0,
    "recruiter_saves_30d":    lambda s: s.saved_by_recruiters_30d > 0,
    "connections":            lambda s: s.connection_count > 0,
    "endorsements":           lambda s: s.endorsements_received > 0,
    "github_linked":          lambda s: s.has_github,
    "offer_history":          lambda s: s.has_offer_history,
    "skill_assessments":      lambda s: len(s.skill_assessment_scores) > 0,
}


@dataclass(frozen=True)
class BehavioralResult:
    """Output of BehavioralScorer.score() for one candidate."""

    candidate_id: str

    behavioral_score: float        # ComponentScores.behavioral_score
    recency_score: float           # ComponentScores.recency_score
    notice_period_score: float     # ComponentScores.notice_period_score
    uncertainty_penalty: float     # ComponentScores.uncertainty_penalty
    signal_count: int              # ComponentScores.signal_count

    # Every input that fed behavioral_score, keyed exactly like
    # config.BEHAVIORAL_WEIGHTS — handy for the trust layer / debug UI.
    sub_scores: dict[str, float] = field(default_factory=dict)


class BehavioralScorer:
    """
    Stateless scorer for the behavioral component of the composite score.

    score() reads only `candidate.signals` (RedrobSignals); none of the
    skill/career fields on CandidateFeatureVector are touched.
    """

    def score(
        self,
        candidate: CandidateFeatureVector,
        today: Optional[date] = None,
    ) -> BehavioralResult:
        
        s = candidate.signals
        _today = today or date.today()

        recency_score = self._recency_score(s, _today)
        notice_period_score = self._notice_period_score(s.notice_period_days)

        sub_scores: dict[str, float] = {
            "recency":              recency_score,
            "response_rate":        float(s.recruiter_response_rate),
            "open_to_work":         float(s.open_to_work_flag),
            "notice_period":        notice_period_score,
            "github_activity":      self._github_score(s),
            "profile_completeness": float(s.profile_completeness_score) / 100.0,
            "interview_completion": float(s.interview_completion_rate),
        }

        behavioral_score = sum(
            config.BEHAVIORAL_WEIGHTS[name] * value
            for name, value in sub_scores.items()
        )
        behavioral_score = float(min(max(behavioral_score, 0.0), 1.0))

        signal_count = sum(1 for check in _SIGNAL_PRESENCE_CHECKS.values() if check(s))
        uncertainty_penalty = self._uncertainty_penalty(signal_count)

        return BehavioralResult(
            candidate_id=candidate.candidate_id,
            behavioral_score=behavioral_score,
            recency_score=recency_score,
            notice_period_score=notice_period_score,
            uncertainty_penalty=uncertainty_penalty,
            signal_count=signal_count,
            sub_scores=sub_scores,
        )

    def score_all(
        self,
        candidates: list[CandidateFeatureVector],
        today: Optional[date] = None,
    ) -> dict[str, BehavioralResult]:
        """Convenience batch wrapper — keyed by candidate_id."""
        return {c.candidate_id: self.score(c, today) for c in candidates}

    # ── Sub-score helpers ────────────────────────────────────────────────────
    # Kept byte-for-byte equivalent to indexing/feature_store.py's
    # FeatureStore._to_vector formulas for dims [0] recency, [3] notice_score,
    # [4] github, [1] response_rate, [2] open_to_work, [5] completeness,
    # [6] interview.

    @staticmethod
    def _recency_score(s: RedrobSignals, today: date) -> float:
        """Exponential recency decay: e^(-λ · days_since_active)."""
        days_inactive = (today - s.last_active_date).days
        return math.exp(-config.RECENCY_LAMBDA * max(days_inactive, 0))

    @staticmethod
    def _notice_period_score(notice_period_days: int) -> float:
        """Tiered linear decay via config NOTICE_PERIOD_* thresholds."""
        nd = notice_period_days
        if nd <= config.NOTICE_PERIOD_IDEAL_MAX:
            return 1.0
        if nd <= config.NOTICE_PERIOD_ACCEPTABLE_MAX:
            return 1.0 - 0.5 * (
                (nd - config.NOTICE_PERIOD_IDEAL_MAX)
                / (config.NOTICE_PERIOD_ACCEPTABLE_MAX - config.NOTICE_PERIOD_IDEAL_MAX)
            )
        if nd <= config.NOTICE_PERIOD_MAX:
            return 0.5 - 0.3 * (
                (nd - config.NOTICE_PERIOD_ACCEPTABLE_MAX)
                / (config.NOTICE_PERIOD_MAX - config.NOTICE_PERIOD_ACCEPTABLE_MAX)
            )
        return 0.1

    @staticmethod
    def _github_score(s: RedrobSignals) -> float:
        """-1 (not linked) → neutral default; otherwise score/100."""
        if not s.has_github:
            return config.GITHUB_NOT_LINKED_DEFAULT
        return float(s.github_activity_score) / 100.0

    @staticmethod
    def _uncertainty_penalty(signal_count: int) -> float:
        """
        Linear interpolation: 0 signals -> UNCERTAINTY_PENALTY_FLOOR,
        >= MIN_SIGNAL_TYPES_FOR_FULL_CONFIDENCE signals -> 1.0.
        """
        floor = config.UNCERTAINTY_PENALTY_FLOOR
        ratio = min(signal_count / config.MIN_SIGNAL_TYPES_FOR_FULL_CONFIDENCE, 1.0)
        return floor + (1.0 - floor) * ratio
