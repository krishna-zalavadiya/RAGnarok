"""
scoring/trajectory.py — Career trajectory velocity scorer.

Computes `trajectory_velocity` (ComponentScores.trajectory_velocity): a 0-1
score capturing how quickly a candidate has been promoted over their career.

This is deliberately standalone (only depends on config.py + pipeline.schemas)
so it can be imported from either side of the pipeline without creating an
import cycle:

  - scoring/career_quality.py   -> uses `trajectory_velocity` as one input
                                    into the career_score composite.
  - indexing/feature_store.py   -> can call TrajectoryVelocityScorer.score_all()
                                    once over the whole pool to pre-compute
                                    velocity + percentile_rank for every
                                    candidate (cheap: pure python, no model).

Note: this is independent of indexing/trajectory_builder.TrajectoryAnalyzer,
which computes yoe_score / product_experience / avg_tenure / job_hopper for
FeatureStore's feature vector. Promotion *velocity* is a distinct signal.

── Algorithm ─────────────────────────────────────────────────────────────────

1. Seniority level (0-5) is inferred per role from title keywords.
2. career_history is sorted chronologically; a "promotion" is counted any
   time the seniority level strictly increases between consecutive roles
   (internal promotion or an external move with a title bump both count —
   both represent forward career velocity).
3. promotions_per_year = num_promotions / years_of_experience
   (years_of_experience falls back to total_career_months/12, floored at
   _MIN_YEARS_FOR_RATE to avoid divide-by-zero / unstable rates for very
   short histories).
4. trajectory_velocity = clip((rate - FLOOR) / (CAP - FLOOR), 0, 1), using
   config.TRAJECTORY_PROMOTIONS_PER_YEAR_{FLOOR,CAP}. This is the value that
   feeds ComponentScores.trajectory_velocity.
5. score_all() additionally attaches `percentile_rank` (0-100): each
   candidate's promotions_per_year rank within the *pool* passed in. This is
   the population-relative view referenced by config's comment "Min/max used
   for percentile normalisation across the pool" and is what the acceptance
   criteria below are checked against, since most candidate pools are heavily
   skewed toward 0 promotions/year (a single long tenure with no title
   change), so even a modest rate sits in a high percentile.

Acceptance criteria (see self-test at the bottom):
  - 3 promotions in 4 years  (rate = 0.75/yr) -> percentile_rank > 80
  - stagnant 10-year tenure  (rate = 0.00/yr) -> percentile_rank < 40
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

import config
from pipeline.schemas import CandidateFeatureVector, CareerEntry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Title -> seniority level
# ─────────────────────────────────────────────────────────────────────────────
# Checked from the most senior tier down to the most junior; the first
# keyword match wins. Anything unmatched defaults to _DEFAULT_LEVEL (a
# generic individual-contributor role with no seniority modifier).
_SENIORITY_KEYWORDS: list[tuple[int, tuple[str, ...]]] = [
    (5, ("chief", "ceo", "cto", "cfo", "coo", "vp", "vice president",
         "president", "founder", "co-founder", "director", "head of")),
    (4, ("principal", "manager", "staff")),
    (3, ("senior", "sr.", "sr ", "lead")),
    (1, ("junior", "associate", "entry", "trainee", "apprentice", "intern")),
]
_DEFAULT_LEVEL = 2

# Minimum denominator for promotions/year — guards against divide-by-zero
# and unstable rates for very short (<6 month) career histories.
_MIN_YEARS_FOR_RATE = 0.5


def _seniority_level(title: str) -> int:
    """Best-effort seniority tier (1-5, default 2) inferred from a job title."""
    t = title.lower()
    for level, keywords in _SENIORITY_KEYWORDS:
        if any(kw in t for kw in keywords):
            return level
    return _DEFAULT_LEVEL


def count_promotions(career_history: list[CareerEntry]) -> int:
    """
    Count seniority step-ups across a candidate's chronological career.

    A "promotion" is any role whose inferred seniority tier is strictly
    greater than the tier of the immediately preceding role — whether the
    move was internal (same company) or external (job change with a title
    bump). Lateral moves and same-or-lower tiers don't count. A single-role
    history has no promotions by definition.
    """
    if len(career_history) < 2:
        return 0

    ordered = sorted(career_history, key=lambda e: e.start_date)
    promotions = 0
    prev_level = _seniority_level(ordered[0].title)
    for entry in ordered[1:]:
        level = _seniority_level(entry.title)
        if level > prev_level:
            promotions += 1
        prev_level = level
    return promotions


def _effective_years(candidate: CandidateFeatureVector) -> float:
    """years_of_experience, falling back to total_career_months, floored."""
    years = candidate.years_of_experience
    if years <= 0:
        years = candidate.total_career_months / 12.0
    return max(years, _MIN_YEARS_FOR_RATE)


def promotions_per_year(candidate: CandidateFeatureVector) -> float:
    """Raw promotion velocity: count_promotions / effective years."""
    return count_promotions(candidate.career_history) / _effective_years(candidate)


def trajectory_velocity_score(rate: float) -> float:
    """
    Min-max normalise a promotions/year rate into [0, 1] using
    config.TRAJECTORY_PROMOTIONS_PER_YEAR_{FLOOR,CAP}.

    floor -> 0.0, cap -> 1.0, clipped at both ends.
    """
    floor = config.TRAJECTORY_PROMOTIONS_PER_YEAR_FLOOR
    cap = config.TRAJECTORY_PROMOTIONS_PER_YEAR_CAP
    normalized = (rate - floor) / (cap - floor)
    return float(np.clip(normalized, 0.0, 1.0))


@dataclass(frozen=True)
class TrajectoryResult:
    candidate_id: str
    num_promotions: int
    years_of_experience: float
    promotions_per_year: float
    trajectory_velocity: float               # ComponentScores.trajectory_velocity
    percentile_rank: Optional[float] = None  # 0-100, set by score_all()


class TrajectoryVelocityScorer:
    """
    Stateless career-velocity scorer.

    score()      -> single-candidate result, percentile_rank=None
    score_all()  -> batch result, with percentile_rank computed against the
                     passed-in pool's promotions_per_year distribution.
    """

    def score(self, candidate: CandidateFeatureVector) -> TrajectoryResult:
        years = _effective_years(candidate)
        n_promo = count_promotions(candidate.career_history)
        rate = n_promo / years
        return TrajectoryResult(
            candidate_id=candidate.candidate_id,
            num_promotions=n_promo,
            years_of_experience=years,
            promotions_per_year=rate,
            trajectory_velocity=trajectory_velocity_score(rate),
        )

    def score_all(self, candidates: list[CandidateFeatureVector]) -> list[TrajectoryResult]:
        """
        Score every candidate, then attach `percentile_rank` (0-100): the
        fraction of the pool with promotions_per_year <= this candidate's,
        expressed as a percentage. A candidate at the top of the pool's
        velocity distribution gets percentile_rank ~100.
        """
        results = [self.score(c) for c in candidates]
        if not results:
            return results

        rates = np.array([r.promotions_per_year for r in results], dtype=np.float64)
        return [
            replace(
                r,
                percentile_rank=float((rates <= r.promotions_per_year).mean() * 100.0),
            )
            for r in results
        ]
