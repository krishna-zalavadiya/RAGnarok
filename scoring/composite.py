from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import config
from pipeline.schemas import CandidateFeatureVector, JDIntent, RRFResult
from scoring.behavioral import BehavioralScorer
from scoring.career_quality import  CareerQualityScorer
from scoring.skill_match import SkillMatchScorer

logger = logging.getLogger(__name__)

# ── Primary component weights (direct from config) ────────────────────────────
_W_SKILL     = config.WEIGHT_SKILL       # 0.40
_W_CAREER    = config.WEIGHT_CAREER      # 0.35
_W_BEHAVIORAL = config.WEIGHT_BEHAVIORAL # 0.25

assert abs(_W_SKILL + _W_CAREER + _W_BEHAVIORAL - 1.0) < config._WEIGHT_SUM_TOLERANCE

# ── Cross-encoder blend factor ────────────────────────────────────────────────
# final = (1 - CE_BLEND) × weighted_sum + CE_BLEND × ce_score
# 0.30 gives cross-encoder meaningful influence without overriding config weights.
_CE_BLEND: float = getattr(config, "CE_BLEND_FACTOR", 0.30)

# ── Uncertainty (sparse-profile) penalty ──────────────────────────────────────
_MIN_SIGNALS: int        = config.MIN_SIGNAL_TYPES_FOR_FULL_CONFIDENCE  # 5
_PENALTY_FLOOR: float    = config.UNCERTAINTY_PENALTY_FLOOR             # 0.70

# ── Preferred locations (per-city float from config) ──────────────────────────
# Keys are already lowercase in config dict.
_PREFERRED_LOCATIONS: dict[str, float] = {
    k.lower().strip(): v for k, v in config.PREFERRED_LOCATIONS.items()
}
_RELOCATION_BONUS: float = config.RELOCATION_BONUS  # 0.03


@dataclass(slots=True)
class ComponentScores:
    """
    Full transparency record for one candidate's final composite score.

    Attributes:
        candidate_id             CAND_XXXXXXX.
        final_score              Composite score in [0.0, 1.0] — primary sort key.
        weighted_sum             (0.40×skill + 0.35×career + 0.25×behavioral)
                                 before CE blend and adjustments.
        cross_encoder_score      From scoring/cross_encoder.py via RRFResult.
        skill_match_score        From scoring/skill_match.py.
        career_quality_score     From scoring/career_quality.py.
        behavioral_score         From scoring/behavioral.py.
        rrf_score                From retrieval/rrf_fusion.py — preserved for eval.
        paths_present            Retrieval paths that surfaced this candidate.
        location_bonus_applied   Additive bonus value applied (0.0 if none).
        uncertainty_multiplier   Confidence multiplier applied (1.0 = no penalty).
        hard_disqualifier        True if skill disqualifier forced score to 0.
        honeypot_override        True if honeypot flag forced score to 0.
    """
    candidate_id:           str
    final_score:            float
    weighted_sum:           float
    cross_encoder_score:    float
    skill_match_score:      float
    career_quality_score:   float
    behavioral_score:       float
    rrf_score:              float
    paths_present:          list[str]
    location_bonus_applied: float
    uncertainty_multiplier: float
    hard_disqualifier:      bool
    honeypot_override:      bool


def _count_signal_types(cfv: CandidateFeatureVector) -> int:
    """
    Count non-empty signal types for uncertainty penalty.

    Each distinct category that has a non-default value counts as one signal type.
    Mirrors config.MIN_SIGNAL_TYPES_FOR_FULL_CONFIDENCE intent.
    """
    s = cfv.signals
    count = 0
    if s.open_to_work_flag:                             count += 1
    if s.github_activity_score >= 0:                    count += 1
    if s.linkedin_connected:                            count += 1
    if s.verified_email or s.verified_phone:            count += 1
    if s.skill_assessment_scores:                       count += 1
    if s.recruiter_response_rate > 0:                   count += 1
    if s.profile_completeness_score > 0:                count += 1
    if cfv.skills:                                      count += 1
    if cfv.career_history:                              count += 1
    if cfv.education:                                   count += 1
    return count


def _uncertainty_multiplier(cfv: CandidateFeatureVector) -> float:
    """
    Linear interpolation from UNCERTAINTY_PENALTY_FLOOR → 1.0 based on
    how many distinct signal types are present.

    0 signals → PENALTY_FLOOR
    ≥ MIN_SIGNALS → 1.0 (full confidence)
    """
    n = _count_signal_types(cfv)
    if n >= _MIN_SIGNALS:
        return 1.0
    frac = n / _MIN_SIGNALS
    return _PENALTY_FLOOR + frac * (1.0 - _PENALTY_FLOOR)


def _location_bonus(cfv: CandidateFeatureVector, jd: JDIntent) -> float:
    """
    Per-city bonus from config.PREFERRED_LOCATIONS, plus relocation bonus.

    City match takes priority — relocation bonus is only added when the
    candidate is NOT already in a preferred city (avoids double-counting).
    """
    city = cfv.location_lower.strip()
    if city in _PREFERRED_LOCATIONS:
        return _PREFERRED_LOCATIONS[city]
    if jd.relocation_accepted and cfv.signals.willing_to_relocate:
        return _RELOCATION_BONUS
    return 0.0


class CompositeScorer:
    """
    Fuses skill-match, career-quality, behavioral, and cross-encoder signals
    into a single ranked list of candidates.

    Weights follow config.py exactly:
        WEIGHT_SKILL=0.40, WEIGHT_CAREER=0.35, WEIGHT_BEHAVIORAL=0.25.
    Cross-encoder blended in post-fusion at CE_BLEND_FACTOR (default 0.30).

    Usage in pipeline/runner.py:
        behavioral_scorer = BehavioralScorer()
        composite = CompositeScorer(jd, candidate_store, behavioral_scorer)
        ranked: list[ComponentScores] = composite.rank(rrf_pool)
    """

    def __init__(
        self,
        jd: JDIntent,
        candidate_store: dict[str, CandidateFeatureVector],
        behavioral_scorer: Optional[BehavioralScorer] = None,
    ) -> None:
        if not isinstance(jd, JDIntent):
            raise TypeError(f"jd must be JDIntent, got {type(jd).__name__}.")
        if not isinstance(candidate_store, dict):
            raise TypeError(
                f"candidate_store must be dict[str, CandidateFeatureVector], "
                f"got {type(candidate_store).__name__}."
            )

        self._jd               = jd
        self._candidate_store  = candidate_store
        # Accept pre-built scorer from runner (avoids double-scoring)
        # or create one internally if not provided.
        self._behavioral       = behavioral_scorer or BehavioralScorer()

        # Lazy — built on first rank() call
        self._career_scorer:  Optional[CareerQualityScorer] = None
        self._skill_scorer:   Optional[SkillMatchScorer]    = None

    def rank(self, pool: list[RRFResult]) -> list[ComponentScores]:
        if not isinstance(pool, list):
            raise TypeError(f"pool must be list[RRFResult], got {type(pool).__name__}.")
        if not pool:
            return []

        t0 = time.perf_counter()

        # ── Resolve candidates ────────────────────────────────────────────
        pool_pairs: list[tuple[RRFResult, CandidateFeatureVector]] = []
        skipped: list[str] = []
        for result in pool:
            cfv = self._candidate_store.get(result.candidate_id)
            if cfv is None:
                logger.warning(
                    "CompositeScorer: '%s' not in candidate_store — skipping.",
                    result.candidate_id,
                )
                skipped.append(result.candidate_id)
                continue
            pool_pairs.append((result, cfv))

        if not pool_pairs:
            logger.error("CompositeScorer: no valid candidates. Returning [].")
            return []

        self._ensure_scorers()
        candidates_only = [cfv for _, cfv in pool_pairs]

        # Score all three primary components in batch
        career_results    = self._career_scorer.score_all(candidates_only)
        skill_results     = self._skill_scorer.score_all(candidates_only)
        behavioral_results = self._behavioral.score_all(candidates_only)

        # ── Fuse ─────────────────────────────────────────────────────────
        output: list[ComponentScores] = []

        for rrf_result, cfv in pool_pairs:
            cid = cfv.candidate_id

            skill_score    = skill_results[cid].skill_match_score
            career_score   = career_results[cid].career_quality_score
            beh_result     = behavioral_results.get(cid)
            beh_score      = beh_result.behavioral_score if beh_result else 0.5
            ce_score       = float(rrf_result.cross_encoder_score)

            # Weighted sum of three primary components
            weighted_sum = (
                _W_SKILL     * skill_score
                + _W_CAREER  * career_score
                + _W_BEHAVIORAL * beh_score
            )

            # Cross-encoder blend: refines without overriding config weights
            blended = (1.0 - _CE_BLEND) * weighted_sum + _CE_BLEND * ce_score

            # Location bonus (additive, clipped to 1.0)
            loc_bonus = _location_bonus(cfv, self._jd)
            blended   = min(1.0, blended + loc_bonus)

            # Uncertainty multiplier (sparse profile penalty)
            unc_mult = _uncertainty_multiplier(cfv)
            blended  *= unc_mult

            blended = float(max(0.0, min(1.0, blended)))

            # Hard overrides — after all arithmetic
            hard_disq    = skill_results[cid].hard_disqualifier
            is_honeypot  = bool(getattr(cfv, "is_honeypot", False))
            honeypot_flag = False

            if hard_disq or is_honeypot:
                blended = 0.0
                if is_honeypot:
                    honeypot_flag = True

            output.append(ComponentScores(
                candidate_id=cid,
                final_score=round(blended, 6),
                weighted_sum=round(weighted_sum, 6),
                cross_encoder_score=round(ce_score, 6),
                skill_match_score=round(skill_score, 6),
                career_quality_score=round(career_score, 6),
                behavioral_score=round(beh_score, 6),
                rrf_score=round(float(rrf_result.rrf_score), 8),
                paths_present=list(rrf_result.paths_present),
                location_bonus_applied=round(loc_bonus, 4),
                uncertainty_multiplier=round(unc_mult, 4),
                hard_disqualifier=hard_disq,
                honeypot_override=honeypot_flag,
            ))

        output.sort(key=lambda r: (-r.final_score, r.candidate_id))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "CompositeScorer: ranked %d/%d candidates in %.1f ms "
            "(top=%.4f, mean=%.4f, loc_bonus=%d, unc_penalised=%d, "
            "zero_scored=%d, skipped=%d).",
            len(output), len(pool), elapsed_ms,
            output[0].final_score if output else 0.0,
            sum(r.final_score for r in output) / len(output) if output else 0.0,
            sum(1 for r in output if r.location_bonus_applied > 0),
            sum(1 for r in output if r.uncertainty_multiplier < 1.0),
            sum(1 for r in output if r.final_score == 0.0),
            len(skipped),
        )
        if elapsed_ms > 500.0:
            logger.warning(
                "CompositeScorer took %.1f ms — expected < 500 ms.", elapsed_ms
            )

        return output

    def _ensure_scorers(self) -> None:
        if self._career_scorer is None:
            self._career_scorer = CareerQualityScorer(self._jd)
        if self._skill_scorer is None:
            self._skill_scorer = SkillMatchScorer(self._jd)

    def __repr__(self) -> str:
        return (
            f"CompositeScorer("
            f"weights=[skill={_W_SKILL}, career={_W_CAREER}, beh={_W_BEHAVIORAL}], "
            f"ce_blend={_CE_BLEND}, store_size={len(self._candidate_store)})"
        )


def rank_candidates(
    pool: list[RRFResult],
    jd: JDIntent,
    candidate_store: dict[str, CandidateFeatureVector],
    behavioral_scorer: Optional[BehavioralScorer] = None,
) -> list[ComponentScores]:
    """
    Convenience wrapper for pipeline/runner.py.

    Pass behavioral_scorer if BehavioralScorer has already been run upstream
    (runner.py calls it before composite) to avoid re-scoring.
    """
    return CompositeScorer(jd, candidate_store, behavioral_scorer).rank(pool)
