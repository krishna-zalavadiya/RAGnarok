from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Union

import config
from pipeline.schemas import CandidateFeatureVector, JDIntent, RRFResult
from scoring.behavioral import BehavioralScorer
from scoring.career_quality import CareerQualityScorer
from scoring.skill_match import SkillMatchScorer
from scoring.trajectory import TrajectoryVelocityScorer

logger = logging.getLogger(__name__)

# ── Primary component weights (direct from config) ────────────────────────────
_W_SKILL      = config.WEIGHT_SKILL       # 0.40
_W_CAREER     = config.WEIGHT_CAREER      # 0.30
_W_BEHAVIORAL = config.WEIGHT_BEHAVIORAL  # 0.20
_W_TRAJECTORY = config.WEIGHT_TRAJECTORY  # 0.10

assert abs(_W_SKILL + _W_CAREER + _W_BEHAVIORAL + _W_TRAJECTORY - 1.0) < config._WEIGHT_SUM_TOLERANCE

# ── Cross-encoder blend factor ────────────────────────────────────────────────
# final = (1 - CE_BLEND) × weighted_sum + CE_BLEND × ce_score
_CE_BLEND: float = getattr(config, "CE_BLEND_FACTOR", 0.30)

# ── Preferred locations (per-city float from config) ──────────────────────────
_PREFERRED_LOCATIONS: dict[str, float] = {
    k.lower().strip(): v for k, v in config.PREFERRED_LOCATIONS.items()
}
_RELOCATION_BONUS: float = config.RELOCATION_BONUS  # 0.03


@dataclass(slots=True)
class ComponentScores:
    
    candidate_id:                str
    final_score:                 float
    weighted_sum:                float
    cross_encoder_score:         float
    skill_match_score:           float
    career_quality_score:        float
    behavioral_score:            float
    trajectory_velocity:         float
    rrf_score:                   float
    paths_present:               list[str]
    location_bonus_applied:      float
    uncertainty_penalty_applied: float
    hard_disqualifier:           bool
    honeypot_override:           bool


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
    
    def __init__(
        self,
        jd: JDIntent,
        candidate_store: Union[list[CandidateFeatureVector], dict[str, CandidateFeatureVector]],
        behavioral_scorer: Optional[BehavioralScorer] = None,
    ) -> None:
        if not isinstance(jd, JDIntent):
            raise TypeError(f"jd must be JDIntent, got {type(jd).__name__}.")

        if isinstance(candidate_store, dict):
            self._candidate_store: dict[str, CandidateFeatureVector] = candidate_store
        elif isinstance(candidate_store, list):
            self._candidate_store = {c.candidate_id: c for c in candidate_store}
        else:
            raise TypeError(
                f"candidate_store must be list[CandidateFeatureVector] or "
                f"dict[str, CandidateFeatureVector], got {type(candidate_store).__name__}."
            )

        self._jd = jd
        # Accept pre-built scorer from runner (avoids double-scoring when
        # runner.py already called bscorer.score_all() for behavioral ranking).
        self._behavioral = behavioral_scorer or BehavioralScorer()

        # Lazy — built on first rank() call.
        # SkillMatchScorer takes no constructor args (jd passed per-call);
        # CareerQualityScorer requires jd upfront;
        # TrajectoryVelocityScorer takes none at all.
        self._career_scorer:     Optional[CareerQualityScorer]     = None
        self._skill_scorer:      Optional[SkillMatchScorer]        = None
        self._trajectory_scorer: Optional[TrajectoryVelocityScorer] = None

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

        # ── Batch score all components ────────────────────────────────────
        # NOTE: SkillMatchScorer.score_all requires jd as a second arg.
        career_results     = self._career_scorer.score_all(candidates_only)
        skill_results      = self._skill_scorer.score_all(candidates_only, self._jd)
        behavioral_results = self._behavioral.score_all(candidates_only)
        trajectory_results = {
            r.candidate_id: r
            for r in self._trajectory_scorer.score_all(candidates_only)
        }

        # ── Fuse ─────────────────────────────────────────────────────────
        output: list[ComponentScores] = []

        for rrf_result, cfv in pool_pairs:
            cid = cfv.candidate_id

            skill_score  = skill_results[cid].skill_match_score
            career_score = career_results[cid].career_quality_score

            beh_result = behavioral_results.get(cid)
            beh_score  = beh_result.behavioral_score if beh_result else 0.5

            # cross_encoder_score must be set by cross_encoder.py before
            # this method is called. If it is still 0.0, the CE blend will
            # pull the score down — that's intentional (CE unavailable = rely
            # more on weighted_sum, but do not silently ignore the blend).
            ce_score = float(rrf_result.cross_encoder_score)

            traj_result         = trajectory_results.get(cid)
            trajectory_velocity = traj_result.trajectory_velocity if traj_result else 0.0

            # ── Step 1: weighted sum of three primary components ──────────
            weighted_sum = (
                _W_SKILL      * skill_score
                + _W_CAREER   * career_score
                + _W_BEHAVIORAL * beh_score
                + _W_TRAJECTORY * trajectory_velocity
            )

            # ── Step 2: cross-encoder blend ───────────────────────────────
            # Refines placement without overriding config weights entirely.
            if ce_score > 0.0:
                blended = (1.0 - _CE_BLEND) * weighted_sum + _CE_BLEND * ce_score
            else:
                blended = weighted_sum

            # ── Step 3: uncertainty penalty ─────────────────────────
            unc_penalty = beh_result.uncertainty_penalty if beh_result else 1.0
            blended *= unc_penalty

            # ── Step 4: location bonus (hard fact, not uncertainty-sensitive) ──
            loc_bonus = _location_bonus(cfv, self._jd)
            blended = min(1.0, blended + loc_bonus)
            
            blended = float(max(0.0, min(1.0, blended)))

            # ── Step 4b ────────────────────────────────────────────────────
            req_coverage = skill_results[cid].required_score
            if req_coverage < config.REQUIRED_SKILL_COVERAGE_THRESHOLD:  # e.g. 0.30
                blended = min(blended, config.REQUIRED_SKILL_COVERAGE_MAX_SCORE)  # e.g. 0.45

            # ── Step 5: hard overrides (after all arithmetic) ─────────────
            hard_disq     = skill_results[cid].hard_disqualifier
            is_honeypot   = bool(getattr(cfv, "is_honeypot", False))
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
                trajectory_velocity=round(trajectory_velocity, 6),
                rrf_score=round(float(rrf_result.rrf_score), 8),
                paths_present=list(rrf_result.paths_present),
                location_bonus_applied=round(loc_bonus, 4),
                uncertainty_penalty_applied=round(unc_penalty, 4),
                hard_disqualifier=hard_disq,
                honeypot_override=honeypot_flag,
            ))

        # Primary sort: final_score descending. Tie-break: candidate_id
        # ascending — spec-compliant (matches submission CSV sort requirement).
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
            sum(1 for r in output if r.uncertainty_penalty_applied < 1.0),
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
            self._skill_scorer = SkillMatchScorer()
        if self._trajectory_scorer is None:
            self._trajectory_scorer = TrajectoryVelocityScorer()

    def __repr__(self) -> str:
        return (
            f"CompositeScorer("
            f"weights=[skill={_W_SKILL}, career={_W_CAREER}, beh={_W_BEHAVIORAL}], "
            f"ce_blend={_CE_BLEND}, store_size={len(self._candidate_store)})"
        )


def rank_candidates(
    pool: list[RRFResult],
    jd: JDIntent,
    candidate_store: Union[list[CandidateFeatureVector], dict[str, CandidateFeatureVector]],
    behavioral_scorer: Optional[BehavioralScorer] = None,
) -> list[ComponentScores]:
    """
    Convenience wrapper for pipeline/runner.py.

    Pass behavioral_scorer if BehavioralScorer has already been run upstream
    (runner.py calls it before composite) to avoid re-scoring.

    NOTE: pool must have cross_encoder_score already set by cross_encoder.py.
    """
    return CompositeScorer(jd, candidate_store, behavioral_scorer).rank(pool)