from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import config
from indexing.bm25_builder import ONTOLOGY
from pipeline.schemas import CandidateFeatureVector, JDIntent, SkillRecord

logger = logging.getLogger(__name__)

# ── Proficiency multipliers — directly from config ────────────────────────────
# Used for required skills at full weight.
_PROF_REQUIRED: dict[str, float] = {
    k: v for k, v in config.PROFICIENCY_MULTIPLIERS.items()
}
# Nice-to-have uses 70% of required multipliers — same ratio as v1 but now
# derived from config so calibration changes propagate automatically.
_PROF_NICE: dict[str, float] = {
    k: v * 0.70 for k, v in config.PROFICIENCY_MULTIPLIERS.items()
}
_PROF_DEFAULT: float = config.PROFICIENCY_MULTIPLIERS.get("beginner", 0.40)

# ── Duration trust factor ─────────────────────────────────────────────────────
# Maps duration_months → trust multiplier in [DURATION_TRUST_MIN, 1.0].
# Linear: 0 months → DURATION_TRUST_MIN, ≥ DURATION_TRUST_MAX_MONTHS → 1.0.
_DURATION_TRUST_MIN: float       = config.DURATION_TRUST_MIN        # 0.50
_DURATION_TRUST_MAX_MONTHS: int  = config.DURATION_TRUST_MAX_MONTHS # 24


def _duration_trust(duration_months: int) -> float:
    if duration_months <= 0:
        return _DURATION_TRUST_MIN
    if duration_months >= _DURATION_TRUST_MAX_MONTHS:
        return 1.0
    frac = duration_months / _DURATION_TRUST_MAX_MONTHS
    return _DURATION_TRUST_MIN + frac * (1.0 - _DURATION_TRUST_MIN)


# ── Endorsement boost ─────────────────────────────────────────────────────────
# Additive boost per skill: log-scaled, capped at ENDORSEMENT_BOOST_MAX (0.10).
_ENDORSEMENT_CAP: int    = config.ENDORSEMENT_BOOST_CAP  # 50
_ENDORSEMENT_MAX: float  = config.ENDORSEMENT_BOOST_MAX  # 0.10


def _endorsement_boost(endorsements: int) -> float:
    if endorsements <= 0:
        return 0.0
    clamped = min(endorsements, _ENDORSEMENT_CAP)
    # log(1 + clamped) / log(1 + cap) → [0, 1], scaled to ENDORSEMENT_MAX
    return _ENDORSEMENT_MAX * math.log1p(clamped) / math.log1p(_ENDORSEMENT_CAP)


# ── Assessment score blending ─────────────────────────────────────────────────
# If skill.assessment_score >= threshold, blend it with proficiency multiplier.
# effective = (1 - ASSESSMENT_SCORE_WEIGHT) × prof + ASSESSMENT_SCORE_WEIGHT × (score/100)
_ASSESSMENT_THRESHOLD: float = config.ASSESSMENT_SCORE_THRESHOLD  # 40.0
_ASSESSMENT_WEIGHT: float    = config.ASSESSMENT_SCORE_WEIGHT      # 0.20


def _effective_proficiency(skill: SkillRecord, prof_multiplier: float) -> float:
    if skill.assessment_score >= _ASSESSMENT_THRESHOLD:
        assessment_norm = min(skill.assessment_score / 100.0, 1.0)
        return (
            (1.0 - _ASSESSMENT_WEIGHT) * prof_multiplier
            + _ASSESSMENT_WEIGHT * assessment_norm
        )
    return prof_multiplier


# ── Disqualifier penalties ────────────────────────────────────────────────────
_DISQUALIFIER_HARD_PENALTY: float = getattr(config, "DISQUALIFIER_HARD_PENALTY", 0.25)
_DISQUALIFIER_SOFT_PENALTY: float = getattr(config, "DISQUALIFIER_SOFT_PENALTY", 0.70)
_DISQUALIFIER_HARD_PROFICIENCY = frozenset(("expert", "advanced", "intermediate"))


@dataclass(slots=True)
class SkillMatchResult:
    candidate_id:          str
    skill_match_score:     float
    required_score:        float
    nice_to_have_score:    float
    matched_required:      list[str]
    matched_nice_to_have:  list[str]
    matched_disqualifiers: list[str]
    hard_disqualifier:     bool
    soft_disqualifier:     bool


class SkillMatchScorer:

    def __init__(self, jd: JDIntent) -> None:
        if not isinstance(jd, JDIntent):
            raise TypeError(f"jd must be JDIntent, got {type(jd).__name__}.")
        self._jd = jd
        self._required_expanded: frozenset[str]    = self._expand_skills(jd.expanded_required)
        self._nice_expanded: frozenset[str]         = self._expand_skills(jd.nice_to_have_skills)
        self._disqualifiers_lower: frozenset[str]   = frozenset(
            s.lower().strip() for s in jd.disqualifier_skills if s
        )
        logger.debug(
            "SkillMatchScorer initialised: %d required, %d nice, %d disqualifiers.",
            len(self._required_expanded),
            len(self._nice_expanded),
            len(self._disqualifiers_lower),
        )

    def score(self, candidate: CandidateFeatureVector) -> SkillMatchResult:
        skills = candidate.skills
        required_score, matched_req   = self._score_required(skills)
        nice_score,     matched_nice  = self._score_nice_to_have(skills)
        hard_disq, soft_disq, matched_disq = self._check_disqualifiers(skills)

        # config.REQUIRED_SKILL_WEIGHT=2.0, NICE_TO_HAVE_SKILL_WEIGHT=1.0
        # Normalised: required gets 2/(2+1)=0.667, nice gets 1/(2+1)=0.333
        req_w  = config.REQUIRED_SKILL_WEIGHT / (
            config.REQUIRED_SKILL_WEIGHT + config.NICE_TO_HAVE_SKILL_WEIGHT
        )
        nice_w = config.NICE_TO_HAVE_SKILL_WEIGHT / (
            config.REQUIRED_SKILL_WEIGHT + config.NICE_TO_HAVE_SKILL_WEIGHT
        )
        raw = req_w * required_score + nice_w * nice_score

        if hard_disq:
            raw *= _DISQUALIFIER_HARD_PENALTY
        elif soft_disq:
            raw *= _DISQUALIFIER_SOFT_PENALTY

        return SkillMatchResult(
            candidate_id=candidate.candidate_id,
            skill_match_score=round(float(max(0.0, min(1.0, raw))), 6),
            required_score=round(required_score, 6),
            nice_to_have_score=round(nice_score, 6),
            matched_required=matched_req,
            matched_nice_to_have=matched_nice,
            matched_disqualifiers=matched_disq,
            hard_disqualifier=hard_disq,
            soft_disqualifier=soft_disq,
        )

    def score_all(
        self,
        candidates: list[CandidateFeatureVector],
    ) -> dict[str, SkillMatchResult]:
        if not isinstance(candidates, list):
            raise TypeError(
                f"candidates must be list[CandidateFeatureVector], "
                f"got {type(candidates).__name__}."
            )
        t0 = time.perf_counter()
        results = {c.candidate_id: self.score(c) for c in candidates}
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logger.info(
            "SkillMatchScorer: scored %d candidates in %.1f ms "
            "(hard_disq=%d, soft_disq=%d, mean=%.3f).",
            len(results),
            elapsed_ms,
            sum(1 for r in results.values() if r.hard_disqualifier),
            sum(1 for r in results.values() if r.soft_disqualifier),
            sum(r.skill_match_score for r in results.values()) / len(results) if results else 0.0,
        )
        return results

    # ── Sub-dimension scorers ─────────────────────────────────────────────────

    def _score_required(
        self, skills: list[SkillRecord]
    ) -> tuple[float, list[str]]:
        required_set = self._required_expanded
        if not required_set:
            return 1.0, []

        # Build two lookups: direct name → SkillRecord, synonym → SkillRecord
        # Direct matches receive full proficiency credit.
        # Synonym matches receive ONTOLOGY_PARTIAL_CREDIT fraction.
        direct_map:  dict[str, SkillRecord] = {}
        synonym_map: dict[str, SkillRecord] = {}

        for skill in skills:
            name_lower = skill.name.lower().strip()
            direct_map[name_lower] = skill
            for synonym in ONTOLOGY.get(name_lower, []):
                if synonym not in direct_map:
                    synonym_map.setdefault(synonym, skill)

        matched: list[str] = []
        total_score = 0.0

        for req_skill in required_set:
            if req_skill in direct_map:
                skill_rec = direct_map[req_skill]
                credit    = 1.0
            elif req_skill in synonym_map:
                skill_rec = synonym_map[req_skill]
                credit    = config.ONTOLOGY_PARTIAL_CREDIT  # 0.60
            else:
                continue

            prof    = skill_rec.proficiency.lower() if skill_rec.proficiency else ""
            prof_m  = _PROF_REQUIRED.get(prof, _PROF_DEFAULT)
            prof_m  = _effective_proficiency(skill_rec, prof_m)
            trust   = _duration_trust(skill_rec.duration_months)
            endorse = _endorsement_boost(skill_rec.endorsements)

            total_score += credit * (prof_m * trust + endorse)
            matched.append(skill_rec.name_raw)

        # Max possible per skill: 1.0 credit × 1.0 prof × 1.0 trust + ENDORSEMENT_MAX
        max_per_skill  = 1.0 * 1.0 * 1.0 + _ENDORSEMENT_MAX
        max_possible   = len(required_set) * max_per_skill
        normalised     = total_score / max_possible if max_possible > 0 else 0.0

        return float(min(1.0, normalised)), matched

    def _score_nice_to_have(
        self, skills: list[SkillRecord]
    ) -> tuple[float, list[str]]:
        nice_set = self._nice_expanded
        if not nice_set:
            return 0.0, []

        direct_map:  dict[str, SkillRecord] = {}
        synonym_map: dict[str, SkillRecord] = {}
        for skill in skills:
            name_lower = skill.name.lower().strip()
            direct_map[name_lower] = skill
            for synonym in ONTOLOGY.get(name_lower, []):
                if synonym not in direct_map:
                    synonym_map.setdefault(synonym, skill)

        matched: list[str] = []
        total_score = 0.0

        for nice_skill in nice_set:
            if nice_skill in direct_map:
                skill_rec = direct_map[nice_skill]
                credit    = 1.0
            elif nice_skill in synonym_map:
                skill_rec = synonym_map[nice_skill]
                credit    = config.ONTOLOGY_PARTIAL_CREDIT
            else:
                continue

            prof    = skill_rec.proficiency.lower() if skill_rec.proficiency else ""
            prof_m  = _PROF_NICE.get(prof, _PROF_DEFAULT * 0.70)
            prof_m  = _effective_proficiency(skill_rec, prof_m)
            trust   = _duration_trust(skill_rec.duration_months)
            endorse = _endorsement_boost(skill_rec.endorsements) * 0.70  # softer

            total_score += credit * (prof_m * trust + endorse)
            matched.append(skill_rec.name_raw)

        # Max per skill for nice: best nice prof (0.70) × trust (1.0) + 0.70×endorse_max
        max_per_skill = 0.70 * 1.0 + 0.70 * _ENDORSEMENT_MAX
        max_possible  = len(nice_set) * max_per_skill
        normalised    = total_score / max_possible if max_possible > 0 else 0.0

        return float(min(1.0, normalised)), matched

    def _check_disqualifiers(
        self, skills: list[SkillRecord]
    ) -> tuple[bool, bool, list[str]]:
        disq_set = self._disqualifiers_lower
        if not disq_set:
            return False, False, []

        disq_expanded: frozenset[str] = self._expand_skills(list(disq_set))

        hard, soft = False, False
        matched: list[str] = []

        for skill in skills:
            name_lower   = skill.name.lower().strip()
            skill_aliases = {name_lower} | set(ONTOLOGY.get(name_lower, []))
            if skill_aliases & disq_expanded:
                matched.append(skill.name_raw)
                prof = skill.proficiency.lower() if skill.proficiency else ""
                if prof in _DISQUALIFIER_HARD_PROFICIENCY:
                    hard = True
                else:
                    soft = True

        return hard, soft, matched

    @staticmethod
    def _expand_skills(skills: list[str]) -> frozenset[str]:
        expanded: set[str] = set()
        for skill in skills:
            norm = skill.lower().strip()
            if not norm:
                continue
            expanded.add(norm)
            expanded.update(ONTOLOGY.get(norm, []))
        return frozenset(expanded)

    def __repr__(self) -> str:
        return (
            f"SkillMatchScorer("
            f"required={len(self._required_expanded)} expanded, "
            f"nice={len(self._nice_expanded)} expanded, "
            f"disqualifiers={len(self._disqualifiers_lower)})"
        )


def score_skill_match(
    candidates: list[CandidateFeatureVector],
    jd: JDIntent,
) -> dict[str, SkillMatchResult]:
    return SkillMatchScorer(jd).score_all(candidates)
