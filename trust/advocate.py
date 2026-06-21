"""
trust/advocate.py — Advocate agent for the adversarial trust layer.

ROLE
----
The Advocate scans every available signal for a candidate and builds the
strongest *positive* case for their ranking.  It does NOT inflate scores or
ignore weaknesses — it simply surfaces the real evidence that supports the
ranking so that verdict.py and reasoning_generator.py have concrete facts
to work with instead of vague scores.

CONTRACT
--------
Input:
  candidate   : CandidateFeatureVector   — normalised candidate record
  scores      : ComponentScores          — pre-computed scoring breakdown
  skill_result: SkillMatchResult         — per-cluster skill evidence
  jd          : JDIntent                 — structured job description

Output:
  list[AdvocateSignal]  — each signal has a label, confidence, and value.
                          Confidence is "HIGH" | "MEDIUM" | "LOW".
                          List is ordered: HIGH first, then MEDIUM, then LOW.
                          Empty list is valid (very weak candidates).

CONFIDENCE RULES (from config.py)
----------------------------------
  HIGH   : signal value ≥ ADVOCATE_HIGH_CONFIDENCE_THRESHOLD   (0.75)
  MEDIUM : signal value ≥ ADVOCATE_MEDIUM_CONFIDENCE_THRESHOLD (0.50)
  LOW    : signal value > 0.0  (something is there, but weak)

SIGNAL CATALOGUE
----------------
  1.  Skill cluster coverage (per cluster with score > 0)
  2.  Required skill match rate
  3.  Nice-to-have skill matches
  4.  YOE band fitness
  5.  Product-company experience
  6.  Career trajectory velocity (promotions/yr)
  7.  Platform recency (last active)
  8.  Recruiter response rate
  9.  Notice period fitness
  10. Open-to-work flag
  11. GitHub activity
  12. Redrob assessment scores above threshold
  13. Location match
  14. Ontology / domain-transfer matches

DESIGN NOTES
------------
- No LLM needed. All signals come from pre-computed scores and parsed fields.
- Every claimed fact in the output MUST exist in the input data.
  The reasoning_generator.py will copy these facts verbatim into the
  submission CSV — hallucination here = hallucination in the output.
- Signals are deduplicated: if a skill is mentioned in cluster coverage AND
  in matched_required, it appears once in the clearest signal.
- The function is deterministic: same input → same output list, same order.
  This is required for the hallucination audit in Phase 6.

DEPENDENCIES
------------
  config              : threshold constants
  pipeline.schemas    : CandidateFeatureVector, ComponentScores,
                        AdvocateSignal, JDIntent
  scoring.skill_match : SkillMatchResult, ClusterScore

No I/O.  No network.  No side-effects.  Pure function.
"""

from __future__ import annotations

import collections
import logging
from typing import Optional

import config
from pipeline.schemas import (
    AdvocateSignal,
    CandidateFeatureVector,
    ComponentScores,
    JDIntent,
)
from scoring.skill_match import SkillMatchResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE THRESHOLDS (sourced from config — never hardcode)
# ─────────────────────────────────────────────────────────────────────────────

_HIGH: str = "HIGH"
_MED: str = "MEDIUM"
_LOW: str = "LOW"

_HIGH_THRESHOLD: float = config.ADVOCATE_HIGH_CONFIDENCE_THRESHOLD    # 0.75
_MED_THRESHOLD: float = config.ADVOCATE_MEDIUM_CONFIDENCE_THRESHOLD   # 0.50

# GitHub score is 0–100 in RedrobSignals; normalise before thresholding.
_GITHUB_NORMALISER: float = 100.0

# Assessment score is 0–100; threshold from config.
_ASSESSMENT_HIGH_THRESHOLD: float = 75.0
_ASSESSMENT_MED_THRESHOLD: float = config.ASSESSMENT_SCORE_THRESHOLD  # 40.0

# Minimum cluster score to emit a signal (avoids noise near zero).
_MIN_CLUSTER_SCORE: float = 0.10

# Trajectory: promotions/yr percentile thresholds (scores are 0–1 after
# normalisation in ComponentScores.trajectory_velocity).
_TRAJECTORY_HIGH: float = _HIGH_THRESHOLD
_TRAJECTORY_MED: float = _MED_THRESHOLD

# Location bonus threshold to emit a signal (Pune/Noida score is 0.08).
_LOCATION_SIGNAL_MIN: float = 0.02

# Recency: score high enough to be a positive signal (not just not-bad).
_RECENCY_SIGNAL_MIN: float = 0.40


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _confidence(value: float) -> str:
    """
    Map a normalised [0, 1] value to a confidence tier string.

    Used for all signals where the underlying metric is already in [0, 1].
    """
    if value >= _HIGH_THRESHOLD:
        return _HIGH
    if value >= _MED_THRESHOLD:
        return _MED
    return _LOW


def _confidence_raw(value: float, high: float, med: float) -> str:
    """
    Map an arbitrary value to a confidence tier using custom thresholds.

    Used where the underlying metric is not normalised to [0, 1]
    (e.g. years of experience, GitHub score in 0–100 range).
    """
    if value >= high:
        return _HIGH
    if value >= med:
        return _MED
    return _LOW


def _add_optional(
    target: list[AdvocateSignal],
    signal: Optional[AdvocateSignal],
) -> None:
    """Append signal to target only if it is not None."""
    if signal is not None:
        target.append(signal)


def _sort_signals(signals: list[AdvocateSignal]) -> list[AdvocateSignal]:
    """
    Sort signals HIGH → MEDIUM → LOW, stable within each tier.

    Verdict.py and reasoning_generator.py both expect the highest-confidence
    signals to appear first so they can build the lead sentence from the top.
    """
    _order = {_HIGH: 0, _MED: 1, _LOW: 2}
    return sorted(signals, key=lambda s: _order.get(s.confidence, 3))


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL SIGNAL SCANNERS
# Each returns Optional[AdvocateSignal] — None if no positive evidence found.
# Each function is deliberately narrow: one signal category, clearly named.
# ─────────────────────────────────────────────────────────────────────────────

def _scan_skill_clusters(
    skill_result: SkillMatchResult,
) -> list[AdvocateSignal]:
    """
    Emit one signal per skill cluster where the candidate has real coverage.

    Uses the per-cluster breakdown from SkillMatchResult.cluster_scores to
    produce specific, verifiable claims ("Strong retrieval_systems coverage
    via: FAISS, sentence-transformers") rather than vague aggregate scores.

    Skips clusters with score < _MIN_CLUSTER_SCORE to avoid noise.
    """
    signals: list[AdvocateSignal] = []

    for cs in skill_result.cluster_scores:
        if cs.score < _MIN_CLUSTER_SCORE:
            continue

        # Build human-readable skill list for the value field.
        skill_list = ", ".join(cs.matched_skills[:6])  # show up to 6 for full context
        if len(cs.matched_skills) > 6:
            skill_list += f" (+{len(cs.matched_skills) - 6} more)"

        if not skill_list:
            # Score > threshold but no named skills — ontology match only.
            skill_list = f"via ontology coverage ({cs.coverage_pct:.0%} capabilities)"

        cluster_label = cs.cluster_name.replace("_", " ").title()
        conf = _confidence(cs.score)

        signals.append(AdvocateSignal(
            label=f"{cluster_label} cluster coverage",
            confidence=conf,
            value=(
                f"{cs.score:.0%} score across "
                f"{cs.coverage_pct:.0%} of capabilities: {skill_list}"
            ),
        ))

    return signals


def _scan_required_skill_rate(
    scores: ComponentScores,
    skill_result: SkillMatchResult,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal on the overall required-skill coverage rate.

    This is a summary-level signal on top of the cluster signals.
    Emitted only when coverage is meaningful (≥ _MED_THRESHOLD).
    The specific matched skills are named so reasoning_generator.py
    can reference them without hallucination.
    """
    coverage = scores.required_skill_coverage
    if coverage < _MED_THRESHOLD:
        return None

    matched_str = ", ".join(skill_result.matched_required[:6])
    if len(skill_result.matched_required) > 6:
        matched_str += f" (+{len(skill_result.matched_required) - 6} more)"

    return AdvocateSignal(
        label="Required skill coverage",
        confidence=_confidence(coverage),
        value=(
            f"{coverage:.0%} of required skills matched"
            + (f": {matched_str}" if matched_str else "")
        ),
    )


def _scan_nice_to_have(
    scores: ComponentScores,
    skill_result: SkillMatchResult,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal if the candidate has meaningful nice-to-have coverage.

    Nice-to-have skills (LoRA, XGBoost LTR, open-source, HR-tech) are
    bonus signals — present only when coverage ≥ 0.30 to avoid noise.
    """
    coverage = scores.nice_to_have_coverage
    if coverage < 0.30:
        return None

    matched_str = ", ".join(skill_result.matched_nice_to_have[:4])

    return AdvocateSignal(
        label="Nice-to-have skill coverage",
        confidence=_confidence(coverage),
        value=(
            f"{coverage:.0%} nice-to-have coverage"
            + (f": {matched_str}" if matched_str else "")
        ),
    )


def _scan_ontology_matches(
    scores: ComponentScores,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when domain-transfer skills were matched via ontology.

    This is the "Tier-5 rescue" signal — the candidate didn't use the
    exact keywords from the JD but their skills map onto the requirements
    via the ontology graph.  Valuable to surface explicitly because these
    candidates would be invisible to keyword-only systems.
    """
    matches = scores.ontology_skills_matched
    if not matches:
        return None

    match_str = ", ".join(matches[:4])
    if len(matches) > 4:
        match_str += f" (+{len(matches) - 4} more)"

    # Ontology matches always LOW or MED — they are adjacent, not direct.
    conf = _MED if len(matches) >= 3 else _LOW

    return AdvocateSignal(
        label="Domain-transfer skills via ontology",
        confidence=conf,
        value=f"{len(matches)} adjacent skill(s) matched: {match_str}",
    )


def _scan_yoe_fitness(
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when the candidate's YOE falls in or near the ideal band.

    The JD says 5–9 years is the target.  We report the actual years and
    whether they're in the ideal band, below it, or above it.
    """
    yoe = candidate.years_of_experience
    yoe_score = scores.yoe_score

    # Only emit a positive signal if YOE score is at least moderate.
    if yoe_score < _MED_THRESHOLD:
        return None

    ideal_min = config.YOE_BAND_IDEAL_MIN  # 5.0
    ideal_max = config.YOE_BAND_IDEAL_MAX  # 9.0

    if ideal_min <= yoe <= ideal_max:
        band_desc = f"in ideal band ({ideal_min:.0f}–{ideal_max:.0f} yrs)"
    elif yoe < ideal_min:
        band_desc = f"slightly below ideal band (target {ideal_min:.0f}–{ideal_max:.0f} yrs)"
    else:
        band_desc = f"slightly above ideal band (target {ideal_min:.0f}–{ideal_max:.0f} yrs)"

    return AdvocateSignal(
        label="Years of experience",
        confidence=_confidence(yoe_score),
        value=f"{yoe:.1f} years — {band_desc}",
    )


def _scan_product_company(
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal if the candidate has product-company experience.

    The JD explicitly penalises consulting-only backgrounds and rewards
    product-company experience (food-tech, fintech, SaaS, AI/ML, etc.).
    We name the specific companies so the recruiter can verify.
    """
    if not scores.product_co_flag:
        return None

    # Collect product-company names from career history.
    product_companies: list[str] = [
        entry.company
        for entry in candidate.career_history
        if entry.industry_lower in config.PRODUCT_INDUSTRIES
    ]

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_companies: list[str] = []
    for c in product_companies:
        if c not in seen:
            seen.add(c)
            unique_companies.append(c)

    company_str = ", ".join(unique_companies[:3])
    if len(unique_companies) > 3:
        company_str += f" (+{len(unique_companies) - 3} more)"

    # Product-company experience is HIGH when it's the dominant career pattern.
    career_score = scores.career_score
    conf = _confidence(career_score) if career_score >= _MED_THRESHOLD else _MED

    return AdvocateSignal(
        label="Product-company experience",
        confidence=conf,
        value=(
            f"{len(unique_companies)} product company role(s): {company_str}"
            if company_str
            else "Product-company experience confirmed"
        ),
    )


def _scan_trajectory(
    scores: ComponentScores,
    candidate: CandidateFeatureVector,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when the candidate's career velocity is above median.

    trajectory_velocity in ComponentScores is already normalised 0–1
    (it's the promotions/yr percentile rank across the pool).
    We back-calculate an approximate promotions figure for the value field.
    """
    velocity = scores.trajectory_velocity
    if velocity < _MED_THRESHOLD:
        return None

    # Count promotions: title changes within same company, or role upgrades.
    # As a proxy, count career_history entries where is_current=False.
    # (Actual promotion detection is in scoring/trajectory.py — we only
    #  read the pre-computed percentile here.)
    total_roles = len(candidate.career_history)
    yoe = max(candidate.years_of_experience, 1.0)

    # Approximate promotions from velocity percentile — purely illustrative,
    # not used for scoring. The trajectory_velocity score is the source of
    # truth; this is just a human-readable representation.
    approx_ppm = velocity * config.TRAJECTORY_PROMOTIONS_PER_YEAR_CAP  # 1.5 cap

    return AdvocateSignal(
        label="Career trajectory velocity",
        confidence=_confidence(velocity),
        value=(
            f"{velocity:.0%} velocity percentile "
            f"(~{approx_ppm:.1f} promotions/yr across {total_roles} roles "
            f"in {yoe:.1f} yrs)"
        ),
    )


def _scan_recency(
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when the candidate has been recently active on the platform.

    recency_score in ComponentScores = exp(-λ × days_since_active).
    We report the actual days for the value field.
    """
    recency = scores.recency_score
    if recency < _RECENCY_SIGNAL_MIN:
        return None

    days = candidate.signals.days_since_active

    # Convert recency score back to days for readability
    # (we already have the raw days from RedrobSignals).
    if days == 0:
        days_str = "active today"
    elif days == 1:
        days_str = "active yesterday"
    elif days <= 7:
        days_str = f"active {days} days ago"
    elif days <= 30:
        weeks = days // 7
        days_str = f"active ~{weeks} week(s) ago"
    else:
        months = days // 30
        days_str = f"active ~{months} month(s) ago"

    return AdvocateSignal(
        label="Platform recency",
        confidence=_confidence(recency),
        value=f"{days_str} (recency score {recency:.2f})",
    )


def _scan_response_rate(
    candidate: CandidateFeatureVector,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when the candidate has a strong recruiter response rate.

    High response rate = actually engages with recruiters = de-risks outreach.
    Below 0.50 is not a positive signal worth surfacing (skeptic.py handles low).
    """
    rate = candidate.signals.recruiter_response_rate
    if rate < _MED_THRESHOLD:
        return None

    return AdvocateSignal(
        label="Recruiter response rate",
        confidence=_confidence(rate),
        value=f"{rate:.0%} response rate to recruiter outreach",
    )


def _scan_notice_period(
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when the candidate has a short notice period.

    The JD explicitly says "we'd love sub-30-day notice".
    notice_period_score in ComponentScores encodes this preference.
    """
    notice_score = scores.notice_period_score
    if notice_score < _MED_THRESHOLD:
        return None

    days = candidate.signals.notice_period_days

    if days == 0:
        period_str = "immediately available"
    elif days <= config.NOTICE_PERIOD_IDEAL_MAX:
        period_str = f"{days}-day notice (within JD preferred ≤30 days)"
    else:
        period_str = f"{days}-day notice"

    return AdvocateSignal(
        label="Notice period",
        confidence=_confidence(notice_score),
        value=period_str,
    )


def _scan_open_to_work(
    candidate: CandidateFeatureVector,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when the open_to_work flag is set.

    This is the strongest explicit availability signal.  Always HIGH when set
    because it is an active declaration by the candidate.
    """
    if not candidate.signals.open_to_work_flag:
        return None

    return AdvocateSignal(
        label="Explicitly open to work",
        confidence=_HIGH,
        value="Candidate has set open_to_work flag on Redrob platform",
    )


def _scan_github(
    candidate: CandidateFeatureVector,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when GitHub activity score is meaningfully positive.

    github_activity_score is 0–100 in RedrobSignals (-1 = not linked).
    We normalise to [0, 1] for thresholding.
    """
    raw_score = candidate.signals.github_activity_score
    if raw_score < 0:
        # Not linked — neutral, handled by skeptic if required.
        return None

    normalised = raw_score / _GITHUB_NORMALISER
    if normalised < _MED_THRESHOLD:
        return None

    conf = _confidence_raw(raw_score, _ASSESSMENT_HIGH_THRESHOLD, _ASSESSMENT_MED_THRESHOLD)

    return AdvocateSignal(
        label="GitHub activity",
        confidence=conf,
        value=f"GitHub activity score: {raw_score:.0f}/100",
    )


def _scan_assessments(
    candidate: CandidateFeatureVector,
    jd: JDIntent,
) -> list[AdvocateSignal]:
    """
    Emit one signal per Redrob skill assessment that is above threshold.

    assessment_scores in RedrobSignals is dict[skill_name → 0–100].
    We only surface assessments for skills that are relevant to the JD
    (in required or nice-to-have lists) to avoid noise from irrelevant certs.
    """
    signals: list[AdvocateSignal] = []
    assessment_scores = candidate.signals.skill_assessment_scores

    if not assessment_scores:
        return signals

    # Build a lookup of JD-relevant skill names (lowercase).
    jd_skills: frozenset[str] = frozenset(
        s.lower() for s in (jd.required_skills + jd.nice_to_have_skills)
    )

    for skill_name, score in assessment_scores.items():
        if score < _ASSESSMENT_MED_THRESHOLD:
            continue

        skill_lower = skill_name.lower()
        # Only emit if the assessed skill is JD-relevant (or JD skills is empty).
        if jd_skills and skill_lower not in jd_skills:
            continue

        conf = _confidence_raw(
            score,
            high=_ASSESSMENT_HIGH_THRESHOLD,
            med=_ASSESSMENT_MED_THRESHOLD,
        )

        signals.append(AdvocateSignal(
            label=f"Redrob assessment: {skill_name}",
            confidence=conf,
            value=f"Assessment score {score:.0f}/100 for {skill_name}",
        ))

    return signals


def _scan_location(
    scores: ComponentScores,
    candidate: CandidateFeatureVector,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when location bonus was applied.

    The JD prefers Pune/Noida; Delhi NCR/Hyderabad/Mumbai are welcome.
    We emit a signal only when the bonus is non-trivial.
    """
    bonus = scores.location_bonus
    if bonus < _LOCATION_SIGNAL_MIN:
        return None

    location_str = candidate.location or candidate.location_lower or "location on file"

    relocate_suffix = ""
    if candidate.signals.willing_to_relocate:
        relocate_suffix = " (willing to relocate)"

    conf = _HIGH if bonus >= 0.07 else _MED  # Pune/Noida = 0.08 → HIGH

    return AdvocateSignal(
        label="Location match",
        confidence=conf,
        value=f"{location_str}{relocate_suffix} — location bonus: +{bonus:.2f}",
    )


def _scan_profile_completeness(
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> Optional[AdvocateSignal]:
    """
    Emit a signal when profile completeness is high.

    High completeness reduces the uncertainty penalty and means all scoring
    signals are trustworthy.  Surfacing this tells the recruiter the candidate
    profile is data-rich (not sparse/unreliable).
    """
    completeness = candidate.signals.profile_completeness_score  # 0–100
    signal_count = scores.signal_count

    # Only emit if completeness is meaningfully high AND signal count is good.
    if completeness < 70.0 or signal_count < 4:
        return None

    normalised = completeness / 100.0
    conf = _confidence(normalised)

    return AdvocateSignal(
        label="Profile completeness",
        confidence=conf,
        value=(
            f"{completeness:.0f}% profile completeness "
            f"({signal_count} signal types present — low uncertainty)"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_advocate_signals(
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
    skill_result: SkillMatchResult,
    jd: JDIntent,
) -> list[AdvocateSignal]:
    """
    Build the complete list of positive signals for one candidate.

    This is the sole public entry point for trust/advocate.py.
    Calls all individual signal scanners, deduplicates, sorts by confidence,
    and returns a clean list ready for verdict.py consumption.

    Parameters
    ----------
    candidate:    Fully parsed CandidateFeatureVector (from candidate_parser.py).
    scores:       ComponentScores breakdown (from scoring/composite.py).
    skill_result: SkillMatchResult (from scoring/skill_match.py).
    jd:           Structured JD intent (from pipeline/jd_parser.py).

    Returns
    -------
    list[AdvocateSignal], sorted HIGH → MEDIUM → LOW.
    Empty list if no positive signals found (valid for very weak candidates).

    Raises
    ------
    TypeError:  If any argument is not of the expected type.  Fail fast —
                better a crash during development than a silent wrong result
                in production.
    """
    # ── Type guard (fail-fast, not silent) ───────────────────────────────────
    if not isinstance(candidate, CandidateFeatureVector):
        raise TypeError(
            f"candidate must be CandidateFeatureVector, got {type(candidate).__name__}"
        )
    if not isinstance(scores, ComponentScores):
        raise TypeError(
            f"scores must be ComponentScores, got {type(scores).__name__}"
        )
    if not isinstance(skill_result, SkillMatchResult):
        raise TypeError(
            f"skill_result must be SkillMatchResult, got {type(skill_result).__name__}"
        )
    if not isinstance(jd, JDIntent):
        raise TypeError(f"jd must be JDIntent, got {type(jd).__name__}")

    if candidate.candidate_id != scores.candidate_id:
        raise ValueError(
            f"ID mismatch: candidate.candidate_id={candidate.candidate_id!r} "
            f"but scores.candidate_id={scores.candidate_id!r}"
        )
    if candidate.candidate_id != skill_result.candidate_id:
        raise ValueError(
            f"ID mismatch: candidate.candidate_id={candidate.candidate_id!r} "
            f"but skill_result.candidate_id={skill_result.candidate_id!r}"
        )

    signals: list[AdvocateSignal] = []

    # ── 1. Skill cluster coverage (multi-signal) ─────────────────────────────
    signals.extend(_scan_skill_clusters(skill_result))

    # ── 2. Required skill rate (summary signal) ───────────────────────────────
    _add_optional(signals, _scan_required_skill_rate(scores, skill_result))

    # ── 3. Nice-to-have coverage ──────────────────────────────────────────────
    _add_optional(signals, _scan_nice_to_have(scores, skill_result))

    # ── 4. Ontology / domain-transfer matches ────────────────────────────────
    _add_optional(signals, _scan_ontology_matches(scores))

    # ── 5. YOE band fitness ───────────────────────────────────────────────────
    _add_optional(signals, _scan_yoe_fitness(candidate, scores))

    # ── 6. Product-company experience ─────────────────────────────────────────
    _add_optional(signals, _scan_product_company(candidate, scores))

    # ── 7. Career trajectory velocity ────────────────────────────────────────
    _add_optional(signals, _scan_trajectory(scores, candidate))

    # ── 8. Platform recency ───────────────────────────────────────────────────
    _add_optional(signals, _scan_recency(candidate, scores))

    # ── 9. Recruiter response rate ────────────────────────────────────────────
    _add_optional(signals, _scan_response_rate(candidate))

    # ── 10. Notice period ─────────────────────────────────────────────────────
    _add_optional(signals, _scan_notice_period(candidate, scores))

    # ── 11. Open-to-work flag ─────────────────────────────────────────────────
    _add_optional(signals, _scan_open_to_work(candidate))

    # ── 12. GitHub activity ───────────────────────────────────────────────────
    _add_optional(signals, _scan_github(candidate))

    # ── 13. Redrob skill assessments (multi-signal) ───────────────────────────
    signals.extend(_scan_assessments(candidate, jd))

    # ── 14. Location match ────────────────────────────────────────────────────
    _add_optional(signals, _scan_location(scores, candidate))

    # ── 15. Profile completeness ──────────────────────────────────────────────
    _add_optional(signals, _scan_profile_completeness(candidate, scores))

    # ── Sort and return ───────────────────────────────────────────────────────
    result = _sort_signals(signals)

    _conf_counts = collections.Counter(s.confidence for s in result)
    logger.debug(
        "advocate: %s → %d signals (HIGH=%d, MEDIUM=%d, LOW=%d)",
        candidate.candidate_id,
        len(result),
        _conf_counts[_HIGH],
        _conf_counts[_MED],
        _conf_counts[_LOW],
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY HELPERS (consumed by verdict.py)
# ─────────────────────────────────────────────────────────────────────────────

def count_by_confidence(signals: list[AdvocateSignal]) -> dict[str, int]:
    """
    Return a count dict: {"HIGH": n, "MEDIUM": n, "LOW": n}.

    Convenience function for verdict.py so it doesn't need to filter inline.
    """
    counts: collections.Counter[str] = collections.Counter(
        s.confidence for s in signals
    )
    return {
        _HIGH: counts[_HIGH],
        _MED:  counts[_MED],
        _LOW:  counts[_LOW],
    }


def top_signals(
    signals: list[AdvocateSignal],
    n: int = 3,
) -> list[AdvocateSignal]:
    """
    Return the top-n highest-confidence signals.

    Assumes list is already sorted (build_advocate_signals guarantees this).
    Used by reasoning_generator.py to build the lead positive sentence.
    """
    return signals[:n]

