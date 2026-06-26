from __future__ import annotations

import math
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import config
from pipeline.schemas import RRFResult
from scoring.behavioral import BehavioralResult, BehavioralScorer
from scoring.honeypot_filter import HoneypotCleanup
from scoring.trajectory import (
    TrajectoryResult,
    TrajectoryVelocityScorer,
    count_promotions,
    trajectory_velocity_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED MOCK BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
# All helpers return plain MagicMock or SimpleNamespace objects so tests
# never need the full dataclass constructor — this avoids coupling tests to
# optional/defaulted field lists that DEV B may still adjust.

_TODAY = date(2026, 6, 16)   # Pinned reference date for all recency tests


def _make_signals(
    *,
    last_active_date: date = _TODAY,
    recruiter_response_rate: float = 0.70,
    open_to_work_flag: bool = True,
    notice_period_days: int = 30,
    github_activity_score: float = 45.0,
    has_github: bool = True,
    profile_completeness_score: float = 80.0,
    interview_completion_rate: float = 0.70,
    profile_views_received_30d: int = 20,
    applications_submitted_30d: int = 3,
    search_appearance_30d: int = 100,
    saved_by_recruiters_30d: int = 5,
    connection_count: int = 300,
    endorsements_received: int = 20,
    has_offer_history: bool = True,
    offer_acceptance_rate: float = 0.5,
    skill_assessment_scores: dict | None = None,
) -> MagicMock:
    """
    Build a mock RedrobSignals with full control over every field.

    Uses MagicMock (no spec) so that both fields AND computed properties
    (has_github, has_offer_history) can be set directly without requiring
    knowledge of whether they are fields or @property definitions.
    """
    s = MagicMock()
    s.last_active_date = last_active_date
    s.recruiter_response_rate = recruiter_response_rate
    s.open_to_work_flag = open_to_work_flag
    s.notice_period_days = notice_period_days
    s.github_activity_score = github_activity_score
    s.has_github = has_github
    s.profile_completeness_score = profile_completeness_score
    s.interview_completion_rate = interview_completion_rate
    s.profile_views_received_30d = profile_views_received_30d
    s.applications_submitted_30d = applications_submitted_30d
    s.search_appearance_30d = search_appearance_30d
    s.saved_by_recruiters_30d = saved_by_recruiters_30d
    s.connection_count = connection_count
    s.endorsements_received = endorsements_received
    s.has_offer_history = has_offer_history
    s.offer_acceptance_rate = offer_acceptance_rate
    s.skill_assessment_scores = skill_assessment_scores or {}
    return s


def _make_cfv(candidate_id: str, signals: MagicMock) -> MagicMock:
    """
    Build a minimal CandidateFeatureVector mock for BehavioralScorer tests.
    BehavioralScorer only reads candidate.candidate_id and candidate.signals.
    """
    cfv = MagicMock()
    cfv.candidate_id = candidate_id
    cfv.signals = signals
    return cfv


def _make_cfv_with_career(
    candidate_id: str,
    career_entries: list[Any],
    years_of_experience: float = 6.0,
    total_career_months: int = 72,
) -> MagicMock:
    """
    Build a minimal CandidateFeatureVector mock for TrajectoryVelocityScorer tests.
    Trajectory scorer reads candidate_id, career_history, years_of_experience,
    total_career_months.
    """
    cfv = MagicMock()
    cfv.candidate_id = candidate_id
    cfv.career_history = career_entries
    cfv.years_of_experience = years_of_experience
    cfv.total_career_months = total_career_months
    return cfv


def _make_cfv_honeypot(candidate_id: str, is_honeypot: bool) -> MagicMock:
    """
    Build a minimal CandidateFeatureVector mock for HoneypotCleanup tests.
    HoneypotCleanup only reads candidate.is_honeypot.
    """
    cfv = MagicMock()
    cfv.candidate_id = candidate_id
    cfv.is_honeypot = is_honeypot
    return cfv


def _career_entry(title: str, start_date: date) -> SimpleNamespace:
    """
    Minimal career entry with only the fields read by count_promotions():
        entry.title       — used by _seniority_level()
        entry.start_date  — used by sorted() ordering
    Using SimpleNamespace avoids depending on the full CareerEntry dataclass.
    """
    return SimpleNamespace(title=title, start_date=start_date)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config-level weight constant tests
#    These run regardless of which scoring files are implemented.
# ─────────────────────────────────────────────────────────────────────────────

class TestScoringWeightConstants:
    """
    Validates all weight constants in config.py that drive the composite score.

    These tests protect against accidental drift when tuning weights during
    calibration (Phase 6, Day 20). If any weight changes break the sum-to-1.0
    invariant, these catch it immediately.

    Sprint acceptance criterion: "Weights sum to 1.0"
    """

    def test_composite_weights_sum_to_one(self):
        """
        WEIGHT_SKILL + WEIGHT_CAREER + WEIGHT_BEHAVIORAL must equal exactly 1.0.

        From config.py: 0.40 + 0.35 + 0.25 = 1.00.
        composite.py uses these as the top-level blend; any deviation means
        final scores are not properly normalised.
        """
        total = config.WEIGHT_SKILL + config.WEIGHT_CAREER + config.WEIGHT_BEHAVIORAL
        assert abs(total - 1.0) < config._WEIGHT_SUM_TOLERANCE, (
            f"Composite weights sum to {total:.8f}, expected 1.0. "
            f"WEIGHT_SKILL={config.WEIGHT_SKILL}, "
            f"WEIGHT_CAREER={config.WEIGHT_CAREER}, "
            f"WEIGHT_BEHAVIORAL={config.WEIGHT_BEHAVIORAL}"
        )

    def test_behavioral_weights_sum_to_one(self):
        """
        sum(BEHAVIORAL_WEIGHTS.values()) must equal exactly 1.0.

        These 7 sub-weights drive behavioral_score inside BehavioralScorer.score().
        Confirmed by behavioral.py: behavioral_score = Σ weight * sub_score.
        """
        total = sum(config.BEHAVIORAL_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, (
            f"BEHAVIORAL_WEIGHTS sum to {total:.9f}, expected 1.0. "
            f"Weights: {config.BEHAVIORAL_WEIGHTS}"
        )

    def test_behavioral_weights_has_seven_keys(self):
        """
        BEHAVIORAL_WEIGHTS must have exactly 7 keys — one per SignalPath column.
        Adding/removing a key without updating the feature matrix breaks alignment.
        """
        assert len(config.BEHAVIORAL_WEIGHTS) == 7, (
            f"Expected 7 BEHAVIORAL_WEIGHTS keys, got {len(config.BEHAVIORAL_WEIGHTS)}: "
            f"{list(config.BEHAVIORAL_WEIGHTS.keys())}"
        )

    def test_behavioral_weights_expected_keys_present(self):
        """All 7 expected behavioral weight keys are present in config."""
        expected_keys = {
            "recency", "response_rate", "open_to_work",
            "notice_period", "github_activity",
            "profile_completeness", "interview_completion",
        }
        actual_keys = set(config.BEHAVIORAL_WEIGHTS.keys())
        missing = expected_keys - actual_keys
        extra   = actual_keys - expected_keys
        assert not missing, f"Missing behavioral weight keys: {missing}"
        assert not extra,   f"Unexpected behavioral weight keys: {extra}"

    def test_all_composite_weights_positive(self):
        """No composite weight should be zero or negative."""
        for name, w in [
            ("WEIGHT_SKILL",     config.WEIGHT_SKILL),
            ("WEIGHT_CAREER",    config.WEIGHT_CAREER),
            ("WEIGHT_BEHAVIORAL",config.WEIGHT_BEHAVIORAL),
        ]:
            assert w > 0.0, f"{name}={w} must be > 0.0"

    def test_all_behavioral_weights_positive(self):
        """No behavioral sub-weight should be zero or negative."""
        for key, val in config.BEHAVIORAL_WEIGHTS.items():
            assert val > 0.0, f"BEHAVIORAL_WEIGHTS['{key}']={val} must be > 0.0"

    def test_proficiency_multipliers_ordered(self):
        """
        Proficiency multipliers must be strictly ordered:
            beginner < intermediate < advanced < expert
        A skill with more proficiency must always contribute more.
        """
        m = config.PROFICIENCY_MULTIPLIERS
        assert m["beginner"]     < m["intermediate"], "beginner must be < intermediate"
        assert m["intermediate"] < m["advanced"],     "intermediate must be < advanced"
        assert m["advanced"]     < m["expert"],       "advanced must be < expert"
        assert m["expert"]       <= 1.0,              "expert multiplier must be <= 1.0"
        assert m["beginner"]     >= 0.0,              "beginner multiplier must be >= 0.0"

    def test_notice_period_thresholds_ordered(self):
        """
        Notice period thresholds must be strictly ordered:
            IDEAL_MAX < ACCEPTABLE_MAX < MAX
        """
        assert config.NOTICE_PERIOD_IDEAL_MAX < config.NOTICE_PERIOD_ACCEPTABLE_MAX, (
            f"NOTICE_PERIOD_IDEAL_MAX ({config.NOTICE_PERIOD_IDEAL_MAX}) "
            f"must be < NOTICE_PERIOD_ACCEPTABLE_MAX ({config.NOTICE_PERIOD_ACCEPTABLE_MAX})"
        )
        assert config.NOTICE_PERIOD_ACCEPTABLE_MAX < config.NOTICE_PERIOD_MAX, (
            f"NOTICE_PERIOD_ACCEPTABLE_MAX ({config.NOTICE_PERIOD_ACCEPTABLE_MAX}) "
            f"must be < NOTICE_PERIOD_MAX ({config.NOTICE_PERIOD_MAX})"
        )

    def test_consulting_only_penalty_is_significant(self):
        """
        CONSULTING_ONLY_PENALTY must be < 0.5 — it is meant to strongly penalise
        candidates with exclusively consulting backgrounds. Per the JD:
        'people who have only worked at consulting firms — we will not move forward.'
        A penalty above 0.5 would be insufficient to signal disqualification.
        """
        assert config.CONSULTING_ONLY_PENALTY < 0.5, (
            f"CONSULTING_ONLY_PENALTY={config.CONSULTING_ONLY_PENALTY} should be < 0.5 "
            "to meaningfully penalise consulting-only backgrounds."
        )

    def test_uncertainty_penalty_floor_in_range(self):
        """UNCERTAINTY_PENALTY_FLOOR must be in [0.5, 1.0) to be meaningful."""
        floor = config.UNCERTAINTY_PENALTY_FLOOR
        assert 0.5 <= floor < 1.0, (
            f"UNCERTAINTY_PENALTY_FLOOR={floor} must be in [0.5, 1.0)"
        )

    def test_recency_lambda_reasonable(self):
        """
        RECENCY_LAMBDA must be small enough that 14-day-old activity scores > 0.90
        (matching the proposal: 'A commit from 14 days ago scores ~0.93').
        """
        score_14d = math.exp(-config.RECENCY_LAMBDA * 14)
        assert score_14d > 0.90, (
            f"14-day recency score = {score_14d:.4f}, expected > 0.90. "
            f"RECENCY_LAMBDA={config.RECENCY_LAMBDA} may be too large."
        )

    def test_rrf_k_is_standard_smoothing_constant(self):
        """RRF_K should be 60 — the universally accepted RRF smoothing constant."""
        assert config.RRF_K == 60, (
            f"RRF_K={config.RRF_K}, expected 60 (standard RRF smoothing constant)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. BehavioralScorer tests — scoring/behavioral.py
# ─────────────────────────────────────────────────────────────────────────────

class TestBehavioralScorer:
    """
    Unit tests for scoring/behavioral.py: BehavioralScorer.score()

    BehavioralScorer is stateless. All tests use a fixed reference date
    (_TODAY = 2026-06-16) to avoid test-time dependency on the system clock.
    """

    @pytest.fixture
    def scorer(self) -> BehavioralScorer:
        return BehavioralScorer()

    @pytest.fixture
    def default_signals(self) -> MagicMock:
        """Well-rounded candidate with good signals across all dimensions."""
        return _make_signals(last_active_date=_TODAY)

    @pytest.fixture
    def default_cfv(self, default_signals) -> MagicMock:
        return _make_cfv("CAND_0000031", default_signals)

    # ── Type and range tests ─────────────────────────────────────────────────

    def test_returns_behavioral_result_type(self, scorer, default_cfv):
        """score() returns a BehavioralResult instance."""
        result = scorer.score(default_cfv, today=_TODAY)
        assert isinstance(result, BehavioralResult), (
            f"Expected BehavioralResult, got {type(result).__name__}"
        )

    def test_candidate_id_preserved(self, scorer, default_cfv):
        """BehavioralResult.candidate_id matches input candidate_id."""
        result = scorer.score(default_cfv, today=_TODAY)
        assert result.candidate_id == "CAND_0000031"

    def test_behavioral_score_in_range(self, scorer, default_cfv):
        """behavioral_score is always in [0.0, 1.0]."""
        result = scorer.score(default_cfv, today=_TODAY)
        assert 0.0 <= result.behavioral_score <= 1.0, (
            f"behavioral_score={result.behavioral_score} out of [0, 1]"
        )

    def test_recency_score_in_range(self, scorer, default_cfv):
        """recency_score is always in [0.0, 1.0]."""
        result = scorer.score(default_cfv, today=_TODAY)
        assert 0.0 <= result.recency_score <= 1.0

    def test_notice_period_score_in_range(self, scorer, default_cfv):
        """notice_period_score is always in [0.0, 1.0]."""
        result = scorer.score(default_cfv, today=_TODAY)
        assert 0.0 <= result.notice_period_score <= 1.0

    def test_uncertainty_penalty_in_range(self, scorer, default_cfv):
        """uncertainty_penalty is in [UNCERTAINTY_PENALTY_FLOOR, 1.0]."""
        result = scorer.score(default_cfv, today=_TODAY)
        assert config.UNCERTAINTY_PENALTY_FLOOR <= result.uncertainty_penalty <= 1.0, (
            f"uncertainty_penalty={result.uncertainty_penalty} out of "
            f"[{config.UNCERTAINTY_PENALTY_FLOOR}, 1.0]"
        )

    # ── Recency decay tests ──────────────────────────────────────────────────

    def test_active_today_has_recency_near_one(self, scorer):
        """Candidate active today should have recency_score ≈ 1.0."""
        signals = _make_signals(last_active_date=_TODAY)
        cfv = _make_cfv("CAND_0000031", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert result.recency_score > 0.99, (
            f"Active today should have recency > 0.99, got {result.recency_score:.4f}"
        )

    def test_active_14_days_ago_has_high_recency(self, scorer):
        """
        Candidate active 14 days ago should score > 0.90.
        From the proposal: 'A commit from 14 days ago scores ~0.93.'
        Formula: e^(-0.005 * 14) ≈ 0.932.
        """
        signals = _make_signals(last_active_date=_TODAY - timedelta(days=14))
        cfv = _make_cfv("CAND_0000031", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert result.recency_score > 0.90, (
            f"14-day inactive recency should be > 0.90, got {result.recency_score:.4f}"
        )

    def test_inactive_6_months_has_low_recency(self, scorer):
        """
        Candidate inactive for 6 months (180 days) should score < 0.45.
        Formula: e^(-0.005 * 180) ≈ 0.407.
        This validates the JD requirement: 'down-weight unavailable candidates.'
        """
        signals = _make_signals(last_active_date=_TODAY - timedelta(days=180))
        cfv = _make_cfv("CAND_0000031", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert result.recency_score < 0.45, (
            f"6-month inactive recency should be < 0.45, got {result.recency_score:.4f}"
        )

    def test_inactive_1_year_has_very_low_recency(self, scorer):
        """
        Candidate inactive for 365 days should score < 0.20.
        Formula: e^(-0.005 * 365) ≈ 0.163.
        """
        signals = _make_signals(last_active_date=_TODAY - timedelta(days=365))
        cfv = _make_cfv("CAND_0000031", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert result.recency_score < 0.20, (
            f"1-year inactive recency should be < 0.20, got {result.recency_score:.4f}"
        )

    def test_recency_is_monotonically_decreasing_with_inactivity(self, scorer):
        """Longer inactivity always produces lower recency scores."""
        day_counts = [0, 14, 30, 90, 180, 365]
        scores = []
        for days in day_counts:
            signals = _make_signals(last_active_date=_TODAY - timedelta(days=days))
            cfv = _make_cfv("CAND_0000031", signals)
            result = scorer.score(cfv, today=_TODAY)
            scores.append(result.recency_score)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Recency not monotone at index {i}: "
                f"days={day_counts[i]}→score={scores[i]:.4f}, "
                f"days={day_counts[i+1]}→score={scores[i+1]:.4f}"
            )

    # ── Notice period tests ──────────────────────────────────────────────────

    def test_notice_at_ideal_max_gets_score_one(self, scorer):
        """
        Notice period exactly at NOTICE_PERIOD_IDEAL_MAX (30 days) → score = 1.0.
        JD says 'we'd love sub-30-day notice'.
        """
        signals = _make_signals(notice_period_days=config.NOTICE_PERIOD_IDEAL_MAX)
        cfv = _make_cfv("CAND_0000031", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert result.notice_period_score == 1.0, (
            f"notice={config.NOTICE_PERIOD_IDEAL_MAX}d should give score=1.0, "
            f"got {result.notice_period_score}"
        )

    def test_notice_below_ideal_max_gets_score_one(self, scorer):
        """Notice period < NOTICE_PERIOD_IDEAL_MAX → score = 1.0."""
        signals = _make_signals(notice_period_days=15)
        cfv = _make_cfv("CAND_0000031", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert result.notice_period_score == 1.0

    def test_notice_above_max_gets_score_0_1(self, scorer):
        """
        Notice period > NOTICE_PERIOD_MAX (90 days) → score = 0.1.
        150-day notice candidates are explicitly in scope per JD but at lower priority.
        """
        signals = _make_signals(notice_period_days=150)
        cfv = _make_cfv("CAND_0000031", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert result.notice_period_score == 0.1, (
            f"150-day notice should give score=0.1, got {result.notice_period_score}"
        )

    def test_notice_decay_between_ideal_and_acceptable(self, scorer):
        """
        Notice period between IDEAL_MAX and ACCEPTABLE_MAX decays from 1.0 to 0.5.
        Midpoint between 30 and 60 days (45 days) should give ~0.75.
        """
        mid = (config.NOTICE_PERIOD_IDEAL_MAX + config.NOTICE_PERIOD_ACCEPTABLE_MAX) // 2
        signals = _make_signals(notice_period_days=mid)
        cfv = _make_cfv("CAND_0000031", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert 0.5 < result.notice_period_score < 1.0, (
            f"{mid}-day notice should give score in (0.5, 1.0), "
            f"got {result.notice_period_score}"
        )

    def test_notice_period_monotonically_decreasing(self, scorer):
        """Longer notice period always produces lower or equal notice score."""
        notice_days = [0, 15, 30, 45, 60, 75, 90, 120, 150]
        scores = []
        for nd in notice_days:
            signals = _make_signals(notice_period_days=nd)
            cfv = _make_cfv("CAND_0000031", signals)
            result = scorer.score(cfv, today=_TODAY)
            scores.append(result.notice_period_score)
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Notice score not monotone at index {i}: "
                f"days={notice_days[i]}→{scores[i]:.4f}, "
                f"days={notice_days[i+1]}→{scores[i+1]:.4f}"
            )

    # ── Open-to-work and response rate tests ─────────────────────────────────

    def test_open_to_work_true_higher_than_false(self, scorer):
        """
        Candidate with open_to_work_flag=True must score higher than one with
        open_to_work_flag=False (all other signals equal).
        """
        signals_open   = _make_signals(open_to_work_flag=True,  last_active_date=_TODAY)
        signals_closed = _make_signals(open_to_work_flag=False, last_active_date=_TODAY)
        cfv_open   = _make_cfv("CAND_OPEN",   signals_open)
        cfv_closed = _make_cfv("CAND_CLOSED", signals_closed)
        result_open   = scorer.score(cfv_open,   today=_TODAY)
        result_closed = scorer.score(cfv_closed, today=_TODAY)
        assert result_open.behavioral_score > result_closed.behavioral_score, (
            f"open_to_work=True ({result_open.behavioral_score:.4f}) should beat "
            f"open_to_work=False ({result_closed.behavioral_score:.4f})"
        )

    def test_high_response_rate_higher_than_low(self, scorer):
        """High recruiter response rate produces higher behavioral score."""
        signals_high = _make_signals(recruiter_response_rate=0.9, last_active_date=_TODAY)
        signals_low  = _make_signals(recruiter_response_rate=0.1, last_active_date=_TODAY)
        cfv_high = _make_cfv("CAND_HIGH", signals_high)
        cfv_low  = _make_cfv("CAND_LOW",  signals_low)
        result_high = scorer.score(cfv_high, today=_TODAY)
        result_low  = scorer.score(cfv_low,  today=_TODAY)
        assert result_high.behavioral_score > result_low.behavioral_score, (
            f"response_rate=0.9 ({result_high.behavioral_score:.4f}) should beat "
            f"response_rate=0.1 ({result_low.behavioral_score:.4f})"
        )

    # ── GitHub score tests ───────────────────────────────────────────────────

    def test_github_not_linked_uses_default(self, scorer):
        """
        GitHub not linked (has_github=False) → github_activity sub-score
        equals config.GITHUB_NOT_LINKED_DEFAULT (neutral, not penalised).
        """
        signals = _make_signals(has_github=False, github_activity_score=-1.0)
        cfv = _make_cfv("CAND_NOGIT", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert abs(result.sub_scores["github_activity"] - config.GITHUB_NOT_LINKED_DEFAULT) < 1e-9, (
            f"Expected github_activity={config.GITHUB_NOT_LINKED_DEFAULT} for not-linked, "
            f"got {result.sub_scores['github_activity']}"
        )

    def test_github_linked_uses_actual_score(self, scorer):
        """
        GitHub linked (has_github=True, github_activity_score=80.0) →
        github_activity sub-score = 80.0 / 100.0 = 0.80.
        """
        signals = _make_signals(has_github=True, github_activity_score=80.0)
        cfv = _make_cfv("CAND_GIT", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert abs(result.sub_scores["github_activity"] - 0.80) < 1e-6, (
            f"Expected github_activity=0.80 for score=80.0, "
            f"got {result.sub_scores['github_activity']}"
        )

    def test_github_zero_score_linked_gives_zero(self, scorer):
        """GitHub linked with activity score = 0 should give sub-score = 0.0."""
        signals = _make_signals(has_github=True, github_activity_score=0.0)
        cfv = _make_cfv("CAND_GITLOW", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert abs(result.sub_scores["github_activity"] - 0.0) < 1e-6

    # ── Sub-scores structure tests ───────────────────────────────────────────

    def test_sub_scores_keys_match_behavioral_weights(self, scorer, default_cfv):
        """
        sub_scores dict must have exactly the same keys as config.BEHAVIORAL_WEIGHTS.
        composite.py iterates both dicts together — key mismatch causes KeyError.
        """
        result = scorer.score(default_cfv, today=_TODAY)
        expected_keys = set(config.BEHAVIORAL_WEIGHTS.keys())
        actual_keys   = set(result.sub_scores.keys())
        assert actual_keys == expected_keys, (
            f"sub_scores keys mismatch.\n"
            f"Expected: {sorted(expected_keys)}\n"
            f"Got:      {sorted(actual_keys)}"
        )

    def test_sub_scores_all_in_0_1_range(self, scorer, default_cfv):
        """All values in sub_scores must be in [0.0, 1.0]."""
        result = scorer.score(default_cfv, today=_TODAY)
        for key, val in result.sub_scores.items():
            assert 0.0 <= val <= 1.0, (
                f"sub_scores['{key}']={val} out of [0, 1]"
            )

    def test_behavioral_score_equals_weighted_sum_of_sub_scores(self, scorer, default_cfv):
        """
        behavioral_score must equal Σ(BEHAVIORAL_WEIGHTS[k] * sub_scores[k])
        clipped to [0, 1]. Validates that BehavioralScorer uses the config
        weights correctly and doesn't have a hardcoded alternative formula.
        """
        result = scorer.score(default_cfv, today=_TODAY)
        expected = sum(
            config.BEHAVIORAL_WEIGHTS[k] * v
            for k, v in result.sub_scores.items()
        )
        expected = min(max(expected, 0.0), 1.0)
        assert abs(result.behavioral_score - expected) < 1e-9, (
            f"behavioral_score={result.behavioral_score:.8f} does not match "
            f"weighted sum={expected:.8f}"
        )

    # ── Uncertainty penalty and signal count tests ───────────────────────────

    def test_zero_signals_gives_penalty_floor(self, scorer):
        """
        A candidate with NO signals populated should get uncertainty_penalty
        equal to config.UNCERTAINTY_PENALTY_FLOOR (0.70).
        """
        signals = _make_signals(
            profile_views_received_30d=0,
            applications_submitted_30d=0,
            search_appearance_30d=0,
            saved_by_recruiters_30d=0,
            connection_count=0,
            endorsements_received=0,
            has_github=False,
            has_offer_history=False,
            skill_assessment_scores={},
        )
        cfv = _make_cfv("CAND_SPARSE", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert result.signal_count == 0, (
            f"Expected signal_count=0, got {result.signal_count}"
        )
        assert abs(result.uncertainty_penalty - config.UNCERTAINTY_PENALTY_FLOOR) < 1e-9, (
            f"Zero signals should give penalty={config.UNCERTAINTY_PENALTY_FLOOR}, "
            f"got {result.uncertainty_penalty}"
        )

    def test_full_signals_gives_penalty_one(self, scorer):
        """
        A candidate with ALL 9 signal types populated should get
        uncertainty_penalty = 1.0 (no penalty; full confidence).
        """
        signals = _make_signals(
            profile_views_received_30d=50,
            applications_submitted_30d=3,
            search_appearance_30d=200,
            saved_by_recruiters_30d=5,
            connection_count=300,
            endorsements_received=20,
            has_github=True,
            has_offer_history=True,
            skill_assessment_scores={"Python": 85.0},
        )
        cfv = _make_cfv("CAND_RICH", signals)
        result = scorer.score(cfv, today=_TODAY)
        assert result.signal_count >= config.MIN_SIGNAL_TYPES_FOR_FULL_CONFIDENCE, (
            f"Expected signal_count >= {config.MIN_SIGNAL_TYPES_FOR_FULL_CONFIDENCE}, "
            f"got {result.signal_count}"
        )
        assert abs(result.uncertainty_penalty - 1.0) < 1e-9, (
            f"Full signals should give uncertainty_penalty=1.0, "
            f"got {result.uncertainty_penalty}"
        )

    def test_uncertainty_penalty_monotone_with_signal_count(self, scorer):
        """More signals always produces equal or higher uncertainty_penalty."""
        from scoring.behavioral import BehavioralScorer as BS
        scores = [BS._uncertainty_penalty(i) for i in range(10)]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"uncertainty_penalty not monotone at signal_count={i}: "
                f"{scores[i]:.4f} > {scores[i + 1]:.4f}"
            )

    # ── Batch scoring test ───────────────────────────────────────────────────

    def test_score_all_returns_dict_keyed_by_candidate_id(self, scorer):
        """score_all() returns dict[candidate_id → BehavioralResult]."""
        cfv1 = _make_cfv("CAND_0000031", _make_signals(last_active_date=_TODAY))
        cfv2 = _make_cfv("CAND_0000043", _make_signals(last_active_date=_TODAY))
        results = scorer.score_all([cfv1, cfv2], today=_TODAY)
        assert isinstance(results, dict)
        assert set(results.keys()) == {"CAND_0000031", "CAND_0000043"}, (
            f"Keys mismatch: {set(results.keys())}"
        )
        for cid, res in results.items():
            assert isinstance(res, BehavioralResult)
            assert res.candidate_id == cid

    def test_score_all_empty_list_returns_empty_dict(self, scorer):
        """score_all([]) returns {}."""
        result = scorer.score_all([], today=_TODAY)
        assert result == {}


# ─────────────────────────────────────────────────────────────────────────────
# 3. TrajectoryVelocityScorer tests — scoring/trajectory.py
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectoryVelocityScorer:
    """
    Unit tests for scoring/trajectory.py.

    Tests cover:
      - count_promotions()          pure function, no mocks needed
      - trajectory_velocity_score() pure function, no mocks needed
      - TrajectoryVelocityScorer.score()     single-candidate
      - TrajectoryVelocityScorer.score_all() batch with percentile_rank
      - Sprint acceptance criteria (3 promos in 4yr / stagnant 10yr)
    """

    @pytest.fixture
    def scorer(self) -> TrajectoryVelocityScorer:
        return TrajectoryVelocityScorer()

    # ── count_promotions() tests ──────────────────────────────────────────────

    def test_count_promotions_zero_for_single_role(self):
        """Single-role career has 0 promotions by definition."""
        history = [_career_entry("Software Engineer", date(2020, 1, 1))]
        assert count_promotions(history) == 0

    def test_count_promotions_zero_for_empty_history(self):
        """Empty career history has 0 promotions."""
        assert count_promotions([]) == 0

    def test_count_promotions_junior_to_mid_to_senior(self):
        """
        Three-role career: Junior → Software Engineer → Senior Engineer.
        Junior (level 1) → SE (level 2): +1 promotion
        SE (level 2) → Senior (level 3): +1 promotion
        Expected: 2 promotions
        """
        history = [
            _career_entry("Junior Software Engineer", date(2018, 1, 1)),
            _career_entry("Software Engineer",        date(2020, 1, 1)),
            _career_entry("Senior Software Engineer", date(2022, 1, 1)),
        ]
        result = count_promotions(history)
        assert result == 2, (
            f"Junior → SE → Senior should give 2 promotions, got {result}"
        )

    def test_count_promotions_no_count_for_lateral_move(self):
        """
        Moving from SE at one company to SE at another is lateral — no promotion.
        """
        history = [
            _career_entry("Software Engineer", date(2018, 1, 1)),
            _career_entry("Software Engineer", date(2021, 1, 1)),  # same level
        ]
        assert count_promotions(history) == 0, (
            "Lateral move (same title) should not count as promotion"
        )

    def test_count_promotions_no_count_for_downgrade(self):
        """
        Title downgrade (Senior → Engineer) counts as 0 promotions at that step.
        """
        history = [
            _career_entry("Senior Engineer",  date(2018, 1, 1)),
            _career_entry("Software Engineer", date(2021, 1, 1)),  # lower level
        ]
        assert count_promotions(history) == 0, (
            "Title downgrade should not count as promotion"
        )

    def test_count_promotions_orders_chronologically(self):
        """
        count_promotions() sorts by start_date before counting.
        Passing entries in reverse order must give the same result.
        """
        forward = [
            _career_entry("Junior Engineer", date(2018, 1, 1)),
            _career_entry("Senior Engineer", date(2021, 1, 1)),
        ]
        reversed_order = [
            _career_entry("Senior Engineer", date(2021, 1, 1)),
            _career_entry("Junior Engineer", date(2018, 1, 1)),
        ]
        assert count_promotions(forward) == count_promotions(reversed_order), (
            "count_promotions must sort by start_date — order of input should not matter"
        )

    def test_count_promotions_manager_after_senior(self):
        """
        Senior Engineer (level 3) → Engineering Manager (level 4): 1 promotion.
        """
        history = [
            _career_entry("Senior Engineer",      date(2019, 6, 1)),
            _career_entry("Engineering Manager",  date(2022, 3, 1)),
        ]
        assert count_promotions(history) == 1

    def test_count_promotions_ic_path_three_steps(self):
        """
        Full IC path: Junior → SE → Senior → Staff → Principal.
        Junior(1) → SE(2) → Senior(3) → Staff(4) → Principal(4): 3 promotions.
        (Staff and Principal both map to level 4 so final step doesn't count.)
        """
        history = [
            _career_entry("Junior Software Engineer", date(2015, 1, 1)),
            _career_entry("Software Engineer",         date(2017, 1, 1)),
            _career_entry("Senior Engineer",           date(2019, 1, 1)),
            _career_entry("Staff Engineer",            date(2021, 6, 1)),
            _career_entry("Principal Engineer",        date(2023, 1, 1)),
        ]
        result = count_promotions(history)
        # Junior(1)→SE(2): +1, SE(2)→Senior(3): +1, Senior(3)→Staff(4): +1,
        # Staff(4)→Principal(4): same, no count
        assert result == 3, f"IC ladder should give 3 promotions, got {result}"

    # ── trajectory_velocity_score() tests ────────────────────────────────────

    def test_velocity_score_at_floor_is_zero(self):
        """0 promotions/year → velocity score = 0.0."""
        score = trajectory_velocity_score(config.TRAJECTORY_PROMOTIONS_PER_YEAR_FLOOR)
        assert score == 0.0, f"At floor: expected 0.0, got {score}"

    def test_velocity_score_at_cap_is_one(self):
        """At or above CAP promotions/year → velocity score = 1.0."""
        score = trajectory_velocity_score(config.TRAJECTORY_PROMOTIONS_PER_YEAR_CAP)
        assert score == 1.0, f"At cap: expected 1.0, got {score}"

    def test_velocity_score_above_cap_clipped_to_one(self):
        """Above CAP promotions/year → clipped to 1.0 (not > 1.0)."""
        score = trajectory_velocity_score(config.TRAJECTORY_PROMOTIONS_PER_YEAR_CAP * 3)
        assert score == 1.0, f"Above cap: expected 1.0, got {score}"

    def test_velocity_score_midpoint(self):
        """
        Midpoint between floor (0.0) and cap (1.5) is 0.75/yr → score = 0.5.
        """
        mid_rate = (
            config.TRAJECTORY_PROMOTIONS_PER_YEAR_FLOOR
            + config.TRAJECTORY_PROMOTIONS_PER_YEAR_CAP
        ) / 2.0
        score = trajectory_velocity_score(mid_rate)
        assert abs(score - 0.5) < 1e-6, (
            f"Midpoint rate {mid_rate:.3f}/yr should give velocity=0.5, got {score:.6f}"
        )

    def test_velocity_score_monotone_increasing(self):
        """Higher promotion rate always produces equal or higher velocity score."""
        rates = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        scores = [trajectory_velocity_score(r) for r in rates]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1], (
                f"velocity_score not monotone at index {i}: "
                f"rate={rates[i]}→{scores[i]:.4f}, "
                f"rate={rates[i+1]}→{scores[i+1]:.4f}"
            )

    # ── TrajectoryVelocityScorer.score() tests ───────────────────────────────

    def test_score_returns_trajectory_result(self, scorer):
        """score() returns a TrajectoryResult instance."""
        cfv = _make_cfv_with_career(
            "CAND_0000031",
            [_career_entry("Software Engineer", date(2018, 1, 1))],
            years_of_experience=4.0,
        )
        result = scorer.score(cfv)
        assert isinstance(result, TrajectoryResult), (
            f"Expected TrajectoryResult, got {type(result).__name__}"
        )

    def test_score_candidate_id_preserved(self, scorer):
        """TrajectoryResult.candidate_id matches input."""
        cfv = _make_cfv_with_career("CAND_0000031", [], years_of_experience=4.0)
        result = scorer.score(cfv)
        assert result.candidate_id == "CAND_0000031"

    def test_score_trajectory_velocity_in_range(self, scorer):
        """trajectory_velocity is always in [0.0, 1.0]."""
        cfv = _make_cfv_with_career(
            "CAND_0000031",
            [
                _career_entry("Junior Engineer", date(2018, 1, 1)),
                _career_entry("Senior Engineer", date(2021, 1, 1)),
            ],
            years_of_experience=6.0,
        )
        result = scorer.score(cfv)
        assert 0.0 <= result.trajectory_velocity <= 1.0, (
            f"trajectory_velocity={result.trajectory_velocity} out of [0, 1]"
        )

    def test_single_role_career_has_zero_promotions(self, scorer):
        """Candidate with one role has num_promotions=0 and low velocity."""
        cfv = _make_cfv_with_career(
            "CAND_STAGNANT",
            [_career_entry("Software Engineer", date(2015, 1, 1))],
            years_of_experience=10.0,
            total_career_months=120,
        )
        result = scorer.score(cfv)
        assert result.num_promotions == 0
        assert result.trajectory_velocity == 0.0

    def test_score_all_attaches_percentile_rank(self, scorer):
        """score_all() attaches percentile_rank (0-100) to every result."""
        cfvs = [
            _make_cfv_with_career(
                f"CAND_{i:07d}",
                [_career_entry("Software Engineer", date(2018, 1, 1))],
                years_of_experience=4.0,
            )
            for i in range(1, 6)
        ]
        results = scorer.score_all(cfvs)
        for r in results:
            assert r.percentile_rank is not None, (
                f"{r.candidate_id}: percentile_rank should not be None after score_all()"
            )
            assert 0.0 <= r.percentile_rank <= 100.0, (
                f"{r.candidate_id}: percentile_rank={r.percentile_rank} out of [0, 100]"
            )

    def test_score_all_empty_returns_empty(self, scorer):
        """score_all([]) returns []."""
        assert scorer.score_all([]) == []

    # ── Sprint acceptance criteria ────────────────────────────────────────────

    def test_acceptance_3_promos_in_4_years_above_80th_percentile(self, scorer):
        """
        KEY ACCEPTANCE TEST: 3 promotions in 4 years → percentile_rank > 80.

        Career: Junior (2019) → SE (2021) → Senior (2022) → Staff (2023)
        promotions_per_year = 3 / 4 = 0.75/yr

        In a realistic pool where most candidates have 0 promotions/yr (single
        long tenure), a rate of 0.75/yr is exceptional and should sit above
        the 80th percentile.

        Per trajectory.py docstring:
          '3 promotions in 4 years (rate = 0.75/yr) -> percentile_rank > 80'
        """
        # HIGH VELOCITY candidate: 3 promotions in 4 years
        high_velocity = _make_cfv_with_career(
            "CAND_HIGH",
            [
                _career_entry("Junior Engineer",    date(2019, 1, 1)),
                _career_entry("Software Engineer",  date(2021, 1, 1)),
                _career_entry("Senior Engineer",    date(2022, 1, 1)),
                _career_entry("Staff Engineer",     date(2023, 1, 1)),
            ],
            years_of_experience=4.0,
            total_career_months=48,
        )

        # Pool of STAGNANT candidates (most common pattern: no promotions)
        stagnant_pool = [
            _make_cfv_with_career(
                f"CAND_STAG_{i:03d}",
                [_career_entry("Software Engineer", date(2014, 1, 1))],
                years_of_experience=10.0,
                total_career_months=120,
            )
            for i in range(8)   # 8 stagnant candidates
        ]

        all_candidates = stagnant_pool + [high_velocity]
        results = scorer.score_all(all_candidates)

        high_result = next(r for r in results if r.candidate_id == "CAND_HIGH")
        assert high_result.percentile_rank > 80.0, (
            f"FAIL: 3 promos in 4yr should be > 80th percentile, "
            f"got {high_result.percentile_rank:.1f}. "
            f"promotions_per_year={high_result.promotions_per_year:.3f}"
        )

    def test_acceptance_stagnant_10_year_below_40th_percentile(self, scorer):
        """
        KEY ACCEPTANCE TEST: Stagnant 10-year tenure → percentile_rank < 40.

        Career: Single role for 10 years with no promotion.
        promotions_per_year = 0.0/yr

        Against a mixed pool (some promoted, some not), a 10-year stagnant
        candidate should rank below the 40th percentile.

        Per trajectory.py docstring:
          'stagnant 10-year tenure (rate = 0.00/yr) -> percentile_rank < 40'
        """
        # STAGNANT candidate: one role, 10 years, zero promotions
        stagnant = _make_cfv_with_career(
            "CAND_STAGNANT",
            [_career_entry("Software Engineer", date(2013, 1, 1))],
            years_of_experience=10.0,
            total_career_months=120,
        )

        # Mix of candidates with various promotion rates
        mixed_pool = [
            _make_cfv_with_career(  # 1 promotion in 5yr
                "CAND_SLOW",
                [
                    _career_entry("Junior Engineer",   date(2018, 1, 1)),
                    _career_entry("Software Engineer", date(2023, 1, 1)),
                ],
                years_of_experience=5.0,
                total_career_months=60,
            ),
            _make_cfv_with_career(  # 2 promotions in 6yr
                "CAND_MED",
                [
                    _career_entry("Junior Engineer",   date(2017, 1, 1)),
                    _career_entry("Software Engineer", date(2019, 1, 1)),
                    _career_entry("Senior Engineer",   date(2021, 1, 1)),
                ],
                years_of_experience=6.0,
                total_career_months=72,
            ),
            _make_cfv_with_career(  # 3 promotions in 4yr
                "CAND_FAST",
                [
                    _career_entry("Junior Engineer",   date(2020, 1, 1)),
                    _career_entry("Software Engineer", date(2021, 6, 1)),
                    _career_entry("Senior Engineer",   date(2022, 6, 1)),
                    _career_entry("Staff Engineer",    date(2023, 6, 1)),
                ],
                years_of_experience=4.0,
                total_career_months=48,
            ),
        ]

        all_candidates = [stagnant] + mixed_pool
        results = scorer.score_all(all_candidates)

        stagnant_result = next(r for r in results if r.candidate_id == "CAND_STAGNANT")
        assert stagnant_result.percentile_rank < 40.0, (
            f"FAIL: Stagnant 10yr should be < 40th percentile, "
            f"got {stagnant_result.percentile_rank:.1f}. "
            f"promotions_per_year={stagnant_result.promotions_per_year:.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. HoneypotCleanup tests — scoring/honeypot_filter.py
# ─────────────────────────────────────────────────────────────────────────────

class TestHoneypotFilter:
    """
    Unit tests for scoring/honeypot_filter.py: HoneypotCleanup.cleanup_candidates()

    Implementation note: the stub uses candidate.is_honeypot (boolean attribute).
    All tests use MagicMock to set this attribute directly — avoids needing to
    know whether it is a dataclass field or computed property.
    """

    @pytest.fixture
    def filter(self) -> HoneypotCleanup:
        return HoneypotCleanup()

    def _make_pool(self, spec: list[tuple[str, bool]]) -> list[MagicMock]:
        """Build a candidate pool from (candidate_id, is_honeypot) tuples."""
        return [_make_cfv_honeypot(cid, is_hp) for cid, is_hp in spec]

    # ── Basic filtering tests ─────────────────────────────────────────────────

    def test_removes_honeypot_candidates(self, filter):
        """Honeypot candidates (is_honeypot=True) are excluded from output."""
        pool = self._make_pool([
            ("CAND_0000031", False),   # clean
            ("CAND_9990001", True),    # honeypot
            ("CAND_9990002", True),    # honeypot
        ])
        result = filter.cleanup_candidates(pool)
        result_ids = [c.candidate_id for c in result]
        assert "CAND_9990001" not in result_ids, "Honeypot CAND_9990001 should be removed"
        assert "CAND_9990002" not in result_ids, "Honeypot CAND_9990002 should be removed"

    def test_preserves_clean_candidates(self, filter):
        """Clean candidates (is_honeypot=False) are preserved in output."""
        pool = self._make_pool([
            ("CAND_0000031", False),
            ("CAND_0000043", False),
            ("CAND_9990001", True),
        ])
        result = filter.cleanup_candidates(pool)
        result_ids = [c.candidate_id for c in result]
        assert "CAND_0000031" in result_ids, "Clean CAND_0000031 should be preserved"
        assert "CAND_0000043" in result_ids, "Clean CAND_0000043 should be preserved"

    def test_mixed_pool_correct_count(self, filter):
        """Output contains exactly the non-honeypot candidates."""
        pool = self._make_pool([
            ("CAND_0000031", False),
            ("CAND_0000043", False),
            ("CAND_0000014", False),
            ("CAND_9990001", True),
            ("CAND_9990002", True),
        ])
        result = filter.cleanup_candidates(pool)
        assert len(result) == 3, (
            f"Expected 3 clean candidates, got {len(result)}"
        )

    def test_empty_pool_returns_empty(self, filter):
        """cleanup_candidates([]) returns []."""
        result = filter.cleanup_candidates([])
        assert result == []

    def test_all_honeypots_returns_empty(self, filter):
        """Pool of only honeypots returns empty list."""
        pool = self._make_pool([
            ("CAND_9990001", True),
            ("CAND_9990002", True),
            ("CAND_9990003", True),
        ])
        result = filter.cleanup_candidates(pool)
        assert result == [], f"All-honeypot pool should return [], got {result}"

    def test_no_honeypots_returns_unchanged_length(self, filter):
        """Pool with zero honeypots returns all candidates."""
        pool = self._make_pool([
            ("CAND_0000031", False),
            ("CAND_0000043", False),
        ])
        result = filter.cleanup_candidates(pool)
        assert len(result) == 2, (
            f"No-honeypot pool should return all 2 candidates, got {len(result)}"
        )

    def test_preserves_order_of_clean_candidates(self, filter):
        """Clean candidates appear in the same relative order as the input."""
        pool = self._make_pool([
            ("CAND_0000031", False),
            ("CAND_9990001", True),    # honeypot in middle
            ("CAND_0000043", False),
        ])
        result = filter.cleanup_candidates(pool)
        result_ids = [c.candidate_id for c in result]
        assert result_ids == ["CAND_0000031", "CAND_0000043"], (
            f"Order should be preserved: got {result_ids}"
        )

    def test_returns_list_type(self, filter):
        """cleanup_candidates() returns a list, not a generator or tuple."""
        pool = self._make_pool([("CAND_0000031", False)])
        result = filter.cleanup_candidates(pool)
        assert isinstance(result, list), (
            f"Expected list, got {type(result).__name__}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. CrossEncoderReranker fallback tests — scoring/cross_encoder.py
#    Tests the NO-MODEL path to keep these fast (no 80MB model loaded).
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossEncoderFallback:
    """
    Tests for scoring/cross_encoder.py that exercise the fallback path
    (model not loaded / unavailable).

    All tests are fast (no model I/O). The full model path is covered by
    the integration test with TEST_LIVE_MODEL=1 in the smoke test.
    """

    @pytest.fixture
    def jd(self, mock_jd_intent):
        return mock_jd_intent

    def _make_pool(self, count: int) -> list[RRFResult]:
        """Build a pool of RRFResult objects with decreasing rrf_scores."""
        return [
            RRFResult(
                candidate_id=f"CAND_{i + 1:07d}",
                rrf_score=1.0 - i * 0.01,
                paths_present=["semantic"],
                cross_encoder_score=0.0,
            )
            for i in range(count)
        ]

    def _make_candidate_store(self, pool: list[RRFResult]) -> dict:
        """Build a mock candidate store for all candidates in pool."""
        store = {}
        for r in pool:
            mock_cfv = MagicMock()
            mock_cfv.candidate_id = r.candidate_id
            mock_cfv.embedding_text = (
                f"ML engineer with embeddings and FAISS experience. "
                f"Candidate {r.candidate_id}."
            )
            mock_cfv.headline = "ML Engineer"
            mock_cfv.current_title = "ML Engineer"
            mock_cfv.current_company = "TestCo"
            mock_cfv.years_of_experience = 6.0
            mock_cfv.skills = []
            store[r.candidate_id] = mock_cfv
        return store

    # ── Fallback path tests ───────────────────────────────────────────────────

    def test_fallback_scores_in_range(self, jd):
        """
        When model is not loaded, fallback assigns normalised rrf_score as
        cross_encoder_score. All values must be in [0.0, 1.0].
        """
        from scoring.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_k=10)
        # Model stays None (not loaded) — triggers fallback
        pool = self._make_pool(5)
        store = self._make_candidate_store(pool)
        result = reranker.rerank(pool, jd, store)
        for r in result:
            assert 0.0 <= r.cross_encoder_score <= 1.0, (
                f"{r.candidate_id}: cross_encoder_score={r.cross_encoder_score} out of [0,1]"
            )

    def test_fallback_sorted_descending(self, jd):
        """Fallback output is sorted by cross_encoder_score descending."""
        from scoring.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_k=10)
        pool = self._make_pool(5)
        store = self._make_candidate_store(pool)
        result = reranker.rerank(pool, jd, store)
        for i in range(len(result) - 1):
            assert result[i].cross_encoder_score >= result[i + 1].cross_encoder_score, (
                f"Not sorted descending at index {i}: "
                f"{result[i].cross_encoder_score} < {result[i+1].cross_encoder_score}"
            )

    def test_empty_pool_returns_empty(self, jd):
        """rerank([], ...) returns [] without error."""
        from scoring.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_k=10)
        result = reranker.rerank([], jd, {})
        assert result == []

    def test_skipped_candidate_scores_zero(self, jd):
        """
        Candidate in pool but missing from candidate_store gets score=0.0
        and sorts to the bottom.
        """
        from scoring.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_k=10)

        pool = [
            RRFResult("CAND_0000001", 0.9, ["semantic"], 0.0),
            RRFResult("CAND_0000099", 0.5, ["keyword"],  0.0),  # will be skipped
        ]
        # Store only has CAND_0000001; CAND_0000099 is missing
        store = self._make_candidate_store([pool[0]])
        result = reranker.rerank(pool, jd, store)

        skipped = next(r for r in result if r.candidate_id == "CAND_0000099")
        assert skipped.cross_encoder_score == 0.0, (
            f"Missing candidate should have score=0.0, got {skipped.cross_encoder_score}"
        )
        assert result[-1].candidate_id == "CAND_0000099", (
            "Missing candidate (score=0.0) should sort to last position"
        )

    def test_raises_type_error_on_non_list_pool(self, jd):
        """TypeError raised when pool is not a list."""
        from scoring.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_k=10)
        with pytest.raises(TypeError):
            reranker.rerank("not-a-list", jd, {})  # type: ignore

    def test_raises_type_error_on_non_jdintent(self):
        """TypeError raised when jd is not a JDIntent."""
        from scoring.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_k=10)
        with pytest.raises(TypeError):
            reranker.rerank([], "not-a-jdintent", {})  # type: ignore

    def test_sigmoid_is_monotone(self):
        """_sigmoid is strictly monotonically increasing."""
        from scoring.cross_encoder import _sigmoid
        logits = [-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0]
        scores = [_sigmoid(x) for x in logits]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"_sigmoid not monotone at index {i}: "
                f"logit={logits[i]}→{scores[i]:.6f}, "
                f"logit={logits[i+1]}→{scores[i+1]:.6f}"
            )

    def test_sigmoid_zero_returns_half(self):
        """_sigmoid(0.0) must equal exactly 0.5."""
        from scoring.cross_encoder import _sigmoid
        assert abs(_sigmoid(0.0) - 0.5) < 1e-10, (
            f"_sigmoid(0.0) should be 0.5, got {_sigmoid(0.0)}"
        )

    def test_sigmoid_output_in_range(self):
        """_sigmoid output is always in (0.0, 1.0)."""
        from scoring.cross_encoder import _sigmoid
        extreme_logits = [-500.0, -100.0, -10.0, 0.0, 10.0, 100.0, 500.0]
        for x in extreme_logits:
            result = _sigmoid(x)
            assert 0.0 < result < 1.0, (
                f"_sigmoid({x}) = {result} — must be strictly in (0, 1)"
            )

    def test_top_k_pool_size_enforced(self, jd):
        """Pool larger than top_k is truncated before scoring."""
        from scoring.cross_encoder import CrossEncoderReranker
        reranker = CrossEncoderReranker(top_k=3)
        pool = self._make_pool(10)   # 10 candidates, but top_k=3
        store = self._make_candidate_store(pool)
        result = reranker.rerank(pool, jd, store)
        assert len(result) <= 3, (
            f"top_k=3 should cap results at 3, got {len(result)}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PENDING: DEV B empty file stubs
# Remove @pytest.mark.skip when the files are implemented.
# The class docstrings specify the EXACT interface expected.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skip(reason=(
    "PENDING DEV B — scoring/skill_match.py is empty. "
    "Remove this skip once SkillMatcher is implemented. "
    "Expected interface: SkillMatcher(skill_map_path).score(candidate, jd_intent) "
    "→ SkillMatchResult(candidate_id, skill_score, matched_required, "
    "matched_nice_to_have, ontology_matches)"
))
class TestSkillMatch:
    """
    Specification tests for scoring/skill_match.py.

    Expected class/interface (from config.py and sprint plan):

        from scoring.skill_match import SkillMatcher, SkillMatchResult

        class SkillMatcher:
            def score(
                self,
                candidate: CandidateFeatureVector,
                jd_intent: JDIntent,
            ) -> SkillMatchResult: ...

        @dataclass
        class SkillMatchResult:
            candidate_id: str
            skill_score: float          # 0-1, weighted sum
            matched_required: list[str]
            matched_nice_to_have: list[str]
            ontology_matches: list[str] # partial credit via ontology

    Acceptance criteria:
        - Missing required skill drops score by > 0.3
        - Adjacent ontology skill gives partial credit (ONTOLOGY_PARTIAL_CREDIT = 0.60)
        - Proficiency multiplier applied: expert=1.0, beginner=0.40
        - Duration trust factor: 0 months → MIN=0.50 multiplier
        - Required skill weight is 2× nice-to-have weight
    """

    def test_required_skills_weighted_double_nice_to_have(self):
        """REQUIRED_SKILL_WEIGHT (2.0) must be 2× NICE_TO_HAVE_SKILL_WEIGHT (1.0)."""
        assert config.REQUIRED_SKILL_WEIGHT == 2.0 * config.NICE_TO_HAVE_SKILL_WEIGHT

    def test_skill_score_in_range(self):
        """skill_score must be in [0.0, 1.0]."""
        pytest.skip("Pending DEV B implementation")

    def test_missing_required_skill_drops_score_significantly(self):
        """Missing a required skill must reduce score by > 0.3."""
        pytest.skip("Pending DEV B implementation")

    def test_expert_proficiency_higher_than_beginner(self):
        """expert proficiency must score higher than beginner for the same skill."""
        pytest.skip("Pending DEV B implementation")

    def test_ontology_partial_credit_for_adjacent_skill(self):
        """
        Candidate with 'recommendation systems' gets partial credit
        for JD requirement 'information retrieval' via ontology edge.
        Credit = ONTOLOGY_PARTIAL_CREDIT = 0.60 × required weight.
        """
        pytest.skip("Pending DEV B implementation")

    def test_duration_zero_months_gets_minimum_trust(self):
        """Skill with duration_months=0 gets DURATION_TRUST_MIN=0.50 multiplier."""
        pytest.skip("Pending DEV B implementation")

    def test_endorsed_skills_get_boost(self):
        """Endorsement count provides additive boost up to ENDORSEMENT_BOOST_MAX."""
        pytest.skip("Pending DEV B implementation")


@pytest.mark.skip(reason=(
    "PENDING DEV B — scoring/career_quality.py is empty. "
    "Remove this skip once CareerQualityScorer is implemented. "
    "Sprint acceptance criterion: 'consulting penalty correct'. "
    "Expected interface: CareerQualityScorer().score(candidate, jd_intent) "
    "→ CareerQualityResult(candidate_id, career_score, yoe_score, "
    "trajectory_velocity, is_consulting_only)"
))
class TestCareerQuality:
    """
    Specification tests for scoring/career_quality.py.

    Expected class/interface (from config.py and sprint plan):

        from scoring.career_quality import CareerQualityScorer, CareerQualityResult

        class CareerQualityScorer:
            def score(
                self,
                candidate: CandidateFeatureVector,
                jd_intent: JDIntent,
            ) -> CareerQualityResult: ...

    Acceptance criteria (from sprint plan):
        - TCS-only career scores < 0.3
        - Swiggy/Zomato product-co scores > 0.7
        - YOE 5-9 gets bonus (within ideal band)
        - Consulting-only candidates penalised by CONSULTING_ONLY_PENALTY=0.35
        - Product-co bonus PRODUCT_CO_BONUS=1.20 applied
    """

    def test_consulting_only_tcs_wipro_scores_below_0_3(self):
        """
        SPRINT ACCEPTANCE: TCS-only career scores < 0.3.
        career_score × CONSULTING_ONLY_PENALTY (0.35) must produce a low score.
        """
        pytest.skip("Pending DEV B implementation")

    def test_product_co_swiggy_scores_above_0_7(self):
        """
        SPRINT ACCEPTANCE: Swiggy/Zomato product-co scores > 0.7.
        PRODUCT_CO_BONUS (1.20) must elevate the career score meaningfully.
        """
        pytest.skip("Pending DEV B implementation")

    def test_ideal_yoe_band_5_to_9_gets_bonus(self):
        """Candidates in YOE_BAND_IDEAL_MIN(5) to YOE_BAND_IDEAL_MAX(9) score higher."""
        pytest.skip("Pending DEV B implementation")

    def test_career_score_in_range(self):
        """career_score must be in [0.0, 1.0] even after bonuses and penalties."""
        pytest.skip("Pending DEV B implementation")

    def test_consulting_penalty_constant_is_correct(self):
        """CONSULTING_ONLY_PENALTY must equal 0.35 (from config)."""
        assert abs(config.CONSULTING_ONLY_PENALTY - 0.35) < 1e-9

    def test_product_co_bonus_constant_is_correct(self):
        """PRODUCT_CO_BONUS must equal 1.20 (from config)."""
        assert abs(config.PRODUCT_CO_BONUS - 1.20) < 1e-9


@pytest.mark.skip(reason=(
    "PENDING DEV B — scoring/composite.py is empty. "
    "Remove this skip once CompositeScorer is implemented. "
    "Sprint acceptance criterion: 'non-increasing enforced'. "
    "Expected interface: CompositeScorer().score_all(candidates, jd_intent) "
    "→ list[ComponentScores] sorted by final_score descending."
))
class TestComposite:
    """
    Specification tests for scoring/composite.py.

    Expected class/interface (from config.py and sprint plan):

        from scoring.composite import CompositeScorer, ComponentScores

        class CompositeScorer:
            def score(
                self,
                candidate: CandidateFeatureVector,
                jd_intent: JDIntent,
                behavioral_result: BehavioralResult,
                career_result: CareerQualityResult,
                skill_result: SkillMatchResult,
            ) -> ComponentScores: ...

            def score_all(
                self,
                candidates: list[CandidateFeatureVector],
                jd_intent: JDIntent,
            ) -> list[ComponentScores]: ...   # sorted final_score desc

        @dataclass
        class ComponentScores:
            candidate_id: str
            skill_score: float         # from SkillMatcher
            career_score: float        # from CareerQualityScorer
            behavioral_score: float    # from BehavioralScorer
            trajectory_velocity: float # from TrajectoryVelocityScorer
            final_score: float         # weighted blend, uncertainty-adjusted
            uncertainty_penalty: float
            signal_count: int

    Formula (from config.py):
        raw = (WEIGHT_SKILL * skill_score
             + WEIGHT_CAREER * career_score
             + WEIGHT_BEHAVIORAL * behavioral_score)
        final_score = raw * uncertainty_penalty (clipped to [0, 1])

    Acceptance criteria:
        - final_score is in [0.0, 1.0]
        - score_all() returns list sorted by final_score descending
        - Tie-break: candidate_id ascending
        - Consulting-only candidate scores lower than product-co IC-riser
        - Sparse profile (low signal_count) gets uncertainty penalty < 1.0
    """

    def test_weights_sum_to_one_config_level(self):
        """
        This test runs even with the skip — it is a pure config test.
        Validates the weights that composite.py will use.
        """
        pytest.skip("See TestScoringWeightConstants.test_composite_weights_sum_to_one")

    def test_final_score_in_range(self):
        """final_score must be in [0.0, 1.0]."""
        pytest.skip("Pending DEV B implementation")

    def test_score_all_sorted_descending(self):
        """
        SPRINT ACCEPTANCE: Non-increasing score enforcement.
        score_all() result must be sorted by final_score descending.
        """
        pytest.skip("Pending DEV B implementation")

    def test_uncertainty_penalty_applied_to_final_score(self):
        """Sparse profile (signal_count < MIN_SIGNAL_TYPES) gets penalised score."""
        pytest.skip("Pending DEV B implementation")

    def test_composite_formula_matches_config_weights(self):
        """
        final_score = (WEIGHT_SKILL * skill + WEIGHT_CAREER * career
                     + WEIGHT_BEHAVIORAL * behavioral) * uncertainty_penalty.
        """
        pytest.skip("Pending DEV B implementation")

    def test_location_bonus_applied_for_preferred_cities(self):
        """
        Candidates in Pune/Noida/Delhi NCR get additive location bonus
        from config.PREFERRED_LOCATIONS. This lifts their final_score.
        """
        pytest.skip("Pending DEV B implementation")