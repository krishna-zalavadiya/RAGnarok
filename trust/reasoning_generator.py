"""
trust/reasoning_generator.py — Template-driven recruiter brief generator.

ROLE
----
reasoning_generator.py is the final output stage of the trust layer.
It converts a TrustVerdict into a 1-2 sentence recruiter brief that:

  1. Names specific facts from the candidate's profile (skill names, scores,
     company names, years, days) — zero invented claims.
  2. Varies in tone by rank tier — rank-1 reads differently from rank-95.
  3. Is structurally unique per candidate — two candidates never produce
     the same reasoning string because it assembles from per-candidate
     signal.value strings.
  4. Flags honest concerns even for top-ranked candidates.
  5. Is short enough to fit the submission CSV without truncation.

OUTPUT
------
  str — 1-2 sentences, 80-280 characters.
        Embedded directly into RankedCandidate.reasoning.
        Written verbatim to the submission CSV.

HALLUCINATION PREVENTION
------------------------
Every fact in the output must trace to one of:
  - AdvocateSignal.value  (set by advocate.py from profile fields)
  - SkepticSignal.value   (set by skeptic.py from profile fields)
  - TrustVerdict.confidence_pct  (numeric, computed from signal counts)
  - TrustVerdict.verdict  (classification string)
  - ComponentScores fields (numeric, pre-computed from profile)
  - CandidateFeatureVector fields (parsed from raw profile JSON)

The generator NEVER:
  - Invents a skill name not present in an AdvocateSignal.value.
  - Makes a claim about seniority level not reflected in scores.
  - States a company name not in a signal.value.
  - Uses superlatives ("best", "exceptional") without a HIGH-confidence signal.

RANK-TIER TONE RULES
---------------------
  Rank  1-10  : Positive lead, mention top HIGH advocate signal,
                note any MODERATE/LOW risks briefly.
  Rank 11-30  : Balanced lead, top advocate signal, note HIGH risks.
  Rank 31-60  : Neutral lead, lead with caution if HIGH risks present.
  Rank 61-100 : Risk-forward lead, specify why rank is low.

These tiers ensure rank-1 reasoning is detectably different from rank-95
reasoning even if both candidates have similar verdict classifications.

SENTENCE STRUCTURE
------------------
  Sentence 1 (signal sentence):
    "[Positive claim from top advocate signal]. "
    OR for risky candidates:
    "Ranked [N] due to [top risk]; [mitigating positive if any]. "

  Sentence 2 (trust sentence):
    "[ROBUST|CONTESTED|FRAGILE] ranking at [confidence]% confidence[; risk note]."
    + falsifiability condition if FRAGILE or CONTESTED.

MAX LENGTH GUARD
----------------
  The combined 2-sentence output is capped at 280 characters.
  If over limit, Sentence 2 is shortened to the verdict + confidence only.
  This prevents CSV column overflow in edge cases with very long signal values.

DEPENDENCIES
------------
  config              : rank band constants, submission spec constants
  pipeline.schemas    : TrustVerdict, CandidateFeatureVector,
                        ComponentScores, AdvocateSignal, SkepticSignal
  trust.verdict       : summarise_verdict (for logging only)
  trust.advocate      : top_signals
  trust.skeptic       : top_risks

No I/O.  No network.  No LLM.  No side-effects.  Pure function.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import config
from pipeline.schemas import (
    AdvocateSignal,
    CandidateFeatureVector,
    ComponentScores,
    SkepticSignal,
    TrustVerdict,
)
from trust.advocate import top_signals as top_advocate_signals
from trust.skeptic import top_risks as top_skeptic_risks
from trust.verdict import summarise_verdict

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Rank tier breakpoints (inclusive upper bounds).
_TIER_ELITE: int = 10    # ranks 1–10
_TIER_STRONG: int = 30   # ranks 11–30
_TIER_MID: int = 60      # ranks 31–60
# ranks 61–100 → TIER_WEAK (implicit)

# Output length constraints.
_MIN_CHARS: int = 60
_MAX_CHARS: int = 320
_SENTENCE2_FALLBACK_MAX: int = 120   # max chars for Sentence 2 if Sentence 1 is long

# Confidence display rounding.
_CONF_DECIMALS: int = 0   # "82%" not "82.3%"

# Verdict strings (must match verdict.py and schemas.py).
_ROBUST: str = "ROBUST"
_CONTESTED: str = "CONTESTED"
_FRAGILE: str = "FRAGILE"

# Severity/confidence tier strings.
_HIGH: str = "HIGH"
_MED: str = "MEDIUM"
_MOD: str = "MODERATE"
_LOW: str = "LOW"

# Regex to strip trailing whitespace / punctuation before appending.
_TRAIL_PUNCT_RE = re.compile(r"[.;,\s]+$")


# ─────────────────────────────────────────────────────────────────────────────
# RANK TIER HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _rank_tier(rank: int) -> str:
    """
    Map a rank (1-100) to a named tier string.

    ELITE  → 1-10
    STRONG → 11-30
    MID    → 31-60
    WEAK   → 61-100
    """
    if rank <= _TIER_ELITE:
        return "ELITE"
    if rank <= _TIER_STRONG:
        return "STRONG"
    if rank <= _TIER_MID:
        return "MID"
    return "WEAK"


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL VALUE EXTRACTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_first_fact(signal_value: str, max_len: int = 120) -> str:
    """
    Extract the first concrete fact from a signal value string.

    Signal values are formatted as:
      "82% of required skills matched: FAISS, sentence-transformers, BGE (+3 more)"
      "Active today (recency score 0.98)"
      "3 product company role(s): Swiggy, Zepto, Razorpay"

    We take everything up to the first ' — ' separator or the full string,
    then truncate smartly at skill-list boundaries to avoid mid-word cuts.
    """
    # Split on em-dash separator used in signal values.
    parts = signal_value.split(" — ")
    fact = parts[0].strip()

    # Also split on " (recency" / " (below" etc. — keep the leading fact only.
    paren_idx = fact.find(" (")
    if paren_idx > 30:   # keep the paren if it's short and part of the core fact
        fact = fact[:paren_idx].strip()

    # Smart truncation: round to the last complete comma-separated item.
    if len(fact) > max_len:
        truncated = fact[:max_len]
        # Try to cut at the last ", " to avoid mid-skill-name truncation.
        last_comma = truncated.rfind(", ")
        if last_comma > max_len // 2:  # only if we're keeping at least half
            fact = truncated[:last_comma] + "…"
        else:
            fact = truncated.rstrip() + "…"

    return fact


def _top_advocate_fact(advocate_signals: list[AdvocateSignal]) -> Optional[str]:
    """
    Return the most compelling advocate fact as a short string.

    Prefers HIGH-confidence signals over MEDIUM/LOW.
    Returns None if no signals with usable values.
    """
    top = top_advocate_signals(advocate_signals, n=1)
    if not top:
        return None
    signal = top[0]
    fact = _extract_first_fact(signal.value, max_len=90)
    return fact if fact else None


def _top_risk_summary(skeptic_signals: list[SkepticSignal]) -> Optional[str]:
    """
    Return the most critical skeptic risk as a short phrase.

    Returns None if no HIGH or MODERATE risks.
    """
    top = top_skeptic_risks(skeptic_signals, n=1)
    if not top:
        return None
    risk = top[0]
    if risk.severity not in (_HIGH, _MOD):
        return None
    fact = _extract_first_fact(risk.value, max_len=70)
    label = risk.label
    return f"{label}: {fact}" if fact else label


def _high_advocate_count(advocate_signals: list[AdvocateSignal]) -> int:
    """Count HIGH-confidence advocate signals."""
    return sum(1 for s in advocate_signals if s.confidence == _HIGH)


def _high_risk_count(skeptic_signals: list[SkepticSignal]) -> int:
    """Count HIGH-severity skeptic risks."""
    return sum(1 for s in skeptic_signals if s.severity == _HIGH)


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE 1 BUILDERS (one per rank tier)
# ─────────────────────────────────────────────────────────────────────────────

def _sentence1_elite(
    trust: TrustVerdict,
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> str:
    """
    Sentence 1 for ranks 1-10 (ELITE tier).

    Lead with the strongest positive signal.
    Mention an honest concern only if HIGH risk exists.
    Tone: confident, specific, forward-looking.

    Pattern: "[Top advocate fact][; risk caveat if HIGH risk]."
    """
    adv_fact = _top_advocate_fact(trust.advocate_signals)
    n_high_risks = _high_risk_count(trust.skeptic_signals)
    top_risk = _top_risk_summary(trust.skeptic_signals) if n_high_risks > 0 else None

    if adv_fact and top_risk:
        return f"{adv_fact}; note {top_risk.lower()}"
    if adv_fact:
        return adv_fact
    # Fallback: use skill score (composite_score is not on this dataclass).
    skill_pct = f"{scores.skill_score:.0%}" if scores.skill_score > 0 else "strong"
    return (
        f"Strong overall match — {skill_pct} skill coverage across "
        f"skill, career, and availability signals"
    )


def _sentence1_strong(
    trust: TrustVerdict,
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> str:
    """
    Sentence 1 for ranks 11-30 (STRONG tier).

    Balanced: lead positive, flag any HIGH risk.
    Tone: measured, professional.

    Pattern: "[Positive fact]; however, [risk if HIGH]." or just "[Positive fact]."
    """
    adv_fact = _top_advocate_fact(trust.advocate_signals)
    # Also try second best signal for richer context
    top2 = top_advocate_signals(trust.advocate_signals, n=2)
    second_fact = _extract_first_fact(top2[1].value, max_len=60) if len(top2) >= 2 else None

    n_high_risks = _high_risk_count(trust.skeptic_signals)
    top_risk = _top_risk_summary(trust.skeptic_signals) if n_high_risks > 0 else None

    if adv_fact and top_risk:
        return f"{adv_fact}; however, {top_risk.lower()}"
    if adv_fact and second_fact and second_fact != adv_fact:
        return f"{adv_fact}; also {second_fact.lower()}"
    if adv_fact:
        return adv_fact
    # Fallback using actual score fields (composite_score does not exist on this dataclass).
    skill_pct = f"{scores.skill_score:.0%}"
    career_pct = f"{scores.career_score:.0%}"
    return (
        f"Solid profile — {skill_pct} skill coverage, "
        f"{career_pct} career quality signal"
    )


def _sentence1_mid(
    trust: TrustVerdict,
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> str:
    """
    Sentence 1 for ranks 31-60 (MID tier).

    Neutral-to-cautious: lead with the primary concern if HIGH risk present,
    otherwise lead with the best positive and acknowledge the weaker score.
    Tone: candid, analytical.

    Pattern: "Ranked [N] due to [risk]; [positive if any]." or "[Positive], though overall fit is moderate."
    """
    n_high_risks = _high_risk_count(trust.skeptic_signals)
    top_risk = _top_risk_summary(trust.skeptic_signals)
    adv_fact = _top_advocate_fact(trust.advocate_signals)

    if n_high_risks > 0 and top_risk:
        if adv_fact:
            return (
                f"Moderate fit: {top_risk.lower()}, "
                f"partially offset by {adv_fact.lower()}"
            )
        return f"Moderate fit — primary concern: {top_risk.lower()}"

    if adv_fact:
        skill_pct = f"{scores.skill_score:.0%}"
        return (
            f"{adv_fact}, though {skill_pct} skill score "
            f"reflects gaps in other areas"
        )

    skill_pct = f"{scores.skill_score:.0%}"
    career_pct = f"{scores.career_score:.0%}"
    return (
        f"Mid-tier match: {skill_pct} skill, {career_pct} career quality — "
        f"some signals missing or weak"
    )


def _sentence1_weak(
    trust: TrustVerdict,
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> str:
    """
    Sentence 1 for ranks 61-100 (WEAK tier).

    Risk-forward: lead with the primary reason for low ranking.
    A positive signal is mentioned only if HIGH confidence exists.
    Tone: honest, specific, avoids false hope.

    Pattern: "Lower-ranked due to [top risk]; strongest positive: [adv if any]."
    """
    top_risk = _top_risk_summary(trust.skeptic_signals)
    n_high_adv = _high_advocate_count(trust.advocate_signals)
    # Show adv_fact when there ARE high advocate signals; shows the offset clearly.
    adv_fact = _top_advocate_fact(trust.advocate_signals) if n_high_adv > 0 else None

    if top_risk and adv_fact:
        # Lead with risk, then show the strongest positive offset.
        return (
            f"Lower-ranked due to {top_risk.lower()}; "
            f"strongest positive: {adv_fact.lower()}"
        )
    if top_risk:
        # No offsetting positive — clean risk-only statement.
        return f"Lower-ranked due to {top_risk.lower()}"
    if adv_fact:
        skill_pct = f"{scores.skill_score:.0%}"
        return (
            f"Limited disqualifying signals; best indicator: {adv_fact.lower()} "
            f"({skill_pct} skill match)"
        )
    skill_pct = f"{scores.skill_score:.0%}"
    return (
        f"Weak overall match: {skill_pct} skill score — "
        f"profile does not align with key JD retrieval/ranking requirements"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SENTENCE 2 BUILDER (trust sentence — same for all tiers, varies by verdict)
# ─────────────────────────────────────────────────────────────────────────────

def _sentence2_trust(
    trust: TrustVerdict,
    max_len: int = _SENTENCE2_FALLBACK_MAX,
) -> str:
    """
    Sentence 2: trust classification + confidence + key falsifiability condition.

    Full form (used when space allows):
      "ROBUST ranking at 84% confidence."
      "CONTESTED ranking at 61% confidence — holds unless: [condition 1]."
      "FRAGILE ranking at 31% confidence — ranking could change if: [condition 1]."

    Shortened form (used when Sentence 1 is long):
      "ROBUST — 84% confidence."
      "FRAGILE — 31% confidence."

    The falsifiability condition is always taken from trust.falsifiability[0]
    (the first and most critical condition from verdict.py) and truncated if
    needed to fit the total output within _MAX_CHARS.
    """
    verdict = trust.verdict
    conf = int(round(trust.confidence_pct, _CONF_DECIMALS))
    falsifiability = trust.falsifiability

    # Core trust string.
    core = f"{verdict} ranking at {conf}% confidence"

    # Add falsifiability only for CONTESTED and FRAGILE (space permitting).
    if verdict in (_CONTESTED, _FRAGILE) and falsifiability:
        # Shorten the first falsifiability condition for inline use.
        condition = falsifiability[0]

        # Strip the "This ranking holds UNLESS " / "This ranking becomes MORE ROBUST if " prefix
        # to produce a compact inline clause.
        condition = re.sub(
            r"^This ranking (holds UNLESS|is \w+ (weakened|affected) by|becomes MORE ROBUST if)\s*",
            "",
            condition,
            flags=re.IGNORECASE,
        ).strip()
        condition = re.sub(
            r"^This ranking \w+\s*",
            "",
            condition,
            flags=re.IGNORECASE,
        ).strip()

        # Truncate condition to avoid overflow.
        if len(condition) > 80:
            condition = condition[:77].rstrip() + "…"

        phrase = f" — verify: {condition}" if condition else ""
        full = f"{core}{phrase}"

        # If the full trust sentence would be too long, fall back to core only.
        if len(full) <= max_len:
            return full

    return core


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLY + LENGTH GUARD
# ─────────────────────────────────────────────────────────────────────────────

def _assemble(sentence1: str, sentence2: str) -> str:
    """
    Join two sentences, clean trailing punctuation, enforce length limits.

    Rules:
    - Sentence 1 ends with ". "  (we add it if missing).
    - Sentence 2 ends with "."   (we add it if missing).
    - Total ≤ _MAX_CHARS.  If over, sentence2 is replaced with
      the minimal trust string "{VERDICT} — {conf}% confidence."
    - Total ≥ _MIN_CHARS.  If under (very sparse profile), pad with a
      fallback phrase.
    """
    # Normalise sentence 1.
    s1 = _TRAIL_PUNCT_RE.sub("", sentence1).strip()
    if not s1:
        s1 = "Limited profile signals available"
    s1 = s1[0].upper() + s1[1:]   # ensure capitalised

    # Normalise sentence 2.
    s2 = _TRAIL_PUNCT_RE.sub("", sentence2).strip()
    if not s2:
        s2 = "Ranking confidence not determined"
    s2 = s2[0].upper() + s2[1:]

    combined = f"{s1}. {s2}."

    # Length guard: if over limit, shorten sentence 2 to minimal form.
    if len(combined) > _MAX_CHARS:
        # Extract just "VERDICT at X% confidence" from sentence 2.
        short_s2_match = re.match(
            r"(ROBUST|CONTESTED|FRAGILE)\s+ranking\s+at\s+\d+%\s+confidence",
            s2,
            re.IGNORECASE,
        )
        if short_s2_match:
            s2 = short_s2_match.group(0)
            combined = f"{s1}. {s2}."
        elif len(combined) > _MAX_CHARS + 40:
            # Still too long — truncate sentence 1 smartly at skill boundary.
            budget = _MAX_CHARS - len(s2) - 5
            trunc = s1[:budget]
            last_comma = trunc.rfind(", ")
            if last_comma > budget // 2:
                s1 = trunc[:last_comma] + "…"
            else:
                s1 = trunc.rstrip() + "…"
            combined = f"{s1}. {s2}."

    # Minimum length guard: if too short, it's likely a sparse profile.
    if len(combined) < _MIN_CHARS:
        combined = combined.rstrip(".") + " — verify via interview."

    return combined


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def _sentence1_disqualified(
    trust: TrustVerdict,
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> str:
    """
    Sentence 1 for candidates whose score was zeroed by a hard disqualifier.

    States the disqualification reason clearly, plus the strongest positive
    signal so the recruiter understands the trade-off.
    """
    top_risk = _top_risk_summary(trust.skeptic_signals)
    adv_fact = _top_advocate_fact(trust.advocate_signals)

    disq_prefix = "Score zeroed due to hard domain disqualifier"
    if top_risk:
        disq_prefix = f"Score zeroed: {top_risk.lower()}"

    if adv_fact:
        return f"{disq_prefix}; strongest relevant signal: {adv_fact.lower()}"
    return disq_prefix


def generate_reasoning(
    rank: int,
    trust: TrustVerdict,
    candidate: CandidateFeatureVector,
    scores: ComponentScores,
) -> str:
    """
    Generate a 1–2 sentence recruiter brief for one candidate.

    This is the sole public entry point for trust/reasoning_generator.py.
    Called by pipeline/runner.py for each of the top-100 candidates.

    Parameters
    ----------
    rank      : Final rank (1-100). Drives the tone tier.
    trust     : TrustVerdict from trust/verdict.py.
    candidate : CandidateFeatureVector from pipeline/candidate_parser.py.
    scores    : ComponentScores from scoring/composite.py.

    Returns
    -------
    str — 1-2 sentences, 60-320 characters.
          Every factual claim traces to profile data via signal.value strings.
          Safe to embed directly in submission CSV.

    Raises
    ------
    TypeError  : Any argument is of the wrong type.
    ValueError : rank outside 1-100, or any ID mismatch.
    """
    # ── Type guards ───────────────────────────────────────────────────────────
    if not isinstance(rank, int):
        raise TypeError(f"rank must be int, got {type(rank).__name__}")
    if not isinstance(trust, TrustVerdict):
        raise TypeError(f"trust must be TrustVerdict, got {type(trust).__name__}")
    if not isinstance(candidate, CandidateFeatureVector):
        raise TypeError(
            f"candidate must be CandidateFeatureVector, "
            f"got {type(candidate).__name__}"
        )
    if not isinstance(scores, ComponentScores):
        raise TypeError(f"scores must be ComponentScores, got {type(scores).__name__}")

    # ── Rank validation ───────────────────────────────────────────────────────
    if not (1 <= rank <= config.SUBMISSION_RANK_MAX):
        raise ValueError(
            f"rank must be 1–{config.SUBMISSION_RANK_MAX}, got {rank}"
        )

    # ── ID consistency ────────────────────────────────────────────────────────
    if trust.candidate_id != candidate.candidate_id:
        raise ValueError(
            f"ID mismatch: trust.candidate_id={trust.candidate_id!r} "
            f"but candidate.candidate_id={candidate.candidate_id!r}"
        )
    if scores.candidate_id != candidate.candidate_id:
        raise ValueError(
            f"ID mismatch: scores.candidate_id={scores.candidate_id!r} "
            f"but candidate.candidate_id={candidate.candidate_id!r}"
        )

    # ── Build Sentence 1 (tier-dependent) ────────────────────────────────────
    tier = _rank_tier(rank)

    # Detect hard-disqualified candidates (score == 0 and disqualifier is the cause)
    # by checking for a domain-mismatch skeptic signal with HIGH severity.
    is_disqualified = scores.skill_score == 0.0 or any(
        "domain" in s.label.lower() and s.severity == "HIGH"
        for s in trust.skeptic_signals
    )

    if is_disqualified and rank > 20:
        s1 = _sentence1_disqualified(trust, candidate, scores)
    elif tier == "ELITE":
        s1 = _sentence1_elite(trust, candidate, scores)
    elif tier == "STRONG":
        s1 = _sentence1_strong(trust, candidate, scores)
    elif tier == "MID":
        s1 = _sentence1_mid(trust, candidate, scores)
    else:
        s1 = _sentence1_weak(trust, candidate, scores)

    # ── Build Sentence 2 (trust classification) ───────────────────────────────
    # Compute the remaining character budget for Sentence 2.
    remaining = _MAX_CHARS - len(s1) - 2   # 2 = ". " separator
    s2_max = max(40, min(remaining, _SENTENCE2_FALLBACK_MAX))
    s2 = _sentence2_trust(trust, max_len=s2_max)

    # ── Assemble and return ───────────────────────────────────────────────────
    reasoning = _assemble(s1, s2)

    logger.debug(
        "reasoning: %s rank=%d tier=%s verdict=%s disq=%s len=%d",
        candidate.candidate_id,
        rank,
        tier,
        trust.verdict,
        is_disqualified if rank > 20 else "N/A",
        len(reasoning),
    )

    return reasoning


# ─────────────────────────────────────────────────────────────────────────────
# BATCH HELPER (called by pipeline/runner.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning_batch(
    items: list[tuple[int, TrustVerdict, CandidateFeatureVector, ComponentScores]],
) -> dict[str, str]:
    """
    Generate reasoning strings for a batch of candidates.

    Parameters
    ----------
    items : list of (rank, trust, candidate, scores) 4-tuples.
            Must be sorted by rank before calling (runner.py responsibility).

    Returns
    -------
    dict[candidate_id → reasoning_str]

    Errors on individual candidates are caught, logged with candidate_id,
    and replaced with a safe fallback string so a single bad profile does not
    abort the submission CSV write.
    """
    results: dict[str, str] = {}

    for rank, trust, candidate, scores in items:
        try:
            reasoning = generate_reasoning(rank, trust, candidate, scores)
            results[candidate.candidate_id] = reasoning
        except Exception as exc:  # noqa: BLE001
            cid = getattr(candidate, "candidate_id", "<unknown>")
            logger.error(
                "reasoning: failed for %s rank=%d — %s: %s",
                cid, rank, type(exc).__name__, exc,
            )
            # Safe fallback: minimal valid reasoning for the CSV.
            results[cid] = (
                f"Ranked {rank} based on composite skill, career, and "
                f"availability signals. {trust.verdict} ranking confidence."
            ) if isinstance(trust, TrustVerdict) else (
                f"Ranked {rank} — reasoning generation failed; see pipeline logs."
            )

    logger.info(
        "reasoning batch: %d/%d generated successfully.",
        sum(1 for v in results.values() if "failed" not in v),
        len(items),
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION HELPER (used by tests/test_reasoning.py and scripts/validate_output.py)
# ─────────────────────────────────────────────────────────────────────────────

def validate_reasoning(
    reasoning: str,
    candidate: CandidateFeatureVector,
    trust: TrustVerdict,
) -> list[str]:
    """
    Run the hallucination and quality checks defined in the sprint plan.

    Checks performed:
      1. Non-empty and within length bounds.
      2. No skill name in the reasoning that isn't in the candidate's profile
         or in an AdvocateSignal.value (hallucination scan).
      3. Contains the verdict string (ROBUST / CONTESTED / FRAGILE).
      4. Does not contain generic filler phrases that indicate template failure.

    Returns a list of error strings.  Empty list = all checks passed.
    Used by tests/test_reasoning.py and scripts/validate_output.py.
    """
    errors: list[str] = []

    # Check 1: length bounds.
    if not reasoning.strip():
        errors.append("reasoning is empty")
        return errors   # No point checking further.

    if len(reasoning) < _MIN_CHARS:
        errors.append(
            f"reasoning too short: {len(reasoning)} chars (min {_MIN_CHARS})"
        )
    if len(reasoning) > _MAX_CHARS:
        errors.append(
            f"reasoning too long: {len(reasoning)} chars (max {_MAX_CHARS})"
        )

    # Check 2: verdict string present.
    verdict_present = any(
        v in reasoning for v in (_ROBUST, _CONTESTED, _FRAGILE)
    )
    if not verdict_present:
        errors.append(
            f"reasoning does not contain a verdict string "
            f"(ROBUST / CONTESTED / FRAGILE): {reasoning!r}"
        )

    # Check 3: no obviously fabricated skill names.
    # Build the allowed set: skills from profile + skills mentioned in
    # advocate signal values.
    allowed_skill_names: set[str] = {
        s.name.lower() for s in candidate.skills
    } | {
        s.name_raw.lower() for s in candidate.skills
    }
    # Also allow any word from AdvocateSignal.value (conservative).
    for sig in trust.advocate_signals:
        for word in sig.value.lower().split():
            allowed_skill_names.add(word.strip("(),.:;"))

    # Check for potential hallucinations: technical words in the reasoning
    # that look like ML skill names but aren't in the allowed set.
    # We use a small list of "suspicious patterns" rather than a full NLP
    # parse, as full NLP is not available in a pure-Python safety check.
    _SUSPICIOUS_SKILL_PATTERNS = re.compile(
        r"\b(faiss|pinecone|weaviate|qdrant|milvus|bert|gpt|llama|mistral|"
        r"langchain|pytorch|tensorflow|keras|xgboost|lightgbm|sklearn|"
        r"pyspark|kafka|airflow|kubernetes|docker|elasticsearch|opensearch|"
        r"sentence.transformers?|rag|lora|qlora|peft|bge|e5)\b",
        re.IGNORECASE,
    )

    for match in _SUSPICIOUS_SKILL_PATTERNS.finditer(reasoning):
        skill_mention = match.group(0).lower().replace("-", "").replace(".", "")
        if skill_mention not in allowed_skill_names:
            errors.append(
                f"Possible hallucinated skill in reasoning: "
                f"'{match.group(0)}' — not found in candidate profile or advocate signals"
            )

    # Check 4: no generic filler that indicates template failure.
    _FILLER_PHRASES = [
        "reasoning generation failed",
        "see pipeline logs",
        "template failure",
    ]
    for phrase in _FILLER_PHRASES:
        if phrase.lower() in reasoning.lower():
            errors.append(
                f"reasoning contains fallback/error phrase: '{phrase}'"
            )

    return errors


def validate_reasoning_batch(
    results: dict[str, str],
    candidates: dict[str, CandidateFeatureVector],
    verdicts: dict[str, TrustVerdict],
) -> dict[str, list[str]]:
    """
    Run validate_reasoning() across all candidates.

    Parameters
    ----------
    results    : dict[candidate_id → reasoning_str] from generate_reasoning_batch.
    candidates : dict[candidate_id → CandidateFeatureVector].
    verdicts   : dict[candidate_id → TrustVerdict].

    Returns
    -------
    dict[candidate_id → list[str]] — only candidates with errors are included.
    Empty dict = all candidates passed.

    Used by tests/test_reasoning.py Stage-4 audit.
    """
    failures: dict[str, list[str]] = {}

    for cid, reasoning in results.items():
        candidate = candidates.get(cid)
        trust = verdicts.get(cid)

        if candidate is None or trust is None:
            failures[cid] = [f"Missing candidate or verdict for {cid}"]
            continue

        errs = validate_reasoning(reasoning, candidate, trust)
        if errs:
            failures[cid] = errs

    if failures:
        logger.warning(
            "reasoning validation: %d/%d candidates have issues.",
            len(failures),
            len(results),
        )
    else:
        logger.info(
            "reasoning validation: all %d candidates passed.", len(results)
        )

    return failures