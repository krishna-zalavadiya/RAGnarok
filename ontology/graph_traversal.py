"""
ontology/graph_traversal.py
---------------------------
BFS-based domain-transfer graph for Tier-5 candidate rescue (Retrieval Path 3).

Problem it solves:
  FAISS (Path 1) and BM25 (Path 2) both rely on skill-text overlap with the JD.
  A candidate whose profile says "Recommendation Systems Engineer @ Swiggy" with
  skills [FAISS, Pinecone, Embeddings] would be caught. But a candidate who says
  "Search Engineer — built product ranking systems" with skills
  ["recommendation systems", "feature engineering", "ranking"] and NO explicit
  "information retrieval" keyword gets missed by both dense and sparse paths.

  This module rescues them. It walks the domain_transfers graph in skill_map.json:
    "recommendation systems" → "information retrieval"  (1-hop)
    "nlp"                    → "information retrieval"  (1-hop)
    "search engineering"     → "information retrieval"  (1-hop)

  A candidate with source-domain skills earns partial credit toward JD target skills.

Graph structure:
  Nodes : skill/domain names (lowercase strings)
  Edges : REVERSE of domain_transfers — we walk FROM jd_required_skills BACKWARD
          to find which candidate skills transfer in.
  Forward : "recommendation systems" → "information retrieval"
  Reverse : "information retrieval"  ← "recommendation systems"

Algorithm (per JD):
  1. build_jd_rescue_map(jd_required_skills):
       For each required skill R, BFS backward through reverse edges to collect
       all source-domain skills S such that S --transfer--> R (within bfs_depth hops).
       Also expand each source via synonyms (e.g. "recsys" ≡ "recommendation systems").
       Result: {R: frozenset(all_source_skills_and_synonyms)}

  2. score_candidate_skills(candidate_skills, jd_rescue_map):
       For each JD required skill R:
         - Direct match    (R in candidate_skills) → full credit  1.0
         - Transfer match  (candidate_skills ∩ rescue_sources[R] ≠ ∅) → 0.7
         - No match        → 0.0
       Final score = coverage_sum / len(jd_required_skills)   ∈ [0, 1]

  3. rank_by_domain_transfer(candidate_skills_map, jd_required_skills, top_k):
       Score all candidates, return top_k sorted by score.

Consumed by:
  - retrieval/ontology_path.py   (Path 3 retrieval — wraps results as RetrievalResult)
  - pipeline/jd_parser.py        (optional: pre-warm rescue_map once per JD)

Does NOT import from pipeline/schemas.py — intentionally decoupled.
Accepts frozenset[str] so it is pure graph logic, testable without candidate objects.

Dependencies:
  config.py, ontology/skill_map.json, stdlib only.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

# Scoring constants — kept here, not in config, as they are internal to this module.
_DIRECT_MATCH_CREDIT: float = 1.0   # candidate skill == JD required skill
_TRANSFER_MATCH_BASE: float = 0.5   # candidate skill 1-hop from JD required skill
_TRANSFER_MATCH_MAX: float = 0.7    # scales up with each additional transfer hit
_TRANSFER_HIT_STEP: float = 0.1     # per-additional-hit bonus (caps at MAX)


class SkillGraph:
    """
    Directed domain-transfer graph for retrieving Tier-5 candidates.

    Usage:
        graph = SkillGraph()

        # Build once per JD (fast — just dict lookups)
        rescue_map = graph.build_jd_rescue_map(jd_intent.required_skills)

        # Score a single candidate
        score = graph.score_candidate_skills(
            candidate.skill_names_lower, rescue_map
        )

        # Or rank a whole pool (Path 3 primary entry point)
        results = graph.rank_by_domain_transfer(
            candidate_skills_map={cid: fvec.skill_names_lower for cid, fvec in pool},
            jd_required_skills=jd_intent.required_skills,
            top_k=config.ONTOLOGY_PATH_TOP_K,
        )
    """

    def __init__(self, skill_map_path: Optional[Path] = None) -> None:
        """
        Load skill_map.json and build internal adjacency structures.

        Args:
            skill_map_path: Override path (for testing). Defaults to
                            config.SKILL_MAP_PATH.

        Raises:
            FileNotFoundError: skill_map.json not found.
            ValueError:        JSON missing required sections.
        """
        self._skill_map_path: Path = skill_map_path or config.SKILL_MAP_PATH

        # Forward adjacency  : source_skill → {target_skill, ...}
        self._forward: dict[str, frozenset[str]] = {}
        # Reverse adjacency  : target_skill → {source_skill, ...}
        self._reverse: dict[str, frozenset[str]] = {}
        # Synonym lookup     : skill → {synonym, ...}
        self._synonyms: dict[str, frozenset[str]] = {}

        self._loaded: bool = False
        self._load()

    # ------------------------------------------------------------------ #
    # Internal loading                                                     #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """
        Parse skill_map.json.
        Build forward adjacency, reverse adjacency, and synonym lookup.
        """
        path = self._skill_map_path

        if not path.exists():
            raise FileNotFoundError(
                f"skill_map.json not found at '{path}'. "
                "Verify ONTOLOGY_DIR in config.py."
            )

        logger.debug("SkillGraph: loading skill_map from '%s'", path)

        with open(path, encoding="utf-8") as fh:
            raw: dict = json.load(fh)

        required_sections = {"synonyms", "domain_transfers"}
        missing = required_sections - set(raw.keys())
        if missing:
            raise ValueError(
                f"skill_map.json missing required sections: {missing}. "
                f"Found: {list(raw.keys())}"
            )

        # ── Build synonym lookup ──────────────────────────────────────────
        raw_synonyms: dict[str, list[str]] = raw.get("synonyms", {})
        self._synonyms = {
            k.lower().strip(): frozenset(v.lower().strip() for v in vals)
            for k, vals in raw_synonyms.items()
        }

        # ── Build forward + reverse adjacency from domain_transfers ───────
        raw_transfers: dict[str, list[str]] = raw.get("domain_transfers", {})

        forward_build: dict[str, set[str]] = {}
        reverse_build: dict[str, set[str]] = {}

        for source, targets in raw_transfers.items():
            src = source.lower().strip()
            forward_build.setdefault(src, set())
            for target in targets:
                tgt = target.lower().strip()
                forward_build[src].add(tgt)
                reverse_build.setdefault(tgt, set()).add(src)

        self._forward = {k: frozenset(v) for k, v in forward_build.items()}
        self._reverse = {k: frozenset(v) for k, v in reverse_build.items()}

        self._loaded = True

        logger.info(
            "SkillGraph loaded: %d forward edges, %d reverse edges, "
            "%d synonym entries",
            len(self._forward),
            len(self._reverse),
            len(self._synonyms),
        )

    # ------------------------------------------------------------------ #
    # Internal BFS helper                                                  #
    # ------------------------------------------------------------------ #

    def _bfs_rescue_sources(
        self,
        target_skill: str,
        depth: int = 1,
    ) -> frozenset[str]:
        """
        BFS backward from target_skill, collecting all source-domain skills
        that can transfer INTO it within `depth` hops.

        For each discovered source, also includes all synonyms of that source
        so that e.g. "recsys" is collected when "recommendation systems" is
        found in the graph.

        The target_skill itself is excluded from the returned set.

        Args:
            target_skill: A JD required skill (lowercase).
            depth:        BFS hop limit. Default 1 (direct transfers only).
                          Depth 2 adds indirect transfers (broader but noisier).

        Returns:
            frozenset of source-skill strings (lowercase) that transfer into
            target_skill within depth hops. Empty frozenset if none found.
        """
        target = target_skill.lower().strip()
        visited: set[str] = {target}
        collected: set[str] = set()  # sources found (excludes target)

        frontier: deque[str] = deque([target])

        for _hop in range(depth):
            next_frontier: deque[str] = deque()

            while frontier:
                node = frontier.popleft()
                # Walk reverse edges from this node
                for source in self._reverse.get(node, frozenset()):
                    if source not in visited:
                        visited.add(source)
                        collected.add(source)
                        next_frontier.append(source)

                        # Include synonyms of this source node
                        for syn in self._synonyms.get(source, frozenset()):
                            if syn not in visited:
                                visited.add(syn)
                                collected.add(syn)

            frontier = next_frontier
            if not frontier:
                # No new nodes found — stop early
                break

        logger.debug(
            "_bfs_rescue_sources('%s', depth=%d) → %d source skills",
            target_skill,
            depth,
            len(collected),
        )
        return frozenset(collected)

    # ------------------------------------------------------------------ #
    # Public API — JD rescue map                                           #
    # ------------------------------------------------------------------ #

    def build_jd_rescue_map(
        self,
        jd_required_skills: list[str],
        bfs_depth: int = 1,
    ) -> dict[str, frozenset[str]]:
        """
        Build the JD-specific rescue map: for each required skill, the full
        set of candidate skills that "transfer in" to it via the domain graph.

        This is computed once per JD and reused for scoring all candidates.

        Args:
            jd_required_skills: Lowercase required skill names from JDIntent.
            bfs_depth:          BFS depth for source discovery. 1 = direct
                                transfers only (recommended for production).

        Returns:
            dict mapping each JD required skill → frozenset of source domain
            skills (and their synonyms) that would satisfy it via transfer.

            Example (depth=1):
              {
                "information retrieval": frozenset({
                    "recommendation systems", "recsys", "recommender systems",
                    "nlp", "natural language processing", "search engineering",
                    "learning to rank", "question answering", ...
                }),
                "ranking": frozenset({
                    "recommendation systems", "recsys",
                    "learning to rank", "ltr", ...
                }),
                ...
              }
        """
        if not jd_required_skills:
            logger.warning("build_jd_rescue_map: empty jd_required_skills")
            return {}

        rescue_map: dict[str, frozenset[str]] = {}

        for raw_skill in jd_required_skills:
            skill = raw_skill.lower().strip()
            sources = self._bfs_rescue_sources(skill, depth=bfs_depth)
            if sources:
                rescue_map[skill] = sources
                logger.debug(
                    "rescue_map['%s'] → %d source skills", skill, len(sources)
                )
            else:
                # No domain transfers found — direct match only
                rescue_map[skill] = frozenset()

        covered = sum(1 for v in rescue_map.values() if v)
        logger.info(
            "build_jd_rescue_map: %d JD skills, %d have transfer sources",
            len(rescue_map),
            covered,
        )
        return rescue_map

    # ------------------------------------------------------------------ #
    # Public API — single candidate scoring                                #
    # ------------------------------------------------------------------ #

    def score_candidate_skills(
        self,
        candidate_skill_names: frozenset[str],
        jd_rescue_map: dict[str, frozenset[str]],
    ) -> float:
        """
        Score a single candidate's skill set against the JD rescue map.

        Scoring logic per JD required skill:
          - Direct match  (skill ∈ candidate_skill_names) → 1.0
          - Transfer match (candidate_skill_names ∩ rescue_sources ≠ ∅) →
              base 0.5, +0.1 per additional hit, capped at 0.7
          - No match                                       → 0.0

        Final score = sum_of_coverages / total_jd_required_skills
        This keeps scores in [0, 1] regardless of JD size.

        Args:
            candidate_skill_names: frozenset of lowercase skill names from
                                   CandidateFeatureVector.skill_names_lower.
            jd_rescue_map:         Output of build_jd_rescue_map().

        Returns:
            Float in [0.0, 1.0]. Higher = stronger domain-transfer alignment.
            Returns 0.0 for empty inputs.
        """
        if not jd_rescue_map or not candidate_skill_names:
            return 0.0

        coverage_sum: float = 0.0
        total: int = len(jd_rescue_map)

        for jd_skill, rescue_sources in jd_rescue_map.items():
            if jd_skill in candidate_skill_names:
                # Direct match — full credit
                coverage_sum += _DIRECT_MATCH_CREDIT
                continue

            if rescue_sources:
                transfer_hits: frozenset[str] = (
                    candidate_skill_names & rescue_sources
                )
                if transfer_hits:
                    # Scale credit by number of matching source domains:
                    # 1 hit → 0.5,  2 hits → 0.6,  3+ hits → 0.7
                    n_hits = len(transfer_hits)
                    credit = min(
                        _TRANSFER_MATCH_MAX,
                        _TRANSFER_MATCH_BASE + _TRANSFER_HIT_STEP * (n_hits - 1),
                    )
                    coverage_sum += credit

        return min(1.0, coverage_sum / total)

    # ------------------------------------------------------------------ #
    # Public API — rank a pool of candidates (Path 3 entry point)         #
    # ------------------------------------------------------------------ #

    def rank_by_domain_transfer(
        self,
        candidate_skills_map: dict[str, frozenset[str]],
        jd_required_skills: list[str],
        top_k: int = 25,
        exclude_ids: Optional[set[str]] = None,
        bfs_depth: int = 1,
    ) -> list[tuple[str, float]]:
        """
        Score and rank all candidates by domain-transfer relevance to the JD.

        This is the primary entry point called by retrieval/ontology_path.py.

        Args:
            candidate_skills_map: {candidate_id: frozenset[skill_names_lower]}
                                  Typically built from the pre-loaded feature store
                                  or all CandidateFeatureVector objects.
            jd_required_skills:   Lowercase required skills from JDIntent.
                                  Pass required_skills (not expanded_required) —
                                  this function performs its own graph expansion.
            top_k:                Number of candidates to return.
                                  Defaults to config.ONTOLOGY_PATH_TOP_K (20).
            exclude_ids:          Optional set of candidate_ids to skip.
                                  Useful when testing for cross-path deduplication,
                                  but Path 3 results are NOT pre-filtered in
                                  practice — deduplication happens in rrf_fusion.py.
            bfs_depth:            Passed to build_jd_rescue_map(). Default 1.

        Returns:
            List of (candidate_id, score) tuples, sorted by score descending,
            length ≤ top_k.

            Only candidates with score > 0.0 are included.
            If fewer than top_k candidates have nonzero scores, returns all
            nonzero-score candidates.

        Raises:
            TypeError:  If candidate_skills_map is not a dict.
            ValueError: If top_k < 1.
        """
        if not isinstance(candidate_skills_map, dict):
            raise TypeError(
                f"candidate_skills_map must be dict, got {type(candidate_skills_map)}"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        if not candidate_skills_map:
            logger.warning("rank_by_domain_transfer: empty candidate_skills_map")
            return []

        if not jd_required_skills:
            logger.warning("rank_by_domain_transfer: empty jd_required_skills")
            return []

        exclude: set[str] = exclude_ids or set()

        # Build rescue map once for this JD
        rescue_map = self.build_jd_rescue_map(
            jd_required_skills, bfs_depth=bfs_depth
        )

        if not any(rescue_map.values()):
            logger.warning(
                "rank_by_domain_transfer: no domain-transfer sources found "
                "for any of %d JD required skills. "
                "Check domain_transfers section in skill_map.json.",
                len(jd_required_skills),
            )

        # Score all candidates
        scored: list[tuple[str, float]] = []

        for candidate_id, skill_names in candidate_skills_map.items():
            if candidate_id in exclude:
                continue

            score = self.score_candidate_skills(skill_names, rescue_map)
            if score > 0.0:
                scored.append((candidate_id, score))

        # Sort by score descending; tie-break by candidate_id ascending
        # (spec-compliant tie-break consistent with composite.py)
        scored.sort(key=lambda x: (-x[1], x[0]))

        result = scored[:top_k]

        logger.info(
            "rank_by_domain_transfer: %d candidates scored, "
            "%d nonzero, returning top %d",
            len(candidate_skills_map) - len(exclude),
            len(scored),
            len(result),
        )

        return result

    # ------------------------------------------------------------------ #
    # Introspection helpers                                                #
    # ------------------------------------------------------------------ #

    def transfers_to(self, source_skill: str) -> list[str]:
        """
        Return the list of target skills that source_skill transfers into.

        Example:
            graph.transfers_to("recommendation systems")
            → ["information retrieval", "search engineering", "ranking", ...]
        """
        return sorted(self._forward.get(source_skill.lower().strip(), frozenset()))

    def transfer_sources_for(self, target_skill: str) -> list[str]:
        """
        Return the list of source skills that transfer into target_skill.

        Example:
            graph.transfer_sources_for("information retrieval")
            → ["recommendation systems", "recsys", "nlp", ...]
        """
        return sorted(self._reverse.get(target_skill.lower().strip(), frozenset()))

    def has_transfer_path(self, source_skill: str, target_skill: str) -> bool:
        """
        Return True if source_skill has a 1-hop transfer path to target_skill.

        Example:
            graph.has_transfer_path("recommendation systems", "information retrieval")
            → True
        """
        src = source_skill.lower().strip()
        tgt = target_skill.lower().strip()
        return tgt in self._forward.get(src, frozenset())

    @property
    def loaded(self) -> bool:
        """True if skill_map.json was loaded successfully."""
        return self._loaded

    def __repr__(self) -> str:
        return (
            f"SkillGraph("
            f"forward_edges={len(self._forward)}, "
            f"reverse_edges={len(self._reverse)}, "
            f"synonyms={len(self._synonyms)}, "
            f"loaded={self._loaded}"
            f")"
        )


# --------------------------------------------------------------------------- #
# Module-level convenience                                                     #
# --------------------------------------------------------------------------- #

def build_rescue_map(
    jd_required_skills: list[str],
    skill_map_path: Optional[Path] = None,
    bfs_depth: int = 1,
) -> dict[str, frozenset[str]]:
    """
    One-shot helper: build a JD rescue map from required skills.

    Creates a SkillGraph on each call. For repeated use (e.g. ranking loop),
    instantiate SkillGraph once and call build_jd_rescue_map() directly.
    """
    graph = SkillGraph(skill_map_path=skill_map_path)
    return graph.build_jd_rescue_map(jd_required_skills, bfs_depth=bfs_depth)


# --------------------------------------------------------------------------- #
# Smoke test — python -m ontology.graph_traversal                             #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)

    print("=" * 60)
    print("SkillGraph — smoke test")
    print("=" * 60)

    try:
        graph = SkillGraph()
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"\nLoaded: {graph}\n")

    # ── Test 1: direct transfer path ──────────────────────────────────────
    assert graph.has_transfer_path(
        "recommendation systems", "information retrieval"
    ), "FAIL: no transfer path recommendation systems → information retrieval"
    print("[PASS] recommendation systems → information retrieval  ✓")

    assert graph.has_transfer_path("nlp", "information retrieval"), (
        "FAIL: no transfer path nlp → information retrieval"
    )
    print("[PASS] nlp → information retrieval  ✓")

    # ── Test 2: rescue map coverage ───────────────────────────────────────
    jd_skills = [
        "information retrieval", "ranking", "embeddings",
        "evaluation framework", "python",
    ]
    rescue_map = graph.build_jd_rescue_map(jd_skills)

    assert "recommendation systems" in rescue_map["information retrieval"], (
        "FAIL: 'recommendation systems' not in rescue sources for 'information retrieval'"
    )
    print("[PASS] rescue_map['information retrieval'] contains "
          "'recommendation systems'  ✓")

    # ── Test 3: candidate scoring — RecSys background ────────────────────
    recsys_candidate_skills: frozenset[str] = frozenset({
        "recommendation systems", "feature engineering",
        "python", "xgboost", "a/b testing",
    })
    recsys_score = graph.score_candidate_skills(recsys_candidate_skills, rescue_map)
    assert recsys_score > 0.0, (
        f"FAIL: RecSys candidate scored 0.0, expected > 0. "
        f"Skills: {recsys_candidate_skills}"
    )
    print(f"[PASS] RecSys candidate score = {recsys_score:.3f} > 0.0  ✓")

    # ── Test 4: irrelevant candidate scores low ───────────────────────────
    irrelevant_skills: frozenset[str] = frozenset({
        "marketing", "accounting", "project management", "tally",
    })
    irrelevant_score = graph.score_candidate_skills(irrelevant_skills, rescue_map)
    assert irrelevant_score < recsys_score, (
        f"FAIL: irrelevant candidate ({irrelevant_score:.3f}) scored "
        f">= RecSys candidate ({recsys_score:.3f})"
    )
    print(f"[PASS] irrelevant candidate score = {irrelevant_score:.3f} "
          f"< RecSys score ({recsys_score:.3f})  ✓")

    # ── Test 5: rank_by_domain_transfer — RecSys rescued in top results ───
    pool: dict[str, frozenset[str]] = {
        "CAND_RECSYS": frozenset({
            "recommendation systems", "feature engineering", "python",
            "xgboost", "a/b testing",
        }),
        "CAND_MARKETING": frozenset({
            "marketing", "project management", "excel",
        }),
        "CAND_NLP": frozenset({
            "nlp", "hugging face transformers", "python", "pytorch",
        }),
        "CAND_CIVIL": frozenset({
            "autocad", "solidworks", "civil engineering",
        }),
    }

    results = graph.rank_by_domain_transfer(
        candidate_skills_map=pool,
        jd_required_skills=jd_skills,
        top_k=4,
    )

    result_ids = [cid for cid, _ in results]
    assert "CAND_RECSYS" in result_ids, (
        f"FAIL: CAND_RECSYS not rescued. Results: {results}"
    )
    assert "CAND_NLP" in result_ids, (
        f"FAIL: CAND_NLP not rescued. Results: {results}"
    )
    assert result_ids.index("CAND_RECSYS") < result_ids.index("CAND_MARKETING") \
        if "CAND_MARKETING" in result_ids else True, (
        "FAIL: CAND_RECSYS ranked below CAND_MARKETING"
    )
    print(f"[PASS] rank_by_domain_transfer results: {results[:3]}  ✓")
    print(f"[PASS] RecSys candidate rescued at rank "
          f"{result_ids.index('CAND_RECSYS') + 1}  ✓")

    # ── Test 6: exclude_ids works ─────────────────────────────────────────
    results_excluded = graph.rank_by_domain_transfer(
        candidate_skills_map=pool,
        jd_required_skills=jd_skills,
        top_k=4,
        exclude_ids={"CAND_RECSYS"},
    )
    excluded_ids = [cid for cid, _ in results_excluded]
    assert "CAND_RECSYS" not in excluded_ids, (
        "FAIL: CAND_RECSYS appeared despite being in exclude_ids"
    )
    print(f"[PASS] exclude_ids correctly filters CAND_RECSYS  ✓")

    print("\nAll smoke-test assertions passed.")
#---Test End--->