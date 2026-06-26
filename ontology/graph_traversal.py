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
    def __init__(self, skill_map_path: Optional[Path] = None) -> None:
        self._skill_map_path: Path = skill_map_path or config.SKILL_MAP_PATH

        # Forward adjacency  : source_skill → {target_skill, ...}
        self._forward: dict[str, frozenset[str]] = {}
        # Reverse adjacency  : target_skill → {source_skill, ...}
        self._reverse: dict[str, frozenset[str]] = {}
        # Synonym lookup     : skill → {synonym, ...}
        self._synonyms: dict[str, frozenset[str]] = {}

        self._loaded: bool = False
        self._load()

    def _load(self) -> None:
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

    # Internal BFS helper 
    def _bfs_rescue_sources(
        self,
        target_skill: str,
        depth: int = 1,
    ) -> frozenset[str]:
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

    # Public API — JD rescue map
    def build_jd_rescue_map(
        self,
        jd_required_skills: list[str],
        bfs_depth: int = 1,
    ) -> dict[str, frozenset[str]]:
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

    # Public API — single candidate scoring
    def score_candidate_skills(
        self,
        candidate_skill_names: frozenset[str],
        jd_rescue_map: dict[str, frozenset[str]],
    ) -> float:
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

    # Public API — rank a pool of candidates (Path 3 entry point)
    def rank_by_domain_transfer(
        self,
        candidate_skills_map: dict[str, frozenset[str]],
        jd_required_skills: list[str],
        top_k: int = 25,
        exclude_ids: Optional[set[str]] = None,
        bfs_depth: int = 1,
    ) -> list[tuple[str, float]]:
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

    # Introspection helpers 
    def transfers_to(self, source_skill: str) -> list[str]:
        return sorted(self._forward.get(source_skill.lower().strip(), frozenset()))

    def transfer_sources_for(self, target_skill: str) -> list[str]:
        return sorted(self._reverse.get(target_skill.lower().strip(), frozenset()))

    def has_transfer_path(self, source_skill: str, target_skill: str) -> bool:
        src = source_skill.lower().strip()
        tgt = target_skill.lower().strip()
        return tgt in self._forward.get(src, frozenset())

    @property
    def loaded(self) -> bool:
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


# Module-level convenience 
def build_rescue_map(
    jd_required_skills: list[str],
    skill_map_path: Optional[Path] = None,
    bfs_depth: int = 1,
) -> dict[str, frozenset[str]]:
    graph = SkillGraph(skill_map_path=skill_map_path)
    return graph.build_jd_rescue_map(jd_required_skills, bfs_depth=bfs_depth)
