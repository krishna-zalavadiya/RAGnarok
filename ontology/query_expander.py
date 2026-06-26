from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


class QueryExpander:

    def __init__(self, skill_map_path: Optional[Path] = None) -> None:
        self._skill_map_path: Path = skill_map_path or config.SKILL_MAP_PATH
        self._synonyms: dict[str, list[str]] = {}
        self._co_skills: dict[str, list[str]] = {}
        self._domain_transfers: dict[str, list[str]] = {}
        # Reverse index: target_skill → [source_domains that transfer in]
        self._reverse_domain_transfers: dict[str, list[str]] = {}
        self._loaded: bool = False
        self._load()

    # Internal loading 
    def _load(self) -> None:
        path = self._skill_map_path

        if not path.exists():
            raise FileNotFoundError(
                f"skill_map.json not found at '{path}'. "
                "Verify ONTOLOGY_DIR in config.py and that the file exists."
            )

        logger.debug("Loading skill_map from '%s'", path)

        with open(path, encoding="utf-8") as fh:
            raw: dict = json.load(fh)

        # Validate required top-level keys exist.
        required_keys = {"synonyms", "co_skills", "domain_transfers"}
        missing = required_keys - set(raw.keys())
        if missing:
            raise ValueError(
                f"skill_map.json is missing required sections: {missing}. "
                f"Present sections: {list(raw.keys())}"
            )

        self._synonyms = raw.get("synonyms", {})
        self._co_skills = raw.get("co_skills", {})
        self._domain_transfers = raw.get("domain_transfers", {})

        # Build reverse domain-transfer index.
        # Forward  : "recommendation systems" → ["information retrieval", ...]
        # Reverse  : "information retrieval"  → ["recommendation systems", ...]
        for source, targets in self._domain_transfers.items():
            for target in targets:
                key = target.lower().strip()
                self._reverse_domain_transfers.setdefault(key, []).append(
                    source.lower().strip()
                )

        self._loaded = True

        logger.info(
            "QueryExpander loaded: %d synonym entries, %d co-skill entries, "
            "%d domain-transfer entries, %d reverse-transfer entries",
            len(self._synonyms),
            len(self._co_skills),
            len(self._domain_transfers),
            len(self._reverse_domain_transfers),
        )

    # Single-skill lookup helpers 
    def get_synonyms(self, skill: str) -> list[str]:
        return list(self._synonyms.get(skill.lower().strip(), []))

    def get_co_skills(self, skill: str) -> list[str]:
        return list(self._co_skills.get(skill.lower().strip(), []))

    def get_domain_transfer_sources(self, target_skill: str) -> list[str]:
        return list(
            self._reverse_domain_transfers.get(target_skill.lower().strip(), [])
        )

    # Multi-skill expansion 
    def expand_skills(
        self,
        skills: list[str],
        include_co_skills: bool = True,
        include_domain_transfer_sources: bool = False,
        depth: int = 1,
    ) -> list[str]:
        if not isinstance(skills, list):
            raise TypeError(
                f"expand_skills expects a list[str], got {type(skills).__name__}"
            )

        if not skills:
            return []

        # Normalise all inputs to lowercase.
        normalised: list[str] = [
            s.lower().strip() for s in skills if isinstance(s, str) and s.strip()
        ]

        expanded: set[str] = set(normalised)

        # ── BFS synonym expansion ──────────────────────────────────────────
        current_level: list[str] = list(normalised)
        visited_synonyms: set[str] = set()

        for _hop in range(depth):
            next_level: list[str] = []
            for skill in current_level:
                if skill in visited_synonyms:
                    continue
                visited_synonyms.add(skill)

                for synonym in self._synonyms.get(skill, []):
                    syn_norm = synonym.lower().strip()
                    if syn_norm not in expanded:
                        expanded.add(syn_norm)
                        next_level.append(syn_norm)

            if not next_level:
                # No new synonyms found — terminate early.
                break
            current_level = next_level

        # ── Co-skill expansion (1-hop from original inputs only) ──────────
        if include_co_skills:
            for skill in normalised:
                for co in self._co_skills.get(skill, []):
                    expanded.add(co.lower().strip())

        # ── Reverse domain-transfer sources ───────────────────────────────
        if include_domain_transfer_sources:
            for skill in normalised:
                for source in self._reverse_domain_transfers.get(skill, []):
                    expanded.add(source.lower().strip())

        result = sorted(expanded)

        logger.debug(
            "expand_skills: %d input → %d expanded (co_skills=%s, dt_sources=%s)",
            len(normalised),
            len(result),
            include_co_skills,
            include_domain_transfer_sources,
        )

        return result

    # BM25-ready token list 
    def build_query_tokens(
        self,
        skills: list[str],
        include_co_skills: bool = True,
        include_domain_transfer_sources: bool = True,
    ) -> list[str]:
        expanded: list[str] = self.expand_skills(
            skills,
            include_co_skills=include_co_skills,
            include_domain_transfer_sources=include_domain_transfer_sources,
        )

        # Tokenise by whitespace. Each multi-word phrase becomes N tokens.
        raw_tokens: list[str] = []
        for term in expanded:
            raw_tokens.extend(term.split())

        # Deduplicate while preserving first-occurrence order.
        seen: set[str] = set()
        unique_tokens: list[str] = []
        for tok in raw_tokens:
            tok_lower = tok.lower()
            if tok_lower not in seen:
                seen.add(tok_lower)
                unique_tokens.append(tok_lower)

        logger.debug(
            "build_query_tokens: %d skills → %d expanded terms → %d tokens",
            len(skills),
            len(expanded),
            len(unique_tokens),
        )

        return unique_tokens

    # Dunder helpers 
    @property
    def loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:
        return (
            f"QueryExpander("
            f"synonyms={len(self._synonyms)}, "
            f"co_skills={len(self._co_skills)}, "
            f"domain_transfers={len(self._domain_transfers)}, "
            f"loaded={self._loaded}"
            f")"
        )


# Module-level convenience function 
def expand_jd_query(
    skills: list[str],
    skill_map_path: Optional[Path] = None,
    include_co_skills: bool = True,
    include_domain_transfer_sources: bool = True,
) -> list[str]:
    expander = QueryExpander(skill_map_path=skill_map_path)
    return expander.build_query_tokens(
        skills,
        include_co_skills=include_co_skills,
        include_domain_transfer_sources=include_domain_transfer_sources,
    )

