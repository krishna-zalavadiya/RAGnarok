"""
ontology/query_expander.py
--------------------------
Expands JD skill terms into richer BM25 query tokens using skill_map.json.

Responsibilities:
  1. Load synonyms, co-skills, and domain-transfer sections from skill_map.json.
  2. Given a list of required skills (from JDIntent), return an expanded set
     that includes synonyms and co-occurring skills.
  3. Tokenise the expanded set for BM25Okapi.get_scores() consumption.
  4. Build a reverse domain-transfer index so that JD target skills can find
     source domains (e.g. "information retrieval" ← "recommendation systems").

Consumed by:
  - pipeline/jd_parser.py       → populates JDIntent.expanded_required
  - retrieval/keyword_path.py   → builds BM25 query via build_query_tokens()

Does NOT handle domain-transfer BFS graph walking — that is graph_traversal.py.

Dependencies:
  - config.py          (SKILL_MAP_PATH)
  - ontology/skill_map.json  (data file)
  - stdlib: json, logging, pathlib, typing

No ML imports. No network calls. Safe to load during the 5-min ranking window.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)


class QueryExpander:
    """
    Expands a set of skill terms using the ontology skill map.

    Expansion modes:
      - Synonyms   : near-equivalent names for the same skill (bidirectional).
      - Co-skills  : skills that commonly co-appear with the input skill.
      - Reverse DT : source domains that transfer *into* the input skill
                     (e.g. "recommendation systems" → "information retrieval",
                      so for target "information retrieval" we surface source
                      "recommendation systems" in the BM25 query).

    All lookups are O(1) dict access after an O(n) JSON load at init time.

    Usage:
        expander = QueryExpander()                    # loads from config.SKILL_MAP_PATH
        tokens = expander.build_query_tokens(jd_intent.required_skills)
        # pass tokens to BM25Okapi.get_scores(tokens)
    """

    def __init__(self, skill_map_path: Optional[Path] = None) -> None:
        """
        Initialise and load skill_map.json.

        Args:
            skill_map_path: Override path for testing. Defaults to
                            config.SKILL_MAP_PATH.

        Raises:
            FileNotFoundError: If skill_map.json does not exist at the
                               resolved path.
            ValueError: If the loaded JSON is missing required sections.
        """
        self._skill_map_path: Path = skill_map_path or config.SKILL_MAP_PATH
        self._synonyms: dict[str, list[str]] = {}
        self._co_skills: dict[str, list[str]] = {}
        self._domain_transfers: dict[str, list[str]] = {}
        # Reverse index: target_skill → [source_domains that transfer in]
        self._reverse_domain_transfers: dict[str, list[str]] = {}
        self._loaded: bool = False
        self._load()

    # ------------------------------------------------------------------ #
    # Internal loading                                                     #
    # ------------------------------------------------------------------ #

    def _load(self) -> None:
        """
        Load skill_map.json and build the reverse domain-transfer index.

        Called once at construction. Thread-safe for read-only usage.
        """
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

    # ------------------------------------------------------------------ #
    # Single-skill lookup helpers                                          #
    # ------------------------------------------------------------------ #

    def get_synonyms(self, skill: str) -> list[str]:
        """
        Return direct synonyms for a skill name.

        Args:
            skill: Skill name (any casing; normalised internally).

        Returns:
            List of synonym strings (lowercase). Empty list if not in map.
        """
        return list(self._synonyms.get(skill.lower().strip(), []))

    def get_co_skills(self, skill: str) -> list[str]:
        """
        Return co-occurring skills for a skill name.

        Args:
            skill: Skill name (any casing; normalised internally).

        Returns:
            List of co-skill strings (lowercase). Empty list if not in map.
        """
        return list(self._co_skills.get(skill.lower().strip(), []))

    def get_domain_transfer_sources(self, target_skill: str) -> list[str]:
        """
        Return source domains that transfer *into* target_skill.

        This is the reverse of the domain_transfers map. Useful for BM25
        recall: if the JD requires "information retrieval", sourcing candidates
        whose profiles mention "recommendation systems" is correct.

        Args:
            target_skill: JD required skill (e.g. "information retrieval").

        Returns:
            List of source domain strings (e.g. ["recommendation systems",
            "recsys", "nlp", ...]). Empty list if none found.
        """
        return list(
            self._reverse_domain_transfers.get(target_skill.lower().strip(), [])
        )

    # ------------------------------------------------------------------ #
    # Multi-skill expansion                                                #
    # ------------------------------------------------------------------ #

    def expand_skills(
        self,
        skills: list[str],
        include_co_skills: bool = True,
        include_domain_transfer_sources: bool = False,
        depth: int = 1,
    ) -> list[str]:
        """
        Expand a list of skill terms using synonyms, co-skills, and optionally
        reverse domain-transfer sources.

        BFS is used for synonym expansion so `depth` controls how many
        synonym hops are traversed. Depth > 1 can cause concept drift and
        is not recommended for production use.

        Co-skills are always added only 1-hop from the original input skills
        (not recursed), to prevent unbounded expansion.

        Args:
            skills: Lowercase skill names (e.g. from JDIntent.required_skills).
            include_co_skills: If True, add co-occurring skills of input terms.
            include_domain_transfer_sources: If True, add reverse-domain-transfer
                source skills. Increases BM25 recall for Tier-5 candidates
                (e.g. adds "recommendation systems" when JD needs "information
                retrieval"). Default False — graph_traversal.py is the primary
                vehicle for domain-transfer rescue; this is supplementary.
            depth: Number of synonym hops. 1 = direct synonyms only (default).

        Returns:
            Sorted, deduplicated list of expanded skill terms (all lowercase).

        Raises:
            TypeError: If `skills` is not a list.

        Example:
            >>> expander.expand_skills(["dense retrieval"])
            ['ann', 'bi-encoder retrieval', 'dense retrieval', 'dpr',
             'embedding search', 'faiss', 'milvus', 'neural search',
             'qdrant', 'semantic search', 'vector search', 'weaviate', ...]
        """
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

    # ------------------------------------------------------------------ #
    # BM25-ready token list                                                #
    # ------------------------------------------------------------------ #

    def build_query_tokens(
        self,
        skills: list[str],
        include_co_skills: bool = True,
        include_domain_transfer_sources: bool = True,
    ) -> list[str]:
        """
        Expand skills and tokenise for BM25Okapi.get_scores() consumption.

        BM25Okapi expects a flat list of string tokens. Multi-word skill
        phrases like "dense retrieval" are split → ["dense", "retrieval"]
        so they can match candidate profiles where those words appear in
        any order or context.

        This method is the main entry point for retrieval/keyword_path.py.

        Args:
            skills: JD required + nice-to-have skill names (lowercase).
            include_co_skills: Expand with co-occurring skills.
            include_domain_transfer_sources: Add reverse-domain-transfer
                sources for broader BM25 recall. Recommended True for the
                keyword retrieval path.

        Returns:
            Flat, deduplicated list of lowercase string tokens.

        Example:
            >>> expander.build_query_tokens(["sentence transformers"])
            ['bi', 'bi-encoder', 'cross', 'cross-encoder', 'dense',
             'encoder', 'faiss', 'hugging', 'face', 'information',
             'retrieval', 'pytorch', 'reranker', 'sbert', 'semantic',
             'sentence', 'transformers', ...]
        """
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

    # ------------------------------------------------------------------ #
    # Dunder helpers                                                       #
    # ------------------------------------------------------------------ #

    @property
    def loaded(self) -> bool:
        """True if skill_map.json was loaded without error."""
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


# --------------------------------------------------------------------------- #
# Module-level convenience function                                            #
# --------------------------------------------------------------------------- #

def expand_jd_query(
    skills: list[str],
    skill_map_path: Optional[Path] = None,
    include_co_skills: bool = True,
    include_domain_transfer_sources: bool = True,
) -> list[str]:
    """
    One-shot convenience: expand JD skills into BM25 query tokens.

    Creates a fresh QueryExpander on each call. For repeated expansion
    (e.g. in a loop or in tests), instantiate QueryExpander once and
    call build_query_tokens() directly to avoid repeated file I/O.

    Args:
        skills: JD required skill names (lowercase).
        skill_map_path: Optional override path to skill_map.json.
        include_co_skills: Expand with co-occurring skills.
        include_domain_transfer_sources: Add reverse-transfer sources.

    Returns:
        Flat, deduplicated list of lowercase BM25 query tokens.
    """
    expander = QueryExpander(skill_map_path=skill_map_path)
    return expander.build_query_tokens(
        skills,
        include_co_skills=include_co_skills,
        include_domain_transfer_sources=include_domain_transfer_sources,
    )

