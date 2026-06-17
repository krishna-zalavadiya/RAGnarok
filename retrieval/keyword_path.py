"""
retrieval/keyword_path.py
--------------------------
Retrieval Path 2: BM25 sparse-retrieval with ontology-expanded query.

Why BM25 alongside FAISS?
    Dense embeddings (FAISS, Path 1) can miss candidates with short or
    sparse profiles — a candidate who simply lists:
        skills: [FAISS, Pinecone, Qdrant, Elasticsearch, Python]
    produces a generic embedding dominated by boilerplate, not their skills.
    BM25 rewards exact term overlap: if the JD mentions "FAISS" and the
    candidate profile says "FAISS", BM25 scores that hit directly.

Ontology expansion prevents the reverse problem: if the JD says
"sentence transformers" but the candidate wrote "sbert" or "bi-encoder",
plain BM25 would miss them. QueryExpander.build_query_tokens() expands the
query so both variants produce token hits.

Algorithm:
    1. Expand jd_intent.required_skills via QueryExpander
         → synonyms + co-skills + reverse domain-transfer sources
         → tokenise to flat list of BM25 query tokens
    2. Load pre-built BM25Okapi model from bm25.pkl  (built by DEV B offline)
    3. bm25.get_scores(query_tokens) → numpy array of relevance scores
    4. Rank by score, normalize to [0, 1], return top-K RetrievalResult

DEV B interface contract (indexing/bm25_builder.py must produce this):
    bm25.pkl  deserialized structure:
    {
        "bm25":          BM25Okapi   — trained on tokenised candidate profiles
        "candidate_ids": list[str]   — CAND_XXXXXXX, same order as bm25 corpus
        "corpus_size":   int         — len(candidate_ids), for validation
    }
    The BM25 corpus must be built from the same text field as
    CandidateFeatureVector.embedding_text  (headline + summary + titles +
    role descriptions + skill names).

Consumed by:
    retrieval/rrf_fusion.py    merges results from all 5 paths
    pipeline/runner.py         Path 2 of the ranking pipeline

Dependencies:
    config.py                  BM25_INDEX_PATH, KEYWORD_PATH_TOP_K
    pipeline/schemas.py        JDIntent, RetrievalResult
    ontology/query_expander.py QueryExpander
    rank_bm25                  Optional import, graceful degradation
    numpy                      Always available (requirements.txt)
    pickle                     stdlib
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np

import config
from ontology.query_expander import QueryExpander
from pipeline.schemas import JDIntent, RetrievalResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional rank_bm25 import — clear error at retrieve() time if missing
# ─────────────────────────────────────────────────────────────────────────────

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi   # type: ignore[import]
    _BM25_AVAILABLE = True
    logger.debug("rank_bm25 imported successfully.")
except ImportError:
    _BM25Okapi = None                                # type: ignore[assignment,misc]
    _BM25_AVAILABLE = False
    logger.warning(
        "rank_bm25 is not installed. KeywordPath.retrieve() will raise "
        "RuntimeError. Install with: pip install rank-bm25==0.2.2"
    )

# Minimum BM25 score to include a result.
# Candidates with score = 0 had zero token overlap with the expanded query.
_MIN_BM25_SCORE: float = 1e-9

# Maximum tokens passed to BM25 — guards against degenerate query expansion.
_MAX_QUERY_TOKENS: int = 500


# ─────────────────────────────────────────────────────────────────────────────
# KeywordPath
# ─────────────────────────────────────────────────────────────────────────────

class KeywordPath:
    """
    BM25 sparse-retrieval path with ontology-expanded query (Path 2 of 5).

    Typical production usage:
        # Load once in runner.py at startup
        path = KeywordPath.from_disk()

        # Call once per JD inside the ranking loop
        results = path.retrieve(jd_intent, top_k=25)

    Unit-test usage (no real index needed):
        from rank_bm25 import BM25Okapi
        corpus = [
            ["embeddings", "faiss", "python", "ranking"],
            ["sales", "marketing", "excel"],
            ["pinecone", "qdrant", "vector", "search", "python"],
        ]
        bm25 = BM25Okapi(corpus)
        ids  = ["CAND_0000001", "CAND_0000002", "CAND_0000003"]
        path = KeywordPath(bm25_model=bm25, candidate_ids=ids)
        results = path.retrieve(jd_intent)
    """

    PATH_NAME: str = "keyword"

    def __init__(
        self,
        bm25_model: Optional[object] = None,
        candidate_ids: Optional[list[str]] = None,
        index_path: Optional[Path] = None,
        query_expander: Optional[QueryExpander] = None,
        skill_map_path: Optional[Path] = None,
    ) -> None:
        """
        Initialise with either pre-loaded objects (for testing) or disk paths.

        Args:
            bm25_model:     Pre-loaded BM25Okapi instance. Takes priority
                            over index_path when supplied.
            candidate_ids:  List of CAND_XXXXXXX strings aligned with bm25
                            corpus order. Required when bm25_model is given.
            index_path:     Path to bm25.pkl.
                            Defaults to config.BM25_INDEX_PATH.
            query_expander: Pre-loaded QueryExpander instance. If None, one
                            is built from skill_map_path.
            skill_map_path: Path to skill_map.json.
                            Defaults to config.SKILL_MAP_PATH.

        Raises:
            ValueError:        bm25_model supplied but candidate_ids is None.
            ValueError:        len(bm25_model.corpus_size) != len(candidate_ids).
            FileNotFoundError: index_path does not exist (raised lazily on
                               first retrieve() when using disk path).
        """
        self._index_path: Path = index_path or config.BM25_INDEX_PATH
        effective_map = skill_map_path or config.SKILL_MAP_PATH

        # Build or accept the query expander
        self._expander: QueryExpander = (
            query_expander if query_expander is not None
            else QueryExpander(effective_map)
        )

        self._bm25 = None
        self._candidate_ids: list[str] = []
        self._loaded: bool = False

        if bm25_model is not None:
            if candidate_ids is None:
                raise ValueError(
                    "candidate_ids must be provided when bm25_model is given."
                )
            self._bm25 = bm25_model
            self._candidate_ids = list(candidate_ids)
            self._validate_alignment()
            self._loaded = True
            logger.debug(
                "KeywordPath initialised with pre-loaded BM25 "
                "(corpus_size=%d).",
                len(self._candidate_ids),
            )

    # ------------------------------------------------------------------ #
    # Factory — production path                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_disk(
        cls,
        index_path: Optional[Path] = None,
        skill_map_path: Optional[Path] = None,
        query_expander: Optional[QueryExpander] = None,
    ) -> "KeywordPath":
        """
        Load BM25 index from bm25.pkl and return a ready instance.

        Call once in pipeline/runner.py at startup; reuse the returned
        instance across retrieve() calls to avoid repeated pickle I/O.

        Args:
            index_path:     Override for config.BM25_INDEX_PATH.
            skill_map_path: Override for config.SKILL_MAP_PATH.
            query_expander: Pre-built expander to share with other paths.

        Returns:
            Fully loaded KeywordPath instance.

        Raises:
            RuntimeError:      rank_bm25 not installed.
            FileNotFoundError: bm25.pkl not found.
            KeyError:          bm25.pkl missing required keys.
            ValueError:        corpus / candidate_id size mismatch.
        """
        instance = cls(
            index_path=index_path,
            skill_map_path=skill_map_path,
            query_expander=query_expander,
        )
        instance._ensure_loaded()
        return instance

    # ------------------------------------------------------------------ #
    # Primary retrieve method                                              #
    # ------------------------------------------------------------------ #

    def retrieve(
        self,
        jd_intent: JDIntent,
        top_k: int = config.KEYWORD_PATH_TOP_K,
    ) -> list[RetrievalResult]:
        """
        Score all candidates in the BM25 index against the expanded JD query.

        Query construction:
            Required skills → QueryExpander → synonyms + co-skills +
            reverse domain-transfer sources → whitespace-tokenised flat list.

        Scoring:
            BM25Okapi.get_scores(query_tokens) → numpy array of TF-IDF
            weighted BM25 scores for every candidate in the corpus.
            Scores are normalised to [0, 1] by dividing by the maximum
            score in the result set.

        Args:
            jd_intent: Parsed JD intent. Uses jd_intent.required_skills
                       as the seed for query expansion.
            top_k:     Maximum candidates to return.
                       Defaults to config.KEYWORD_PATH_TOP_K (25).

        Returns:
            list[RetrievalResult] sorted by BM25 score descending,
            length ≤ top_k. Candidates with zero BM25 score are excluded.

            path_name    = "keyword"
            path_score   ∈ [0.0, 1.0]  (normalised BM25 score)
            rank_in_path = 1-indexed position in this path's results

        Raises:
            RuntimeError:  rank_bm25 not installed.
            ValueError:    top_k < 1.
            ValueError:    jd_intent.required_skills is empty.
        """
        self._assert_bm25_available()
        self._ensure_loaded()

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        if not jd_intent.required_skills:
            logger.warning(
                "KeywordPath.retrieve: jd_intent.required_skills is empty. "
                "Returning no results."
            )
            return []

        # ── Build expanded BM25 query tokens ──────────────────────────────
        query_tokens: list[str] = self._build_query_tokens(jd_intent)
        if not query_tokens:
            logger.warning(
                "KeywordPath: query expansion produced 0 tokens. "
                "Check skill_map.json and required_skills: %s",
                jd_intent.required_skills[:5],
            )
            return []

        logger.debug(
            "BM25 query: %d tokens (sample: %s …)",
            len(query_tokens),
            query_tokens[:6],
        )

        # ── BM25 scoring ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        scores: np.ndarray = self._bm25.get_scores(query_tokens)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logger.debug(
            "BM25 scored %d candidates in %.1f ms "
            "(max=%.3f, nonzero=%d)",
            len(scores),
            elapsed_ms,
            float(scores.max()) if len(scores) > 0 else 0.0,
            int(np.sum(scores > _MIN_BM25_SCORE)),
        )

        # ── Rank and normalise ────────────────────────────────────────────
        results = self._build_results(scores, top_k)

        logger.info(
            "KeywordPath.retrieve: %d results (top_k=%d, %.1f ms)",
            len(results),
            top_k,
            elapsed_ms,
        )
        return results

    # ------------------------------------------------------------------ #
    # Query building                                                       #
    # ------------------------------------------------------------------ #

    def _build_query_tokens(self, jd_intent: JDIntent) -> list[str]:
        """
        Build BM25 query tokens from JD required skills.

        Strategy:
          1. Expand required skills via QueryExpander with full options:
               - synonyms (bidirectional near-equivalents)
               - co-skills (commonly paired skills)
               - reverse domain-transfer sources (for Tier-5 recall:
                 "recommendation systems" found when JD needs "information
                 retrieval")
          2. Also add nice-to-have skill tokens for supplementary coverage
             (without co-skill expansion to avoid excessive noise).
          3. Deduplicate and cap at _MAX_QUERY_TOKENS.

        Returns:
            Flat list of lowercase string tokens for BM25Okapi.get_scores().
        """
        # Primary: required skills with full expansion
        primary_tokens: list[str] = self._expander.build_query_tokens(
            jd_intent.required_skills,
            include_co_skills=True,
            include_domain_transfer_sources=True,
        )

        # Supplement: nice-to-have skills — synonyms only, no co-skills
        # (prevents NTH from dominating over required skills)
        nth_tokens: list[str] = []
        if jd_intent.nice_to_have_skills:
            nth_tokens = self._expander.build_query_tokens(
                jd_intent.nice_to_have_skills,
                include_co_skills=False,
                include_domain_transfer_sources=False,
            )

        # Merge: required tokens first (higher implicit priority in BM25
        # because they appear earlier in the merged document — BM25 does
        # not differentiate position, but deduplication preserves required
        # tokens over NTH duplicates).
        seen: set[str] = set()
        merged: list[str] = []
        for tok in primary_tokens + nth_tokens:
            if tok not in seen:
                seen.add(tok)
                merged.append(tok)
            if len(merged) >= _MAX_QUERY_TOKENS:
                break

        return merged

    # ------------------------------------------------------------------ #
    # Result building                                                      #
    # ------------------------------------------------------------------ #

    def _build_results(
        self,
        scores: np.ndarray,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Convert raw BM25 score array into sorted, normalised RetrievalResult list.

        Score normalisation: divide all scores by the maximum score so that
        the top result always has path_score = 1.0. Candidates with score
        <= _MIN_BM25_SCORE (effectively zero) are excluded entirely —
        they had no BM25 vocabulary overlap with the query.

        Args:
            scores:  numpy array shape (N,) of raw BM25 scores.
            top_k:   Maximum results to return.

        Returns:
            list[RetrievalResult] sorted by score descending, len <= top_k.
        """
        if len(scores) == 0:
            return []

        max_score: float = float(scores.max())
        if max_score <= _MIN_BM25_SCORE:
            logger.warning(
                "BM25: all scores are effectively zero. "
                "Check that the index was built from the correct text fields."
            )
            return []

        # Partial sort — O(N log K) instead of O(N log N)
        # Only compute top_k+buffer to handle ties and zero exclusion
        top_k_clamped = min(top_k, len(scores))
        top_indices: np.ndarray = np.argpartition(
            scores, -top_k_clamped
        )[-top_k_clamped:]
        # Sort the top_k_clamped candidates by score descending
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: list[RetrievalResult] = []
        rank = 0

        for idx in top_indices:
            raw_score = float(scores[idx])
            if raw_score <= _MIN_BM25_SCORE:
                continue  # Zero-score candidates excluded

            if idx >= len(self._candidate_ids):
                logger.warning(
                    "BM25 returned out-of-range index %d "
                    "(candidate_ids length=%d). Skipping.",
                    int(idx),
                    len(self._candidate_ids),
                )
                continue

            rank += 1
            normalised_score = min(1.0, raw_score / max_score)

            results.append(
                RetrievalResult(
                    candidate_id=self._candidate_ids[int(idx)],
                    path_score=normalised_score,
                    path_name=self.PATH_NAME,
                    rank_in_path=rank,
                )
            )

            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------ #
    # Internal loading                                                     #
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Load BM25 index from disk if not already loaded."""
        if self._loaded:
            return
        self._assert_bm25_available()
        bm25_model, candidate_ids = self._load_bm25_index(self._index_path)
        self._bm25 = bm25_model
        self._candidate_ids = candidate_ids
        self._validate_alignment()
        self._loaded = True
        logger.info(
            "KeywordPath loaded from disk: corpus_size=%d",
            len(self._candidate_ids),
        )

    @staticmethod
    def _load_bm25_index(path: Path) -> tuple[object, list[str]]:
        """
        Load and validate the BM25 index pickle.

        Expected pickle structure (produced by indexing/bm25_builder.py):
            {
                "bm25":          BM25Okapi   — trained model
                "candidate_ids": list[str]   — CAND_XXXXXXX, aligned with corpus
                "corpus_size":   int         — validation count (optional)
            }

        Security note: pickle.load is inherently unsafe with untrusted files.
        This file is written by our own pipeline (bm25_builder.py) and read
        back in a controlled environment, so the risk is acceptable.

        Raises:
            FileNotFoundError: bm25.pkl not found at path.
            KeyError:          Pickle missing "bm25" or "candidate_ids" keys.
            ValueError:        candidate_ids contains invalid ID format.
            RuntimeError:      Pickle deserialization failed.
        """
        if not path.exists():
            raise FileNotFoundError(
                f"BM25 index not found: '{path}'. "
                "Run precompute.py to build the index, or verify "
                "config.BM25_INDEX_PATH."
            )

        t0 = time.perf_counter()
        try:
            with open(path, "rb") as fh:
                data: dict = pickle.load(fh)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load BM25 index from '{path}': {exc}. "
                "The file may be corrupt. Delete it and re-run precompute.py."
            ) from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Validate required keys
        required_keys = {"bm25", "candidate_ids"}
        missing = required_keys - set(data.keys())
        if missing:
            raise KeyError(
                f"bm25.pkl is missing required keys: {missing}. "
                f"Found keys: {list(data.keys())}. "
                "Check indexing/bm25_builder.py output format."
            )

        bm25_model = data["bm25"]
        candidate_ids: list[str] = list(data["candidate_ids"])

        # Validate corpus_size matches if present
        if "corpus_size" in data:
            expected = int(data["corpus_size"])
            if expected != len(candidate_ids):
                raise ValueError(
                    f"bm25.pkl corpus_size={expected} does not match "
                    f"len(candidate_ids)={len(candidate_ids)}. "
                    "Re-run precompute.py."
                )

        # Spot-check candidate_id format
        import re as _re
        _cand_re = _re.compile(r"^CAND_[0-9]{7}$")
        for cid in candidate_ids[:5]:
            if not _cand_re.match(str(cid)):
                raise ValueError(
                    f"Unexpected candidate_id format in bm25.pkl: '{cid}'. "
                    "Expected CAND_XXXXXXX."
                )

        logger.debug(
            "BM25 index loaded in %.0f ms (corpus_size=%d).",
            elapsed_ms,
            len(candidate_ids),
        )
        return bm25_model, candidate_ids

    def _validate_alignment(self) -> None:
        """
        Verify that corpus size matches candidate_ids length.

        BM25Okapi stores corpus_size as an attribute. The candidate_ids list
        must be the same length, or index lookups will be incorrect.

        Raises:
            ValueError: Size mismatch between BM25 corpus and candidate_ids.
        """
        if self._bm25 is None or not self._candidate_ids:
            return

        bm25_size = getattr(self._bm25, "corpus_size", None)
        if bm25_size is None:
            # Older rank_bm25 versions may not have corpus_size
            return

        if bm25_size != len(self._candidate_ids):
            raise ValueError(
                f"BM25 corpus_size={bm25_size} does not match "
                f"len(candidate_ids)={len(self._candidate_ids)}. "
                "Re-run precompute.py to rebuild aligned indexes."
            )

    # ------------------------------------------------------------------ #
    # Assertion helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _assert_bm25_available() -> None:
        if not _BM25_AVAILABLE:
            raise RuntimeError(
                "rank_bm25 is not installed. "
                "Run: pip install rank-bm25==0.2.2"
            )

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def loaded(self) -> bool:
        """True if BM25 model and candidate IDs are ready for queries."""
        return self._loaded

    @property
    def corpus_size(self) -> int:
        """Number of candidates in the BM25 corpus (0 if not loaded)."""
        return len(self._candidate_ids) if self._loaded else 0

    def __repr__(self) -> str:
        status = (
            f"corpus_size={self.corpus_size}"
            if self._loaded else "not loaded"
        )
        return f"KeywordPath({status})"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_keyword(
    jd_intent: JDIntent,
    top_k: int = config.KEYWORD_PATH_TOP_K,
    index_path: Optional[Path] = None,
    skill_map_path: Optional[Path] = None,
) -> list[RetrievalResult]:
    """
    One-shot convenience: load BM25 index and retrieve top-K candidates.

    Creates a new KeywordPath on each call (incurs pickle I/O).
    For repeated calls, use KeywordPath.from_disk() once and reuse.

    Args:
        jd_intent:      Parsed JDIntent (required_skills must be populated).
        top_k:          Number of results to return.
        index_path:     Override for config.BM25_INDEX_PATH.
        skill_map_path: Override for config.SKILL_MAP_PATH.

    Returns:
        list[RetrievalResult] sorted by BM25 score descending.
    """
    path = KeywordPath.from_disk(
        index_path=index_path, skill_map_path=skill_map_path
    )
    return path.retrieve(jd_intent, top_k=top_k)

