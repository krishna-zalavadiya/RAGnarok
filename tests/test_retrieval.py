"""
tests/test_retrieval.py
-----------------------
Unit and integration tests for all 5 retrieval paths + RRF fusion.

Sprint acceptance criteria (Phase 3, DEV A, Day 10):
  ✓ Tier-5 recall test passes:
      A candidate with RecSys skills but zero IR keywords is rescued by
      OntologyPath and survives RRF fusion.
  ✓ Sparse-profile BM25 rescue test passes:
      A candidate whose profile is ONLY a list of skill tokens (no summary,
      no descriptions) ranks high via BM25 even though FAISS would miss them.
  ✓ All 5 paths return expected result counts within their configured top-K.

Test strategy:
  - All paths accept pre-loaded data in their constructors, so NO disk I/O,
    NO real indexes, and NO model downloads are needed during testing.
  - SemanticPath: in-memory FAISS IndexFlatIP (skipped if faiss-cpu absent)
  - KeywordPath:  in-memory BM25Okapi  +  MockQueryExpander (no ontology I/O)
  - OntologyPath: MagicMock SkillGraph (controls rank_by_domain_transfer output)
  - TrajectoryPath: numpy array literal (no file I/O)
  - SignalPath:    numpy array literal (no file I/O, no JDIntent)
  - RRF:          RetrievalResult lists built inline (no path infrastructure)

Running:
    pytest tests/test_retrieval.py -v
    pytest tests/test_retrieval.py -v -m "tier5"           # Tier-5 recall only
    pytest tests/test_retrieval.py -v -m "bm25_rescue"     # BM25 rescue only
    pytest tests/test_retrieval.py -v -m "integration"     # RRF integration only

Dependencies:
    config.py               constants (SEMANTIC_PATH_TOP_K, etc.)
    pipeline/schemas.py     RetrievalResult, RRFResult, JDIntent
    retrieval/semantic_path.py   SemanticPath
    retrieval/keyword_path.py    KeywordPath
    retrieval/ontology_path.py   OntologyPath
    retrieval/trajectory_path.py TrajectoryPath
    retrieval/signal_path.py     SignalPath
    retrieval/rrf_fusion.py      RRFFusion, fuse_results
    tests/conftest.py       mock_jd_intent, good_candidate_ids, bad_candidate_ids,
                            good_candidate_raw, app_config
"""

from __future__ import annotations

import copy
import re
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import config
from pipeline.schemas import JDIntent, RetrievalResult, RRFResult
from retrieval.rrf_fusion import RRFFusion, fuse_results
from retrieval.ontology_path import OntologyPath
from retrieval.signal_path import SignalPath
from retrieval.trajectory_path import (
    TrajectoryPath,
    COL_PROMOTIONS_PER_YEAR,
    COL_YOE,
    COL_HAS_PRODUCT_CO,
    COL_IS_IC_RISER,
    COL_CONSULTING_ONLY,
)

# ─────────────────────────────────────────────────────────────────────────────
# Optional import guards — skip relevant test classes if libraries missing
# ─────────────────────────────────────────────────────────────────────────────

try:
    import faiss as _faiss  # type: ignore[import]
    FAISS_AVAILABLE = True
except ImportError:
    _faiss = None  # type: ignore[assignment]
    FAISS_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi as _BM25Okapi  # type: ignore[import]
    BM25_AVAILABLE = True
except ImportError:
    _BM25Okapi = None  # type: ignore[assignment]
    BM25_AVAILABLE = False

# Candidate ID format validator
_CAND_ID_RE = re.compile(r"^CAND_[0-9]{7}$")

# ─────────────────────────────────────────────────────────────────────────────
# Custom pytest marks
# ─────────────────────────────────────────────────────────────────────────────

# Register marks to avoid PytestUnknownMarkWarning. Add to pytest.ini if needed.
# Usage: pytest -m tier5 | pytest -m bm25_rescue | pytest -m integration

pytestmark = []   # No module-wide marks — each test/class uses own marks


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FIXTURES (module-scoped to avoid rebuilding for every test)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def jd_with_embedding(mock_jd_intent: JDIntent) -> JDIntent:
    """
    JDIntent with a synthetic 384-dim L2-normalised embedding vector.

    mock_jd_intent from conftest has embedding=None.
    SemanticPath.retrieve() requires jd_intent.embedding to be populated.
    This fixture deepcopies the intent and adds a deterministic random vector.

    Seed=42 ensures the same vector is used across all SemanticPath tests in
    this module — consistent for reproducible test assertions.
    """
    jd = copy.deepcopy(mock_jd_intent)
    rng = np.random.default_rng(seed=42)
    vec = rng.standard_normal(config.EMBEDDING_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)   # L2-normalise to match FAISS IndexFlatIP
    jd.embedding = vec.tolist()
    return jd


class MockQueryExpander:
    """
    Minimal stub for ontology/query_expander.QueryExpander.

    Returns the input skills as whitespace-split tokens with underscores
    replacing spaces (mimicking what the real expander would do with no
    ontology expansion). This keeps KeywordPath tests self-contained —
    no skill_map.json required.

    Injection point: KeywordPath(query_expander=MockQueryExpander())
    Called by: KeywordPath._build_query_tokens() which calls
               self._expander.build_query_tokens(skills, include_co_skills=...,
                                                  include_domain_transfer_sources=...)
    """

    def build_query_tokens(
        self,
        skills: list[str],
        include_co_skills: bool = True,
        include_domain_transfer_sources: bool = True,
    ) -> list[str]:
        """Return skills as lowercase underscore-delimited tokens."""
        tokens: list[str] = []
        for skill in skills:
            tokens.extend(skill.lower().replace(" ", "_").split("_"))
        return list(dict.fromkeys(tokens))   # deduplicate, preserve order


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _assert_retrieval_result_structure(
    results: list[RetrievalResult],
    expected_path_name: str,
    top_k: int,
) -> None:
    """
    Shared structural assertions for any path's result list.

    Checks:
      1. len(results) <= top_k
      2. path_name on every result matches expected_path_name
      3. rank_in_path is 1-indexed sequential (1, 2, 3, …)
      4. path_score in [0.0, 1.0] for every result
      5. candidate_id format matches CAND_XXXXXXX on every result
      6. Results are sorted by path_score descending
    """
    assert len(results) <= top_k, (
        f"Expected ≤ {top_k} results, got {len(results)}"
    )

    for r in results:
        assert r.path_name == expected_path_name, (
            f"path_name mismatch: expected '{expected_path_name}', "
            f"got '{r.path_name}' for {r.candidate_id}"
        )
        assert 0.0 <= r.path_score <= 1.0, (
            f"path_score out of [0, 1]: {r.path_score} for {r.candidate_id}"
        )
        assert _CAND_ID_RE.match(r.candidate_id), (
            f"Invalid candidate_id format: '{r.candidate_id}'"
        )

    expected_ranks = list(range(1, len(results) + 1))
    actual_ranks = [r.rank_in_path for r in results]
    assert actual_ranks == expected_ranks, (
        f"rank_in_path not 1-indexed sequential: {actual_ranks}"
    )

    for i in range(len(results) - 1):
        assert results[i].path_score >= results[i + 1].path_score, (
            f"Results not sorted descending at index {i}: "
            f"{results[i].path_score} < {results[i + 1].path_score}"
        )


def _make_rrf_result(cid: str, rrf_score: float, paths: list[str]) -> RRFResult:
    """Build a minimal RRFResult for RRF integration tests."""
    return RRFResult(
        candidate_id=cid,
        rrf_score=rrf_score,
        paths_present=paths,
        cross_encoder_score=0.0,
    )


def _make_retrieval_result(cid: str, score: float, path: str, rank: int) -> RetrievalResult:
    """Build a minimal RetrievalResult for RRF integration tests."""
    return RetrievalResult(
        candidate_id=cid,
        path_score=score,
        path_name=path,
        rank_in_path=rank,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. SemanticPath Tests
#    Skipped if faiss-cpu not installed.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu not installed")
class TestSemanticPath:
    """
    Unit tests for retrieval/semantic_path.py.

    All tests use an in-memory FAISS IndexFlatIP with synthetic 384-dim vectors.
    No disk I/O. No sentence-transformer model needed.

    The JD is represented by a known vector. Candidate vectors are constructed
    so some are guaranteed to be similar (dot-product > 0.8) and some orthogonal.
    """

    # Import inside class so the skip decorator fires before import is attempted.
    @pytest.fixture(scope="class")
    def semantic_path_class_imports(self):
        """Lazy import SemanticPath inside test class to honour skip."""
        from retrieval.semantic_path import SemanticPath
        return SemanticPath

    N_CANDIDATES: int = 20
    DIM: int = config.EMBEDDING_DIM
    TOP_K: int = config.SEMANTIC_PATH_TOP_K

    @pytest.fixture(scope="class")
    def synthetic_faiss_index_and_ids(self):
        """
        Build a synthetic 20-candidate FAISS IndexFlatIP and aligned ID array.

        Candidate 0 is constructed to be very similar to the JD query vector
        (used in jd_with_embedding fixture, seed=42). All other candidates
        are random orthogonal-ish vectors.
        """
        rng = np.random.default_rng(seed=1)
        vecs = rng.standard_normal((self.N_CANDIDATES, self.DIM)).astype(np.float32)
        _faiss.normalize_L2(vecs)

        index = _faiss.IndexFlatIP(self.DIM)
        index.add(vecs)

        ids = np.array(
            [f"CAND_{i + 1:07d}" for i in range(self.N_CANDIDATES)],
            dtype=object,
        )
        return index, ids

    @pytest.fixture(scope="class")
    def semantic_path(self, semantic_path_class_imports, synthetic_faiss_index_and_ids):
        """SemanticPath with pre-loaded in-memory index."""
        SemanticPath = semantic_path_class_imports
        index, ids = synthetic_faiss_index_and_ids
        return SemanticPath(index=index, candidate_ids=ids)

    # ── Structural tests ──────────────────────────────────────────────────

    def test_returns_at_most_top_k(self, semantic_path, jd_with_embedding):
        """retrieve() returns ≤ SEMANTIC_PATH_TOP_K results."""
        results = semantic_path.retrieve(jd_with_embedding, top_k=self.TOP_K)
        assert len(results) <= self.TOP_K, (
            f"Expected ≤ {self.TOP_K} results, got {len(results)}"
        )

    def test_results_sorted_descending(self, semantic_path, jd_with_embedding):
        """Results are ordered by cosine similarity descending."""
        results = semantic_path.retrieve(jd_with_embedding)
        for i in range(len(results) - 1):
            assert results[i].path_score >= results[i + 1].path_score, (
                f"Not sorted descending at index {i}: "
                f"{results[i].path_score} < {results[i + 1].path_score}"
            )

    def test_path_name_is_semantic(self, semantic_path, jd_with_embedding):
        """All results have path_name='semantic'."""
        results = semantic_path.retrieve(jd_with_embedding)
        assert all(r.path_name == "semantic" for r in results), (
            f"Found non-semantic path names: "
            f"{[r.path_name for r in results if r.path_name != 'semantic']}"
        )

    def test_rank_in_path_is_one_indexed_sequential(self, semantic_path, jd_with_embedding):
        """rank_in_path values are 1, 2, 3, … without gaps."""
        results = semantic_path.retrieve(jd_with_embedding)
        expected = list(range(1, len(results) + 1))
        actual = [r.rank_in_path for r in results]
        assert actual == expected, (
            f"rank_in_path not 1-indexed sequential: {actual}"
        )

    def test_path_scores_in_valid_range(self, semantic_path, jd_with_embedding):
        """All path_score values are in [0.0, 1.0]."""
        results = semantic_path.retrieve(jd_with_embedding)
        out_of_range = [r for r in results if not 0.0 <= r.path_score <= 1.0]
        assert not out_of_range, (
            f"Scores out of [0,1]: "
            f"{[(r.candidate_id, r.path_score) for r in out_of_range]}"
        )

    def test_candidate_ids_match_cand_format(self, semantic_path, jd_with_embedding):
        """All candidate_id values match CAND_XXXXXXX (7 digits)."""
        results = semantic_path.retrieve(jd_with_embedding)
        bad = [r.candidate_id for r in results if not _CAND_ID_RE.match(r.candidate_id)]
        assert not bad, f"Invalid candidate_id formats: {bad}"

    def test_no_duplicate_candidate_ids(self, semantic_path, jd_with_embedding):
        """No candidate_id appears more than once in results."""
        results = semantic_path.retrieve(jd_with_embedding)
        ids = [r.candidate_id for r in results]
        assert len(ids) == len(set(ids)), (
            f"Duplicate candidate_ids in results: "
            f"{[cid for cid in ids if ids.count(cid) > 1]}"
        )

    def test_top_k_respected_when_smaller_than_index(self, semantic_path, jd_with_embedding):
        """retrieve(top_k=5) returns exactly 5 results when index has ≥5."""
        results = semantic_path.retrieve(jd_with_embedding, top_k=5)
        assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    def test_top_k_clamped_when_larger_than_index(self, semantic_path_class_imports, jd_with_embedding):
        """
        retrieve(top_k=1000) returns at most ntotal results when index is small.
        Uses a tiny 3-candidate index to force the clamp.
        """
        SemanticPath = semantic_path_class_imports
        rng = np.random.default_rng(seed=99)
        vecs = rng.standard_normal((3, self.DIM)).astype(np.float32)
        _faiss.normalize_L2(vecs)
        index = _faiss.IndexFlatIP(self.DIM)
        index.add(vecs)
        ids = np.array(["CAND_0000001", "CAND_0000002", "CAND_0000003"], dtype=object)

        tiny_path = SemanticPath(index=index, candidate_ids=ids)
        results = tiny_path.retrieve(jd_with_embedding, top_k=1000)
        assert len(results) <= 3, (
            f"Expected ≤ 3 results (index size), got {len(results)}"
        )

    def test_raises_value_error_on_missing_embedding(self, semantic_path, mock_jd_intent):
        """
        retrieve() raises ValueError if jd_intent.embedding is None.
        mock_jd_intent has embedding=None by design (no encoder run).
        """
        assert mock_jd_intent.embedding is None
        with pytest.raises(ValueError, match="embedding is None"):
            semantic_path.retrieve(mock_jd_intent)

    def test_raises_value_error_on_wrong_embedding_dimension(
        self, semantic_path_class_imports, mock_jd_intent
    ):
        """
        retrieve() raises ValueError if jd_intent.embedding has wrong dimension.
        The FAISS index expects config.EMBEDDING_DIM (384) dimensions.
        """
        SemanticPath = semantic_path_class_imports
        rng = np.random.default_rng(seed=7)
        vecs = rng.standard_normal((3, self.DIM)).astype(np.float32)
        _faiss.normalize_L2(vecs)
        index = _faiss.IndexFlatIP(self.DIM)
        index.add(vecs)
        ids = np.array(["CAND_0000001", "CAND_0000002", "CAND_0000003"], dtype=object)
        path = SemanticPath(index=index, candidate_ids=ids)

        bad_jd = copy.deepcopy(mock_jd_intent)
        bad_jd.embedding = [0.1] * 128   # Wrong: 128-dim instead of 384-dim

        with pytest.raises(ValueError, match=r"(embedding|dimension|dim)"):
            path.retrieve(bad_jd)

    def test_path_name_constant(self, semantic_path_class_imports):
        """SemanticPath.PATH_NAME is 'semantic' (used by RRF fusion for bonus lookup)."""
        SemanticPath = semantic_path_class_imports
        assert SemanticPath.PATH_NAME == "semantic"

    def test_is_loaded_property(self, semantic_path):
        """loaded property returns True after construction with pre-loaded data."""
        assert semantic_path.loaded is True


# ─────────────────────────────────────────────────────────────────────────────
# 2. KeywordPath Tests
#    Skipped if rank-bm25 not installed.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not BM25_AVAILABLE, reason="rank-bm25 not installed")
class TestKeywordPath:
    """
    Unit tests for retrieval/keyword_path.py.

    Uses an in-memory BM25Okapi index built from controlled token lists.
    MockQueryExpander is injected so no skill_map.json is needed.

    Key test: test_sparse_profile_bm25_rescue — the core BM25 recall story.
    """

    TOP_K: int = config.KEYWORD_PATH_TOP_K

    @pytest.fixture(scope="class")
    def keyword_path_class_import(self):
        """Lazy import inside class to respect the skip decorator."""
        from retrieval.keyword_path import KeywordPath
        return KeywordPath

    @pytest.fixture(scope="class")
    def mixed_corpus_path(self, keyword_path_class_import):
        """
        KeywordPath with a 6-candidate corpus covering the key BM25 scenarios:

          CAND_0000001  — dense IR/ML profile: many matching tokens
          CAND_0000002  — SPARSE profile: ONLY skill tokens, no prose (rescue target)
          CAND_0000003  — wrong domain: marketing/sales, no ML
          CAND_0000004  — partial match: some ML, some unrelated
          CAND_0000005  — zero ML keywords: accountant/finance
          CAND_0000006  — empty/blank profile: no tokens
        """
        KeywordPath = keyword_path_class_import
        corpus = [
            # CAND_0000001: Dense IR profile — should rank near top
            ["machine", "learning", "engineer", "faiss", "pinecone", "embeddings",
             "sentence", "transformers", "vector", "search", "information",
             "retrieval", "ranking", "evaluation", "ndcg", "python", "production",
             "swiggy", "recommendation", "systems"],
            # CAND_0000002: SPARSE — only skill keywords, no prose (BM25 rescue target)
            ["faiss", "pinecone", "embeddings", "python", "qdrant", "weaviate",
             "elasticsearch", "vector", "search"],
            # CAND_0000003: Wrong domain — marketing/sales, no ML keywords
            ["marketing", "manager", "sales", "excel", "powerpoint", "campaigns",
             "brand", "leads", "crm", "social", "media"],
            # CAND_0000004: Partial match — some ML, mostly unrelated
            ["software", "engineer", "python", "django", "rest", "api",
             "postgres", "docker", "kubernetes"],
            # CAND_0000005: Zero ML — accountant/finance
            ["accountant", "gaap", "ind-as", "tally", "tax", "filing", "audit",
             "balance", "sheet", "excel"],
            # CAND_0000006: Blank-ish — tokenizer returns empty
            [""],
        ]
        candidate_ids = [
            "CAND_0000001", "CAND_0000002", "CAND_0000003",
            "CAND_0000004", "CAND_0000005", "CAND_0000006",
        ]
        bm25 = _BM25Okapi(corpus)
        return KeywordPath(
            bm25_model=bm25,
            candidate_ids=candidate_ids,
            query_expander=MockQueryExpander(),
        )

    # ── Structural tests ──────────────────────────────────────────────────

    def test_returns_at_most_top_k(self, mixed_corpus_path, mock_jd_intent):
        """retrieve() returns ≤ KEYWORD_PATH_TOP_K results."""
        results = mixed_corpus_path.retrieve(mock_jd_intent, top_k=self.TOP_K)
        assert len(results) <= self.TOP_K

    def test_path_name_is_keyword(self, mixed_corpus_path, mock_jd_intent):
        """All results have path_name='keyword'."""
        results = mixed_corpus_path.retrieve(mock_jd_intent)
        assert all(r.path_name == "keyword" for r in results)

    def test_rank_in_path_is_sequential(self, mixed_corpus_path, mock_jd_intent):
        """rank_in_path is 1-indexed sequential."""
        results = mixed_corpus_path.retrieve(mock_jd_intent)
        assert [r.rank_in_path for r in results] == list(range(1, len(results) + 1))

    def test_path_scores_in_range(self, mixed_corpus_path, mock_jd_intent):
        """All path_score values are in [0.0, 1.0]."""
        results = mixed_corpus_path.retrieve(mock_jd_intent)
        for r in results:
            assert 0.0 <= r.path_score <= 1.0, (
                f"{r.candidate_id} score={r.path_score} out of range"
            )

    def test_results_sorted_descending(self, mixed_corpus_path, mock_jd_intent):
        """Results are sorted by BM25 score descending."""
        results = mixed_corpus_path.retrieve(mock_jd_intent)
        for i in range(len(results) - 1):
            assert results[i].path_score >= results[i + 1].path_score, (
                f"Not sorted at index {i}: "
                f"{results[i].path_score} < {results[i + 1].path_score}"
            )

    def test_zero_score_candidates_excluded(self, mixed_corpus_path, mock_jd_intent):
        """
        Candidates with zero BM25 score (no token overlap) are not in results.
        CAND_0000003, CAND_0000005 have marketing/finance keywords only — no
        overlap with ML/retrieval JD query.
        """
        results = mixed_corpus_path.retrieve(mock_jd_intent)
        result_ids = {r.candidate_id for r in results}
        # Marketing manager and accountant have zero ML token overlap
        for bad_id in ("CAND_0000003", "CAND_0000005"):
            assert bad_id not in result_ids, (
                f"{bad_id} (non-ML profile) appeared in BM25 results — "
                "it has zero token overlap with the ML JD query."
            )

    def test_top_scorer_has_score_one(self, mixed_corpus_path, mock_jd_intent):
        """
        The highest-scoring candidate is normalised to path_score = 1.0.
        KeywordPath normalises BM25 scores by dividing by max_score.
        """
        results = mixed_corpus_path.retrieve(mock_jd_intent)
        assert results, "Expected at least one result"
        assert abs(results[0].path_score - 1.0) < 1e-6, (
            f"Top result score should be 1.0 (normalised), "
            f"got {results[0].path_score}"
        )

    def test_raises_on_empty_required_skills(self, keyword_path_class_import):
        """
        retrieve() returns [] (with warning) when jd_intent.required_skills is empty.
        No exception — just empty results, as documented in keyword_path.py.
        """
        KeywordPath = keyword_path_class_import
        bm25 = _BM25Okapi([["python"], ["java"]])
        path = KeywordPath(
            bm25_model=bm25,
            candidate_ids=["CAND_0000001", "CAND_0000002"],
            query_expander=MockQueryExpander(),
        )
        empty_jd = JDIntent(
            required_skills=[],   # Empty — triggers warning and returns []
            nice_to_have_skills=[],
            disqualifier_skills=[],
            expanded_required=[],
            yoe_min=5.0, yoe_max=9.0, yoe_ideal_min=5.0, yoe_ideal_max=9.0,
            preferred_locations=["noida"],
            relocation_accepted=True,
            disqualify_consulting_only=True,
            disqualify_no_production=True,
            raw_text="",
        )
        results = path.retrieve(empty_jd)
        assert results == [], (
            f"Expected [] for empty required_skills, got {results}"
        )

    def test_path_name_constant(self, keyword_path_class_import):
        """KeywordPath.PATH_NAME is 'keyword' (used by RRF fusion)."""
        KeywordPath = keyword_path_class_import
        assert KeywordPath.PATH_NAME == "keyword"

    # ── KEY ACCEPTANCE TEST ───────────────────────────────────────────────

    @pytest.mark.bm25_rescue
    def test_sparse_profile_bm25_rescue(
        self, keyword_path_class_import, mock_jd_intent
    ):
        """
        KEY ACCEPTANCE TEST: Sparse-profile BM25 rescue.

        Scenario:
          - CAND_SPARSE: Profile = ONLY skill tokens: "faiss pinecone embeddings
            qdrant weaviate python". No headline, no summary, no descriptions.
            A FAISS semantic search would return a generic, near-zero-content
            embedding for this profile — it would be missed.
          - CAND_PROSE: Rich profile with lots of text but in the wrong domain
            (frontend, marketing). Their embedding might be non-trivial but
            wrong-domain.

        Expected:
          CAND_SPARSE ranks in the top results because BM25 directly rewards
          exact token overlap: the JD asks for "faiss", "pinecone", "embeddings"
          and CAND_SPARSE's profile contains all three.

        This validates the core BM25 recall story from the proposal:
          "BM25 rewards exact term overlap: if the JD mentions 'FAISS' and the
          candidate profile says 'FAISS', BM25 scores that hit directly."
        """
        KeywordPath = keyword_path_class_import

        corpus = [
            # CAND_SPARSE: Only skill tokens, no prose — the rescue target
            ["faiss", "pinecone", "embeddings", "qdrant", "weaviate",
             "python", "elasticsearch", "vector", "search"],

            # CAND_PROSE_WRONG: Rich prose but wrong domain (frontend)
            ["senior", "frontend", "engineer", "react", "typescript", "webpack",
             "jest", "css", "html", "animations", "accessibility", "ui", "design",
             "component", "library", "migration", "angularjs", "redux", "next"],

            # CAND_MARKETING: Wrong domain entirely (control group)
            ["marketing", "brand", "campaign", "seo", "content", "social",
             "leads", "crm", "kpi", "pipeline", "demand", "generation"],
        ]
        ids = ["CAND_0000031", "CAND_0000014", "CAND_0000004"]
        bm25 = _BM25Okapi(corpus)
        path = KeywordPath(
            bm25_model=bm25,
            candidate_ids=ids,
            query_expander=MockQueryExpander(),
        )

        results = path.retrieve(mock_jd_intent, top_k=25)
        result_ids = [r.candidate_id for r in results]

        # CAND_SPARSE (CAND_0000031) must appear in results
        assert "CAND_0000031" in result_ids, (
            "FAIL: Sparse-profile candidate (CAND_0000031) not rescued by BM25. "
            "A profile of only skill tokens should rank high via exact token match. "
            f"Results: {result_ids}"
        )

        # CAND_SPARSE should rank #1 or #2 (it has the most required-skill token overlap)
        sparse_rank = next(r.rank_in_path for r in results if r.candidate_id == "CAND_0000031")
        assert sparse_rank <= 2, (
            f"FAIL: Sparse-profile candidate ranked {sparse_rank}, expected ≤ 2. "
            "BM25 should heavily favour direct skill keyword matches."
        )

        # CAND_MARKETING should NOT appear (zero ML token overlap)
        assert "CAND_0000004" not in result_ids, (
            "FAIL: Marketing manager candidate appeared in BM25 results. "
            "Zero overlap with ML JD query tokens expected."
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. OntologyPath Tests
#    Uses MagicMock SkillGraph — no skill_map.json needed.
# ─────────────────────────────────────────────────────────────────────────────

class TestOntologyPath:
    """
    Unit tests for retrieval/ontology_path.py.

    OntologyPath delegates all graph logic to SkillGraph.rank_by_domain_transfer().
    We mock SkillGraph entirely to control outputs and test OntologyPath's
    adapter logic (RetrievalResult construction, sort order, top_k clamping).

    KEY TEST: test_tier5_recall — RecSys candidate rescued via ontology.
    """

    TOP_K: int = config.ONTOLOGY_PATH_TOP_K

    @pytest.fixture
    def mock_skill_graph(self) -> MagicMock:
        """
        Minimal SkillGraph mock. Tests configure rank_by_domain_transfer
        return values per scenario.
        """
        graph = MagicMock()
        graph.__repr__ = MagicMock(return_value="MockSkillGraph()")
        return graph

    @pytest.fixture
    def ontology_path(self, mock_skill_graph: MagicMock) -> OntologyPath:
        """OntologyPath with injected mock SkillGraph."""
        return OntologyPath(skill_graph=mock_skill_graph)

    def _set_graph_return(
        self,
        mock_graph: MagicMock,
        ranked: list[tuple[str, float]],
    ) -> None:
        """Configure mock_graph.rank_by_domain_transfer to return `ranked`."""
        mock_graph.rank_by_domain_transfer.return_value = ranked

    # ── Structural tests ──────────────────────────────────────────────────

    def test_returns_at_most_top_k(
        self, ontology_path, mock_skill_graph, mock_jd_intent
    ):
        """retrieve() returns ≤ ONTOLOGY_PATH_TOP_K results."""
        # Graph returns more than top_k — OntologyPath must clamp via top_k arg
        # (the clamp happens in SkillGraph, but we pass top_k down and verify count)
        self._set_graph_return(
            mock_skill_graph,
            [(f"CAND_{i+1:07d}", 0.9 - i * 0.01) for i in range(self.TOP_K)],
        )
        candidate_skills_map = {
            f"CAND_{i+1:07d}": frozenset({"recommendation systems"})
            for i in range(self.TOP_K)
        }
        results = ontology_path.retrieve(
            mock_jd_intent,
            candidate_skills_map=candidate_skills_map,
            top_k=self.TOP_K,
        )
        assert len(results) <= self.TOP_K, (
            f"Expected ≤ {self.TOP_K} results, got {len(results)}"
        )

    def test_path_name_is_ontology(
        self, ontology_path, mock_skill_graph, mock_jd_intent
    ):
        """All results have path_name='ontology'."""
        self._set_graph_return(
            mock_skill_graph,
            [("CAND_0000001", 0.75), ("CAND_0000002", 0.50)],
        )
        results = ontology_path.retrieve(
            mock_jd_intent,
            candidate_skills_map={
                "CAND_0000001": frozenset({"recommendation systems"}),
                "CAND_0000002": frozenset({"nlp"}),
            },
        )
        assert all(r.path_name == "ontology" for r in results)

    def test_rank_in_path_sequential(
        self, ontology_path, mock_skill_graph, mock_jd_intent
    ):
        """rank_in_path is 1-indexed sequential from 1."""
        self._set_graph_return(
            mock_skill_graph,
            [("CAND_0000031", 0.85), ("CAND_0000043", 0.60), ("CAND_0000014", 0.40)],
        )
        results = ontology_path.retrieve(
            mock_jd_intent,
            candidate_skills_map={
                "CAND_0000031": frozenset({"recommendation systems"}),
                "CAND_0000043": frozenset({"nlp"}),
                "CAND_0000014": frozenset({"faiss"}),
            },
        )
        assert [r.rank_in_path for r in results] == [1, 2, 3]

    def test_path_scores_match_graph_output(
        self, ontology_path, mock_skill_graph, mock_jd_intent
    ):
        """path_score on each result matches the score returned by SkillGraph."""
        self._set_graph_return(
            mock_skill_graph,
            [("CAND_0000031", 0.80), ("CAND_0000043", 0.55)],
        )
        results = ontology_path.retrieve(
            mock_jd_intent,
            candidate_skills_map={
                "CAND_0000031": frozenset({"recommendation systems"}),
                "CAND_0000043": frozenset({"nlp"}),
            },
        )
        assert results[0].candidate_id == "CAND_0000031"
        assert abs(results[0].path_score - 0.80) < 1e-6
        assert results[1].candidate_id == "CAND_0000043"
        assert abs(results[1].path_score - 0.55) < 1e-6

    def test_empty_candidate_map_returns_empty(
        self, ontology_path, mock_jd_intent
    ):
        """retrieve() returns [] when candidate_skills_map is empty."""
        results = ontology_path.retrieve(
            mock_jd_intent,
            candidate_skills_map={},
        )
        assert results == [], (
            f"Expected [] for empty skills map, got {results}"
        )

    def test_empty_required_skills_returns_empty(
        self, ontology_path, mock_skill_graph
    ):
        """retrieve() returns [] when jd_intent.required_skills is empty."""
        empty_skills_jd = JDIntent(
            required_skills=[],   # Empty — OntologyPath should short-circuit
            nice_to_have_skills=[],
            disqualifier_skills=[],
            expanded_required=[],
            yoe_min=5.0, yoe_max=9.0, yoe_ideal_min=5.0, yoe_ideal_max=9.0,
            preferred_locations=["noida"],
            relocation_accepted=True,
            disqualify_consulting_only=True,
            disqualify_no_production=True,
            raw_text="",
        )
        results = ontology_path.retrieve(
            empty_skills_jd,
            candidate_skills_map={"CAND_0000001": frozenset({"python"})},
        )
        assert results == [], (
            f"Expected [] when required_skills is empty, got {results}"
        )

    def test_raises_type_error_on_non_dict_skills_map(
        self, ontology_path, mock_jd_intent
    ):
        """TypeError raised when candidate_skills_map is not a dict."""
        with pytest.raises(TypeError):
            ontology_path.retrieve(
                mock_jd_intent,
                candidate_skills_map=["not", "a", "dict"],  # type: ignore
            )

    def test_path_name_constant(self):
        """OntologyPath.PATH_NAME is 'ontology' (used by RRF for 1.3x bonus)."""
        assert OntologyPath.PATH_NAME == "ontology"

    def test_build_skills_map_static(self):
        """
        OntologyPath.build_skills_map() returns correct dict from feature vectors.
        Verifies the static helper used in pipeline/runner.py.
        """
        # We need minimal CandidateFeatureVector objects.
        # Only candidate_id and skill_names_lower are used by build_skills_map.
        mock_fv1 = MagicMock()
        mock_fv1.candidate_id = "CAND_0000001"
        mock_fv1.skill_names_lower = frozenset({"faiss", "python", "embeddings"})

        mock_fv2 = MagicMock()
        mock_fv2.candidate_id = "CAND_0000002"
        mock_fv2.skill_names_lower = frozenset({"recommendation systems", "xgboost"})

        result = OntologyPath.build_skills_map([mock_fv1, mock_fv2])

        assert result == {
            "CAND_0000001": frozenset({"faiss", "python", "embeddings"}),
            "CAND_0000002": frozenset({"recommendation systems", "xgboost"}),
        }

    # ── KEY ACCEPTANCE TEST ───────────────────────────────────────────────

    @pytest.mark.tier5
    def test_tier5_recall_recsys_candidate_rescued(
        self, mock_skill_graph, mock_jd_intent
    ):
        """
        KEY ACCEPTANCE TEST: Tier-5 domain-transfer rescue.

        The problem being solved:
          A recommendation-systems engineer at Swiggy has skills:
            {recommendation systems, xgboost, python, a/b testing, clickthrough}
          The JD requires:
            {information retrieval, embeddings, ranking, evaluation framework}
          There is ZERO direct keyword overlap. FAISS (semantic) would score them
          low because their profile text is about RecSys, not IR. BM25 would also
          score them low because none of the JD keywords appear in their profile.

          The ontology domain-transfer edge:
            "recommendation systems" → "information retrieval"
            "a/b testing"           → "evaluation framework"
          means this candidate IS a genuine fit — they've done the same work
          under different industry nomenclature.

        What we're testing:
          OntologyPath correctly passes CAND_TIER5's skills to the SkillGraph,
          the graph returns them with a non-zero score, and they appear in the
          final RetrievalResult list.

        Acceptance: CAND_TIER5 is in results and has path_score > 0.
        """
        # Configure SkillGraph to rescue the Tier-5 candidate
        self._set_graph_return(
            mock_skill_graph,
            [
                ("CAND_0000031", 0.75),   # Tier-5 RecSys engineer — rescued
            ],
        )

        path = OntologyPath(skill_graph=mock_skill_graph)

        tier5_skills = frozenset({
            "recommendation systems",   # → "information retrieval" via transfer
            "xgboost",                  # → "learning to rank" via transfer
            "python",
            "a/b testing",              # → "evaluation framework" via transfer
            "clickthrough",
        })
        ir_candidate_skills = frozenset({
            "faiss",               # Direct match — would also appear in FAISS/BM25
            "pinecone",
            "embeddings",
            "python",
        })
        candidate_skills_map = {
            "CAND_0000031": tier5_skills,    # Tier-5 rescue target
            "CAND_0000043": ir_candidate_skills,  # Already findable via other paths
        }

        results = path.retrieve(
            mock_jd_intent,
            candidate_skills_map=candidate_skills_map,
            top_k=20,
        )
        result_ids = [r.candidate_id for r in results]

        assert "CAND_0000031" in result_ids, (
            "FAIL: Tier-5 RecSys candidate (CAND_0000031) NOT rescued by OntologyPath. "
            "Domain-transfer graph should map 'recommendation systems' → "
            "'information retrieval' to surface this candidate. "
            f"Got results: {result_ids}"
        )

        tier5_result = next(r for r in results if r.candidate_id == "CAND_0000031")
        assert tier5_result.path_score > 0.0, (
            "FAIL: Tier-5 candidate has path_score = 0.0 — should be > 0.0"
        )
        assert tier5_result.path_name == "ontology"

        # Verify rank is assigned (1-indexed)
        assert tier5_result.rank_in_path >= 1


# ─────────────────────────────────────────────────────────────────────────────
# 4. TrajectoryPath Tests
#    Pure numpy — no external library needed.
# ─────────────────────────────────────────────────────────────────────────────

class TestTrajectoryPath:
    """
    Unit tests for retrieval/trajectory_path.py.

    Uses in-memory numpy arrays. No disk I/O. No external dependencies.

    Column layout (from trajectory_path.py constants):
        Col 0: promotions_per_year  (float >= 0)
        Col 1: years_of_experience  (float >= 0)
        Col 2: has_product_co       (0.0 or 1.0)
        Col 3: is_ic_riser          (0.0 or 1.0)
        Col 4: consulting_only      (0.0 or 1.0)
    """

    TOP_K: int = config.TRAJECTORY_PATH_TOP_K

    @pytest.fixture(scope="class")
    def scored_trajectory_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Controlled trajectory data with 5 candidates covering all key scenarios.

        Candidate profiles:
          CAND_0000031 — IC-riser, product-co, good YOE, active promotions
                         → Expected: high score
          CAND_0000043 — product-co, good YOE, moderate promotions
                         → Expected: medium-high score
          CAND_0000002 — consulting-only (Wipro), stagnant
                         → Expected: penalised score (× 0.35)
          CAND_0000010 — under-experienced (4yr), product-co, moderate trajectory
                         → Expected: lower due to YOE penalty
          CAND_0000005 — accountant, no product-co, consulting-only, no promotions
                         → Expected: lowest score
        """
        data = np.array(
            [
                # prom/yr  YOE   prod_co  ic_riser  consult_only
                [0.75,     6.0,  1.0,     1.0,      0.0],   # CAND_0000031 — IC riser
                [0.33,     8.0,  1.0,     0.0,      0.0],   # CAND_0000043 — product-co
                [0.00,     12.0, 0.0,     0.0,      1.0],   # CAND_0000002 — consulting only
                [0.25,     4.0,  1.0,     0.0,      0.0],   # CAND_0000010 — junior
                [0.00,     11.0, 0.0,     0.0,      1.0],   # CAND_0000005 — accountant
            ],
            dtype=np.float32,
        )
        ids = np.array(
            ["CAND_0000031", "CAND_0000043", "CAND_0000002",
             "CAND_0000010", "CAND_0000005"],
            dtype=object,
        )
        return data, ids

    @pytest.fixture(scope="class")
    def trajectory_path(self, scored_trajectory_data) -> TrajectoryPath:
        """TrajectoryPath with pre-loaded in-memory numpy data."""
        data, ids = scored_trajectory_data
        return TrajectoryPath(trajectory_data=data, candidate_ids=ids)

    # ── Structural tests ──────────────────────────────────────────────────

    def test_returns_at_most_top_k(self, trajectory_path, mock_jd_intent):
        """retrieve() returns ≤ TRAJECTORY_PATH_TOP_K results."""
        results = trajectory_path.retrieve(mock_jd_intent, top_k=self.TOP_K)
        assert len(results) <= self.TOP_K

    def test_path_name_is_trajectory(self, trajectory_path, mock_jd_intent):
        """All results have path_name='trajectory'."""
        results = trajectory_path.retrieve(mock_jd_intent)
        assert all(r.path_name == "trajectory" for r in results), (
            f"Non-trajectory path names found: "
            f"{[r.path_name for r in results if r.path_name != 'trajectory']}"
        )

    def test_rank_in_path_sequential(self, trajectory_path, mock_jd_intent):
        """rank_in_path is 1-indexed sequential."""
        results = trajectory_path.retrieve(mock_jd_intent)
        expected = list(range(1, len(results) + 1))
        assert [r.rank_in_path for r in results] == expected

    def test_scores_in_range(self, trajectory_path, mock_jd_intent):
        """All path_score values are in [0.0, 1.0]."""
        results = trajectory_path.retrieve(mock_jd_intent)
        out_of_range = [r for r in results if not 0.0 <= r.path_score <= 1.0]
        assert not out_of_range, (
            f"Scores out of [0,1]: "
            f"{[(r.candidate_id, r.path_score) for r in out_of_range]}"
        )

    def test_results_sorted_descending(self, trajectory_path, mock_jd_intent):
        """Results are sorted by trajectory score descending."""
        results = trajectory_path.retrieve(mock_jd_intent)
        for i in range(len(results) - 1):
            assert results[i].path_score >= results[i + 1].path_score, (
                f"Not sorted at index {i}: "
                f"{results[i].path_score} < {results[i + 1].path_score}"
            )

    def test_candidate_ids_in_results(self, trajectory_path, mock_jd_intent):
        """All candidate_ids in results are valid CAND_XXXXXXX format."""
        results = trajectory_path.retrieve(mock_jd_intent)
        bad = [r.candidate_id for r in results if not _CAND_ID_RE.match(r.candidate_id)]
        assert not bad, f"Invalid candidate_id formats: {bad}"

    def test_path_name_constant(self):
        """TrajectoryPath.PATH_NAME is 'trajectory'."""
        assert TrajectoryPath.PATH_NAME == "trajectory"

    # ── Business logic tests ──────────────────────────────────────────────

    def test_ic_riser_ranks_above_stagnant_consulting_only(
        self, trajectory_path, mock_jd_intent
    ):
        """
        CAND_0000031 (IC-riser, product-co, 0.75 promotions/yr) must rank
        higher than CAND_0000002 (consulting-only, 0 promotions/yr).

        This is the primary JD signal: 'Title-chasers' and consulting-only
        backgrounds are explicitly penalised. IC-risers at product companies
        are the target profile.
        """
        results = trajectory_path.retrieve(mock_jd_intent)
        result_map = {r.candidate_id: r for r in results}

        assert "CAND_0000031" in result_map, "IC-riser (CAND_0000031) not in results"
        assert "CAND_0000002" in result_map, "Consulting-only (CAND_0000002) not in results"

        ic_riser_score   = result_map["CAND_0000031"].path_score
        consulting_score = result_map["CAND_0000002"].path_score

        assert ic_riser_score > consulting_score, (
            f"FAIL: IC-riser ({ic_riser_score:.4f}) should score higher than "
            f"consulting-only ({consulting_score:.4f}). "
            "Consulting penalty (× 0.35) should strongly penalise CAND_0000002."
        )

    def test_consulting_only_penalty_is_significant(
        self, trajectory_path, mock_jd_intent
    ):
        """
        Consulting-only candidates receive a × 0.35 penalty (config.CONSULTING_ONLY_PENALTY).
        Their score should be less than half of the top non-consulting candidate.
        """
        results = trajectory_path.retrieve(mock_jd_intent)
        result_map = {r.candidate_id: r for r in results}

        if "CAND_0000002" not in result_map:
            pytest.skip("CAND_0000002 not in results (score may be 0)")

        consulting_score = result_map["CAND_0000002"].path_score
        non_consulting = [r for r in results if r.candidate_id != "CAND_0000002"]
        if not non_consulting:
            pytest.skip("No non-consulting candidates to compare")
        top_non_consulting = non_consulting[0].path_score

        assert consulting_score < top_non_consulting * 0.6, (
            f"FAIL: Consulting-only score ({consulting_score:.4f}) too close to "
            f"non-consulting score ({top_non_consulting:.4f}). "
            "CONSULTING_ONLY_PENALTY (× 0.35) should create a large gap."
        )

    def test_raises_on_missing_candidate_ids(self):
        """
        ValueError raised when trajectory_data is supplied without candidate_ids.
        """
        data = np.zeros((3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="candidate_ids"):
            TrajectoryPath(trajectory_data=data, candidate_ids=None)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SignalPath Tests
#    Pure numpy — no external library, no JDIntent.
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalPath:
    """
    Unit tests for retrieval/signal_path.py.

    SignalPath is unique: retrieve() takes NO JDIntent — engagement is JD-agnostic.

    Feature column order (from signal_path.py constants):
        Col 0: recency_score          exp(-λ × days_since_active)  [0,1]
        Col 1: response_rate          recruiter_response_rate       [0,1]
        Col 2: open_to_work           1.0 or 0.0
        Col 3: notice_period_score    1.0 (≤30d) to 0.0 (>90d)
        Col 4: github_activity        github_score/100 or DEFAULT   [0,1]
        Col 5: profile_completeness   completeness/100              [0,1]
        Col 6: interview_completion   rate                          [0,1]
    """

    TOP_K: int = config.SIGNAL_PATH_TOP_K

    @pytest.fixture(scope="class")
    def signal_feature_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Controlled feature data for 6 candidates.

        Candidate profiles:
          CAND_0000031 — highly engaged: open, recent, responsive, short notice
                         → Expected: rank 1 or 2
          CAND_0000043 — engaged: open, recent but slower response
                         → Expected: top 3
          CAND_0000022 — moderate: not open_to_work, decent recency
                         → Expected: middle tier
          CAND_0000002 — inactive: old last_active, no OTW, slow response
                         → Expected: low rank
          CAND_0000030 — ghost: zero engagement on all signals
                         → Expected: score = 0 or near-zero (may be excluded)
          CAND_0000005 — minimal: low completeness, no github, no OTW
                         → Expected: below mid tier
        """
        data = np.array(
            [
                # recency  resp_rate  otw  notice  github  completeness  interview
                [0.95,     0.91,      1.0, 1.0,    0.33,   0.83,         0.60],  # CAND_0000031
                [0.85,     0.04,      0.0, 0.5,    0.00,   0.57,         0.72],  # CAND_0000043
                [0.70,     0.27,      1.0, 0.8,    0.50,   0.63,         0.45],  # CAND_0000022
                [0.16,     0.29,      1.0, 0.0,    0.50,   0.79,         0.74],  # CAND_0000002
                [0.01,     0.05,      0.0, 0.0,    0.50,   0.10,         0.20],  # CAND_0000030
                [0.30,     0.37,      1.0, 0.5,    0.50,   0.85,         0.37],  # CAND_0000005
            ],
            dtype=np.float32,
        )
        ids = np.array(
            ["CAND_0000031", "CAND_0000043", "CAND_0000022",
             "CAND_0000002", "CAND_0000030", "CAND_0000005"],
            dtype=object,
        )
        return data, ids

    @pytest.fixture(scope="class")
    def signal_path(self, signal_feature_data) -> SignalPath:
        """SignalPath with pre-loaded in-memory feature data."""
        data, ids = signal_feature_data
        return SignalPath(feature_data=data, candidate_ids=ids)

    # ── Structural tests ──────────────────────────────────────────────────

    def test_returns_at_most_top_k(self, signal_path):
        """retrieve() returns ≤ SIGNAL_PATH_TOP_K results."""
        results = signal_path.retrieve(top_k=self.TOP_K)
        assert len(results) <= self.TOP_K

    def test_path_name_is_signal(self, signal_path):
        """All results have path_name='signal'."""
        results = signal_path.retrieve()
        assert all(r.path_name == "signal" for r in results), (
            f"Non-signal path names: "
            f"{[r.path_name for r in results if r.path_name != 'signal']}"
        )

    def test_rank_in_path_sequential(self, signal_path):
        """rank_in_path values are 1-indexed sequential."""
        results = signal_path.retrieve()
        expected = list(range(1, len(results) + 1))
        assert [r.rank_in_path for r in results] == expected

    def test_scores_in_range(self, signal_path):
        """All path_score values are in [0.0, 1.0]."""
        results = signal_path.retrieve()
        out_of_range = [r for r in results if not 0.0 <= r.path_score <= 1.0]
        assert not out_of_range, (
            f"Scores out of [0,1]: "
            f"{[(r.candidate_id, r.path_score) for r in out_of_range]}"
        )

    def test_results_sorted_descending(self, signal_path):
        """Results are ordered by behavioral engagement score descending."""
        results = signal_path.retrieve()
        for i in range(len(results) - 1):
            assert results[i].path_score >= results[i + 1].path_score, (
                f"Not sorted at index {i}: "
                f"{results[i].path_score} < {results[i+1].path_score}"
            )

    def test_retrieve_takes_no_jd_intent(self, signal_path):
        """
        retrieve() succeeds with NO JDIntent argument — behavioral signals
        are JD-agnostic. The same candidate is equally reachable regardless
        of the JD being ranked against.
        """
        # This test would fail at type checking or runtime if JDIntent were required.
        results = signal_path.retrieve(top_k=5)
        assert isinstance(results, list), "retrieve() must return list"

    def test_path_name_constant(self):
        """SignalPath.PATH_NAME is 'signal' (used by RRF for 1.1x bonus)."""
        assert SignalPath.PATH_NAME == "signal"

    def test_raises_on_missing_candidate_ids(self):
        """ValueError raised when feature_data is supplied without candidate_ids."""
        data = np.zeros((3, 7), dtype=np.float32)
        with pytest.raises(ValueError, match="candidate_ids"):
            SignalPath(feature_data=data, candidate_ids=None)

    # ── Business logic tests ──────────────────────────────────────────────

    def test_highly_engaged_candidate_ranks_above_inactive(self, signal_path):
        """
        CAND_0000031 (recency=0.95, open_to_work, notice≤30d, response_rate=0.91)
        must rank above CAND_0000030 (recency=0.01, no OTW, zero response).

        This validates the core Signal Path value proposition: the JD says
        'a perfect-on-paper candidate who hasn't logged in for 6 months and
        has a 5% recruiter response rate is not actually available — down-weight
        them appropriately.'
        """
        results = signal_path.retrieve()
        result_map = {r.candidate_id: r for r in results}

        engaged_result = result_map.get("CAND_0000031")
        inactive_result = result_map.get("CAND_0000030")

        # Ghost candidate may be excluded (score ≤ 0) — that is ALSO acceptable
        if inactive_result is None:
            # CAND_0000030 excluded entirely — even better outcome
            assert engaged_result is not None, (
                "CAND_0000031 (highly engaged) should appear in results"
            )
            return

        assert engaged_result is not None, "CAND_0000031 not in results"
        assert engaged_result.path_score > inactive_result.path_score, (
            f"FAIL: Highly engaged CAND_0000031 ({engaged_result.path_score:.4f}) "
            f"should outrank inactive CAND_0000030 ({inactive_result.path_score:.4f})"
        )

    def test_zero_engagement_candidates_excluded_or_ranked_last(self, signal_path):
        """
        Candidates with effectively zero engagement score (≤ 0.0) should be
        excluded from results. If included, they must rank last.
        """
        results = signal_path.retrieve()

        if not results:
            pytest.skip("Empty results — no data to validate")

        # If the ghost candidate (CAND_0000030) is in results, it must be last
        ghost = next(
            (r for r in results if r.candidate_id == "CAND_0000030"), None
        )
        if ghost is not None:
            assert ghost.rank_in_path == len(results), (
                f"Ghost candidate (score={ghost.path_score:.4f}) should be last, "
                f"got rank {ghost.rank_in_path} of {len(results)}"
            )

    def test_top_k_one_returns_single_result(self, signal_path):
        """retrieve(top_k=1) returns exactly 1 result."""
        results = signal_path.retrieve(top_k=1)
        assert len(results) == 1, f"Expected 1 result, got {len(results)}"
        assert results[0].rank_in_path == 1

    def test_raises_on_invalid_top_k(self, signal_path):
        """ValueError raised for top_k < 1."""
        with pytest.raises(ValueError, match="top_k"):
            signal_path.retrieve(top_k=0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. RRF Integration Tests
#    Tests across multiple paths through RRFFusion.
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestRRFIntegration:
    """
    Integration tests for retrieval/rrf_fusion.py consuming results from
    all 5 retrieval paths.

    These tests build RetrievalResult lists by hand (no real path infrastructure)
    to focus on RRF behaviour: score accumulation, bonus multipliers, dedup,
    sort order, and the key Tier-5 recall scenario.
    """

    # ── Tier-5 recall through RRF ─────────────────────────────────────────

    @pytest.mark.tier5
    def test_ontology_only_candidate_survives_rrf(self):
        """
        KEY ACCEPTANCE TEST: End-to-end Tier-5 recall through RRF fusion.

        Scenario:
          CAND_RICH  — appears in semantic (rank 1) + keyword (rank 1)
                       → highest multi-path RRF score
          CAND_TIER5 — appears ONLY in ontology path (rank 1)
                       → gets 1.3x bonus = 1.3 / (60+1) ≈ 0.02131
                       → MUST survive in top-60 pool

        Acceptance: CAND_TIER5 in pool even though it appears in only ONE path.
        This is the critical design goal: "A candidate ranked highly in ANY
        path is probably relevant" (from rrf_fusion.py docstring).
        """
        CAND_RICH  = "CAND_0000031"
        CAND_TIER5 = "CAND_0000099"

        semantic_results  = [_make_retrieval_result(CAND_RICH,  0.95, "semantic",  1)]
        keyword_results   = [_make_retrieval_result(CAND_RICH,  0.90, "keyword",   1)]
        ontology_results  = [_make_retrieval_result(CAND_TIER5, 0.80, "ontology",  1)]

        pool = RRFFusion().fuse({
            "semantic":  semantic_results,
            "keyword":   keyword_results,
            "ontology":  ontology_results,
        })

        pool_ids = [r.candidate_id for r in pool]

        assert CAND_TIER5 in pool_ids, (
            f"FAIL: Tier-5 candidate (CAND_TIER5={CAND_TIER5}) not in RRF pool. "
            f"Ontology-only candidates with 1.3x bonus MUST survive. "
            f"Pool: {pool_ids}"
        )

        assert CAND_RICH in pool_ids, (
            f"FAIL: Multi-path candidate (CAND_RICH={CAND_RICH}) not in pool. "
            f"Pool: {pool_ids}"
        )

    @pytest.mark.tier5
    def test_ontology_bonus_contributes_meaningfully_to_rrf_score(self):
        """
        The 1.3x ontology path bonus (config.RRF_ONTOLOGY_PATH_BONUS) should
        give an ontology-only rank-1 candidate a score comparable to a
        single-path semantic-only rank-5 candidate.

        Manually verified:
          ontology rank-1 with 1.3x: 1.3/(60+1) ≈ 0.02131
          semantic rank-5:           1.0/(60+5)  ≈ 0.01538

        The ontology-only candidate should beat a mediocre semantic-only match.
        """
        CAND_ONTOLOGY_RANK1  = "CAND_0000031"
        CAND_SEMANTIC_RANK10 = "CAND_0000043"

        pool = RRFFusion().fuse({
            "ontology":  [_make_retrieval_result(CAND_ONTOLOGY_RANK1,  0.75, "ontology",  1)],
            "semantic":  [_make_retrieval_result(CAND_SEMANTIC_RANK10, 0.50, "semantic", 10)],
        })

        result_map = {r.candidate_id: r for r in pool}
        assert CAND_ONTOLOGY_RANK1  in result_map
        assert CAND_SEMANTIC_RANK10 in result_map

        expected_ontology = config.RRF_ONTOLOGY_PATH_BONUS / (config.RRF_K + 1)
        expected_semantic = 1.0 / (config.RRF_K + 10)

        assert abs(result_map[CAND_ONTOLOGY_RANK1].rrf_score - expected_ontology) < 1e-7
        assert result_map[CAND_ONTOLOGY_RANK1].rrf_score > expected_semantic, (
            f"Ontology rank-1 ({expected_ontology:.6f}) should beat "
            f"semantic rank-10 ({expected_semantic:.6f})"
        )

    # ── Multi-path ranking ────────────────────────────────────────────────

    def test_multi_path_candidate_scores_higher_than_single_path(self):
        """
        A candidate appearing in semantic + keyword + signal (3 paths) should
        have a higher RRF score than a candidate appearing in only 1 path.
        """
        CAND_MULTI  = "CAND_0000031"
        CAND_SINGLE = "CAND_0000043"

        pool = RRFFusion().fuse({
            "semantic":  [
                _make_retrieval_result(CAND_MULTI,  0.9, "semantic",  2),
                _make_retrieval_result(CAND_SINGLE, 0.7, "semantic",  5),
            ],
            "keyword":   [_make_retrieval_result(CAND_MULTI, 0.8, "keyword",   2)],
            "signal":    [_make_retrieval_result(CAND_MULTI, 0.7, "signal",    2)],
        })

        result_map = {r.candidate_id: r for r in pool}
        assert CAND_MULTI  in result_map, "Multi-path candidate missing from pool"
        assert CAND_SINGLE in result_map, "Single-path candidate missing from pool"

        assert result_map[CAND_MULTI].rrf_score > result_map[CAND_SINGLE].rrf_score, (
            f"Multi-path ({result_map[CAND_MULTI].rrf_score:.5f}) should beat "
            f"single-path ({result_map[CAND_SINGLE].rrf_score:.5f})"
        )

    # ── Pool properties ───────────────────────────────────────────────────

    def test_pool_size_within_rrf_pool_size_limit(self):
        """Pool size ≤ config.RRF_POOL_SIZE (60) even with many input candidates."""
        # Create many unique candidates across paths
        semantic  = [_make_retrieval_result(f"CAND_{i:07d}", 0.9, "semantic",  i + 1) for i in range(1, 26)]
        keyword   = [_make_retrieval_result(f"CAND_{i:07d}", 0.8, "keyword",   i + 1) for i in range(26, 51)]
        ontology  = [_make_retrieval_result(f"CAND_{i:07d}", 0.7, "ontology",  i + 1) for i in range(51, 71)]
        trajectory = [_make_retrieval_result(f"CAND_{i:07d}", 0.6, "trajectory", i + 1) for i in range(71, 86)]
        signal    = [_make_retrieval_result(f"CAND_{i:07d}", 0.5, "signal",    i + 1) for i in range(86, 101)]

        pool = RRFFusion().fuse({
            "semantic":   semantic,
            "keyword":    keyword,
            "ontology":   ontology,
            "trajectory": trajectory,
            "signal":     signal,
        })
        assert len(pool) <= config.RRF_POOL_SIZE, (
            f"Pool size {len(pool)} exceeds RRF_POOL_SIZE={config.RRF_POOL_SIZE}"
        )

    def test_no_duplicate_candidate_ids_in_pool(self):
        """
        A candidate appearing in multiple paths appears only once in the pool
        (cross-path deduplication via RRF score accumulation).
        """
        CAND_A = "CAND_0000001"
        pool = RRFFusion().fuse({
            "semantic":  [_make_retrieval_result(CAND_A, 0.9, "semantic",  1)],
            "keyword":   [_make_retrieval_result(CAND_A, 0.8, "keyword",   1)],
            "ontology":  [_make_retrieval_result(CAND_A, 0.7, "ontology",  1)],
        })
        pool_ids = [r.candidate_id for r in pool]
        count = pool_ids.count(CAND_A)
        assert count == 1, (
            f"FAIL: {CAND_A} appears {count}× in pool (expected 1 — dedup required)"
        )

    def test_pool_sorted_by_rrf_score_descending(self):
        """Pool is sorted by rrf_score descending, candidate_id ascending for ties."""
        pool = RRFFusion().fuse({
            "semantic":  [
                _make_retrieval_result("CAND_0000003", 0.9, "semantic", 1),
                _make_retrieval_result("CAND_0000001", 0.7, "semantic", 2),
                _make_retrieval_result("CAND_0000002", 0.5, "semantic", 3),
            ],
        })
        for i in range(len(pool) - 1):
            assert pool[i].rrf_score >= pool[i + 1].rrf_score, (
                f"Pool not sorted descending at index {i}: "
                f"{pool[i].rrf_score} < {pool[i + 1].rrf_score}"
            )

    def test_cross_encoder_score_initialised_to_zero(self):
        """
        All RRFResult.cross_encoder_score values are 0.0 on output from RRF.
        cross_encoder.py populates this field in the next pipeline stage.
        """
        pool = RRFFusion().fuse({
            "semantic": [_make_retrieval_result("CAND_0000001", 0.9, "semantic", 1)],
        })
        assert all(r.cross_encoder_score == 0.0 for r in pool), (
            "All RRFResult.cross_encoder_score should be 0.0 on exit from RRF. "
            "cross_encoder.py populates this field."
        )

    def test_paths_present_correctly_reflects_multi_path_membership(self):
        """
        RRFResult.paths_present lists all paths where each candidate appeared.
        A candidate in semantic + keyword should have paths_present = ["keyword", "semantic"].
        """
        CAND_A = "CAND_0000001"
        pool = RRFFusion().fuse({
            "semantic": [_make_retrieval_result(CAND_A, 0.9, "semantic", 1)],
            "keyword":  [_make_retrieval_result(CAND_A, 0.8, "keyword",  1)],
        })
        result = next(r for r in pool if r.candidate_id == CAND_A)
        assert sorted(result.paths_present) == ["keyword", "semantic"], (
            f"paths_present = {result.paths_present}, "
            f"expected ['keyword', 'semantic']"
        )

    def test_empty_path_results_silently_skipped(self):
        """
        Passing empty lists for some paths doesn't crash RRF.
        Only non-empty paths contribute to scores.
        """
        pool = fuse_results(
            semantic=[_make_retrieval_result("CAND_0000001", 0.9, "semantic", 1)],
            keyword=[],      # Empty — should be silently skipped
            ontology=None,   # None — should be silently skipped
        )
        assert len(pool) == 1
        assert pool[0].candidate_id == "CAND_0000001"

    def test_signal_path_bonus_applied_correctly(self):
        """
        Signal path bonus (config.RRF_SIGNAL_PATH_BONUS = 1.1) is applied
        so that a signal-only rank-1 scores higher than semantic-only rank-1
        per-path-bonus math:
            signal:   1.1 / (60 + 1) ≈ 0.01803
            This is correct per rrf_fusion.py: bonus / (RRF_K + rank)
        """
        CAND_SIGNAL   = "CAND_0000001"
        CAND_SEMANTIC = "CAND_0000002"

        pool = RRFFusion().fuse({
            "signal":   [_make_retrieval_result(CAND_SIGNAL,   0.9, "signal",   1)],
            "semantic": [_make_retrieval_result(CAND_SEMANTIC, 0.9, "semantic", 1)],
        })
        result_map = {r.candidate_id: r for r in pool}

        expected_signal   = config.RRF_SIGNAL_PATH_BONUS / (config.RRF_K + 1)
        expected_semantic = 1.0 / (config.RRF_K + 1)

        assert abs(result_map[CAND_SIGNAL].rrf_score   - expected_signal)   < 1e-7
        assert abs(result_map[CAND_SEMANTIC].rrf_score - expected_semantic) < 1e-7
        assert result_map[CAND_SIGNAL].rrf_score > result_map[CAND_SEMANTIC].rrf_score, (
            f"Signal-path rank-1 ({expected_signal:.6f}) should beat "
            f"semantic-path rank-1 ({expected_semantic:.6f}) due to 1.1x bonus"
        )

    def test_rrf_tie_break_is_candidate_id_ascending(self):
        """
        When two candidates have identical RRF scores, tie-break is
        candidate_id ascending (lexicographic). Matches spec requirement.
        """
        pool = fuse_results(
            semantic=[
                _make_retrieval_result("CAND_0000099", 0.9, "semantic", 5),
                _make_retrieval_result("CAND_0000001", 0.9, "semantic", 5),
            ]
        )
        ids = [r.candidate_id for r in pool]
        # Both have same rank in same path → same RRF score → tie-break ascending
        assert ids.index("CAND_0000001") < ids.index("CAND_0000099"), (
            f"Tie-break should be ascending by candidate_id, got: {ids}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 7. RetrievalResult and RRFResult contract tests
#    Validates that schema dataclasses behave as expected.
# ─────────────────────────────────────────────────────────────────────────────

class TestOutputContractSchemas:
    """
    Tests that RetrievalResult and RRFResult dataclasses produced by the
    retrieval layer satisfy the contracts consumed by rrf_fusion.py and
    scoring/cross_encoder.py.
    """

    def test_retrieval_result_fields(self):
        """RetrievalResult holds all required fields with correct types."""
        r = RetrievalResult(
            candidate_id="CAND_0000031",
            path_score=0.85,
            path_name="semantic",
            rank_in_path=1,
        )
        assert r.candidate_id == "CAND_0000031"
        assert r.path_score   == 0.85
        assert r.path_name    == "semantic"
        assert r.rank_in_path == 1

    def test_rrf_result_cross_encoder_score_defaults_zero(self):
        """RRFResult.cross_encoder_score defaults to 0.0 (as required by pipeline)."""
        r = RRFResult(
            candidate_id="CAND_0000031",
            rrf_score=0.049,
            paths_present=["semantic", "keyword"],
        )
        assert r.cross_encoder_score == 0.0, (
            "cross_encoder_score must default to 0.0 — "
            "scoring/cross_encoder.py sets this field later."
        )

    def test_rrf_result_cross_encoder_score_is_mutable(self):
        """
        RRFResult.cross_encoder_score can be mutated in-place.
        cross_encoder.py does in-place mutation — this must work.
        """
        r = RRFResult(
            candidate_id="CAND_0000031",
            rrf_score=0.049,
            paths_present=["semantic"],
        )
        r.cross_encoder_score = 0.876
        assert r.cross_encoder_score == 0.876

    def test_rrf_result_paths_present_is_sorted_list(self, mock_jd_intent):
        """
        RRFResult.paths_present from rrf_fusion.py is a sorted list.
        Tests that confirm paths_present can use index lookups or sorted comparisons.
        """
        pool = RRFFusion().fuse({
            "semantic": [_make_retrieval_result("CAND_0000001", 0.9, "semantic", 1)],
            "keyword":  [_make_retrieval_result("CAND_0000001", 0.8, "keyword",  1)],
        })
        result = pool[0]
        assert result.paths_present == sorted(result.paths_present), (
            f"paths_present should be sorted: {result.paths_present}"
        )