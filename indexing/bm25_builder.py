from __future__ import annotations

import re
import json
import math
import heapq
import pickle
import logging
import collections
from pathlib import Path
from typing import Optional

import config
from pipeline.schemas import CandidateFeatureVector

logger = logging.getLogger(__name__)

# Compiled once at import time — avoids recompiling on every _tokenize() call
_TOKEN_RE = re.compile(r'\b[a-z0-9][a-z0-9\+\#\.]*\b')


# ── Ontology loader ───────────────────────────────────────────────────────────

def _load_ontology(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        logger.warning("skill_map.json not found at '%s'. Ontology expansion disabled.", path)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        ontology = {}
        for key, val in raw.items():
            if isinstance(val, list):
                ontology[key.lower().strip()] = [
                    v.lower().strip() for v in val if isinstance(v, str)
                ]
        logger.info("Loaded ontology: %d skill entries from '%s'", len(ontology), path)
        return ontology
    except Exception as e:
        logger.warning("Failed to parse skill_map.json: %s. Ontology expansion disabled.", e)
        return {}


ONTOLOGY: dict[str, list[str]] = _load_ontology(config.SKILL_MAP_PATH)


# ── Text utilities ────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def _expand_query(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    seen = set(tokens)
    for token in tokens:
        if token in ONTOLOGY:
            for synonym in ONTOLOGY[token]:
                if synonym not in seen:
                    expanded.append(synonym)
                    seen.add(synonym)
    return expanded


def _build_candidate_text(c: CandidateFeatureVector) -> str:
    parts: list[str] = []

    # Current role — repeated for higher weight in BM25 term frequency
    parts.append(c.current_title)
    parts.append(c.current_title)
    parts.append(c.current_company)
    parts.append(c.current_industry)

    if c.headline:
        parts.append(c.headline)
    if c.summary:
        parts.append(c.summary)

    # Skills repeated by proficiency so expert skills score higher
    repeat_map = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}
    for skill in c.skills:
        repeats = repeat_map.get(skill.proficiency, 1)
        parts.extend([skill.name_raw] * repeats)

    for job in c.career_history:
        parts.append(job.title)
        parts.append(job.company)
        parts.append(job.industry)
        if job.description:
            parts.append(job.description)

    for edu in c.education:
        parts.append(f"{edu.degree} {edu.field_of_study} {edu.institution}")

    parts.append(c.location)

    return " ".join(p for p in parts if p and p.strip())


# ── BM25 core ─────────────────────────────────────────────────────────────────

class _BM25Core:
    """Pure Okapi BM25 over a pre-tokenized corpus."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(corpus)
        self.avg_dl = sum(len(d) for d in corpus) / self.n if self.n else 1.0
        self.dl = [len(d) for d in corpus]

        # inv[token][doc_idx] = term frequency
        self.inv: dict[str, dict[int, int]] = {}
        # Precomputed IDF per token — eliminates redundant log() on every search call
        self.idf: dict[str, float] = {}

        self._build(corpus)

    def _build(self, corpus: list[list[str]]) -> None:
        n = self.n
        inv = self.inv

        for idx, doc in enumerate(corpus):
            for token, freq in collections.Counter(doc).items():
                if token not in inv:
                    inv[token] = {}
                inv[token][idx] = freq

        # Precompute IDF for every token now, once, rather than per search
        self.idf = {
            token: math.log((n - len(postings) + 0.5) / (len(postings) + 0.5) + 1.0)
            for token, postings in inv.items()
        }

    def score(self, query_tokens: list[str]) -> dict[int, float]:
        scores: dict[int, float] = {}
        k1 = self.k1
        b = self.b
        avg_dl = self.avg_dl
        dl = self.dl
        inv = self.inv
        idf = self.idf
        k1_plus1 = k1 + 1

        for token in query_tokens:
            postings = inv.get(token)
            if postings is None:
                continue
            token_idf = idf[token]
            for doc_idx, tf in postings.items():
                tf_norm = (tf * k1_plus1) / (
                    tf + k1 * (1 - b + b * (dl[doc_idx] / avg_dl))
                )
                # Plain dict get+set is faster than defaultdict in a tight loop
                scores[doc_idx] = scores.get(doc_idx, 0.0) + token_idf * tf_norm

        return scores


# ── Public class ──────────────────────────────────────────────────────────────

class BM25Index:
    """
    Keyword retrieval index over CandidateFeatureVector.

    API mirrors FaissIndex:
        .build(candidates, save=True)
        .load()
        .search(query_text, top_k) → list[tuple[str, float]]
    """

    def __init__(self, index_path: Path = config.BM25_INDEX_PATH) -> None:
        self.index_path = index_path
        self._core: Optional[_BM25Core] = None
        self._id_map: Optional[list[str]] = None

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, candidates: list[CandidateFeatureVector], save: bool = True) -> None:
        if not candidates:
            raise ValueError("candidates list is empty — nothing to index.")

        logger.info("Building BM25 index for %d candidates...", len(candidates))

        corpus = [_tokenize(_build_candidate_text(c)) for c in candidates]
        self._id_map = [c.candidate_id for c in candidates]
        self._core = _BM25Core(corpus)

        logger.info(
            "BM25 index built: %d candidates, %d unique tokens, avg_dl=%.1f",
            len(candidates),
            len(self._core.inv),
            self._core.avg_dl,
        )

        if save:
            self._save()

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index not found at '{self.index_path}'. Run .build() first."
            )
        with open(self.index_path, "rb") as f:
            payload = pickle.load(f)
        self._core = payload["core"]
        self._id_map = payload["id_map"]
        logger.info(
            "Loaded BM25 index: %d candidates from '%s'",
            len(self._id_map), self.index_path,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_text: str,
        top_k: int = config.KEYWORD_PATH_TOP_K,
        expand: bool = True,
    ) -> list[tuple[str, float]]:
        """
        Keyword search with optional ontology expansion.

        Returns:
            list of (candidate_id, bm25_score) sorted descending
        """
        self._require_loaded()

        tokens = _tokenize(query_text)
        if expand:
            tokens = _expand_query(tokens)

        raw_scores = self._core.score(tokens)

        # heapq.nlargest is O(n log k) vs sorted O(n log n) — matters when k << n
        ranked = heapq.nlargest(top_k, raw_scores.items(), key=lambda x: x[1])
        results = [(self._id_map[idx], float(score)) for idx, score in ranked]

        logger.debug("BM25 top result: %s", results[0] if results else None)
        return results

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._core is not None and self._id_map is not None

    @property
    def vocab_size(self) -> int:
        self._require_loaded()
        return len(self._core.inv)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"core": self._core, "id_map": self._id_map}, f)
        logger.info("Saved BM25 index → %s", self.index_path)

    # ── Guards ────────────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError(
                "BM25Index not loaded. Call .build() or .load() first."
            )

    def __repr__(self) -> str:
        status = (
            f"{len(self._id_map)} candidates, {self.vocab_size} tokens"
            if self.is_loaded else "not loaded"
        )
        return f"BM25Index({status})"