from __future__ import annotations

import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import config

from pipeline.schemas import CandidateFeatureVector

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME    = config.BI_ENCODER_MODEL
EMBEDDING_DIM = config.EMBEDDING_DIM
N_CLUSTERS    = config.FAISS_NLIST
N_PROBE       = config.FAISS_NPROBE
BATCH_SIZE    = 128
MAX_TEXT_CHARS = 2000

INDEX_PATH  = config.FAISS_INDEX_PATH
ID_MAP_PATH = config.FAISS_ID_MAP_PATH

# Proficiency buckets as frozensets — O(1) lookup vs tuple/list `in`
_EXPERT_SET       = frozenset(("advanced", "expert"))
_INTERMEDIATE_SET = frozenset(("intermediate",))


class FaissIndex:
    """
    Dense semantic FAISS index over a CandidateFeatureVector list.

    Usage:
        fi = FaissIndex()
        fi.build(candidates, save=True)

        fi = FaissIndex()
        fi.load()
        results = fi.search(query_text, top_k=100)
        # → [("CAND_0000001", 0.91), ("CAND_0000042", 0.87), ...]
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        index_path: Path = INDEX_PATH,
        id_map_path: Path = ID_MAP_PATH,
    ) -> None:
        self.model_name  = model_name
        self.index_path  = index_path
        self.id_map_path = id_map_path

        self._model: Optional[SentenceTransformer] = None  # lazy loaded
        self._index: Optional[faiss.Index] = None
        self._id_map: Optional[list[str]] = None           # position → candidate_id

    def build(self, candidates: list[CandidateFeatureVector], save: bool = True) -> None:
        """
        Encode all candidates and build FAISS index.

        Automatically selects:
          - IndexIVFFlat when len(candidates) >= N_CLUSTERS  [production]
          - IndexFlatIP  when len(candidates) <  N_CLUSTERS  [dev/testing]
        """
        if not candidates:
            raise ValueError("candidates list is empty — nothing to index.")

        logger.info("Building FAISS index for %d candidates...", len(candidates))

        texts  = [self._build_embedding_text(c) for c in candidates]
        id_map = [c.candidate_id for c in candidates]

        embeddings = self._encode_batch(texts)

        if len(candidates) >= N_CLUSTERS:
            index = self._build_ivf_index(embeddings)
            logger.info("Built IVF256 index (%d vectors)", index.ntotal)
        else:
            index = self._build_flat_index(embeddings)
            logger.warning(
                "Candidate pool (%d) < N_CLUSTERS (%d). "
                "Using IndexFlatIP (exact search). Switch to IVF for production.",
                len(candidates), N_CLUSTERS,
            )

        self._index  = index
        self._id_map = id_map

        if save:
            self._save(index, id_map)

    def load(self) -> None:
        """Load pre-built index and id_map from disk."""
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{self.index_path}'. Run .build() first."
            )
        self._index = faiss.read_index(str(self.index_path))
        self._index.nprobe = N_PROBE
        with open(self.id_map_path, "rb") as f:
            self._id_map = pickle.load(f)
        logger.info(
            "Loaded FAISS index: %d vectors from '%s'",
            self._index.ntotal, self.index_path,
        )

    def search(self, query_text: str, top_k: int = 100) -> list[tuple[str, float]]:
        """
        Semantic search over the index.

        Returns:
            list of (candidate_id, cosine_score) sorted by score descending
        """
        self._require_loaded()

        # Bind model to local — avoids repeated _get_model() attr chain in hot path
        model = self._get_model()
        query_vec = model.encode(
            [query_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # encode() already returns np.ndarray — reshape in-place, no copy
        query_vec = query_vec.reshape(1, -1).astype(np.float32)

        scores, indices = self._index.search(query_vec, top_k)

        results = []
        id_map = self._id_map
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS pads with -1 when fewer results exist
                continue
            results.append((id_map[idx], float(score)))

        logger.debug("semantic_search top result: %s", results[0] if results else None)
        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None and self._id_map is not None

    @property
    def total_vectors(self) -> int:
        self._require_loaded()
        return self._index.ntotal

    # ── Embedding text construction ───────────────────────────────────────────

    @staticmethod
    def _build_embedding_text(c: CandidateFeatureVector) -> str:
        """
        Build a rich text representation of a candidate using ALL available fields.

        Sections (in semantic importance order):
          1. Current role + headline
          2. Summary
          3. Skills (name + proficiency)
          4. Career history (title + company + industry + description)
          5. Education (institution + degree + field)
          6. Location + country
        """
        parts: list[str] = []

        parts.append(f"{c.current_title} at {c.current_company}.")
        parts.append(c.headline)

        if c.summary:
            parts.append(c.summary)

        # Single pass over skills — avoids 3 separate list comprehensions
        if c.skills:
            advanced, intermediate, beginner = [], [], []
            for s in c.skills:
                if s.proficiency in _EXPERT_SET:
                    advanced.append(s.name_raw)
                elif s.proficiency in _INTERMEDIATE_SET:
                    intermediate.append(s.name_raw)
                else:
                    beginner.append(s.name_raw)

            if advanced:
                parts.append("Expert skills: " + ", ".join(advanced) + ".")
            if intermediate:
                parts.append("Intermediate skills: " + ", ".join(intermediate) + ".")
            if beginner:
                parts.append("Familiar with: " + ", ".join(beginner) + ".")

        for job in c.career_history:
            job_parts = [f"{job.title} at {job.company} ({job.industry})"]
            if job.description:
                job_parts.append(job.description)
            parts.append(" — ".join(job_parts))

        for edu in c.education:
            parts.append(f"{edu.degree} in {edu.field_of_study} from {edu.institution}.")

        parts.append(f"Location: {c.location}, {c.country}.")

        full_text = " ".join(p.strip() for p in parts if p and p.strip())
        return full_text[:MAX_TEXT_CHARS]

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """Encode texts in batches, return float32 normalised embeddings."""
        model = self._get_model()
        logger.info("Encoding %d candidates (batch_size=%d)...", len(texts), BATCH_SIZE)
        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,   # ensures ndarray directly — skips internal tensor copy
        )
        # encode() with convert_to_numpy=True already returns float32 ndarray
        return embeddings.astype(np.float32, copy=False)

    # ── Index constructors ────────────────────────────────────────────────────

    @staticmethod
    def _build_ivf_index(embeddings: np.ndarray) -> faiss.IndexIVFFlat:
        """IVF256 index — fast approximate search for large pools."""
        quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
        index = faiss.IndexIVFFlat(
            quantizer, EMBEDDING_DIM, N_CLUSTERS, faiss.METRIC_INNER_PRODUCT
        )
        index.train(embeddings)
        index.add(embeddings)
        index.nprobe = N_PROBE
        return index

    @staticmethod
    def _build_flat_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """Exact flat index — for small datasets only."""
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        index.add(embeddings)
        return index

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self, index: faiss.Index, id_map: list[str]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        with open(self.id_map_path, "wb") as f:
            pickle.dump(id_map, f)
        logger.info("Saved index → %s  |  id_map → %s", self.index_path, self.id_map_path)

    # ── Model cache ───────────────────────────────────────────────────────────

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load and cache the sentence transformer model."""
        if self._model is None:
            logger.info("Loading sentence transformer: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    # ── Guards ────────────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Index not loaded. Call .build() or .load() first.")

    def __repr__(self) -> str:
        status = f"{self._index.ntotal} vectors" if self.is_loaded else "not loaded"
        return f"FaissIndex(model={self.model_name}, status={status})"