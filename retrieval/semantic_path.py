from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

import config
from pipeline.schemas import JDIntent, RetrievalResult

logger = logging.getLogger(__name__)

# Optional FAISS import — raises clear error at retrieve() time, not at import
try:
    import faiss as _faiss          # type: ignore[import]
    _FAISS_AVAILABLE = True
    logger.debug("faiss-cpu imported successfully.")
except ImportError:
    _faiss = None                   # type: ignore[assignment]
    _FAISS_AVAILABLE = False
    logger.warning(
        "faiss-cpu is not installed. SemanticPath.retrieve() will raise "
        "RuntimeError. Install with: pip install faiss-cpu==1.8.0"
    )


# SemanticPath
class SemanticPath:
    PATH_NAME: str = "semantic"

    def __init__(
        self,
        index: Optional[Any] = None,
        candidate_ids: Optional[np.ndarray] = None,
        index_path: Optional[Path] = None,
        id_map_path: Optional[Path] = None,
    ) -> None:
        self._index_path: Path = index_path or config.FAISS_INDEX_PATH
        self._id_map_path: Path = id_map_path or config.FAISS_ID_MAP_PATH

        if index is not None:
            # Pre-loaded objects supplied — use directly (test / hot-reload path)
            self._index = index
            self._candidate_ids: np.ndarray = (
                candidate_ids if candidate_ids is not None
                else self._load_id_map(self._id_map_path)
            )
            self._configure_nprobe()
            self._validate_index_id_alignment()
            logger.debug(
                "SemanticPath initialised with pre-loaded index "
                "(ntotal=%d, ids=%d).",
                self._index.ntotal,
                len(self._candidate_ids),
            )
        else:
            # Defer I/O to _ensure_loaded() — caller must call from_disk()
            # or retrieve() will trigger lazy load.
            self._index = None
            self._candidate_ids = np.empty(0, dtype=object)

        # Unified assignment: True if pre-loaded index was supplied, False otherwise.
        self._loaded: bool = index is not None

    # Factory — production path 
    @classmethod
    def from_disk(
        cls,
        index_path: Optional[Path] = None,
        id_map_path: Optional[Path] = None,
    ) -> "SemanticPath":
        instance = cls(index_path=index_path, id_map_path=id_map_path)
        instance._ensure_loaded()
        return instance

    # Primary retrieve method
    def retrieve(
        self,
        jd_intent: JDIntent,
        top_k: int = config.SEMANTIC_PATH_TOP_K,
    ) -> list[RetrievalResult]:
        self._assert_faiss_available()
        self._ensure_loaded()
        self._assert_embedding_present(jd_intent)

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # ── Build query vector ────────────────────────────────────────────
        query_vec = np.array(
            jd_intent.embedding, dtype=np.float32
        ).reshape(1, -1)

        # Ensure the query is L2-normalised (it should already be from
        # JDParser._encode, but we re-normalise defensively).
        _faiss.normalize_L2(query_vec)

        # Clamp top_k to index size to avoid FAISS returning -1 padding
        effective_top_k = min(top_k, self._index.ntotal)
        if effective_top_k < top_k:
            logger.debug(
                "top_k clamped from %d to %d (index only has %d candidates).",
                top_k, effective_top_k, self._index.ntotal,
            )

        # ── FAISS search ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        distances, indices = self._index.search(query_vec, effective_top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        logger.debug(
            "FAISS search: top_k=%d, ntotal=%d, elapsed=%.1f ms",
            effective_top_k,
            self._index.ntotal,
            elapsed_ms,
        )

        if elapsed_ms > 50.0:
            logger.warning(
                "FAISS search took %.1f ms (> 50 ms budget). "
                "Check nprobe setting or index type.",
                elapsed_ms,
            )

        # ── Map FAISS rows → candidate_ids ────────────────────────────────
        raw_indices: np.ndarray = indices[0]   # shape (effective_top_k,)
        raw_scores: np.ndarray = distances[0]  # shape (effective_top_k,)

        results: list[RetrievalResult] = []
        rank = 0

        for faiss_idx, raw_score in zip(raw_indices, raw_scores):
            # FAISS uses -1 to pad when index has fewer candidates than top_k
            if faiss_idx < 0:
                continue

            if faiss_idx >= len(self._candidate_ids):
                logger.warning(
                    "FAISS returned out-of-range index %d "
                    "(candidate_ids length=%d). Skipping.",
                    faiss_idx,
                    len(self._candidate_ids),
                )
                continue

            candidate_id: str = str(self._candidate_ids[faiss_idx])
            # Cosine similarity for L2-normalised vectors ∈ [-1, 1].
            # Clip to [0, 1] — negative similarity means anti-similar, not relevant.
            score: float = float(max(0.0, raw_score))
            rank += 1

            results.append(
                RetrievalResult(
                    candidate_id=candidate_id,
                    path_score=score,
                    path_name=self.PATH_NAME,
                    rank_in_path=rank,
                )
            )

        logger.info(
            "SemanticPath.retrieve: top_k=%d → %d results  (%.1f ms)",
            top_k,
            len(results),
            elapsed_ms,
        )
        return results

    # Internal loading 
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        self._assert_faiss_available()

        self._index = self._load_faiss_index(self._index_path)
        self._candidate_ids = self._load_id_map(self._id_map_path)
        self._configure_nprobe()
        self._validate_index_id_alignment()
        self._loaded = True

        logger.info(
            "SemanticPath loaded from disk: "
            "ntotal=%d, id_map=%d, nprobe=%s",
            self._index.ntotal,
            len(self._candidate_ids),
            getattr(self._index, "nprobe", "N/A (Flat)"),
        )

    @staticmethod
    def _load_faiss_index(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: '{path}'. "
                "Run precompute.py to build the index, or verify "
                "config.FAISS_INDEX_PATH."
            )
        t0 = time.perf_counter()
        try:
            index = _faiss.read_index(str(path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read FAISS index from '{path}': {exc}"
            ) from exc

        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.debug("FAISS index loaded in %.0f ms (ntotal=%d).", elapsed, index.ntotal)
        return index

    @staticmethod
    def _load_id_map(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(
                f"Candidate ID map not found: '{path}'. "
                "Run precompute.py to build the indexes, or verify "
                "config.FAISS_ID_MAP_PATH."
            )
        try:
            # allow_pickle=True required for object-dtype (string) arrays.
            # We immediately validate content to mitigate injection risk.
            arr: np.ndarray = np.load(str(path), allow_pickle=True)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load candidate_ids from '{path}': {exc}"
            ) from exc

        if arr.ndim != 1:
            raise ValueError(
                f"candidate_ids array must be 1-D, got shape {arr.shape}."
            )

        # Validate that all entries look like CAND_XXXXXXX.
        # Sample at least 1% of the array (min 100, max len(arr)) so that
        # corrupt ID maps in large indexes are not silently missed.
        import re as _re
        _cand_re = _re.compile(r"^CAND_[0-9]{7}$")
        sample_size = min(len(arr), max(100, int(len(arr) * 0.01)))
        for entry in arr[:sample_size]:
            entry_str = str(entry)
            if not _cand_re.match(entry_str):
                raise ValueError(
                    f"Unexpected candidate_id format in id_map: '{entry_str}'. "
                    "Expected CAND_XXXXXXX. Check faiss_builder.py output."
                )

        logger.debug("candidate_ids loaded: %d entries.", len(arr))
        return arr

    def _configure_nprobe(self) -> None:
        if self._index is not None and hasattr(self._index, "nprobe"):
            self._index.nprobe = config.FAISS_NPROBE
            logger.debug("FAISS nprobe set to %d.", config.FAISS_NPROBE)

    def _validate_index_id_alignment(self) -> None:
        if self._index is None or len(self._candidate_ids) == 0:
            return
        if self._index.ntotal != len(self._candidate_ids):
            raise ValueError(
                f"FAISS index ntotal={self._index.ntotal} does not match "
                f"candidate_ids length={len(self._candidate_ids)}. "
                "Re-run precompute.py to rebuild aligned indexes."
            )

    # Assertion helpers
    @staticmethod
    def _assert_faiss_available() -> None:
        if not _FAISS_AVAILABLE:
            raise RuntimeError(
                "faiss-cpu is not installed. "
                "Run: pip install faiss-cpu==1.8.0"
            )

    @staticmethod
    def _assert_embedding_present(jd_intent: JDIntent) -> None:
        if jd_intent.embedding is None:
            raise ValueError(
                "jd_intent.embedding is None. "
                "Parse the JD with encode=True: "
                "JDParser().parse(jd_text, encode=True). "
                "The FAISS semantic path requires a pre-computed embedding."
            )
        if len(jd_intent.embedding) != config.EMBEDDING_DIM:
            raise ValueError(
                f"jd_intent.embedding has {len(jd_intent.embedding)} dimensions "
                f"but expected {config.EMBEDDING_DIM}. "
                "Ensure the JD was encoded with the same model as the FAISS index "
                f"({config.BI_ENCODER_MODEL})."
            )

    # Properties  
    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def ntotal(self) -> int:
        if not self._loaded or self._index is None:
            return 0
        return int(self._index.ntotal)

    def __repr__(self) -> str:
        status = (
            f"ntotal={self.ntotal}, "
            f"nprobe={getattr(self._index, 'nprobe', 'N/A')}"
            if self._loaded else "not loaded"
        )
        return f"SemanticPath({status})"


# Module-level convenience
def retrieve_semantic(
    jd_intent: JDIntent,
    top_k: int = config.SEMANTIC_PATH_TOP_K,
    index_path: Optional[Path] = None,
    id_map_path: Optional[Path] = None,
) -> list[RetrievalResult]:
    path = SemanticPath.from_disk(index_path=index_path, id_map_path=id_map_path)
    return path.retrieve(jd_intent, top_k=top_k)

