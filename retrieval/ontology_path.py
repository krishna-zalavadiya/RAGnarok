from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import config
from ontology.graph_traversal import SkillGraph
from pipeline.schemas import CandidateFeatureVector, JDIntent, RetrievalResult

logger = logging.getLogger(__name__)


class OntologyPath:
    PATH_NAME: str = "ontology"

    def __init__(
        self,
        skill_graph: Optional[SkillGraph] = None,
        skill_map_path: Optional[Path] = None,
    ) -> None:
        if skill_graph is not None:
            self._graph: SkillGraph = skill_graph
        else:
            effective_path = skill_map_path or config.SKILL_MAP_PATH
            self._graph = SkillGraph(skill_map_path=effective_path)

        logger.info("OntologyPath initialised — %r", self._graph)

   
    # Factory
    @classmethod
    def from_skill_map(
        cls,
        skill_map_path: Optional[Path] = None,
    ) -> "OntologyPath":
        return cls(skill_map_path=skill_map_path)

    # Primary retrieve method
    def retrieve(
        self,
        jd_intent: JDIntent,
        candidate_skills_map: dict[str, frozenset[str]],
        top_k: int = config.ONTOLOGY_PATH_TOP_K,
        exclude_ids: Optional[set[str]] = None,
        bfs_depth: int = 1,
    ) -> list[RetrievalResult]:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}.")

        if not isinstance(candidate_skills_map, dict):
            raise TypeError(
                "candidate_skills_map must be dict[str, frozenset[str]], "
                f"got {type(candidate_skills_map).__name__}."
            )

        if not jd_intent.required_skills:
            logger.warning(
                "OntologyPath.retrieve: jd_intent.required_skills is empty. "
                "No domain-transfer scoring possible. Returning []."
            )
            return []

        if not candidate_skills_map:
            logger.warning(
                "OntologyPath.retrieve: candidate_skills_map is empty. "
                "Returning []."
            )
            return []

        t0 = time.perf_counter()

        # Delegate all graph logic to SkillGraph
        ranked: list[tuple[str, float]] = self._graph.rank_by_domain_transfer(
            candidate_skills_map=candidate_skills_map,
            jd_required_skills=jd_intent.required_skills,
            top_k=top_k,
            exclude_ids=exclude_ids,
            bfs_depth=bfs_depth,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Adapt (candidate_id, score) tuples → RetrievalResult objects
        results: list[RetrievalResult] = [
            RetrievalResult(
                candidate_id=candidate_id,
                path_score=score,
                path_name=self.PATH_NAME,
                rank_in_path=rank + 1,
            )
            for rank, (candidate_id, score) in enumerate(ranked)
        ]

        logger.info(
            "OntologyPath.retrieve: %d candidates evaluated, "
            "%d rescued (top_k=%d, bfs_depth=%d, %.1f ms)",
            len(candidate_skills_map) - len(exclude_ids or set()),
            len(results),
            top_k,
            bfs_depth,
            elapsed_ms,
        )

        return results

    # Single-candidate scoring (testing / debugging) 
    def score_single(
        self,
        candidate_skills: frozenset[str],
        jd_intent: JDIntent,
        bfs_depth: int = 1,
    ) -> float:
        if not jd_intent.required_skills:
            return 0.0
        rescue_map = self._graph.build_jd_rescue_map(
            jd_intent.required_skills, bfs_depth=bfs_depth
        )
        return self._graph.score_candidate_skills(candidate_skills, rescue_map)

    # Static helper — used by pipeline/runner.py 
    @staticmethod
    def build_skills_map(
        feature_vectors: list[CandidateFeatureVector],
    ) -> dict[str, frozenset[str]]:
        return {
            fv.candidate_id: fv.skill_names_lower
            for fv in feature_vectors
        }

    # Introspection
    def explain_candidate(
        self,
        candidate_id: str,
        candidate_skills: frozenset[str],
        jd_intent: JDIntent,
    ) -> dict[str, object]:
        if not jd_intent.required_skills:
            return {
                "score": 0.0,
                "matched_via": [],
                "covered_jd_skills": [],
                "rescue_sources": {},
            }

        rescue_map = self._graph.build_jd_rescue_map(jd_intent.required_skills)
        score = self._graph.score_candidate_skills(candidate_skills, rescue_map)

        matched_via: list[str] = []
        covered_jd_skills: list[str] = []

        for jd_skill, sources in rescue_map.items():
            if jd_skill in candidate_skills:
                covered_jd_skills.append(jd_skill)
                matched_via.append(f"{jd_skill} (direct)")
            else:
                hits = candidate_skills & sources
                if hits:
                    covered_jd_skills.append(jd_skill)
                    for h in sorted(hits):
                        matched_via.append(f"{h} → {jd_skill} (transfer)")

        return {
            "score": score,
            "matched_via": matched_via,
            "covered_jd_skills": covered_jd_skills,
            "rescue_sources": {k: sorted(v) for k, v in rescue_map.items() if v},
        }

    def __repr__(self) -> str:
        return f"OntologyPath(graph={self._graph!r})"


# Module-level convenience
def retrieve_ontology(
    jd_intent: JDIntent,
    candidate_skills_map: dict[str, frozenset[str]],
    top_k: int = config.ONTOLOGY_PATH_TOP_K,
    skill_map_path: Optional[Path] = None,
) -> list[RetrievalResult]:
    path = OntologyPath.from_skill_map(skill_map_path=skill_map_path)
    return path.retrieve(jd_intent, candidate_skills_map, top_k=top_k)

