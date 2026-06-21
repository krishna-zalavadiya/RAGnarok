from __future__ import annotations

import logging
import time
from typing import Optional

import config
from pipeline.schemas import (
    JDIntent,
    CandidateFeatureVector,
    RankedCandidate,
    ComponentScores,
)

logger = logging.getLogger(__name__)


class PipelineRunner:
    """
    Orchestrates the full ranking pipeline.

    Args:
        jd:         Parsed JDIntent (from pipeline/jd_parser.py).
        candidates: List of CandidateFeatureVector (from pipeline/candidate_parser.py).

    Usage:
        runner = PipelineRunner(jd=intent, candidates=candidates)
        ranked, timings = runner.run(top_k=100)
    """

    def __init__(
        self,
        jd: JDIntent,
        candidates: list[CandidateFeatureVector],
    ) -> None:
        self._jd = jd
        self._candidates = candidates
        self._candidate_store: dict[str, CandidateFeatureVector] = {
            c.candidate_id: c for c in candidates
        }

    def run(
        self,
        top_k: int = 100,
    ) -> tuple[list[RankedCandidate], dict[str, float]]:
        """
        Run the full pipeline. Returns (ranked_candidates, stage_timings_ms).
        """
        timings: dict[str, float] = {}
        total_start = time.perf_counter()

        # ── 1. Honeypot filter (mark flagged candidates) ──────────────────
        t0 = time.perf_counter()
        try:
            from indexing.honeypot_registry import HoneypotFilter
            honeypot_ids = HoneypotFilter.load_honeypots()
            for c in self._candidates:
                if c.candidate_id in honeypot_ids:
                    c.is_honeypot = True
        except Exception as e:
            logger.warning("Honeypot filter unavailable: %s", e)
        timings["honeypot_filter"] = (time.perf_counter() - t0) * 1000

        # Remove honeypots from active pool (keep reference for output)
        clean_candidates = [c for c in self._candidates if not c.is_honeypot]
        honeypot_ids = {c.candidate_id for c in self._candidates if c.is_honeypot}
        logger.info("Honeypot filter: %d removed, %d clean", len(honeypot_ids), len(clean_candidates))

        # ── 2. Load indexes ───────────────────────────────────────────────
        t0 = time.perf_counter()
        # Indexes are now loaded lazily inside the paths via from_disk()
        timings["load_indexes"] = (time.perf_counter() - t0) * 1000

        # ── 3. Run 5 retrieval paths ──────────────────────────────────────
        t0 = time.perf_counter()
        all_retrieval_results = []
        path_results = {}

        # Path 1: Semantic (FAISS)
        try:
            from retrieval.semantic_path import SemanticPath
            sp = SemanticPath.from_disk()
            res = sp.retrieve(self._jd, top_k=config.SEMANTIC_PATH_TOP_K)
            path_results["semantic"] = res
            all_retrieval_results.extend(res)
        except Exception as e:
            logger.error("Semantic path failed: %s", e)
            raise RuntimeError(f"Semantic path failed: {e}") from e

        # Path 2: Keyword (BM25)
        try:
            from retrieval.keyword_path import KeywordPath
            kp = KeywordPath.from_disk()
            res = kp.retrieve(self._jd, top_k=config.KEYWORD_PATH_TOP_K)
            path_results["keyword"] = res
            all_retrieval_results.extend(res)
        except Exception as e:
            logger.error("Keyword path failed: %s", e)
            raise RuntimeError(f"Keyword path failed: {e}") from e

        # Path 3: Ontology
        try:
            from retrieval.ontology_path import OntologyPath
            op = OntologyPath.from_skill_map()
            skills_map = OntologyPath.build_skills_map(self._candidates)
            res = op.retrieve(self._jd, candidate_skills_map=skills_map, top_k=config.ONTOLOGY_PATH_TOP_K)
            path_results["ontology"] = res
            all_retrieval_results.extend(res)
        except Exception as e:
            logger.error("Ontology path failed: %s", e)
            raise RuntimeError(f"Ontology path failed: {e}") from e

        # Path 4: Trajectory
        try:
            from retrieval.trajectory_path import TrajectoryPath
            tp = TrajectoryPath.from_disk()
            res = tp.retrieve(self._jd, top_k=config.TRAJECTORY_PATH_TOP_K)
            path_results["trajectory"] = res
            all_retrieval_results.extend(res)
        except Exception as e:
            logger.error("Trajectory path failed: %s", e)
            raise RuntimeError(f"Trajectory path failed: {e}") from e

        # Path 5: Signal (behavioral)
        try:
            from retrieval.signal_path import SignalPath
            sigp = SignalPath.from_disk()
            res = sigp.retrieve(top_k=config.SIGNAL_PATH_TOP_K)
            path_results["signal"] = res
            all_retrieval_results.extend(res)
        except Exception as e:
            logger.error("Signal path failed: %s", e)
            raise RuntimeError(f"Signal path failed: {e}") from e

        timings["retrieval_paths"] = (time.perf_counter() - t0) * 1000
        logger.info("Retrieval: %d total results across all paths", len(all_retrieval_results))

        # ── 4. RRF Fusion ─────────────────────────────────────────────────
        t0 = time.perf_counter()
        rrf_pool = []
        try:
            from retrieval.rrf_fusion import RRFFusion
            rrf = RRFFusion()
            rrf_pool = rrf.fuse(path_results)
        except Exception as e:
            logger.error("RRF fusion failed: %s", e)
            raise RuntimeError(f"RRF fusion failed: {e}") from e
        timings["rrf_fusion"] = (time.perf_counter() - t0) * 1000
        logger.info("RRF pool: %d candidates", len(rrf_pool))

        # Remove honeypots from RRF pool
        rrf_pool = [r for r in rrf_pool if r.candidate_id not in honeypot_ids]
        rrf_pool = rrf_pool[:config.CROSS_ENCODER_TOP_K]

        # ── 5. Cross-encoder rerank ───────────────────────────────────────
        t0 = time.perf_counter()
        try:
            from scoring.cross_encoder import CrossEncoderReranker
            ce = CrossEncoderReranker()
            rrf_pool = ce.rerank(rrf_pool, self._jd, self._candidate_store)
        except Exception as e:
            logger.error("Cross-encoder unavailable: %s", e)
            raise RuntimeError(f"Cross-encoder unavailable: {e}") from e
        timings["cross_encoder"] = (time.perf_counter() - t0) * 1000

        # ── 6. Composite scoring ──────────────────────────────────────────
        t0 = time.perf_counter()
        composite_results = []
        # Cache behavioral_scorer so trust layer can reuse it (avoids double-scoring).
        _behavioral_scorer_instance = None
        try:
            from scoring.behavioral import BehavioralScorer
            from scoring.composite import CompositeScorer
            _behavioral_scorer_instance = BehavioralScorer()
            scorer = CompositeScorer(self._jd, self._candidate_store, _behavioral_scorer_instance)
            composite_results = scorer.rank(rrf_pool)
        except Exception as e:
            logger.error("Composite scoring failed: %s", e)
            raise RuntimeError(f"Composite scoring error: {e}") from e
        timings["composite_scoring"] = (time.perf_counter() - t0) * 1000
        logger.info("Composite scored %d candidates", len(composite_results))

        # ── 6b. Min-max score normalization ──────────────────────────────
        # Spreads the score range to [0.1, 1.0] for non-disqualified candidates,
        # making the score column more discriminative and useful for evaluation.
        # Disqualified (score == 0.0) candidates stay at 0.0.
        import dataclasses
        non_zero_scores = [r.final_score for r in composite_results if r.final_score > 0.0]
        if len(non_zero_scores) >= 2:
            s_min = min(non_zero_scores)
            s_max = max(non_zero_scores)
            score_range = s_max - s_min
            if score_range > 1e-6:
                composite_results = [
                    dataclasses.replace(
                        r,
                        final_score=round(
                            0.10 + 0.90 * (r.final_score - s_min) / score_range, 6
                        ),
                    ) if r.final_score > 0.0 else r
                    for r in composite_results
                ]
                logger.info(
                    "Score normalization: range [%.4f, %.4f] → [0.10, 1.00] "
                    "(%d non-zero candidates)",
                    s_min, s_max, len(non_zero_scores),
                )

        # ── 7. Trust layer & Reasoning ────────────────────────────────────
        t0 = time.perf_counter()
        trust_verdicts: dict = {}
        reasonings: dict[str, str] = {}
        schema_components: dict[str, ComponentScores] = {}

        top_candidates = []
        for cs in composite_results[:top_k]:
            cfv = self._candidate_store.get(cs.candidate_id)
            if cfv:
                top_candidates.append(cfv)

        try:
            from trust.advocate import build_advocate_signals
            from trust.skeptic import build_skeptic_signals
            from trust.verdict import build_verdict
            from trust.reasoning_generator import generate_reasoning
            from scoring.skill_match import SkillMatchScorer
            from scoring.career_quality import CareerQualityScorer

            skill_results = SkillMatchScorer().score_all(top_candidates, self._jd)
            career_results = CareerQualityScorer(self._jd).score_all(top_candidates)
            # Reuse the already-computed behavioral scorer instance from step 6
            # to avoid rescoring the same candidates a second time.
            behavioral_results = _behavioral_scorer_instance.score_all(top_candidates)

            for rank_pos, cs in enumerate(composite_results[:top_k], start=1):
                cid = cs.candidate_id
                cfv = self._candidate_store.get(cid)
                if not cfv:
                    continue

                s_res = skill_results.get(cid)
                c_res = career_results.get(cid)
                b_res = behavioral_results.get(cid)

                if s_res and c_res and b_res:
                    schema_comp = ComponentScores(
                        candidate_id=cid,
                        skill_score=cs.skill_match_score,
                        career_score=cs.career_quality_score,
                        behavioral_score=cs.behavioral_score,
                        required_skill_coverage=s_res.required_score,
                        nice_to_have_coverage=s_res.nice_to_have_score,
                        ontology_skills_matched=[],
                        yoe_score=c_res.yoe_score,
                        trajectory_velocity=cs.trajectory_velocity,
                        product_co_flag=cfv.has_product_co_experience,
                        consulting_only_flag=cfv.is_consulting_only,
                        location_bonus=cs.location_bonus_applied,
                        recency_score=b_res.recency_score,
                        notice_period_score=b_res.notice_period_score,
                        uncertainty_penalty=cs.uncertainty_penalty_applied,
                        signal_count=b_res.signal_count,
                    )
                    schema_components[cid] = schema_comp

                    adv_signals = build_advocate_signals(cfv, schema_comp, s_res, self._jd)
                    skep_signals = build_skeptic_signals(cfv, schema_comp, c_res, b_res, s_res, self._jd)
                    verdict = build_verdict(cfv, schema_comp, adv_signals, skep_signals)
                    trust_verdicts[cid] = verdict

                    reasonings[cid] = generate_reasoning(rank_pos, verdict, cfv, schema_comp)
        except Exception as e:
            logger.error("Trust layer unavailable: %s", e)
            raise RuntimeError(f"Trust layer unavailable: {e}") from e

        for cs in composite_results[:top_k]:
            cid = cs.candidate_id
            if cid not in reasonings:
                cfv = self._candidate_store.get(cid)
                yoe = f"{cfv.years_of_experience:.0f}y" if cfv else "?"
                reasonings[cid] = (
                    f"Ranked #{cid} with composite score {cs.final_score:.4f}. "
                    f"Skill: {cs.skill_match_score:.2f}, Career: {cs.career_quality_score:.2f}, "
                    f"Behavioral: {cs.behavioral_score:.2f}. "
                    f"Experience: {yoe}. "
                    f"Retrieved via: {', '.join(cs.paths_present)}."
                )

        timings["trust_layer"] = (time.perf_counter() - t0) * 1000

        # ── 9. Assemble RankedCandidate list ──────────────────────────────
        t0 = time.perf_counter()
        ranked: list[RankedCandidate] = []

        for rank_pos, cs in enumerate(composite_results[:top_k], start=1):
            cid = cs.candidate_id
            cfv = self._candidate_store.get(cid)
            
            schema_comp = schema_components.get(cid)
            if not schema_comp:
                schema_comp = ComponentScores(
                    candidate_id=cid,
                    skill_score=cs.skill_match_score,
                    career_score=cs.career_quality_score,
                    behavioral_score=cs.behavioral_score,
                    required_skill_coverage=0.0,
                    nice_to_have_coverage=0.0,
                    ontology_skills_matched=[],
                    yoe_score=0.0,
                    trajectory_velocity=cs.trajectory_velocity,
                    product_co_flag=cfv.has_product_co_experience if cfv else False,
                    consulting_only_flag=cfv.is_consulting_only if cfv else False,
                    location_bonus=cs.location_bonus_applied,
                    recency_score=0.0,
                    notice_period_score=0.0,
                    uncertainty_penalty=cs.uncertainty_penalty_applied,
                    signal_count=5,
                )

            ranked.append(RankedCandidate(
                candidate_id=cid,
                rank=rank_pos,
                final_score=cs.final_score,
                reasoning=reasonings.get(cid, ""),
                components=schema_comp,
                trust=trust_verdicts.get(cid),
                feature_vector=cfv,
            ))

        timings["assemble"] = (time.perf_counter() - t0) * 1000
        timings["total"] = (time.perf_counter() - total_start) * 1000

        logger.info(
            "Pipeline complete: %d candidates ranked in %.1fms (top score=%.4f)",
            len(ranked),
            timings["total"],
            ranked[0].final_score if ranked else 0.0,
        )
        return ranked, timings
