import config
from pipeline.schemas import CandidateFeatureVector
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Config sets built once at import time — avoids per-call getattr cache checks
_CONSULTING_FIRMS    = frozenset(f.lower().strip() for f in config.CONSULTING_FIRMS)
_PRODUCT_INDUSTRIES  = frozenset(i.lower().strip() for i in config.PRODUCT_INDUSTRIES)


class TrajectoryAnalyzer:

    # ── Tenure ────────────────────────────────────────────────────────────────

    @staticmethod
    def calculate_tenure_metrics(candidate: CandidateFeatureVector) -> tuple[float, float, float]:
        """
        Single pass over career history.
        Returns: (avg_tenure_years, stability_score, job_hopper_flag)
        """
        history = candidate.career_history
        if not history:
            return 0.0, 0.0, 1.0

        total_months = sum(job.duration_months for job in history)
        avg_tenure = (total_months / len(history)) / 12.0

        stability_score = min(avg_tenure / 3.0, 1.0)
        is_job_hopper = float(avg_tenure < 1.5)

        return avg_tenure, stability_score, is_job_hopper

    # ── Consulting & product experience ───────────────────────────────────────

    @staticmethod
    def analyze_career_history(candidate: CandidateFeatureVector) -> tuple[float, float]:
        """
        Single pass over career history for both consulting and product signals.
        Returns: (consulting_only_flag, has_product_exp_flag)
        """
        history = candidate.career_history
        if not history:
            return 0.0, 0.0

        has_companies = False
        all_companies_are_consulting = True
        has_product_experience = 0.0

        for job in history:
            if job.company:
                has_companies = True
                if job.company.lower().strip() not in _CONSULTING_FIRMS:
                    all_companies_are_consulting = False

            if job.industry and job.industry.lower().strip() in _PRODUCT_INDUSTRIES:
                has_product_experience = 1.0

        is_consulting_only = float(has_companies and all_companies_are_consulting)
        return is_consulting_only, has_product_experience

    # ── YOE score ─────────────────────────────────────────────────────────────

    @staticmethod
    def yoe_score(candidate: CandidateFeatureVector) -> float:
        yoe = candidate.years_of_experience

        if config.YOE_BAND_IDEAL_MIN <= yoe <= config.YOE_BAND_IDEAL_MAX:
            return 1.0

        if yoe < config.YOE_BAND_IDEAL_MIN:
            return max(0.0, (yoe / config.YOE_BAND_IDEAL_MIN) ** 2)

        excess = yoe - config.YOE_BAND_IDEAL_MAX
        width = config.YOE_BAND_MAX - config.YOE_BAND_IDEAL_MAX

        if width <= 0:
            return 0.30

        return max(0.30, 1.0 - (excess / width))

    # ── Career score ──────────────────────────────────────────────────────────

    def career_score(self, candidate: CandidateFeatureVector) -> float:
        traj = self.build_feature_vector(candidate)

        score = (
            0.40 * traj["yoe_score"]
            + 0.30 * traj["product_experience"]
            + 0.30 * traj["stability_score"]
        )
        if traj["consulting_only"] == 1.0:
            score *= config.CONSULTING_ONLY_PENALTY

        return round(score, 4)

    # ── Feature vector ────────────────────────────────────────────────────────

    def build_feature_vector(self, candidate: CandidateFeatureVector) -> dict:
        avg_tenure, stability, job_hopper = self.calculate_tenure_metrics(candidate)
        is_consulting, product_exp = self.analyze_career_history(candidate)

        return {
            "yoe_score":          self.yoe_score(candidate),
            "avg_tenure":         avg_tenure,
            "stability_score":    stability,
            "job_hopper":         job_hopper,
            "consulting_only":    is_consulting,
            "product_experience": product_exp,
        }

    def build_all_feature_vector(self, candidates: list[CandidateFeatureVector]) -> list[dict]:
        return [self.build_feature_vector(c) for c in candidates]


class TrajectoryIndex:
    """
    Builds and saves the (N, 5) float32 matrix expected by trajectory_path.py.
    """
    def __init__(self, index_path: Path = None, ids_path: Path = None) -> None:
        self.index_path = index_path or config.TRAJECTORY_PATH
        self.ids_path = ids_path or config.TRAJECTORY_IDS_PATH

    def build(self, candidates: list[CandidateFeatureVector], save: bool = True) -> None:
        from scoring.trajectory import promotions_per_year, count_promotions, _effective_years
        
        logger.info("Building Trajectory Index for %d candidates...", len(candidates))
        
        analyzer = TrajectoryAnalyzer()
        
        # Keywords that indicate management/executive roles. Principal is often IC, but in 
        # _SENIORITY_KEYWORDS it's lumped with manager (level 4). We'll assume IC means never manager/exec.
        management_keywords = ("manager", "director", "head of", "chief", "ceo", "cto", "vp", "president", "founder")
        
        rows = []
        ids = []
        
        for c in candidates:
            # 1. promotions_per_year
            ppy = promotions_per_year(c)
            
            # 2. years_of_experience
            yoe = _effective_years(c)
            
            # 3 & 5. product_co and consulting_only
            is_consulting, product_exp = analyzer.analyze_career_history(c)
            
            # 4. is_ic_riser
            num_promotions = count_promotions(c.career_history)
            is_ic_riser = 0.0
            if num_promotions > 0:
                has_management = False
                for job in c.career_history:
                    title = job.title.lower()
                    if any(kw in title for kw in management_keywords):
                        has_management = True
                        break
                if not has_management:
                    is_ic_riser = 1.0
                    
            rows.append([ppy, yoe, product_exp, is_ic_riser, is_consulting])
            ids.append(c.candidate_id)
            
        matrix = np.array(rows, dtype=np.float32)
        
        if save:
            self._save(matrix, ids)
            
    def _save(self, matrix: np.ndarray, ids: list[str]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(self.index_path), matrix)
        np.save(str(self.ids_path), np.array(ids, dtype=object))
        logger.info("Saved Trajectory Index → %s | ids → %s", self.index_path, self.ids_path)