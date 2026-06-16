import config
from indexing.honeypot_registry import HoneypotFilter
from indexing.trajectory_builder import TrajectoryAnalyzer
import time
from pipeline.jd_parser import JDParser
from pipeline.candidate_parser import CandidateParser
from pathlib import Path
from indexing.faiss_builder import FaissIndex
from indexing.bm25_builder import BM25Index
from indexing.feature_store import FeatureStore
from scoring.behavioral import BehavioralScorer
from scoring.trajectory import TrajectoryVelocityScorer
from scoring.honeypot_filter import HoneypotCleanup
from scoring.career_quality import CareerQualityScorer


DATASET_PATH = Path("sample_candidates.json")

time1 = time.perf_counter()


candidate_parser = CandidateParser() 
candidates = candidate_parser.build_candidate_list(DATASET_PATH)
print("Candidates loaded successfully")

honeypot_filter = HoneypotFilter()
honeypot_filter.run_honeypot_filters(candidates)
print("Honeypot run successfully")

honeypot_cleanup = HoneypotCleanup()
candidates = honeypot_cleanup.cleanup_candidates(candidates)

trajectory_analyzer = TrajectoryAnalyzer()
trajectory_analyzer.build_all_feature_vector(candidates)
print("Trajectory Analyzer run successfully")

parser = JDParser()
intent = parser.parse(Path("job_description.md"), encode=False)  # encode=False skips model load
print("Job description parser run successfully")

fi = FaissIndex()
fi.build(candidates, save=True)
print("Faiss index built successfully")

bm25 = BM25Index()
bm25.build(candidates, save=True)
print("BM25 index built successfully")

fs = FeatureStore()
matrix = fs.build(candidates, save=True)
print("Feature store run successfully")

bscorer = BehavioralScorer()
beh_results = bscorer.score_all(candidates)
sorted_beh_results = sorted(
    beh_results.values(), 
    key=lambda x: x.behavioral_score, 
    reverse=True
)
print("--- TOP 10 BEHAVIORAL CANDIDATES ---")
for rank, result in enumerate(sorted_beh_results[:10], start=1):
    print(f"{rank}. ID: {result.candidate_id} | Score: {result.behavioral_score:.4f}")
print("Behavioual Score run successfully")

tscorer = TrajectoryVelocityScorer()
traj_results = tscorer.score_all(candidates)
sorted_trajectory = sorted(
    traj_results, 
    key=lambda x: x.trajectory_velocity, 
    reverse=True
)

print("--- TOP 10 CAREER TRAJECTORY VELOCITY CANDIDATES ---")
for rank, res in enumerate(sorted_trajectory[:5], start=1):
    print(
        f"{rank}. ID: {res.candidate_id} | "
        f"Velocity Score: {res.trajectory_velocity:.4f} | "
        f"Percentile: {res.percentile_rank:.1f}% | "
        f"Promotions: {res.num_promotions} over {res.years_of_experience:.1f} YOE "
        f"({res.promotions_per_year:.2f}/yr)"
    )
print("Trajectory Score run successfully")

career_quality_scorer = CareerQualityScorer(intent)
 
results = career_quality_scorer.score_all(candidates)

for r in results.values():
    print(
        f"  {r.candidate_id:<20} "
        f"final={r.career_quality_score:.4f}  "
        f"product_co={r.product_co_score:.3f}  "
        f"yoe={r.yoe_score:.3f}  "
        f"stability={r.stability_score:.3f}  "
        f"domain={r.domain_match_score:.3f}  "
        f"consulting={r.is_consulting_only}"
    )


time2 = time.perf_counter()
print(time2 - time1)