import json
from pipeline.schemas import CandidateFeatureVector
from datetime import date
from .schemas import (
    CandidateFeatureVector,
    SkillRecord,
    CareerEntry,
    EducationEntry,
    RedrobSignals,
)
import config
from indexing.honeypot_registry import HoneypotFilter
from indexing.trajectory_builder import TrajectoryAnalyzer
import time
from pipeline.jd_parser import JDParser
from pathlib import Path
from indexing.faiss_builder import FaissIndex

def parse_candidate(item):

    profile = item["profile"]
    signals_raw = item["redrob_signals"]

    skills = [
        SkillRecord(
            name=s["name"].lower(),
            name_raw=s["name"],
            proficiency=s["proficiency"],
            endorsements=s["endorsements"],
            duration_months=s["duration_months"],
            assessment_score=signals_raw
                .get("skill_assessment_scores", {})
                .get(s["name"], -1.0),
        )
        for s in item["skills"]
    ]

    career_history = [
        CareerEntry(
            company=c["company"],
            company_lower=c["company"].lower(),
            title=c["title"],
            start_date=date.fromisoformat(c["start_date"]),
            end_date=None if c["end_date"] is None
                     else date.fromisoformat(c["end_date"]),
            duration_months=c["duration_months"],
            is_current=c["is_current"],
            industry=c["industry"],
            industry_lower=c["industry"].lower(),
            company_size=c["company_size"],
            description=c["description"],
        )
        for c in item["career_history"]
    ]

    education = [
        EducationEntry(**e)
        for e in item["education"]
    ]

    signals = RedrobSignals(
        profile_completeness_score=signals_raw["profile_completeness_score"],
        signup_date=date.fromisoformat(signals_raw["signup_date"]),
        last_active_date=date.fromisoformat(signals_raw["last_active_date"]),
        open_to_work_flag=signals_raw["open_to_work_flag"],
        profile_views_received_30d=signals_raw["profile_views_received_30d"],
        applications_submitted_30d=signals_raw["applications_submitted_30d"],
        recruiter_response_rate=signals_raw["recruiter_response_rate"],
        avg_response_time_hours=signals_raw["avg_response_time_hours"],
        skill_assessment_scores=signals_raw["skill_assessment_scores"],
        connection_count=signals_raw["connection_count"],
        endorsements_received=signals_raw["endorsements_received"],
        notice_period_days=signals_raw["notice_period_days"],
        expected_salary_min_lpa=signals_raw["expected_salary_range_inr_lpa"]["min"],
        expected_salary_max_lpa=signals_raw["expected_salary_range_inr_lpa"]["max"],
        preferred_work_mode=signals_raw["preferred_work_mode"],
        willing_to_relocate=signals_raw["willing_to_relocate"],
        github_activity_score=signals_raw["github_activity_score"],
        search_appearance_30d=signals_raw["search_appearance_30d"],
        saved_by_recruiters_30d=signals_raw["saved_by_recruiters_30d"],
        interview_completion_rate=signals_raw["interview_completion_rate"],
        offer_acceptance_rate=signals_raw["offer_acceptance_rate"],
        verified_email=signals_raw["verified_email"],
        verified_phone=signals_raw["verified_phone"],
        linkedin_connected=signals_raw["linkedin_connected"],
    )
    companies = {
        job.company_lower
        for job in career_history
    }

    is_consulting_only = (
        len(companies) > 0
        and all(company in config.CONSULTING_FIRMS for company in companies)
    )

    has_product_co_experience = any(
        company not in config.CONSULTING_FIRMS
        for company in companies
    )

    return CandidateFeatureVector(
        candidate_id=item["candidate_id"],
        headline=profile["headline"],
        summary=profile["summary"],
        location=profile["location"],
        location_lower=profile["location"].lower(),
        country=profile["country"],
        years_of_experience=profile["years_of_experience"],
        current_title=profile["current_title"],
        current_title_lower=profile["current_title"].lower(),
        current_company=profile["current_company"],
        current_company_lower=profile["current_company"].lower(),
        current_company_size=profile["current_company_size"],
        current_industry=profile["current_industry"],
        current_industry_lower=profile["current_industry"].lower(),
        skills=skills,
        career_history=career_history,
        education=education,
        signals=signals,
        is_consulting_only=is_consulting_only,
        has_product_co_experience=has_product_co_experience,
        total_career_months=sum(c.duration_months for c in career_history),
        skill_names_lower=frozenset(s.name for s in skills),
        embedding_text=""
    )

with open("sample_candidates.json", "r") as file:
    data_list = json.load(file)
    candidates = [parse_candidate(item) for item in data_list]
    # print(candidates[0])


# Run using: python -m pipeline.candidate_parser

# ------- Test Start--------
# always keep track of time because it will help improve
time1 = time.perf_counter()

count = 0

for c in candidates:
    if (c.is_consulting_only):
        print(c.candidate_id)
        for ch in c.career_history:
            print(ch.company_lower)
    else:
        count += 1

print(count) # count = 44 out of 50

count = 0
HoneypotFilter.run_honeypot_filters(candidates)
for c in candidates:
    if c.is_honeypot:
        count += 1

print(count) # count = 24 out of 50 i.e 22 honeypot candidates detected

for c in candidates:
    traj = TrajectoryAnalyzer.build_feature_vector(c)
    
    stability_score = min(
        traj["avg_tenure"] / 3.0,
        1.0
    )

    career_score = (
        0.40 * traj["yoe_score"]
        + 0.30 * traj["product_experience"]
        + 0.30 * stability_score
    )

    if traj["consulting_only"] == 1.0:
        career_score *= config.CONSULTING_ONLY_PENALTY
        
    # print("=" * 50)
    # print(c.candidate_id)
    # print("YOE:", c.years_of_experience)
    # print("YOE Score:", traj["yoe_score"])
    # print("Avg Tenure:", traj["avg_tenure"])
    # print("Job Hopper:", traj["job_hopper"])
    # print("Consulting Only:", traj["consulting_only"])
    # print("Product Experience:", traj["product_experience"])
    # print("Career Score:", career_score)

# print("\n \n \n \n \n \n")
parser = JDParser()
intent = parser.parse(Path("job_description.md"), encode=False)  # encode=False skips model load
# print(intent)

# with open("parsed_job_description.json", "w") as f:
#     json.dump(intent.__dict__, f, indent=2)

fi = FaissIndex()
fi.build(candidates, save=True)

fi.search("Job Description: Senior AI Engineer - Founding Team\n\nCompany: Redrob AI (Series A AI-native talent intelligence platform)\n\nLocation: Pune/Noida, India (Hybrid - flexible cadence) | Open to relocation candidates from Tier-1 Indian cities\n\nEmployment Type: Full-time\n\nExperience Required: 5-9 years (see \"what we mean by this\" below)\n\nLet's be honest about this role\n\nWe're going to write this JD differently from most. We're a Series A company that just raised our round and we're building a new AI Engineering org from scratch. This is the kind of role where the JD changes every six months because the company changes every six months. So instead of pretending we have a fixed checklist, we're going to tell you what we actually need and what we've gotten wrong before.\n\nIf you've spent your career at Google or Meta and you want a well-scoped role with a defined ladder, this isn't it.\n\nIf you've spent your career bouncing between early-stage startups and you want to \"just code\" without having to think about product or recruiter workflows or eval frameworks, this also isn't it.\n\nWe need someone who is simultaneously comfortable with two things that sound contradictory:\n\nDeep technical depth in modern ML systems - embeddings, retrieval, ranking, LLMs, fine-tuning.\n\nScrappy product-engineering attitude - willing to ship a working ranker in a week even if the underlying ML is \"obviously suboptimal,\" because we need to learn from real users before we know what to actually optimize for.\n\nThese are not contradictory in real life. They feel contradictory because of how engineering culture sorted itself into \"researcher\" vs \"shipper\" archetypes. We need both modes available in the same person, and we'd rather you tilt slightly toward shipper than toward researcher.\n\nWhat you'd actually be doing\n\nThe high-level mandate: own the intelligence layer of Redrob's product. That means the ranking, retrieval, and matching systems that decide what recruiters see when they search for candidates and what candidates see when they search for roles.\n\nIn practical terms, your first 90 days will probably look like:\n\nWeeks 1-3: Audit what we currently have (it's mostly BM25 + rule-based scoring, working but not great). Identify the 3-4 highest-leverage things to fix.\n\nWeeks 4-8: Ship a v2 ranking system that demonstrably improves recruiter-engagement metrics. This will involve embeddings, hybrid retrieval, and probably some LLM-based re-ranking, but the architecture is your call.\n\nWeeks 9-12: Set up the evaluation infrastructure - offline benchmarks, online A/B testing, recruiter-feedback loops - so we can keep improving without flying blind.\n\nBeyond that, you'll be driving the long-term architecture of how we do candidate-JD matching at scale, mentoring the next round of hires (we're growing the team from 4 to 12 engineers in the next year), and working closely with our recruiter-experience PM on what to build.\n\nWhat we mean by \"5-9 years\"\n\nThis is a range, not a requirement. Some people hit \"senior engineer\" judgment at 4 years; some never hit it after 15. We've used 5-9 because it's roughly where people we've hired into this kind of role have landed, but we'll seriously consider candidates outside the band if other signals are strong.\n\nThat said, here are the disqualifiers we actually apply:\n\nIf you've spent your career in pure research environments (academic labs, research-only roles) without any production deployment - we will not move forward. We are explicit about this. We've tried it twice and it didn't work for either side.\n\nIf your \"AI experience\" consists primarily of recent (under 12 months) projects using LangChain to call OpenAI - we will probably not move forward, unless you can demonstrate substantial pre-LLM-era ML production experience. We're looking for people who understood retrieval and ranking before it became fashionable.\n\nIf you are a senior engineer who hasn't written production code in the last 18 months because you've moved into \"architecture\" or \"tech lead\" roles - we will probably not move forward. This role writes code.\n\nThe skills inventory (please read carefully)\n\nMost JDs list 20 skills and you're supposed to have all of them. We're going to do this differently.\n\nThings you absolutely need\n\nProduction experience with embeddings-based retrieval systems (sentence-transformers, OpenAI embeddings, BGE, E5, or similar) deployed to real users. We don't care which model - we care that you've handled embedding drift, index refresh, retrieval-quality regression in production.\n\nProduction experience with vector databases or hybrid search infrastructure - Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS, or something similar. Again, the specific tech doesn't matter; the operational experience does.\n\nStrong Python. Yes really, we care about code quality.\n\nHands-on experience designing evaluation frameworks for ranking systems - NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation. If you've never thought about how to evaluate a ranking system rigorously, this role will be very painful.\n\nThings we'd like you to have but won't reject you for\n\nLLM fine-tuning experience (LoRA, QLoRA, PEFT)\n\nExperience with learning-to-rank models (XGBoost-based or neural)\n\nPrior exposure to HR-tech, recruiting tech, or marketplace products\n\nBackground in distributed systems or large-scale inference optimization\n\nOpen-source contributions in the AI/ML space\n\nThings we explicitly do NOT want\n\nThis is the section most JDs skip but we think it's the most important:\n\nTitle-chasers. If your career trajectory shows you optimizing for \"Senior\" \u2192 \"Staff\" \u2192 \"Principal\" titles by switching companies every 1.5 years, we're not a fit. We need someone who plans to be here for 3+ years.\n\nFramework enthusiasts. If your GitHub is full of LangChain tutorials and your blog posts are \"How I used [hot framework] to build [demo]\" - that's fine but it's not what we need. We need people who think about systems, not frameworks.\n\nPeople who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.) in their entire career. We've had bad fit experiences in both directions. If you're currently at one of these companies but have prior product-company experience, that's fine.\n\nPeople whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure. We respect your work but you'd be re-learning fundamentals here.\n\nPeople whose work has been entirely on closed-source proprietary systems for 5+ years without external validation (papers, talks, open-source). We need to see how you think, not just trust that you can think.\n\nOn location, comp, and logistics\n\nLocation: Pune/Noida-preferred but flexible. We have offices in Noida and Pune(mostly used Tue/Thu). We don't require any specific number of in-office days but we expect quarterly travel for offsites. Candidates in Hyderabad, Pune, Mumbai, Delhi NCR welcome to apply. Outside India: case-by-case, but we don't sponsor work visas.\n\nNotice period: We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher.\n\nThe vibe check\n\nWe genuinely believe culture-fit matters more at this stage than skills-fit. Skills are teachable; the rest mostly isn't.\n\nWe work async-first and write a lot. If you find writing painful, you'll find this role painful.\n\nWe disagree openly and decide quickly. If you find that style abrasive, you'll find this role abrasive.\n\nWe move fast and break things, with the caveat that \"things\" are usually our internal assumptions, not user-facing systems. If you need a stable, mature codebase to be productive, you'll find this role unstable.\n\nHow to read between the lines\n\nThe \"ideal candidate\" we're imagining is roughly:\n\n6-8 years total experience, of which 4-5 are in applied ML/AI roles at product companies (not pure services).\n\nHas shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.\n\nHas strong opinions about retrieval (hybrid vs dense), evaluation (offline vs online), and LLM integration (when to fine-tune vs prompt) - and can defend them with reference to systems they actually built.\n\nLocated in or willing to relocate to Noida or Pune.\n\nActive on Redrob platform (or has clear signal of being in the job market) so we can actually talk to them.\n\nWe are aware this is a narrow profile. We're not expecting to find many matches in a 100K candidate pool. We're explicitly OK with that - we'd rather see 10 great matches than 1000 maybes.\n\nFinal note for the participants of the Redrob hackathon\n\nIf you're reading this in the context of the Intelligent Candidate Discovery & Ranking Challenge:\n\nThe \"right answer\" to this JD is not \"find candidates whose skills section contains the most AI keywords.\" That's a trap we've explicitly built into the dataset.\n\nThe right answer involves reasoning about the gap between what the JD says and what the JD means. A Tier 5 candidate may not use the words \"RAG\" or \"Pinecone\" in their profile, but if their career history shows they built a recommendation system at a product company, they're a fit. A candidate who has all the AI keywords listed as skills but whose title is \"Marketing Manager\" is not a fit, no matter how perfect their skill list looks.\n\nYour ranking system should also weigh behavioral signals - a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% recruiter response rate is, for hiring purposes, not actually available. Down-weight them appropriately")

time2 = time.perf_counter()
print(time2 - time1)
# ------- Test End--------