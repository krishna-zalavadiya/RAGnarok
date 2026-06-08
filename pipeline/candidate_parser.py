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
count = 0

for c in candidates:
    if (c.is_consulting_only):
        print(c.candidate_id)
        for ch in c.career_history:
            print(ch.company_lower)
    else:
        count += 1


print(count) # count = 44 out of 50


# ------- Test End--------