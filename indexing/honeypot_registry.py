from pipeline.schemas import CandidateFeatureVector
import config

# Proficiency levels that count as meaningful skill claims — O(1) lookup per skill
_MEANINGFUL_PROFICIENCY = frozenset(("expert", "advanced", "intermediate"))


class HoneypotFilter:

    # 2. Check if skills are actually mentioned in career descriptions
    def __filter2(self, candidate: CandidateFeatureVector) -> bool:
        skills = candidate.skills
        if not skills or not candidate.career_history:
            return False

        # Pre-filter to meaningful skills before paying the cost of building career text
        qualifying_skills = [
            skill.name.lower()
            for skill in skills
            if skill.proficiency
            and skill.proficiency.lower() in _MEANINGFUL_PROFICIENCY
            and skill.name
        ]
        if not qualifying_skills:
            return False

        full_career_text = " ".join(
            career.description.lower()
            for career in candidate.career_history
            if career.description
        )
        if not full_career_text:
            return False

        for skill_name in qualifying_skills:
            if skill_name in full_career_text:
                return True
        return False

    # 3. Profile completeness score < threshold with suspiciously many skills
    def __filter3(self, candidate: CandidateFeatureVector) -> bool:
        profile_incomplete = candidate.signals.profile_completeness_score < config.HONEYPOT_COMPLETENESS_THRESHOLD
        too_many_skills = len(candidate.skills) > config.HONEYPOT_SKILLS_STUFFING_COUNT
        return profile_incomplete or too_many_skills

    # 4. Salary min > max anomaly
    def __filter4(self, candidate: CandidateFeatureVector) -> bool:
        return candidate.signals.expected_salary_min_lpa > candidate.signals.expected_salary_max_lpa

    # 5. Experience discrepancy check
    def __filter5(self, candidate: CandidateFeatureVector) -> bool:
        total_duration_months = sum(
            career.duration_months
            for career in candidate.career_history
            if career.duration_months
        )
        total_duration_years = total_duration_months / 12.0
        return (total_duration_years - config.HONEYPOT_YOE_DISCREPANCY_YEARS) > candidate.years_of_experience

    def run_honeypot_filters(self, candidates: list[CandidateFeatureVector]) -> None:
        for candidate in candidates:
            # Ordered cheapest → most expensive so we short-circuit early:
            # f4: 2 attr lookups (O(1))
            # f3: 2 attr lookups + len() (O(1))
            # f5: sum over career_history (O(jobs))
            # f2: join + substring scan (O(jobs * text + skills * text))
            if self.__filter4(candidate):
                candidate.is_honeypot = True
                continue
            if self.__filter3(candidate):
                candidate.is_honeypot = True
                continue
            if self.__filter5(candidate):
                candidate.is_honeypot = True
                continue
            if self.__filter2(candidate):
                candidate.is_honeypot = True
                continue