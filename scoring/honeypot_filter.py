from pipeline.schemas import CandidateFeatureVector

class HoneypotCleanup:

    def cleanup_candidates(
            self,
            candidates : list[CandidateFeatureVector]
            ) -> list[CandidateFeatureVector]:
        new_candidates_list = [candidate for candidate in candidates if not candidate.is_honeypot]
        return new_candidates_list
