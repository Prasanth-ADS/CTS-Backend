import asyncio

from app.schemas.diagnosis import AnswerRequest, DiagnosisMetadata, DiagnosisResult, Question

CandidateDistribution = dict[str, float]
ObservationMap = dict[str, bool | None]

_MOCK_QUESTIONS = [
    Question(question_id="q_water_soaked", text="Do you see water-soaked patches?"),
    Question(question_id="q_yellow_halo", text="Do the spots have a yellow halo?"),
]


class ReasoningServiceClientError(RuntimeError):
    pass


class ReasoningServiceClient:
    async def expand_candidates(
        self,
        image_distribution: CandidateDistribution,
        observations: ObservationMap,
        metadata: DiagnosisMetadata,
    ) -> CandidateDistribution:
        """Return candidate probabilities from the external reasoning service."""
        await asyncio.sleep(0)
        expanded_distribution = dict(image_distribution)
        symptom_matches = {
            "wet_lesions": "Potato___Late_blight",
            "brown_patches": "Alternaria_Solani",
            "yellow_halo": "Tomato___Late_blight",
            "white_mold": "Tomato___Leaf_Mold",
            "leaf_wilting": "Tomato___Bacterial_wilt",
        }
        for observation, disease in symptom_matches.items():
            if observations.get(observation) is True:
                expanded_distribution.setdefault(disease, 0.05)
        return _normalize(expanded_distribution)

    async def next_question(
        self,
        candidate_distribution: CandidateDistribution,
        previous_answers: list[AnswerRequest],
    ) -> Question | None:
        """Return the next external-reasoning follow-up question, if any."""
        await asyncio.sleep(0)
        if len(previous_answers) >= len(_MOCK_QUESTIONS):
            return None
        return _MOCK_QUESTIONS[len(previous_answers)]

    async def finalize(
        self,
        candidate_distribution: CandidateDistribution,
        answers: list[AnswerRequest],
    ) -> DiagnosisResult:
        """Return the final diagnosis from the external reasoning service."""
        await asyncio.sleep(0)
        diagnosed_disease, confidence_score = next(iter(candidate_distribution.items()), ("Unknown", 0.0))
        return DiagnosisResult(
            diagnosed_disease=diagnosed_disease,
            confidence_score=confidence_score,
            confidence_note="mock_external_reasoning_service",
            management=["Remove heavily infected leaves", "Avoid overhead irrigation"],
            prevention=["Use certified disease-free seed", "Improve field air circulation"],
            precautions=["Confirm with an agricultural specialist before treatment"],
            references=["https://example.org/reasoning-service/mock-guidance"],
            explanation="Mock result returned by the reasoning-service adapter.",
        )

    async def answer_followup(self, disease: str, question: str) -> str:
        """Answer diagnosis follow-up questions through the reasoning service."""
        await asyncio.sleep(0)
        return f"Mock reasoning-service answer for {disease}: verify guidance with an agricultural specialist."


def get_reasoning_service_client() -> ReasoningServiceClient:
    return ReasoningServiceClient()


def _normalize(distribution: CandidateDistribution) -> CandidateDistribution:
    total = sum(distribution.values())
    if total <= 0:
        return {}
    return {
        disease: probability / total
        for disease, probability in sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    }
