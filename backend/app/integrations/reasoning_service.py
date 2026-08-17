from typing import Protocol

import httpx


class ReasoningServiceClient(Protocol):
    async def symptom_match(self, observations: dict) -> list[str]: ...
    async def qa_knowledge(self, diseases: list[str]) -> dict: ...
    async def fuse(self, ensemble_belief: dict, symptom_evidence: dict, qa_evidence: list[dict]) -> dict: ...
    async def get_remedies(self, disease: str) -> dict: ...


class ReasoningServiceError(RuntimeError):
    pass


class MockReasoningServiceClient(ReasoningServiceClient):
    async def symptom_match(self, observations: dict) -> list[str]:
        return ["Potato___Late_blight", "Tomato___Late_blight", "Alternaria_Solani"]

    async def qa_knowledge(self, diseases: list[str]) -> dict:
        return {
            "questions": [
                {
                    "question_id": "q_water_soaked",
                    "text": "Do you see water-soaked patches?",
                    "support": {
                        "Potato___Late_blight": {"yes": 0.9, "no": 0.1, "not_sure": 0.5},
                        "Tomato___Late_blight": {"yes": 0.85, "no": 0.15, "not_sure": 0.5},
                    },
                },
                {
                    "question_id": "q_stem_lesions",
                    "text": "Are there dark lesions on the stem too?",
                    "support": {
                        "Potato___Late_blight": {"yes": 0.7, "no": 0.3, "not_sure": 0.5},
                        "Tomato___Late_blight": {"yes": 0.4, "no": 0.6, "not_sure": 0.5},
                    },
                },
            ]
        }

    async def fuse(self, ensemble_belief: dict, symptom_evidence: dict, qa_evidence: list[dict]) -> dict:
        turns = len(qa_evidence)
        belief = {
            "Potato___Late_blight": min(0.5 + 0.2 * turns, 0.95),
            "Tomato___Late_blight": max(0.3 - 0.1 * turns, 0.05),
        }
        return {
            "belief": belief,
            "plausibility": {disease: min(value + 0.05, 1.0) for disease, value in belief.items()},
            "uncertainty": {disease: 0.05 for disease in belief},
            "conflict": max(0.3 - 0.1 * turns, 0.02),
        }

    async def get_remedies(self, disease: str) -> dict:
        return {
            "management": [{"title": "Remove infected leaves", "priority": 1}],
            "prevention": [{"title": "Improve air circulation", "priority": 1}],
            "precautions": [{"title": "Use recommended fungicide as labeled", "priority": 1}],
            "references": ["https://extension.example.edu/late-blight"],
        }


class HTTPReasoningServiceClient(ReasoningServiceClient):
    def __init__(self, base_url: str, timeout_seconds: float = 3.0):
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout_seconds)

    async def symptom_match(self, observations: dict) -> list[str]:
        raise NotImplementedError("Wire up once DST/KB team's contract is confirmed")

    async def qa_knowledge(self, diseases: list[str]) -> dict:
        raise NotImplementedError("Wire up once DST/KB team's contract is confirmed")

    async def fuse(self, ensemble_belief: dict, symptom_evidence: dict, qa_evidence: list[dict]) -> dict:
        raise NotImplementedError("Wire up once DST/KB team's contract is confirmed")

    async def get_remedies(self, disease: str) -> dict:
        raise NotImplementedError("Wire up once DST/KB team's contract is confirmed")
