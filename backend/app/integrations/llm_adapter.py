from backend.app.integrations.protocols import LLMAdapter


class MockLLMAdapter(LLMAdapter):
    async def extract_observations(self, description: str) -> dict[str, bool | None]:
        normalized = " ".join(description.lower().strip().split())
        return {
            "brown_patches": _contains_any(normalized, ("brown patch", "brown spot")),
            "wet_lesions": _contains_any(normalized, ("water-soaked", "water soaked", "wet lesion")),
            "yellow_halo": False if "no yellow halo" in normalized else None,
        }

    async def generate_explanation(self, diagnosis: dict, recommendations: dict | None) -> str:
        disease = diagnosis["diagnosed_disease"]
        return f"Mock explanation: the diagnosis flow converged on {disease} based on image and farmer evidence."

    async def answer_followup(self, session_context: dict, question: str) -> str:
        disease = session_context.get("result", {}).get("diagnosed_disease", "the diagnosed disease")
        return f"Mock follow-up answer for {disease}: consult local extension guidance for details."


class SLMAdapter(LLMAdapter):
    def __init__(self, base_url: str, model_name: str):
        import httpx

        self.client = httpx.AsyncClient(base_url=base_url)
        self.model_name = model_name

    async def extract_observations(self, description: str) -> dict[str, bool | None]:
        response = await self.client.post(
            "/extract-observations",
            json={"model": self.model_name, "description": description},
        )
        if response.status_code >= 400:
            return {}
        try:
            payload = response.json()
        except ValueError:
            return {}
        observations = payload.get("observations", payload)
        return observations if isinstance(observations, dict) else {}

    async def generate_explanation(self, diagnosis: dict, recommendations: dict | None) -> str:
        response = await self.client.post(
            "/generate-explanation",
            json={"model": self.model_name, "diagnosis": diagnosis, "recommendations": recommendations},
        )
        if response.status_code >= 400:
            return "Explanation unavailable."
        payload = response.json()
        return str(payload.get("explanation", "Explanation unavailable."))

    async def answer_followup(self, session_context: dict, question: str) -> str:
        response = await self.client.post(
            "/answer-followup",
            json={"model": self.model_name, "session_context": session_context, "question": question},
        )
        if response.status_code >= 400:
            return "Follow-up answer unavailable."
        payload = response.json()
        return str(payload.get("answer", "Follow-up answer unavailable."))


def _contains_any(value: str, needles: tuple[str, ...]) -> bool | None:
    if any(needle in value for needle in needles):
        return True
    return None
