import asyncio
import json
from typing import Any
from urllib import error, parse, request

from app.core.config import get_settings

ObservationMap = dict[str, bool | None]
QuestionKnowledge = dict[str, object]


_MOCK_QA_KNOWLEDGE: list[QuestionKnowledge] = [
    {
        "question_id": "q_water_soaked",
        "text": "Do you see water-soaked patches?",
        "support": {
            "Potato___Late_blight": {"yes": 0.90, "no": 0.10, "not_sure": 0.50},
            "Tomato___Late_blight": {"yes": 0.80, "no": 0.20, "not_sure": 0.50},
            "Alternaria_Solani": {"yes": 0.30, "no": 0.70, "not_sure": 0.50},
            "Tomato___Leaf_Mold": {"yes": 0.40, "no": 0.60, "not_sure": 0.50},
            "Tomato___Bacterial_wilt": {"yes": 0.20, "no": 0.80, "not_sure": 0.50},
        },
    },
    {
        "question_id": "q_yellow_halo",
        "text": "Do the spots have a yellow halo?",
        "support": {
            "Potato___Late_blight": {"yes": 0.20, "no": 0.80, "not_sure": 0.50},
            "Tomato___Late_blight": {"yes": 0.35, "no": 0.65, "not_sure": 0.50},
            "Alternaria_Solani": {"yes": 0.85, "no": 0.15, "not_sure": 0.50},
            "Tomato___Leaf_Mold": {"yes": 0.30, "no": 0.70, "not_sure": 0.50},
            "Tomato___Bacterial_wilt": {"yes": 0.10, "no": 0.90, "not_sure": 0.50},
        },
    },
]

_MOCK_SYMPTOM_MATCHES = {
    "wet_lesions": "Potato___Late_blight",
    "brown_patches": "Alternaria_Solani",
    "yellow_halo": "Tomato___Late_blight",
    "white_mold": "Tomato___Leaf_Mold",
    "leaf_wilting": "Tomato___Bacterial_wilt",
}


class Team2KBError(RuntimeError):
    pass


async def match_symptoms(observations: ObservationMap) -> list[str]:
    """Return diseases matching extracted farmer observations.

    If TEAM2_KB_URL is unset, this uses a deterministic local fallback that keeps
    the step-4 candidate-expansion flow runnable while the external KB contract is
    finalized. Once Team 2 confirms the API, set TEAM2_KB_URL and the same public
    function will issue `POST /symptom-match`.
    """
    settings = get_settings()
    if not settings.team2_kb_url:
        return _mock_match_symptoms(observations)

    return await asyncio.to_thread(_post_symptom_match, settings.team2_kb_url, observations, settings.team2_kb_timeout_seconds)


def _mock_match_symptoms(observations: ObservationMap) -> list[str]:
    matched_diseases = [
        disease
        for observation, disease in _MOCK_SYMPTOM_MATCHES.items()
        if observations.get(observation) is True
    ]
    return list(dict.fromkeys(matched_diseases))


def _post_symptom_match(base_url: str, observations: ObservationMap, timeout_seconds: float) -> list[str]:
    url = f"{base_url.rstrip('/')}/symptom-match"
    payload = json.dumps({"observations": observations}).encode("utf-8")
    http_request = request.Request(
        url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, error.URLError, json.JSONDecodeError) as exc:
        raise Team2KBError("Team 2 symptom-match API is unavailable") from exc

    matched_diseases = response_payload.get("matched_diseases")
    if not isinstance(matched_diseases, list) or not all(isinstance(disease, str) for disease in matched_diseases):
        raise Team2KBError("Team 2 symptom-match API returned an invalid payload")
    return matched_diseases


async def fetch_qa_knowledge(diseases: list[str]) -> list[QuestionKnowledge]:
    """Fetch Team 2 question-bank knowledge once per session after expansion."""
    settings = get_settings()
    if not settings.team2_kb_url:
        return _mock_qa_knowledge(diseases)

    return await asyncio.to_thread(_get_qa_knowledge, settings.team2_kb_url, diseases, settings.team2_kb_timeout_seconds)


def _mock_qa_knowledge(diseases: list[str]) -> list[QuestionKnowledge]:
    disease_set = set(diseases)
    questions: list[QuestionKnowledge] = []
    for question in _MOCK_QA_KNOWLEDGE:
        support = question["support"]
        questions.append(
            {
                "question_id": question["question_id"],
                "text": question["text"],
                "support": {disease: values for disease, values in support.items() if disease in disease_set},
            }
        )
    return questions


def _get_qa_knowledge(base_url: str, diseases: list[str], timeout_seconds: float) -> list[QuestionKnowledge]:
    query = parse.urlencode({"diseases": ",".join(diseases)})
    url = f"{base_url.rstrip('/')}/qa-knowledge?{query}"
    http_request = request.Request(url, method="GET", headers={"Accept": "application/json"})

    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, error.URLError, json.JSONDecodeError) as exc:
        raise Team2KBError("Team 2 qa-knowledge API is unavailable") from exc

    questions = response_payload.get("questions")
    if not isinstance(questions, list):
        raise Team2KBError("Team 2 qa-knowledge API returned an invalid payload")
    return questions
