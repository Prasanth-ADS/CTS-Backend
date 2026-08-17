import asyncio
import json
from typing import Any
from urllib import error, request

from app.core.config import get_settings

ObservationMap = dict[str, bool | None]

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
