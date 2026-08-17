from typing import Literal

from backend.app.config import Settings

Decision = Literal["confirmed", "uncertain"]


def decide(fusion: dict, settings: Settings) -> Decision:
    ranked = sorted(fusion["belief"].items(), key=lambda item: -item[1])
    best, second = ranked[0], (ranked[1] if len(ranked) > 1 else (None, 0))
    if (
        best[1] >= settings.min_belief
        and best[1] - second[1] >= settings.min_margin
        and fusion["conflict"] <= settings.max_conflict
    ):
        return "confirmed"
    return "uncertain"


def diagnosed_disease(fusion: dict) -> str:
    return max(fusion["belief"].items(), key=lambda item: item[1])[0]


def displayed_confidence(fusion: dict, disease: str) -> float:
    raw = float(fusion["belief"].get(disease, 0.0))
    conflict = float(fusion.get("conflict", 0.0))
    return max(0.0, min(raw * (1 - conflict), 1.0))
