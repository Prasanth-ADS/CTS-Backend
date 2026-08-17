from dataclasses import dataclass
from typing import Any

from app.core.config import load_model_registry


@dataclass(frozen=True)
class ModelCandidate:
    id: str
    path: str
    arch: str
    eval_accuracy: float
    fusion_weight: float | None = None
    mock_top_k: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class ModelRegistry:
    active_ensemble: tuple[str, ...]
    fusion_strategy: str
    candidates: tuple[ModelCandidate, ...]

    @property
    def candidates_by_id(self) -> dict[str, ModelCandidate]:
        return {candidate.id: candidate for candidate in self.candidates}

    @property
    def active_candidates(self) -> tuple[ModelCandidate, ...]:
        candidates_by_id = self.candidates_by_id
        return tuple(candidates_by_id[model_id] for model_id in self.active_ensemble)


def get_model_registry() -> ModelRegistry:
    raw_registry = load_model_registry()
    return ModelRegistry(
        active_ensemble=tuple(raw_registry["active_ensemble"]),
        fusion_strategy=raw_registry["fusion_strategy"],
        candidates=tuple(_parse_candidate(candidate) for candidate in raw_registry["candidates"]),
    )


def _parse_candidate(candidate: dict[str, Any]) -> ModelCandidate:
    return ModelCandidate(
        id=candidate["id"],
        path=candidate["path"],
        arch=candidate["arch"],
        eval_accuracy=float(candidate["eval_accuracy"]),
        fusion_weight=candidate.get("fusion_weight"),
        mock_top_k=tuple((disease, float(score)) for disease, score in candidate.get("mock_top_k", [])),
    )
