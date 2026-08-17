from fastapi import APIRouter
from pydantic import BaseModel

from app.core.model_registry import get_model_registry

router = APIRouter(prefix="/api/v1/models", tags=["models"])


class ModelCandidate(BaseModel):
    id: str
    arch: str
    eval_accuracy: float
    fusion_weight: float | None = None


class ModelPerformanceResponse(BaseModel):
    active_ensemble: list[str]
    fusion_strategy: str
    candidates: list[ModelCandidate]


@router.get("/performance", response_model=ModelPerformanceResponse)
async def model_performance() -> ModelPerformanceResponse:
    registry = get_model_registry()
    return ModelPerformanceResponse(
        active_ensemble=list(registry.active_ensemble),
        fusion_strategy=registry.fusion_strategy,
        candidates=[
            ModelCandidate(
                id=candidate.id,
                arch=candidate.arch,
                eval_accuracy=candidate.eval_accuracy,
                fusion_weight=candidate.fusion_weight,
            )
            for candidate in registry.candidates
        ],
    )
