from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import load_model_registry

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
    registry = load_model_registry()
    return ModelPerformanceResponse(
        active_ensemble=registry["active_ensemble"],
        fusion_strategy=registry["fusion_strategy"],
        candidates=[ModelCandidate(**candidate) for candidate in registry["candidates"]],
    )
