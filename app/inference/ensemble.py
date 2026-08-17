import asyncio

from app.core.model_registry import ModelCandidate, get_model_registry

PredictionTopK = list[tuple[str, float]]
EnsemblePrediction = dict[str, PredictionTopK]

_DEFAULT_MOCK_TOP_K: PredictionTopK = [
    ("Potato___Late_blight", 0.46),
    ("Tomato___Late_blight", 0.31),
    ("Alternaria_Solani", 0.23),
]


async def predict_ensemble(image_bytes: bytes) -> EnsemblePrediction:
    """Run every active model concurrently and return raw per-model top-K predictions.

    This implements the step-2 execution shape from the architecture. The actual
    model runner is still mocked until model artifacts and preprocessing are added.
    """
    registry = get_model_registry()
    results = await asyncio.gather(
        *(run_model(candidate, image_bytes) for candidate in registry.active_candidates)
    )
    return dict(zip(registry.active_ensemble, results, strict=True))


async def run_model(candidate: ModelCandidate, image_bytes: bytes) -> PredictionTopK:
    """Placeholder model runner with the same async contract as real inference.

    The image bytes are accepted now so the endpoint contract does not change when
    real CNN loading/preprocessing replaces this mock implementation.
    """
    await asyncio.sleep(0)
    if candidate.mock_top_k:
        return list(candidate.mock_top_k)
    return list(_DEFAULT_MOCK_TOP_K)
