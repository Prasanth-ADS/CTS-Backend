from app.core.model_registry import get_model_registry
from app.inference.ensemble import EnsemblePrediction

FusedDistribution = dict[str, float]


class UnsupportedFusionStrategyError(ValueError):
    pass


def fuse_predictions(per_model_topk: EnsemblePrediction) -> FusedDistribution:
    """Fuse per-model top-K predictions using the configured strategy.

    FUSION STRATEGY: weighted_average (config-selectable). This takes the union
    of each model's top-K, weights probabilities by configured fusion_weight
    values, and normalizes the output to a distribution that sums to 1.0.
    """
    registry = get_model_registry()
    if registry.fusion_strategy != "weighted_average":
        raise UnsupportedFusionStrategyError(f"Unsupported fusion strategy: {registry.fusion_strategy}")

    candidates_by_id = registry.candidates_by_id
    weighted_scores: dict[str, float] = {}
    total_weight = 0.0

    for model_id in registry.active_ensemble:
        candidate = candidates_by_id[model_id]
        weight = candidate.fusion_weight if candidate.fusion_weight is not None else candidate.eval_accuracy
        total_weight += weight
        for disease, probability in per_model_topk.get(model_id, []):
            weighted_scores[disease] = weighted_scores.get(disease, 0.0) + probability * weight

    if not weighted_scores or total_weight <= 0:
        return {}

    averaged_scores = {disease: score / total_weight for disease, score in weighted_scores.items()}
    return _normalize(averaged_scores)


def _normalize(scores: dict[str, float]) -> FusedDistribution:
    total = sum(scores.values())
    if total <= 0:
        return {}
    return {disease: score / total for disease, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
