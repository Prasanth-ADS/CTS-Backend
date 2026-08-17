FusedDistribution = dict[str, float]
PerModelTopK = dict[str, list[tuple[str, float]]]

EXPANDED_CANDIDATE_PRIOR = 0.05


def merge_model_predictions(per_model_topk: PerModelTopK, weights: dict[str, float] | None = None) -> FusedDistribution:
    weights = weights or {}
    model_ids = list(per_model_topk)
    if not model_ids:
        return {}

    weighted_scores: dict[str, float] = {}
    total_weight = 0.0
    for model_id, predictions in per_model_topk.items():
        weight = weights.get(model_id, 1.0)
        total_weight += weight
        for disease, probability in predictions:
            weighted_scores[disease] = weighted_scores.get(disease, 0.0) + probability * weight

    if total_weight <= 0:
        return {}
    return _normalize({disease: score / total_weight for disease, score in weighted_scores.items()})


def expand_candidates(base_distribution: FusedDistribution, matched_diseases: list[str]) -> FusedDistribution:
    expanded = dict(base_distribution)
    for disease in matched_diseases:
        expanded.setdefault(disease, EXPANDED_CANDIDATE_PRIOR)
    return _normalize(expanded)


def _normalize(distribution: FusedDistribution) -> FusedDistribution:
    total = sum(distribution.values())
    if total <= 0:
        return {}
    return {
        disease: probability / total
        for disease, probability in sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    }
