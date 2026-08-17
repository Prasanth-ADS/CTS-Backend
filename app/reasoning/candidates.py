CandidateDistribution = dict[str, float]

EXPANDED_CANDIDATE_PRIOR = 0.05


def expand_candidate_distribution(
    fused_distribution: CandidateDistribution,
    matched_diseases: list[str],
) -> CandidateDistribution:
    """Merge KB symptom matches into the fused ensemble distribution.

    Diseases already in the ensemble distribution keep their current mass. New KB
    matches receive a small configurable-style prior for the MVP, then the full
    set is renormalized. This captures the v4 behavior that candidates can grow,
    not only shrink, after symptom matching.
    """
    expanded_distribution = dict(fused_distribution)
    for disease in matched_diseases:
        expanded_distribution.setdefault(disease, EXPANDED_CANDIDATE_PRIOR)
    return _normalize(expanded_distribution)


def _normalize(distribution: CandidateDistribution) -> CandidateDistribution:
    total = sum(distribution.values())
    if total <= 0:
        return {}
    return {
        disease: probability / total
        for disease, probability in sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    }
