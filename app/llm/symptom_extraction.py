ObservationMap = dict[str, bool | None]

_OBSERVATION_KEYWORDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "brown_patches": (("brown patch", "brown patches", "brown spot", "brown spots"), ("no brown", "without brown")),
    "wet_lesions": (("wet lesion", "wet lesions", "water-soaked", "water soaked", "soaked patch"), ("dry lesion", "dry lesions", "not wet")),
    "yellow_halo": (("yellow halo", "yellow ring", "yellow edge"), ("no yellow halo", "without yellow halo")),
    "leaf_wilting": (("wilt", "wilting", "drooping", "droop"), ("not wilt", "no wilting")),
    "white_mold": (("white mold", "white mould", "fuzzy white", "white growth"), ("no white mold", "no white mould", "or white mold", "or white mould")),
}


def extract_observations(description: str) -> ObservationMap:
    """Extract farmer-observed symptoms from free text.

    This is the v4 live-SLM adapter boundary. The current implementation is a
    deterministic local placeholder so `/describe` is wired once per session
    without calling a network service. Replace this function body with the SLM
    provider call when the model/provider contract is selected, preserving this
    input/output shape.
    """
    normalized_description = _normalize(description)
    return {
        observation: _detect_observation(normalized_description, positive_terms, negative_terms)
        for observation, (positive_terms, negative_terms) in _OBSERVATION_KEYWORDS.items()
    }


def _detect_observation(
    normalized_description: str,
    positive_terms: tuple[str, ...],
    negative_terms: tuple[str, ...],
) -> bool | None:
    if any(term in normalized_description for term in negative_terms):
        return False
    if any(term in normalized_description for term in positive_terms):
        return True
    return None


def _normalize(description: str) -> str:
    return " ".join(description.lower().strip().split())
