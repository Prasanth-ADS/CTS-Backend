import os
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field


class Settings(BaseModel):
    use_mock_model_serving: bool = True
    use_mock_reasoning_service: bool = True
    use_mock_llm: bool = True

    model_serving_url: str | None = None
    reasoning_service_url: str | None = None
    llm_base_url: str | None = None
    llm_model_name: str = "local-slm"

    redis_url: str = "redis://redis:6379"
    model_serving_timeout_seconds: float = 5.0
    reasoning_timeout_seconds: float = 3.0
    max_turns: int = 3

    min_belief_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    min_margin_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    max_conflict_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    fusion_weights: dict[str, float] = {"alexnet_v1": 0.5, "vgg16_v3": 0.5}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


@lru_cache
def get_settings() -> Settings:
    return Settings(
        use_mock_model_serving=_env_bool("USE_MOCK_MODEL_SERVING", True),
        use_mock_reasoning_service=_env_bool("USE_MOCK_REASONING_SERVICE", True),
        use_mock_llm=_env_bool("USE_MOCK_LLM", True),
        model_serving_url=os.getenv("MODEL_SERVING_URL"),
        reasoning_service_url=os.getenv("REASONING_SERVICE_URL"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "local-slm"),
        redis_url=os.getenv("REDIS_URL", "redis://redis:6379"),
        model_serving_timeout_seconds=_env_float("MODEL_SERVING_TIMEOUT_SECONDS", 5.0),
        reasoning_timeout_seconds=_env_float("REASONING_TIMEOUT_SECONDS", 3.0),
        max_turns=_env_int("MAX_TURNS", 3),
        min_belief_threshold=_env_float("MIN_BELIEF_THRESHOLD", 0.70),
        min_margin_threshold=_env_float("MIN_MARGIN_THRESHOLD", 0.20),
        max_conflict_threshold=_env_float("MAX_CONFLICT_THRESHOLD", 0.30),
    )


DecisionStatus = Literal["confirmed", "uncertain"]
