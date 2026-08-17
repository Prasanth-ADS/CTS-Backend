from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

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

    min_belief: float = Field(default=0.70, ge=0.0, le=1.0)
    min_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    max_conflict: float = Field(default=0.30, ge=0.0, le=1.0)
    fusion_weights: dict[str, float] = {"alexnet_v1": 0.5, "vgg16_v3": 0.5}


@lru_cache
def get_settings() -> Settings:
    return Settings()


DecisionStatus = Literal["confirmed", "uncertain"]
