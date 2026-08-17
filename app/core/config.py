from dataclasses import dataclass
from functools import lru_cache
import os


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Adaptive AI Disease Diagnosis API")


@lru_cache
def get_settings() -> Settings:
    return Settings()

