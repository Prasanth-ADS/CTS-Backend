from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Adaptive AI Disease Diagnosis API")
    models_config_path: Path = Path(os.getenv("MODELS_CONFIG_PATH", "config/models.yaml"))
    team2_kb_url: str | None = os.getenv("TEAM2_KB_URL")
    team2_kb_timeout_seconds: float = float(os.getenv("TEAM2_KB_TIMEOUT_SECONDS", "5"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_model_registry() -> dict[str, Any]:
    import yaml

    settings = get_settings()
    with settings.models_config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
