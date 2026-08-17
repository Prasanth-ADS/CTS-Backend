from functools import lru_cache
from pathlib import Path
import os
from typing import Any

import yaml
from pydantic import BaseModel


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "Adaptive AI Disease Diagnosis API")
    models_config_path: Path = Path(os.getenv("MODELS_CONFIG_PATH", "config/models.yaml"))


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_model_registry() -> dict[str, Any]:
    settings = get_settings()
    with settings.models_config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
