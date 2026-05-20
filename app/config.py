from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class AppSettings:
    model_path: Path
    scaler_path: Path
    metadata_path: Path
    log_level: str = "INFO"


def load_yaml_config(path: Path = Path("configs/default.yaml")) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_settings() -> AppSettings:
    config = load_yaml_config()
    artifacts = config["artifacts"]
    return AppSettings(
        model_path=Path(os.getenv("MODEL_PATH", artifacts["model_path"])),
        scaler_path=Path(os.getenv("SCALER_PATH", artifacts["scaler_path"])),
        metadata_path=Path(os.getenv("METADATA_PATH", artifacts["metadata_path"])),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
