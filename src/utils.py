from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(payload: dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"Metric arrays must have the same shape, got {true.shape} and {pred.shape}.")

    errors = true - pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    denominator = np.where(np.abs(true) < 1e-8, np.nan, np.abs(true))
    mape = float(np.nanmean(np.abs(errors) / denominator) * 100)
    return {"mae": mae, "rmse": rmse, "mape": mape}


def inverse_close_values(
    scaled_values: np.ndarray,
    scaler: Any,
    feature_names: list[str],
    target_column: str,
) -> np.ndarray:
    values = np.asarray(scaled_values, dtype=float).reshape(-1)
    target_index = feature_names.index(target_column)
    placeholder = np.zeros((len(values), len(feature_names)), dtype=float)
    placeholder[:, target_index] = values
    inversed = scaler.inverse_transform(placeholder)
    return inversed[:, target_index]
