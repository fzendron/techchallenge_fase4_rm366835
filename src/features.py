from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class FeatureDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_train_actual: np.ndarray
    y_val_actual: np.ndarray
    y_test_actual: np.ndarray
    naive_test_actual: np.ndarray
    scaler: MinMaxScaler
    metadata: dict[str, Any]


def select_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}.")
    selected = df.loc[:, feature_names].copy()
    if selected.isna().any().any():
        raise ValueError("Feature data contains missing values.")
    return selected


def chronological_sample_splits(
    n_rows: int,
    window_size: int,
    horizon: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    first_target_index = window_size + horizon - 1
    target_indices = np.arange(first_target_index, n_rows)
    if len(target_indices) < 3:
        raise ValueError(
            "Not enough rows to create train/validation/test windows. "
            f"Need more than {first_target_index + 2} rows."
        )

    train_end = max(1, int(len(target_indices) * train_ratio))
    val_end = max(train_end + 1, int(len(target_indices) * (train_ratio + validation_ratio)))
    if val_end >= len(target_indices):
        val_end = len(target_indices) - 1
    if train_end >= val_end:
        raise ValueError("Not enough samples for the requested chronological split.")

    return target_indices[:train_end], target_indices[train_end:val_end], target_indices[val_end:]


def _build_windows_from_indices(
    scaled_values: np.ndarray,
    raw_values: np.ndarray,
    target_indices: np.ndarray,
    window_size: int,
    horizon: int,
    target_feature_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows: list[np.ndarray] = []
    scaled_targets: list[float] = []
    actual_targets: list[float] = []
    naive_targets: list[float] = []
    for target_index in target_indices:
        window_end = target_index - horizon + 1
        window_start = window_end - window_size
        if window_start < 0:
            raise ValueError("Invalid window configuration produced a negative start index.")
        windows.append(scaled_values[window_start:window_end, :])
        scaled_targets.append(float(scaled_values[target_index, target_feature_index]))
        actual_targets.append(float(raw_values[target_index, target_feature_index]))
        naive_index = target_index - horizon
        naive_targets.append(float(raw_values[naive_index, target_feature_index]))

    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(scaled_targets, dtype=np.float32).reshape(-1, 1),
        np.asarray(actual_targets, dtype=np.float32).reshape(-1, 1),
        np.asarray(naive_targets, dtype=np.float32).reshape(-1, 1),
    )


def build_feature_dataset(
    df: pd.DataFrame,
    feature_names: list[str],
    target_column: str,
    window_size: int,
    horizon: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    fitted_scaler: MinMaxScaler | None = None,
) -> FeatureDataset:
    if target_column not in feature_names:
        raise ValueError("target_column must be present in feature_names.")

    selected = select_features(df, feature_names)
    train_indices, val_indices, test_indices = chronological_sample_splits(
        n_rows=len(selected),
        window_size=window_size,
        horizon=horizon,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

    train_fit_end = int(train_indices[-1]) + 1
    scaler = fitted_scaler or MinMaxScaler()
    if fitted_scaler is None:
        scaler.fit(selected.iloc[:train_fit_end].to_numpy(dtype=float))
    scaled_values = scaler.transform(selected.to_numpy(dtype=float))
    raw_values = selected.to_numpy(dtype=float)
    target_feature_index = feature_names.index(target_column)

    X_train, y_train, y_train_actual, naive_train = _build_windows_from_indices(
        scaled_values, raw_values, train_indices, window_size, horizon, target_feature_index
    )
    X_val, y_val, y_val_actual, naive_val = _build_windows_from_indices(
        scaled_values, raw_values, val_indices, window_size, horizon, target_feature_index
    )
    X_test, y_test, y_test_actual, naive_test = _build_windows_from_indices(
        scaled_values, raw_values, test_indices, window_size, horizon, target_feature_index
    )

    metadata: dict[str, Any] = {
        "feature_names": feature_names,
        "target_column": target_column,
        "target_feature_index": target_feature_index,
        "window_size": window_size,
        "horizon": horizon,
        "n_features": len(feature_names),
        "split": {
            "train_samples": int(len(train_indices)),
            "validation_samples": int(len(val_indices)),
            "test_samples": int(len(test_indices)),
            "train_target_start": str(selected.index[train_indices[0]].date()),
            "train_target_end": str(selected.index[train_indices[-1]].date()),
            "validation_target_start": str(selected.index[val_indices[0]].date()),
            "validation_target_end": str(selected.index[val_indices[-1]].date()),
            "test_target_start": str(selected.index[test_indices[0]].date()),
            "test_target_end": str(selected.index[test_indices[-1]].date()),
            "scaler_fit_end": str(selected.index[train_fit_end - 1].date()),
        },
    }

    return FeatureDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        y_train_actual=y_train_actual,
        y_val_actual=y_val_actual,
        y_test_actual=y_test_actual,
        naive_test_actual=naive_test,
        scaler=scaler,
        metadata=metadata,
    )
