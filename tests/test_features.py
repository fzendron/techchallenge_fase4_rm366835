from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import build_feature_dataset


def _sample_frame(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="B")
    base = np.arange(rows, dtype=float) + 100
    return pd.DataFrame(
        {
            "Open": base,
            "High": base + 2,
            "Low": base - 2,
            "Close": base + 1,
            "Volume": base * 1000,
        },
        index=index,
    )


def test_build_feature_dataset_shapes_and_chronological_split() -> None:
    df = _sample_frame()
    dataset = build_feature_dataset(
        df=df,
        feature_names=["Open", "High", "Low", "Close", "Volume"],
        target_column="Close",
        window_size=10,
        horizon=1,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
    )

    assert dataset.X_train.shape[1:] == (10, 5)
    assert dataset.y_train.shape[1:] == (1,)
    assert dataset.X_val.shape[1:] == (10, 5)
    assert dataset.X_test.shape[1:] == (10, 5)
    assert dataset.metadata["split"]["train_target_end"] < dataset.metadata["split"]["validation_target_start"]
    assert dataset.metadata["split"]["validation_target_end"] < dataset.metadata["split"]["test_target_start"]


def test_scaler_is_fit_only_on_training_period() -> None:
    df = _sample_frame()
    dataset = build_feature_dataset(
        df=df,
        feature_names=["Open", "High", "Low", "Close", "Volume"],
        target_column="Close",
        window_size=10,
        horizon=1,
        train_ratio=0.7,
        validation_ratio=0.15,
        test_ratio=0.15,
    )

    train_end_date = pd.Timestamp(dataset.metadata["split"]["scaler_fit_end"])
    train_rows = df.loc[:train_end_date, ["Open", "High", "Low", "Close", "Volume"]]
    np.testing.assert_allclose(dataset.scaler.data_min_, train_rows.min().to_numpy(dtype=float))
    np.testing.assert_allclose(dataset.scaler.data_max_, train_rows.max().to_numpy(dtype=float))
