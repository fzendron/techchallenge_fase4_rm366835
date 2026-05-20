from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import yaml

from src.data import download_price_data
from src.features import build_feature_dataset
from src.utils import inverse_close_values, load_json, regression_metrics, save_json


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def evaluate(config_path: Path) -> dict[str, Any]:
    import tensorflow as tf

    config = load_config(config_path)
    artifact_config = config["artifacts"]
    metadata_path = Path(artifact_config["metadata_path"])
    model_path = Path(artifact_config["model_path"])
    scaler_path = Path(artifact_config["scaler_path"])
    metrics_path = Path(artifact_config["metrics_path"])

    if not model_path.exists() or not scaler_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model, scaler, and metadata artifacts must exist before evaluation.")

    metadata = load_json(metadata_path)
    scaler = joblib.load(scaler_path)
    model = tf.keras.models.load_model(model_path)

    data_config = config["data"]
    feature_config = config["features"]
    df = download_price_data(
        ticker=data_config["ticker"],
        start_date=data_config["start_date"],
        end_date=data_config.get("end_date"),
        required_columns=data_config["features"],
    )
    dataset = build_feature_dataset(
        df=df,
        feature_names=list(metadata["feature_names"]),
        target_column=metadata["target_column"],
        window_size=int(feature_config["window_size"]),
        horizon=int(feature_config["horizon"]),
        train_ratio=float(feature_config["train_ratio"]),
        validation_ratio=float(feature_config["validation_ratio"]),
        test_ratio=float(feature_config["test_ratio"]),
        fitted_scaler=scaler,
    )

    pred_scaled = model.predict(dataset.X_test, verbose=0)
    pred_actual = inverse_close_values(
        pred_scaled, scaler, metadata["feature_names"], metadata["target_column"]
    )
    y_test_actual = dataset.y_test_actual.reshape(-1)
    naive_actual = dataset.naive_test_actual.reshape(-1)
    metrics = {
        "lstm": regression_metrics(y_test_actual, pred_actual),
        "naive_baseline": regression_metrics(y_test_actual, naive_actual),
    }
    save_json(metrics, metrics_path)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained LSTM stock forecasting model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(evaluate(args.config))


if __name__ == "__main__":
    main()
