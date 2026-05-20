from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

from src.data import download_price_data
from src.features import build_feature_dataset
from src.model import build_lstm_model
from src.utils import ensure_parent_dir, inverse_close_values, regression_metrics, save_json, set_random_seed


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _path_from_config(config: dict[str, Any], key: str) -> Path:
    return Path(config["artifacts"][key])


def _plot_training_history(history: Any, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_parent_dir(output_path)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history.get("loss", []), label="train_loss")
    plt.plot(history.history.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    ensure_parent_dir(output_path)
    plt.figure(figsize=(10, 5))
    plt.plot(y_true.reshape(-1), label="actual")
    plt.plot(y_pred.reshape(-1), label="lstm_prediction")
    plt.plot(baseline.reshape(-1), label="naive_baseline", alpha=0.75)
    plt.xlabel("Test sample")
    plt.ylabel("Close price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train(config_path: Path) -> dict[str, Any]:
    import tensorflow as tf

    config = load_config(config_path)
    set_random_seed(int(config.get("project", {}).get("random_seed", 42)))

    data_config = config["data"]
    feature_config = config["features"]
    artifact_config = config["artifacts"]

    df = download_price_data(
        ticker=data_config["ticker"],
        start_date=data_config["start_date"],
        end_date=data_config.get("end_date"),
        required_columns=data_config["features"],
    )
    dataset = build_feature_dataset(
        df=df,
        feature_names=list(data_config["features"]),
        target_column=data_config["target"],
        window_size=int(feature_config["window_size"]),
        horizon=int(feature_config["horizon"]),
        train_ratio=float(feature_config["train_ratio"]),
        validation_ratio=float(feature_config["validation_ratio"]),
        test_ratio=float(feature_config["test_ratio"]),
    )

    model_path = Path(artifact_config["model_path"])
    scaler_path = Path(artifact_config["scaler_path"])
    metadata_path = Path(artifact_config["metadata_path"])
    metrics_path = Path(artifact_config["metrics_path"])
    ensure_parent_dir(model_path)

    model = build_lstm_model(
        window_size=dataset.metadata["window_size"],
        n_features=dataset.metadata["n_features"],
        config=config,
    )
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=int(config["model"].get("patience", 8)),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=int(config["model"].get("reduce_lr_patience", 4)),
            factor=0.5,
            min_lr=1e-6,
        ),
    ]

    history = model.fit(
        dataset.X_train,
        dataset.y_train,
        validation_data=(dataset.X_val, dataset.y_val),
        epochs=int(config["model"].get("epochs", 30)),
        batch_size=int(config["model"].get("batch_size", 32)),
        callbacks=callbacks,
        verbose=1,
    )
    model.save(model_path)

    test_pred_scaled = model.predict(dataset.X_test, verbose=0)
    test_pred_actual = inverse_close_values(
        test_pred_scaled, dataset.scaler, dataset.metadata["feature_names"], dataset.metadata["target_column"]
    )
    y_test_actual = dataset.y_test_actual.reshape(-1)
    naive_actual = dataset.naive_test_actual.reshape(-1)

    metrics = {
        "lstm": regression_metrics(y_test_actual, test_pred_actual),
        "naive_baseline": regression_metrics(y_test_actual, naive_actual),
    }

    metadata = {
        **dataset.metadata,
        "ticker": data_config["ticker"],
        "data_start": str(df.index.min().date()),
        "data_end": str(df.index.max().date()),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "metrics_path": str(metrics_path),
        "training_config": config,
        "metrics": metrics,
    }

    ensure_parent_dir(scaler_path)
    joblib.dump(dataset.scaler, scaler_path)
    save_json(metadata, metadata_path)
    save_json(metrics, metrics_path)
    _plot_training_history(history, Path(artifact_config["training_plot_path"]))
    _plot_predictions(
        y_true=y_test_actual,
        y_pred=test_pred_actual,
        baseline=naive_actual,
        output_path=Path(artifact_config["prediction_plot_path"]),
    )
    return {"metrics": metrics, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LSTM stock forecasting model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train(args.config)
    print(result["metrics"])


if __name__ == "__main__":
    main()
