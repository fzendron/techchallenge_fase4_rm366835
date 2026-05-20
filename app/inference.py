from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.schemas import PriceRow, PredictionRequest, PredictionResponse
from src.utils import inverse_close_values, load_json


class InferenceError(ValueError):
    """Raised when a prediction request cannot be transformed for inference."""


class StockPredictor:
    def __init__(self, model_path: Path, scaler_path: Path, metadata_path: Path) -> None:
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.metadata: dict[str, Any] | None = None
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None and self.metadata is not None

    def load(self) -> None:
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler artifact not found: {self.scaler_path}")
            if not self.metadata_path.exists():
                raise FileNotFoundError(f"Metadata artifact not found: {self.metadata_path}")

            import tensorflow as tf

            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.metadata = load_json(self.metadata_path)
            self.load_error = None
        except Exception as exc:  # noqa: BLE001 - surface artifact problems through /model-info.
            self.model = None
            self.scaler = None
            self.metadata = None
            self.load_error = str(exc)

    def model_info(self) -> dict[str, Any] | None:
        return self.metadata

    def _request_to_frame(self, rows: list[PriceRow]) -> pd.DataFrame:
        frame = pd.DataFrame([row.model_dump() for row in rows])
        if "date" in frame.columns and frame["date"].notna().any():
            frame["date"] = pd.to_datetime(frame["date"])
            frame = frame.sort_values("date")
        return frame.reset_index(drop=True)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        if not self.is_loaded:
            raise RuntimeError(self.load_error or "Model artifacts are not loaded.")
        assert self.model is not None
        assert self.scaler is not None
        assert self.metadata is not None

        feature_order = list(self.metadata["feature_names"])
        target_column = str(self.metadata["target_column"])
        window_size = int(self.metadata["window_size"])
        horizon = int(self.metadata.get("horizon", 1))
        frame = self._request_to_frame(request.historical_prices)

        if len(frame) < window_size:
            raise InferenceError(f"At least {window_size} historical rows are required.")
        missing = [feature for feature in feature_order if feature not in frame.columns]
        if missing:
            raise InferenceError(f"Request is missing required features: {missing}.")

        feature_frame = frame.loc[:, feature_order].tail(window_size)
        if feature_frame.isna().any().any():
            raise InferenceError("Historical price rows contain missing feature values.")

        scaled_window = self.scaler.transform(feature_frame.to_numpy(dtype=float))
        model_input = scaled_window.reshape(1, window_size, len(feature_order))
        pred_scaled = np.asarray(self.model.predict(model_input, verbose=0)).reshape(-1)
        pred_actual = inverse_close_values(pred_scaled, self.scaler, feature_order, target_column)

        return PredictionResponse(
            ticker=self.metadata.get("ticker"),
            horizon=horizon,
            predicted_close=float(pred_actual[0]),
            currency=self.metadata.get("currency"),
            model_version=self.metadata.get("trained_at_utc"),
            input_rows=len(frame),
            feature_order=feature_order,
            request_id=request.request_id,
        )
