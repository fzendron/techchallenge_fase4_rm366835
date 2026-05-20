from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from app.inference import InferenceError, StockPredictor
from app.schemas import PredictionRequest, PriceRow


class FakeModel:
    def predict(self, model_input, verbose: int = 0):
        assert model_input.shape == (1, 3, 5)
        return np.array([[0.5]], dtype=float)


def _rows(count: int) -> list[PriceRow]:
    return [
        PriceRow(Open=100 + i, High=101 + i, Low=99 + i, Close=100.5 + i, Volume=1000 + i)
        for i in range(count)
    ]


def _predictor() -> StockPredictor:
    scaler = MinMaxScaler()
    scaler.fit(
        np.array(
            [
                [100, 101, 99, 100, 1000],
                [110, 111, 109, 110, 2000],
            ],
            dtype=float,
        )
    )
    predictor = StockPredictor(Path("missing.keras"), Path("missing.joblib"), Path("missing.json"))
    predictor.model = FakeModel()
    predictor.scaler = scaler
    predictor.metadata = {
        "ticker": "AAPL",
        "feature_names": ["Open", "High", "Low", "Close", "Volume"],
        "target_column": "Close",
        "window_size": 3,
        "horizon": 1,
        "trained_at_utc": "test",
    }
    return predictor


def test_predict_rejects_too_few_rows() -> None:
    request = PredictionRequest(historical_prices=_rows(2))
    predictor = _predictor()

    try:
        predictor.predict(request)
    except InferenceError as exc:
        assert "At least 3 historical rows" in str(exc)
    else:
        raise AssertionError("Expected InferenceError")


def test_predict_returns_inverse_scaled_close() -> None:
    request = PredictionRequest(historical_prices=_rows(3), request_id="abc")
    predictor = _predictor()

    response = predictor.predict(request)

    assert response.ticker == "AAPL"
    assert response.request_id == "abc"
    assert response.predicted_close == 105.0
    assert response.feature_order == ["Open", "High", "Low", "Close", "Volume"]
