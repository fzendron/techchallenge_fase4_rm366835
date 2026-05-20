from __future__ import annotations

from app.main import create_app
from app.schemas import PredictionRequest, PredictionResponse


class FakePredictor:
    is_loaded = True
    load_error = None

    def load(self) -> None:
        return None

    def model_info(self):
        return {
            "ticker": "AAPL",
            "window_size": 3,
            "feature_names": ["Open", "High", "Low", "Close", "Volume"],
        }

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        return PredictionResponse(
            ticker="AAPL",
            horizon=1,
            predicted_close=123.45,
            input_rows=len(request.historical_prices),
            feature_order=["Open", "High", "Low", "Close", "Volume"],
            request_id=request.request_id,
        )


def _payload(rows: int = 3):
    return {
        "request_id": "test-request",
        "historical_prices": [
            {
                "date": f"2024-01-0{index + 1}",
                "Open": 100 + index,
                "High": 101 + index,
                "Low": 99 + index,
                "Close": 100.5 + index,
                "Volume": 1000 + index,
            }
            for index in range(rows)
        ],
    }


def test_health_and_model_info() -> None:
    from fastapi.testclient import TestClient

    with TestClient(create_app(FakePredictor())) as client:
        health = client.get("/health")
        model_info = client.get("/model-info")

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": True}
    assert model_info.status_code == 200
    assert model_info.json()["metadata"]["ticker"] == "AAPL"


def test_predict_endpoint() -> None:
    from fastapi.testclient import TestClient

    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/predict", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["ticker"] == "AAPL"
    assert body["predicted_close"] == 123.45
    assert body["request_id"] == "test-request"


def test_metrics_endpoint() -> None:
    from fastapi.testclient import TestClient

    with TestClient(create_app(FakePredictor())) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    assert "stock_api_requests_total" in response.text
