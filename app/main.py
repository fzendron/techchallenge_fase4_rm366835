from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pythonjsonlogger.json import JsonFormatter

from app.config import get_settings
from app.inference import InferenceError, StockPredictor
from app.monitoring import PREDICTION_COUNT, metrics_response, prometheus_middleware
from app.schemas import HealthResponse, ModelInfoResponse, PredictionRequest, PredictionResponse


def configure_logging(level: str) -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.handlers = [handler]


def create_app(predictor: StockPredictor | None = None) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = logging.getLogger("stock-api")
    stock_predictor = predictor or StockPredictor(
        model_path=settings.model_path,
        scaler_path=settings.scaler_path,
        metadata_path=settings.metadata_path,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if not stock_predictor.is_loaded:
            stock_predictor.load()
        if stock_predictor.is_loaded:
            logger.info("model_loaded", extra={"metadata": stock_predictor.model_info()})
        else:
            logger.warning("model_not_loaded", extra={"error": stock_predictor.load_error})
        yield

    app = FastAPI(
        title="FIAP Fase 4 LSTM Stock Forecast API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.predictor = stock_predictor
    app.middleware("http")(prometheus_middleware)

    @app.exception_handler(InferenceError)
    async def inference_exception_handler(_: Request, exc: InferenceError) -> JSONResponse:
        logger.warning("prediction_validation_error", extra={"error": str(exc)})
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error", extra={"error": str(exc)})
        return JSONResponse(status_code=500, content={"detail": "Internal server error."})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=stock_predictor.is_loaded)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return ModelInfoResponse(
            model_loaded=stock_predictor.is_loaded,
            metadata=stock_predictor.model_info(),
            error=stock_predictor.load_error,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        if not stock_predictor.is_loaded:
            raise HTTPException(status_code=503, detail=stock_predictor.load_error or "Model not loaded.")
        response = stock_predictor.predict(request)
        PREDICTION_COUNT.inc()
        logger.info(
            "prediction_completed",
            extra={
                "request_id": response.request_id,
                "ticker": response.ticker,
                "predicted_close": response.predicted_close,
            },
        )
        return response

    @app.get("/metrics")
    def metrics():
        return metrics_response(stock_predictor.is_loaded)

    return app


app = create_app()
