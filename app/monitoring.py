from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

REQUEST_COUNT = Counter(
    "stock_api_requests_total",
    "Total HTTP requests.",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "stock_api_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
)
ERROR_COUNT = Counter(
    "stock_api_errors_total",
    "Total HTTP errors.",
    ["method", "endpoint", "status_code"],
)
PREDICTION_COUNT = Counter("stock_api_predictions_total", "Total successful predictions.")
MODEL_LOADED = Gauge("stock_api_model_loaded", "Whether the model artifacts are loaded.")
PROCESS_MEMORY_BYTES = Gauge("stock_api_process_memory_bytes", "Resident memory used by the API process.")


def update_resource_metrics(model_loaded: bool) -> None:
    MODEL_LOADED.set(1 if model_loaded else 0)
    try:
        import psutil

        PROCESS_MEMORY_BYTES.set(psutil.Process().memory_info().rss)
    except Exception:  # noqa: BLE001 - metrics endpoint should remain available without psutil.
        PROCESS_MEMORY_BYTES.set(0)


async def prometheus_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    start = time.perf_counter()
    status_code = "500"
    try:
        response = await call_next(request)
        status_code = str(response.status_code)
        return response
    finally:
        endpoint = request.url.path
        latency = time.perf_counter() - start
        REQUEST_COUNT.labels(request.method, endpoint, status_code).inc()
        REQUEST_LATENCY.labels(request.method, endpoint).observe(latency)
        if status_code.startswith(("4", "5")):
            ERROR_COUNT.labels(request.method, endpoint, status_code).inc()


def metrics_response(model_loaded: bool) -> Response:
    update_resource_metrics(model_loaded)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
