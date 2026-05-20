from __future__ import annotations

from datetime import date as Date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PriceRow(BaseModel):
    date: Date | None = None
    Open: float = Field(..., gt=0)
    High: float = Field(..., gt=0)
    Low: float = Field(..., gt=0)
    Close: float = Field(..., gt=0)
    Volume: float = Field(..., ge=0)


class PredictionRequest(BaseModel):
    historical_prices: list[PriceRow] = Field(..., min_length=1)
    request_id: str | None = None
    window_size: int | None = Field(
        default=None,
        description="Optional client-side validation size. The API also validates against model metadata.",
    )

    @model_validator(mode="after")
    def validate_optional_window_size(self) -> "PredictionRequest":
        if self.window_size is not None and len(self.historical_prices) < self.window_size:
            raise ValueError(f"At least {self.window_size} historical rows are required.")
        return self


class PredictionResponse(BaseModel):
    ticker: str | None
    horizon: int
    predicted_close: float
    currency: str | None = None
    model_version: str | None = None
    input_rows: int
    feature_order: list[str]
    request_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    metadata: dict[str, Any] | None = None
    error: str | None = None

    model_config = ConfigDict(protected_namespaces=())
