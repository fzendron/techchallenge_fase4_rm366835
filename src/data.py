from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def validate_ticker(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("Ticker must not be empty.")
    if len(normalized) > 15:
        raise ValueError("Ticker is unexpectedly long.")
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-")
    if any(character not in allowed for character in normalized):
        raise ValueError("Ticker contains invalid characters.")
    return normalized


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Downloaded data is missing required columns: {missing}.")


def clean_price_data(df: pd.DataFrame, required_columns: Iterable[str] = REQUIRED_COLUMNS) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No price data was returned for the requested ticker/date range.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    validate_required_columns(df, required_columns)
    cleaned = df.loc[:, list(required_columns)].copy()
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]
    cleaned = cleaned.dropna(how="any")

    numeric_columns = list(required_columns)
    cleaned[numeric_columns] = cleaned[numeric_columns].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(how="any")

    if cleaned.empty:
        raise ValueError("Price data became empty after cleaning missing or non-numeric values.")
    if (cleaned["Volume"] < 0).any():
        raise ValueError("Price data contains negative volume values.")
    price_columns = [column for column in numeric_columns if column != "Volume"]
    if (cleaned[price_columns] <= 0).any().any():
        raise ValueError("Price data contains non-positive OHLC prices.")

    return cleaned


def download_price_data(
    ticker: str,
    start_date: str | date,
    end_date: str | date | None = None,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> pd.DataFrame:
    normalized_ticker = validate_ticker(ticker)
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date else None
    if end is not None and start >= end:
        raise ValueError("start_date must be earlier than end_date.")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required to download market data.") from exc

    raw = yf.download(
        normalized_ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d") if end is not None else None,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=False,
    )
    cleaned = clean_price_data(raw, required_columns=required_columns)
    if len(cleaned) < 2:
        raise ValueError("At least two trading days are required for next-day forecasting.")
    return cleaned
