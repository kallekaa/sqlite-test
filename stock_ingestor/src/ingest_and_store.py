"""Fetch OHLCV data from yfinance and store it in the SQLite database."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from utils import (
    QUERIES_DIR,
    ROOT_DIR,
    get_trading_calendar,
    load_schema,
    load_sql,
)

PRICE_COLUMNS = [
    "ticker",
    "date",
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]


def parse_date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    return datetime.fromisoformat(value).date()


def fetch_ticker_frame(
    ticker: str,
    start: date,
    end: date,
    interval: str,
) -> pd.DataFrame:
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    print(f"Fetching data for {ticker} ({start_str} -> {end_str}, {interval})")
    data = yf.download(
        tickers=ticker,
        start=start_str,
        end=end_str,
        interval=interval,
        auto_adjust=False,
        progress=False,
        actions=True,
        threads=False,
    )

    if data.empty:
        print(f"No data returned for {ticker}")
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.columns = [
        str(column).lower().replace(" ", "_") for column in data.columns
    ]

    # Drop action columns after adjustments are captured.
    for extra in ("dividends", "stock_splits"):
        if extra in data.columns:
            data = data.drop(columns=extra)

    required = {"open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        print(f"Skipping {ticker}: missing columns {sorted(missing)}")
        return pd.DataFrame()

    if "adj_close" not in data.columns:
        data["adj_close"] = data["close"]
    if "volume" not in data.columns:
        data["volume"] = 0

    index = pd.to_datetime(data.index)
    if index.tz is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")
    data.index = index

    calendar = get_trading_calendar(start, end)
    data = data.reindex(calendar)

    adj_close = data["adj_close"]
    close = data["close"].replace(0, np.nan)
    adj_factor = (adj_close / close).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    for column in ("open", "high", "low", "close"):
        data[column] = (data[column] * adj_factor).astype("float64")
    data["close"] = adj_close.astype("float64")
    data["adj_close"] = adj_close.astype("float64")
    safe_factor = adj_factor.replace(0, 1.0)
    data["volume"] = (data["volume"] / safe_factor).fillna(0)
    data["volume"] = data["volume"].clip(lower=0)

    float_columns = ["open", "high", "low", "close", "adj_close"]
    data[float_columns] = data[float_columns].ffill()
    data["volume"] = data["volume"].fillna(0)

    data = data.dropna(subset=float_columns)

    data = data.reset_index().rename(columns={"index": "timestamp_utc"})
    timestamp_series = pd.to_datetime(data["timestamp_utc"]).dt.tz_convert("UTC")
    data["timestamp_utc"] = timestamp_series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    data["date"] = timestamp_series.dt.strftime("%Y-%m-%d")

    data.insert(0, "ticker", ticker)
    data = data[PRICE_COLUMNS]
    data["volume"] = data["volume"].round().astype("int64")
    return data


def iter_records(dataframe: pd.DataFrame, columns: List[str]) -> Iterable[Dict[str, Any]]:
    sanitized = dataframe.where(pd.notnull(dataframe), None)
    for record in sanitized.to_dict(orient="records"):
        yield {column: record[column] for column in columns}


def store_frames(
    connection: sqlite3.Connection,
    table_name: str,
    frames: List[pd.DataFrame],
    insert_sql: str,
) -> None:
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        print("No data to insert.")
        return

    rows = list(iter_records(merged, PRICE_COLUMNS))
    print(f"Inserting {len(rows)} rows into {table_name}")
    connection.executemany(insert_sql, rows)
    connection.commit()


def main() -> None:
    schema = load_schema()
    ingest_settings = schema.get("ingest", {})
    query_cfg = schema.get("query", {})

    table_name = query_cfg.get("table")
    if not table_name:
        raise ValueError("Query configuration must specify 'table'.")

    tickers: List[str] = ingest_settings.get("tickers", [])
    end_date_cfg = ingest_settings.get("end_date")
    start_date_cfg = ingest_settings.get("start_date")

    end_date_dt = parse_date(end_date_cfg) or date.today()
    start_date_dt = parse_date(start_date_cfg) or (end_date_dt - timedelta(days=365))

    if start_date_dt >= end_date_dt:
        raise ValueError("Ingest start_date must be earlier than end_date.")

    interval: str = ingest_settings.get("interval", "1d")

    if not tickers:
        raise ValueError("At least one ticker must be configured for ingestion.")

    db_path = ROOT_DIR / schema["database"]
    insert_sql_path = QUERIES_DIR / f"insert_{table_name}.sql"
    insert_sql = load_sql(insert_sql_path)

    connection = sqlite3.connect(db_path)

    try:
        frames = [
            frame
            for frame in (
                fetch_ticker_frame(ticker, start_date_dt, end_date_dt, interval)
                for ticker in tickers
            )
            if not frame.empty
        ]
        store_frames(connection, table_name, frames, insert_sql)
    finally:
        connection.close()
        print("Database connection closed.")


if __name__ == "__main__":
    main()
