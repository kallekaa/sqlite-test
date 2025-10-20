"""Fetch OHLCV data from yfinance and store it in the SQLite database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import yfinance as yf


ROOT_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT_DIR / "config" / "schema.json"


def load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_ticker_frame(ticker: str, period: str, interval: str) -> pd.DataFrame:
    print(f"Fetching data for {ticker} ({period}, {interval})")
    data = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        actions=False,
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

    required = {"open", "high", "low", "close"}
    missing = required.difference(data.columns)
    if missing:
        print(f"Skipping {ticker}: missing columns {sorted(missing)}")
        return pd.DataFrame()

    if "adj_close" not in data.columns:
        data["adj_close"] = data["close"]
    if "volume" not in data.columns:
        data["volume"] = 0

    data = data.reset_index()
    if "Date" in data.columns:
        data = data.rename(columns={"Date": "date"})
    elif "index" in data.columns:
        data = data.rename(columns={"index": "date"})
    if pd.api.types.is_datetime64_any_dtype(data["date"]):
        data["date"] = data["date"].dt.strftime("%Y-%m-%d")

    data.insert(0, "ticker", ticker)
    data = data[["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]]
    data = data.dropna(subset=["open", "high", "low", "close", "adj_close"])
    data["volume"] = data["volume"].fillna(0).astype("int64")
    return data


def iter_rows(dataframe: pd.DataFrame, columns: List[str]) -> Iterable[Tuple[Any, ...]]:
    sanitized = dataframe.where(pd.notnull(dataframe), None)
    for record in sanitized.to_dict(orient="records"):
        yield tuple(record[column] for column in columns)


def store_frames(
    connection: sqlite3.Connection,
    table_schema: Dict[str, Any],
    frames: List[pd.DataFrame],
) -> None:
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if merged.empty:
        print("No data to insert.")
        return

    columns = [column["name"] for column in table_schema["columns"]]
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT OR REPLACE INTO {table_schema['name']} ({', '.join(columns)}) VALUES ({placeholders})"

    rows = list(iter_rows(merged, columns))
    print(f"Inserting {len(rows)} rows into {table_schema['name']}")
    connection.executemany(sql, rows)
    connection.commit()


def main() -> None:
    schema = load_schema()
    ingest_settings = schema.get("ingest", {})
    tickers: List[str] = ingest_settings.get("tickers", [])
    period: str = ingest_settings.get("period", "1mo")
    interval: str = ingest_settings.get("interval", "1d")

    if not tickers:
        raise ValueError("At least one ticker must be configured for ingestion.")

    db_path = ROOT_DIR / schema["database"]
    connection = sqlite3.connect(db_path)

    table_name = schema["query"]["table"]
    table_schema = next(
        (table for table in schema["tables"] if table["name"] == table_name),
        None,
    )
    if table_schema is None:
        connection.close()
        raise ValueError(f"Table {table_name} defined in query config is missing from schema.")

    try:
        frames = [
            frame
            for frame in (
                fetch_ticker_frame(ticker, period, interval)
                for ticker in tickers
            )
            if not frame.empty
        ]
        store_frames(connection, table_schema, frames)
    finally:
        connection.close()
        print("Database connection closed.")


if __name__ == "__main__":
    main()
