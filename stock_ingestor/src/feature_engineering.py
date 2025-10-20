"""Generate lag-based features from price history and store them in the database."""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

from utils import QUERIES_DIR, ROOT_DIR, load_schema, load_sql

FEATURE_COLUMNS = [
    "ticker",
    "date",
    "close_lag_1",
    "close_lag_3",
    "close_lag_5",
    "volume_lag_1",
    "volume_lag_3",
    "return_1d",
    "return_5d",
    "return_10d",
    "rolling_mean_return_5d",
    "rolling_std_return_5d",
    "momentum_5d",
    "price_change",
    "sma_5",
    "sma_10",
    "sma_20",
    "ema_5",
    "ema_10",
    "ema_20",
    "close_to_sma_5",
    "sma_ratio_5_20",
    "rolling_std_close_5d",
    "rolling_std_close_10d",
    "true_range",
    "avg_true_range_5d",
    "volatility_ratio",
    "volume_change",
    "volume_sma_5",
    "volume_sma_10",
    "volume_zscore_5d",
    "price_volume_trend",
    "high_low_spread",
    "close_position_in_range",
    "rolling_high_20d",
    "rolling_low_20d",
    "close_to_rolling_high_20d",
    "rsi_14",
]


def fetch_price_history(
    connection: sqlite3.Connection,
    tickers: List[str] | None,
) -> pd.DataFrame:
    base_sql = """
        SELECT ticker, date, open, high, low, close, volume
        FROM prices
    """
    params: List[Any] = []
    if tickers:
        placeholders = ", ".join(["?"] * len(tickers))
        base_sql = f"{base_sql} WHERE ticker IN ({placeholders})"
        params.extend(tickers)

    base_sql = f"{base_sql} ORDER BY ticker ASC, date ASC"
    df = pd.read_sql_query(base_sql, connection, params=params)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker", group_keys=False)

    # Lag features
    df["close_lag_1"] = grouped["close"].shift(1)
    df["close_lag_3"] = grouped["close"].shift(3)
    df["close_lag_5"] = grouped["close"].shift(5)
    df["volume_lag_1"] = grouped["volume"].shift(1)
    df["volume_lag_3"] = grouped["volume"].shift(3)

    # Return and momentum features
    df["return_1d"] = grouped["close"].pct_change(1)
    df["return_5d"] = grouped["close"].pct_change(5)
    df["return_10d"] = grouped["close"].pct_change(10)
    df["rolling_mean_return_5d"] = grouped["return_1d"].transform(lambda s: s.rolling(window=5).mean())
    df["rolling_std_return_5d"] = grouped["return_1d"].transform(lambda s: s.rolling(window=5).std())
    df["momentum_5d"] = df["close"] / grouped["close"].shift(5)
    df["price_change"] = df["close"] - df["open"]

    # Moving averages and ratios
    df["sma_5"] = grouped["close"].transform(lambda s: s.rolling(window=5).mean())
    df["sma_10"] = grouped["close"].transform(lambda s: s.rolling(window=10).mean())
    df["sma_20"] = grouped["close"].transform(lambda s: s.rolling(window=20).mean())
    df["ema_5"] = grouped["close"].transform(lambda s: s.ewm(span=5, adjust=False).mean())
    df["ema_10"] = grouped["close"].transform(lambda s: s.ewm(span=10, adjust=False).mean())
    df["ema_20"] = grouped["close"].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    df["close_to_sma_5"] = df["close"] / df["sma_5"] - 1
    df["sma_ratio_5_20"] = df["sma_5"] / df["sma_20"]

    # Volatility features
    df["rolling_std_close_5d"] = grouped["close"].transform(lambda s: s.rolling(window=5).std())
    df["rolling_std_close_10d"] = grouped["close"].transform(lambda s: s.rolling(window=10).std())
    df["true_range"] = df["high"] - df["low"]
    df["avg_true_range_5d"] = grouped["true_range"].transform(lambda s: s.rolling(window=5).mean())
    df["volatility_ratio"] = df["rolling_std_close_5d"] / df["sma_5"]

    # Volume-based features
    df["volume_change"] = grouped["volume"].pct_change(1)
    df["volume_sma_5"] = grouped["volume"].transform(lambda s: s.rolling(window=5).mean())
    df["volume_sma_10"] = grouped["volume"].transform(lambda s: s.rolling(window=10).mean())
    df["volume_zscore_5d"] = grouped["volume"].transform(
        lambda s: (s - s.rolling(window=5).mean()) / s.rolling(window=5).std()
    )
    df["pvt_component"] = df["return_1d"].fillna(0) * df["volume"].fillna(0)
    df["price_volume_trend"] = grouped["pvt_component"].cumsum()

    # Price range and relative strength features
    price_range = df["high"] - df["low"]
    df["high_low_spread"] = price_range / df["close"]
    df["close_position_in_range"] = (df["close"] - df["low"]) / price_range.replace(0, np.nan)
    df["rolling_high_20d"] = grouped["high"].transform(lambda s: s.rolling(window=20).max())
    df["rolling_low_20d"] = grouped["low"].transform(lambda s: s.rolling(window=20).min())
    df["close_to_rolling_high_20d"] = df["close"] / df["rolling_high_20d"]

    delta = grouped["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    df["gain"] = gain
    df["loss"] = loss
    avg_gain = grouped["gain"].transform(lambda s: s.rolling(window=14).mean())
    avg_loss = grouped["loss"].transform(lambda s: s.rolling(window=14).mean())
    rs = avg_gain / avg_loss
    df["rsi_14"] = 100 - (100 / (1 + rs))
    df.loc[avg_loss == 0, "rsi_14"] = 100
    df.loc[(avg_gain == 0) & (avg_loss == 0), "rsi_14"] = 50

    df["close_to_rolling_high_20d"] = df["close_to_rolling_high_20d"].replace([np.inf, -np.inf], np.nan)
    df["volatility_ratio"] = df["volatility_ratio"].replace([np.inf, -np.inf], np.nan)
    df["volume_zscore_5d"] = df["volume_zscore_5d"].replace([np.inf, -np.inf], np.nan)
    df["close_position_in_range"] = df["close_position_in_range"].replace([np.inf, -np.inf], np.nan)

    feature_df = df[FEATURE_COLUMNS].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    feature_df["date"] = feature_df["date"].dt.strftime("%Y-%m-%d")
    feature_df["volume_lag_1"] = feature_df["volume_lag_1"].round().astype("Int64")
    feature_df["volume_lag_3"] = feature_df["volume_lag_3"].round().astype("Int64")

    return feature_df[FEATURE_COLUMNS]


def iter_records(dataframe: pd.DataFrame) -> Iterable[Dict[str, Any]]:
    sanitized = dataframe.where(pd.notnull(dataframe), None)
    for record in sanitized.to_dict(orient="records"):
        yield record


def store_features(
    connection: sqlite3.Connection,
    features: pd.DataFrame,
    insert_sql: str,
) -> None:
    if features.empty:
        print("No feature rows to insert.")
        return

    rows = list(iter_records(features))
    print(f"Inserting {len(rows)} feature rows.")
    connection.executemany(insert_sql, rows)
    connection.commit()


def main() -> None:
    schema = load_schema()
    ingest_cfg = schema.get("ingest", {})
    tickers: List[str] = ingest_cfg.get("tickers", [])

    db_path = ROOT_DIR / schema["database"]
    insert_sql_path = QUERIES_DIR / "insert_features.sql"
    insert_sql = load_sql(insert_sql_path)

    connection = sqlite3.connect(db_path)
    try:
        price_df = fetch_price_history(connection, tickers or None)
        features = compute_features(price_df)
        store_features(connection, features, insert_sql)
    finally:
        connection.close()
        print("Feature engineering complete.")


if __name__ == "__main__":
    main()
