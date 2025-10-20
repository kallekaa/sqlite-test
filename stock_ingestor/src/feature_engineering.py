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
    "log_return_1d",
    "log_return_5d",
    "log_return_10d",
    "rolling_mean_return_5d",
    "rolling_std_return_5d",
    "rolling_sharpe_5d",
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
    "rolling_std_close_20d",
    "bollinger_upper_20d",
    "bollinger_lower_20d",
    "bollinger_bandwidth_20d",
    "macd_line",
    "macd_signal",
    "macd_hist",
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
    "drawdown",
    "max_drawdown_20d",
    "rsi_14",
    "streak_up",
    "streak_down",
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
    df["volume"] = df["volume"].clip(lower=0)
    grouped = df.groupby("ticker", group_keys=False)

    def _winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
        if series.count() == 0:
            return series
        lower_q = series.quantile(lower)
        upper_q = series.quantile(upper)
        return series.clip(lower_q, upper_q)

    # Lag features
    df["close_lag_1"] = grouped["close"].shift(1)
    df["close_lag_3"] = grouped["close"].shift(3)
    df["close_lag_5"] = grouped["close"].shift(5)
    df["volume_lag_1"] = grouped["volume"].shift(1)
    df["volume_lag_3"] = grouped["volume"].shift(3)

    # Return and momentum features
    prev_close = grouped["close"].shift(1).replace(0, np.nan)
    close_lag_5 = grouped["close"].shift(5).replace(0, np.nan)
    close_lag_10 = grouped["close"].shift(10).replace(0, np.nan)

    df["return_1d"] = (df["close"] / prev_close) - 1
    df["return_5d"] = (df["close"] / close_lag_5) - 1
    df["return_10d"] = (df["close"] / close_lag_10) - 1

    df["log_return_1d"] = np.log(df["close"] / prev_close)
    df["log_return_5d"] = np.log(df["close"] / close_lag_5)
    df["log_return_10d"] = np.log(df["close"] / close_lag_10)

    for column in [
        "return_1d",
        "return_5d",
        "return_10d",
        "log_return_1d",
        "log_return_5d",
        "log_return_10d",
    ]:
        df[column] = grouped[column].transform(_winsorize)

    df["rolling_mean_return_5d"] = grouped["return_1d"].transform(lambda s: s.rolling(window=5).mean())
    df["rolling_std_return_5d"] = grouped["return_1d"].transform(lambda s: s.rolling(window=5).std())
    df["rolling_sharpe_5d"] = df["rolling_mean_return_5d"] / (df["rolling_std_return_5d"].replace(0, np.nan))

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
    df["rolling_std_close_20d"] = grouped["close"].transform(lambda s: s.rolling(window=20).std())
    df["bollinger_upper_20d"] = df["sma_20"] + 2 * df["rolling_std_close_20d"]
    df["bollinger_lower_20d"] = df["sma_20"] - 2 * df["rolling_std_close_20d"]
    df["bollinger_bandwidth_20d"] = (df["bollinger_upper_20d"] - df["bollinger_lower_20d"]) / df["sma_20"]
    df["true_range"] = df["high"] - df["low"]
    df["avg_true_range_5d"] = grouped["true_range"].transform(lambda s: s.rolling(window=5).mean())
    df["volatility_ratio"] = df["rolling_std_close_5d"] / df["sma_5"]

    ema_12 = grouped["close"].transform(lambda s: s.ewm(span=12, adjust=False).mean())
    ema_26 = grouped["close"].transform(lambda s: s.ewm(span=26, adjust=False).mean())
    df["macd_line"] = ema_12 - ema_26
    df["macd_signal"] = grouped["macd_line"].transform(lambda s: s.ewm(span=9, adjust=False).mean())
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]

    # Volume-based features
    df["volume_change"] = grouped["volume"].pct_change(1)
    df["volume_change"] = grouped["volume_change"].transform(_winsorize)
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

    rolling_max = grouped["close"].cummax()
    df["drawdown"] = (df["close"] / rolling_max) - 1
    df["max_drawdown_20d"] = grouped["drawdown"].transform(lambda s: s.rolling(window=20).min())

    def _streaks(returns: pd.Series) -> pd.DataFrame:
        up = np.zeros(len(returns), dtype=int)
        down = np.zeros(len(returns), dtype=int)
        last_up = 0
        last_down = 0
        for idx, value in enumerate(returns):
            if pd.isna(value) or value == 0:
                last_up = 0
                last_down = 0
            elif value > 0:
                last_up += 1
                last_down = 0
            else:
                last_down += 1
                last_up = 0
            up[idx] = last_up
            down[idx] = last_down
        return pd.DataFrame({"streak_up": up, "streak_down": down}, index=returns.index)

    streaks = grouped["return_1d"].apply(_streaks)
    if isinstance(streaks.index, pd.MultiIndex):
        streaks = streaks.reset_index(level=0, drop=True)
    df["streak_up"] = streaks["streak_up"]
    df["streak_down"] = streaks["streak_down"]

    df["close_to_rolling_high_20d"] = df["close_to_rolling_high_20d"].replace([np.inf, -np.inf], np.nan)
    df["volatility_ratio"] = df["volatility_ratio"].replace([np.inf, -np.inf], np.nan)
    df["volume_zscore_5d"] = df["volume_zscore_5d"].replace([np.inf, -np.inf], np.nan)
    df["close_position_in_range"] = df["close_position_in_range"].replace([np.inf, -np.inf], np.nan)
    df["bollinger_bandwidth_20d"] = df["bollinger_bandwidth_20d"].replace([np.inf, -np.inf], np.nan)
    df["rolling_sharpe_5d"] = df["rolling_sharpe_5d"].replace([np.inf, -np.inf], np.nan)

    feature_df = df[FEATURE_COLUMNS].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    feature_df["date"] = feature_df["date"].dt.strftime("%Y-%m-%d")
    feature_df["volume_lag_1"] = feature_df["volume_lag_1"].round().astype("Int64")
    feature_df["volume_lag_3"] = feature_df["volume_lag_3"].round().astype("Int64")
    feature_df["streak_up"] = feature_df["streak_up"].fillna(0).round().astype("Int64")
    feature_df["streak_down"] = feature_df["streak_down"].fillna(0).round().astype("Int64")

    float_columns = [
        column
        for column in FEATURE_COLUMNS
        if column
        not in {"ticker", "date", "volume_lag_1", "volume_lag_3", "streak_up", "streak_down"}
    ]
    feature_df[float_columns] = feature_df[float_columns].astype("float64")

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
