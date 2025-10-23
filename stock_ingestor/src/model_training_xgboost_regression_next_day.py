"""Train per-ticker XGBoost regressors to predict next-day closing price."""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from utils import REPORTS_DIR, ROOT_DIR, ensure_directory, load_schema


FEATURE_COLUMNS: List[str] = [
    "close_lag_1",
    "close_lag_3",
    "return_1d",
    "rolling_mean_return_5d",
    "rsi_14",
    "volume_lag_1",
]

SCHEMA = load_schema()
DATABASE_PATH = ROOT_DIR / SCHEMA["database"]
TARGET_TICKERS: List[str] = SCHEMA.get("ingest", {}).get("tickers", ["AAPL", "MSFT", "GOOGL"])

TARGET_COLUMN = "target_close_next_day"
MODEL_DIR = REPORTS_DIR / "models" / "xgboost_regression"
PLOTS_DIR = MODEL_DIR / "plots"
METRICS_PATH = MODEL_DIR / "xgboost_regression_next_day_metrics.json"


def load_dataset(feature_columns: List[str], tickers: List[str]) -> pd.DataFrame:
    """Fetch feature rows and align them with the next-day close target."""
    column_sql = ", ".join(feature_columns)
    feature_sql = f"SELECT ticker, date, {column_sql} FROM features"
    params: List[str] = []

    if tickers:
        placeholders = ", ".join(["?"] * len(tickers))
        feature_sql = f"{feature_sql} WHERE ticker IN ({placeholders})"
        params.extend(tickers)

    feature_sql = f"{feature_sql} ORDER BY ticker ASC, date ASC"

    price_sql = "SELECT ticker, date, close FROM prices"
    if tickers:
        placeholders = ", ".join(["?"] * len(tickers))
        price_sql = f"{price_sql} WHERE ticker IN ({placeholders}) ORDER BY ticker ASC, date ASC"
    else:
        price_sql = f"{price_sql} ORDER BY ticker ASC, date ASC"

    with sqlite3.connect(DATABASE_PATH) as connection:
        features = pd.read_sql_query(feature_sql, connection, params=params or None)
        prices = pd.read_sql_query(price_sql, connection, params=params or None)

    if features.empty or prices.empty:
        return pd.DataFrame()

    features["date"] = pd.to_datetime(features["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    prices = prices.sort_values(["ticker", "date"])
    prices[TARGET_COLUMN] = prices.groupby("ticker")["close"].shift(-1)

    dataset = features.merge(
        prices[["ticker", "date", TARGET_COLUMN]],
        on=["ticker", "date"],
        how="inner",
    )

    dataset = dataset.dropna(subset=feature_columns + [TARGET_COLUMN])
    dataset = dataset.sort_values(["ticker", "date"]).reset_index(drop=True)
    dataset[feature_columns] = dataset[feature_columns].astype(float)

    return dataset


def temporal_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into chronological train/test sets."""
    if df.empty:
        return df, df

    split_index = int(len(df) * (1 - test_size))
    split_index = max(1, min(split_index, len(df) - 1))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
    }


def descriptive_stats(series: pd.Series) -> Dict[str, float]:
    """Return descriptive statistics for a numeric series."""
    if series.empty:
        return {}
    return {
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
    }


def plot_predictions(
    ticker: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
) -> Path:
    """Plot actual vs predicted closes for train and test segments."""
    ensure_directory(PLOTS_DIR)
    plt.figure(figsize=(12, 6))

    plt.plot(train_df["date"], train_df[TARGET_COLUMN], label="Train Actual", color="tab:blue")
    plt.plot(train_df["date"], train_pred, label="Train Predicted", color="tab:cyan", linestyle="--")

    if not test_df.empty:
        plt.plot(test_df["date"], test_df[TARGET_COLUMN], label="Test Actual", color="tab:orange")
        plt.plot(test_df["date"], test_pred, label="Test Predicted", color="tab:red", linestyle="--")

    plt.title(f"XGBoost Next-Day Close Prediction - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    plot_path = PLOTS_DIR / f"xgb_next_day_predictions_{ticker}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def save_model(model: XGBRegressor, scaler: StandardScaler, ticker: str) -> Dict[str, str]:
    """Persist the trained model and scaler."""
    ensure_directory(MODEL_DIR)
    model_path = MODEL_DIR / f"xgb_regression_next_day_{ticker}.json"
    scaler_path = MODEL_DIR / f"xgb_regression_next_day_{ticker}_scaler.pkl"

    model.save_model(model_path)
    with scaler_path.open("wb") as fh:
        pickle.dump(scaler, fh)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }


def main() -> None:
    ensure_directory(MODEL_DIR)
    ensure_directory(PLOTS_DIR)

    dataset = load_dataset(FEATURE_COLUMNS, TARGET_TICKERS)
    if dataset.empty:
        print("No feature rows available after preprocessing. Aborting.")
        return

    metrics_report: Dict[str, Dict[str, object]] = {}

    for ticker in TARGET_TICKERS:
        ticker_df = dataset[dataset["ticker"] == ticker].copy()
        if len(ticker_df) < 2:
            print(f"Skipping {ticker}: not enough observations.")
            continue

        train_df, test_df = temporal_split(ticker_df)
        if train_df.empty or test_df.empty:
            print(f"Skipping {ticker}: unable to create a valid train/test split.")
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_df[FEATURE_COLUMNS])
        X_test = scaler.transform(test_df[FEATURE_COLUMNS])
        y_train = train_df[TARGET_COLUMN].values
        y_test = test_df[TARGET_COLUMN].values

        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        train_metrics = compute_metrics(y_train, train_pred)
        test_metrics = compute_metrics(y_test, test_pred)

        train_stats = descriptive_stats(train_df[TARGET_COLUMN])
        test_stats = descriptive_stats(test_df[TARGET_COLUMN])

        plot_path = plot_predictions(ticker, train_df, test_df, train_pred, test_pred)
        artifact_paths = save_model(model, scaler, ticker)

        metrics_report[ticker] = {
            "train": {
                "size": len(train_df),
                "metrics": train_metrics,
                "target_stats": train_stats,
            },
            "test": {
                "size": len(test_df),
                "metrics": test_metrics,
                "target_stats": test_stats,
            },
            "artifacts": {
                **artifact_paths,
                "plot_path": str(plot_path),
            },
        }

        print(f"Ticker {ticker}: train={len(train_df)}, test={len(test_df)}")
        print(
            f"  Train Metrics -> RMSE: {train_metrics['rmse']:.4f}, "
            f"MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}"
        )
        print(
            f"  Test Metrics  -> RMSE: {test_metrics['rmse']:.4f}, "
            f"MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}"
        )

    if not metrics_report:
        print("No models were trained.")
        return

    ensure_directory(MODEL_DIR)
    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics_report, fh, indent=2)

    print(f"Metrics report saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
