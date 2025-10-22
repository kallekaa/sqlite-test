"""Train per-ticker logistic regression models to classify next-day price direction."""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

TARGET_COLUMN = "target_direction_next_day"
MODEL_DIR = REPORTS_DIR / "models" / "classification"
PLOTS_DIR = MODEL_DIR / "plots"
METRICS_PATH = MODEL_DIR / "logistic_regression_next_day_metrics.json"


def load_dataset(feature_columns: List[str], tickers: List[str]) -> pd.DataFrame:
    """Fetch feature rows and construct binary targets for next-day price direction."""
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
    prices["next_close"] = prices.groupby("ticker")["close"].shift(-1)
    prices[TARGET_COLUMN] = (prices["next_close"] > prices["close"]).astype("Int64")

    dataset = features.merge(
        prices[["ticker", "date", TARGET_COLUMN]],
        on=["ticker", "date"],
        how="inner",
    )

    dataset = dataset.dropna(subset=feature_columns + [TARGET_COLUMN])
    dataset = dataset.sort_values(["ticker", "date"]).reset_index(drop=True)
    dataset[feature_columns] = dataset[feature_columns].astype(float)
    dataset[TARGET_COLUMN] = dataset[TARGET_COLUMN].astype(int)

    return dataset


def build_pipeline() -> Pipeline:
    """Create the preprocessing and logistic regression pipeline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def temporal_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into chronological train/test sets."""
    if df.empty:
        return df, df

    split_index = int(len(df) * (1 - test_size))
    split_index = max(1, min(split_index, len(df) - 1))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def compute_classification_metrics(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    metrics = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }
    if y_true.nunique() > 1 and not y_proba.empty:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def class_distribution(y: pd.Series) -> Dict[str, float]:
    """Summarize class distribution."""
    if y.empty:
        return {}
    positives = y.sum()
    total = len(y)
    return {
        "positives": int(positives),
        "negatives": int(total - positives),
        "positive_rate": float(positives / total if total else 0.0),
    }


def save_model(model: Pipeline, feature_columns: List[str], ticker: str) -> Path:
    """Persist the trained model and metadata for future inference."""
    ensure_directory(MODEL_DIR)
    model_path = MODEL_DIR / f"logistic_regression_next_day_{ticker}.pkl"
    payload = {
        "model": model,
        "feature_columns": feature_columns,
        "ticker": ticker,
    }
    with model_path.open("wb") as fh:
        pickle.dump(payload, fh)
    return model_path


def plot_roc_curve(ticker: str, y_true: pd.Series, y_proba: pd.Series) -> Path:
    """Plot ROC curve for the test split."""
    ensure_directory(PLOTS_DIR)
    plt.figure(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=f"{ticker} ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title(f"ROC Curve - {ticker}")
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"roc_curve_{ticker}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def plot_confusion_matrix(ticker: str, y_true: pd.Series, y_pred: pd.Series) -> Path:
    """Plot confusion matrix for the test split."""
    ensure_directory(PLOTS_DIR)
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title(f"Confusion Matrix - {ticker}")
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"confusion_matrix_{ticker}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


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

        if train_df[TARGET_COLUMN].nunique() < 2:
            print(f"Skipping {ticker}: training set lacks class diversity.")
            continue

        model = build_pipeline()
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

        train_pred = pd.Series(model.predict(train_df[FEATURE_COLUMNS]), index=train_df.index)
        train_proba = pd.Series(model.predict_proba(train_df[FEATURE_COLUMNS])[:, 1], index=train_df.index)

        test_pred = pd.Series(model.predict(test_df[FEATURE_COLUMNS]), index=test_df.index)
        test_proba = pd.Series(model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1], index=test_df.index)

        train_metrics = compute_classification_metrics(train_df[TARGET_COLUMN], train_pred, train_proba)
        test_metrics = compute_classification_metrics(test_df[TARGET_COLUMN], test_pred, test_proba)

        train_distribution = class_distribution(train_df[TARGET_COLUMN])
        test_distribution = class_distribution(test_df[TARGET_COLUMN])

        roc_path = plot_roc_curve(ticker, test_df[TARGET_COLUMN], test_proba)
        cm_path = plot_confusion_matrix(ticker, test_df[TARGET_COLUMN], test_pred)
        model_path = save_model(model, FEATURE_COLUMNS, ticker)

        metrics_report[ticker] = {
            "train": {
                "size": len(train_df),
                "metrics": train_metrics,
                "class_distribution": train_distribution,
            },
            "test": {
                "size": len(test_df),
                "metrics": test_metrics,
                "class_distribution": test_distribution,
            },
            "artifacts": {
                "model_path": str(model_path),
                "roc_curve_path": str(roc_path),
                "confusion_matrix_path": str(cm_path),
            },
        }

        print(f"Ticker {ticker}: train={len(train_df)}, test={len(test_df)}")
        print(
            f"  Train Metrics -> Precision: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, F1: {train_metrics['f1']:.4f}, "
            f"ROC AUC: {train_metrics.get('roc_auc', float('nan')):.4f}"
        )
        print(
            f"  Test Metrics  -> Precision: {test_metrics['precision']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, "
            f"ROC AUC: {test_metrics.get('roc_auc', float('nan')):.4f}"
        )

    if not metrics_report:
        print("No models were trained.")
        return

    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics_report, fh, indent=2)

    print(f"Metrics report saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
