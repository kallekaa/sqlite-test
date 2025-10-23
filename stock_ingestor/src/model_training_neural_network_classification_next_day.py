"""Train per-ticker neural network classifiers (Keras) for next-day price direction."""

from __future__ import annotations

import json
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

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
MODEL_DIR = REPORTS_DIR / "models" / "neural_network_classification"
PLOTS_DIR = MODEL_DIR / "plots"
METRICS_PATH = MODEL_DIR / "neural_network_classification_next_day_metrics.json"

RANDOM_SEED = 42


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


def temporal_split(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into chronological train/test sets."""
    if df.empty:
        return df, df

    split_index = int(len(df) * (1 - test_size))
    split_index = max(1, min(split_index, len(df) - 1))
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    return train_df, test_df


def prepare_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Scale features and return numpy arrays for Keras."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[feature_columns])
    X_test = scaler.transform(test_df[feature_columns])

    y_train = train_df[target_column].astype(float).values
    y_test = test_df[target_column].astype(float).values
    return X_train, X_test, y_train, y_test, scaler


def build_model(input_dim: int) -> keras.Model:
    """Create a small dense network for binary classification."""
    keras.utils.set_random_seed(RANDOM_SEED)
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    """Compute standard binary classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def class_distribution(y: pd.Series) -> Dict[str, float]:
    """Summarize class distribution."""
    if y.empty:
        return {}
    positives = int(y.sum())
    total = len(y)
    return {
        "positives": positives,
        "negatives": int(total - positives),
        "positive_rate": float(positives / total if total else 0.0),
    }


def plot_roc_curve(ticker: str, y_true: np.ndarray, y_proba: np.ndarray) -> Path:
    """Plot ROC curve for the test split."""
    ensure_directory(PLOTS_DIR)
    plt.figure(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, y_proba, name=f"{ticker} ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title(f"ROC Curve - {ticker}")
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"nn_classification_roc_{ticker}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def plot_confusion_matrix(ticker: str, y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    """Plot confusion matrix for the test split."""
    ensure_directory(PLOTS_DIR)
    plt.figure(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {ticker}")
    plt.tight_layout()
    plot_path = PLOTS_DIR / f"nn_classification_confusion_{ticker}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def save_model(model: keras.Model, scaler: StandardScaler, ticker: str) -> Dict[str, str]:
    """Persist the trained Keras model alongside the fitted scaler."""
    ensure_directory(MODEL_DIR)
    model_path = MODEL_DIR / f"nn_classification_next_day_{ticker}.keras"
    scaler_path = MODEL_DIR / f"nn_classification_next_day_{ticker}_scaler.pkl"

    # Save the TensorFlow model (includes architecture + weights).
    model.save(model_path)

    # Persist the scaler used for feature normalization.
    with scaler_path.open("wb") as fh:
        pickle.dump(scaler, fh)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }


def main() -> None:
    ensure_directory(MODEL_DIR)
    ensure_directory(PLOTS_DIR)

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

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

        X_train, X_test, y_train, y_test, scaler = prepare_arrays(train_df, test_df, FEATURE_COLUMNS, TARGET_COLUMN)

        model = build_model(len(FEATURE_COLUMNS))

        # Early stopping prevents overfitting and restores the best checkpoint.
        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=min(64, len(X_train)),
            callbacks=[early_stopping],
            verbose=0,
        )

        train_proba = model.predict(X_train, verbose=0).ravel()
        test_proba = model.predict(X_test, verbose=0).ravel()

        train_pred = (train_proba >= 0.5).astype(int)
        test_pred = (test_proba >= 0.5).astype(int)

        train_metrics = compute_metrics(y_train, train_pred, train_proba)
        test_metrics = compute_metrics(y_test, test_pred, test_proba)

        train_distribution = class_distribution(train_df[TARGET_COLUMN])
        test_distribution = class_distribution(test_df[TARGET_COLUMN])

        roc_path = plot_roc_curve(ticker, y_test, test_proba)
        cm_path = plot_confusion_matrix(ticker, y_test, test_pred)
        artifact_paths = save_model(model, scaler, ticker)

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
                **artifact_paths,
                "roc_curve_path": str(roc_path),
                "confusion_matrix_path": str(cm_path),
            },
            "training_history": history.history,
        }

        print(f"Ticker {ticker}: train={len(train_df)}, test={len(test_df)}")
        print(
            f"  Train Metrics -> Accuracy: {train_metrics['accuracy']:.4f}, "
            f"Precision: {train_metrics['precision']:.4f}, "
            f"Recall: {train_metrics['recall']:.4f}, "
            f"F1: {train_metrics['f1']:.4f}"
        )
        print(
            f"  Test Metrics  -> Accuracy: {test_metrics['accuracy']:.4f}, "
            f"Precision: {test_metrics['precision']:.4f}, "
            f"Recall: {test_metrics['recall']:.4f}, "
            f"F1: {test_metrics['f1']:.4f}, "
            f"ROC AUC: {test_metrics.get('roc_auc', float('nan')):.4f}"
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
