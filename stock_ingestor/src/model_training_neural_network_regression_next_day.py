"""Train a Keras neural network to predict next-day close for a single ticker."""

from __future__ import annotations

import json
import pickle
import sqlite3
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
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
ALL_TICKERS: List[str] = SCHEMA.get("ingest", {}).get("tickers", ["AAPL", "MSFT", "GOOGL"])
# Restrict training to a single representative ticker (the first configured symbol).
TARGET_TICKER = ALL_TICKERS[0] if ALL_TICKERS else "AAPL"

TARGET_COLUMN = "target_close_next_day"
MODEL_DIR = REPORTS_DIR / "models" / "neural_network_regression"
PLOTS_DIR = MODEL_DIR / "plots"
METRICS_PATH = MODEL_DIR / "neural_network_regression_next_day_metrics.json"

RANDOM_SEED = 42


def load_dataset(feature_columns: List[str], ticker: str) -> pd.DataFrame:
    """Fetch feature rows and align them with the next-day close target for a single ticker."""
    column_sql = ", ".join(feature_columns)
    feature_sql = f"SELECT ticker, date, {column_sql} FROM features WHERE ticker = ? ORDER BY date ASC"
    price_sql = "SELECT ticker, date, close FROM prices WHERE ticker = ? ORDER BY date ASC"

    with sqlite3.connect(DATABASE_PATH) as connection:
        features = pd.read_sql_query(feature_sql, connection, params=(ticker,))
        prices = pd.read_sql_query(price_sql, connection, params=(ticker,))

    if features.empty or prices.empty:
        return pd.DataFrame()

    features["date"] = pd.to_datetime(features["date"])
    prices["date"] = pd.to_datetime(prices["date"])

    prices = prices.sort_values("date")
    prices[TARGET_COLUMN] = prices["close"].shift(-1)

    dataset = features.merge(
        prices[["ticker", "date", TARGET_COLUMN]],
        on=["ticker", "date"],
        how="inner",
    )

    dataset = dataset.dropna(subset=feature_columns + [TARGET_COLUMN])
    dataset = dataset.sort_values("date").reset_index(drop=True)
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
    """Assemble a moderately deep regression network."""
    keras.utils.set_random_seed(RANDOM_SEED)
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="linear"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = float(np.mean((y_true - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if len(y_true) > 1:
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    else:
        r2 = float("nan")
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "r2": float(r2),
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

    plt.title(f"Keras NN Next-Day Close Prediction - {ticker}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()

    plot_path = PLOTS_DIR / f"nn_regression_predictions_{ticker}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    return plot_path


def save_model(model: keras.Model, scaler: StandardScaler, ticker: str) -> Dict[str, str]:
    """Persist the trained model and the fitted scaler."""
    ensure_directory(MODEL_DIR)
    model_path = MODEL_DIR / f"nn_regression_next_day_{ticker}.keras"
    scaler_path = MODEL_DIR / f"nn_regression_next_day_{ticker}_scaler.pkl"

    # Save the TensorFlow model (architecture + weights).
    model.save(model_path)

    # Persist the scaler so future predictions can reuse the same normalization.
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

    dataset = load_dataset(FEATURE_COLUMNS, TARGET_TICKER)
    if dataset.empty:
        print(f"No feature rows available for ticker {TARGET_TICKER}. Aborting.")
        return

    train_df, test_df = temporal_split(dataset)
    if train_df.empty or test_df.empty:
        print(f"Unable to create a valid train/test split for ticker {TARGET_TICKER}.")
        return

    X_train, X_test, y_train, y_test, scaler = prepare_arrays(train_df, test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    model = build_model(len(FEATURE_COLUMNS))

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=min(64, len(X_train)),
        callbacks=[early_stopping],
        verbose=1,  # Emit progress logs to the CLI.
    )

    train_pred = model.predict(X_train, verbose=0).ravel()
    test_pred = model.predict(X_test, verbose=0).ravel()

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    train_stats = descriptive_stats(train_df[TARGET_COLUMN])
    test_stats = descriptive_stats(test_df[TARGET_COLUMN])

    plot_path = plot_predictions(TARGET_TICKER, train_df, test_df, train_pred, test_pred)
    artifact_paths = save_model(model, scaler, TARGET_TICKER)

    metrics_report: Dict[str, Dict[str, object]] = {
        TARGET_TICKER: {
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
            "training_history": history.history,
        }
    }

    with METRICS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(metrics_report, fh, indent=2)

    print(f"Ticker {TARGET_TICKER}: train={len(train_df)}, test={len(test_df)}")
    print(
        f"  Train Metrics -> RMSE: {train_metrics['rmse']:.4f}, "
        f"MAE: {train_metrics['mae']:.4f}, R2: {train_metrics['r2']:.4f}"
    )
    print(
        f"  Test Metrics  -> RMSE: {test_metrics['rmse']:.4f}, "
        f"MAE: {test_metrics['mae']:.4f}, R2: {test_metrics['r2']:.4f}"
    )
    print(f"Metrics report saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
