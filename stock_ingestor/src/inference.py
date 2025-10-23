"""Generate next-day predictions using saved linear and logistic regression models."""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd

from utils import REPORTS_DIR, ROOT_DIR, load_schema


SCHEMA = load_schema()
DATABASE_PATH = ROOT_DIR / SCHEMA["database"]

REGRESSION_MODEL_DIR = REPORTS_DIR / "models"
CLASSIFICATION_MODEL_DIR = REPORTS_DIR / "models" / "classification"

DEFAULT_TICKER = SCHEMA.get("ingest", {}).get("tickers", ["AAPL"])[0]
DEFAULT_ROWS = 5


def load_pickle_model(path: Path) -> Tuple[object, List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict) and "model" in payload:
        feature_columns = payload.get("feature_columns", [])
        return payload["model"], feature_columns
    raise ValueError(f"Unexpected model format in {path}. Expected dict with 'model' key.")


def fetch_feature_rows(
    database: Path, ticker: str, feature_columns: Iterable[str], rows: int
) -> pd.DataFrame:
    if rows <= 0:
        raise ValueError("rows must be a positive integer.")

    columns = ", ".join(feature_columns)
    query = f"""
        SELECT date, {columns}
        FROM features
        WHERE ticker = ?
        ORDER BY date DESC
        LIMIT ?
    """
    with sqlite3.connect(database) as connection:
        df = pd.read_sql_query(query, connection, params=(ticker, rows))
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    ticker = DEFAULT_TICKER

    regression_path = REGRESSION_MODEL_DIR / f"linear_regression_next_day_{ticker}.pkl"
    classification_path = CLASSIFICATION_MODEL_DIR / f"logistic_regression_next_day_{ticker}.pkl"

    regression_model, regression_features = load_pickle_model(regression_path)
    classification_model, classification_features = load_pickle_model(classification_path)

    all_features: List[str] = sorted(set(regression_features) | set(classification_features))
    if not all_features:
        raise ValueError("The loaded models did not expose any feature columns.")

    feature_rows = fetch_feature_rows(DATABASE_PATH, ticker, all_features, DEFAULT_ROWS)
    if feature_rows.empty:
        print(f"No feature rows available for ticker {ticker}.")
        return

    regression_input = feature_rows[regression_features]
    classification_input = feature_rows[classification_features]

    regression_pred = regression_model.predict(regression_input)
    classification_pred = classification_model.predict(classification_input)
    classification_proba = classification_model.predict_proba(classification_input)[:, 1]

    output = feature_rows[["date"]].copy()
    output["predicted_next_close"] = regression_pred
    output["predicted_up"] = classification_pred
    output["predicted_up_probability"] = classification_proba

    print(f"Predictions for {ticker}:")
    print(output.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()
