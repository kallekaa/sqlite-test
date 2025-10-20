"""Query the SQLite database and perform simple exploratory analysis."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from utils import QUERIES_DIR, ROOT_DIR, load_schema, load_sql

DEFAULT_QUERY_TICKERS = ["AAPL"]


def build_query(base_sql: str, query_cfg: Dict[str, Any]) -> Tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []

    tickers = query_cfg.get("tickers") or []
    if tickers:
        placeholders = ", ".join(["?"] * len(tickers))
        clauses.append(f"ticker IN ({placeholders})")
        params.extend(tickers)

    start_date = query_cfg.get("start_date")
    if start_date:
        clauses.append("date >= ?")
        params.append(start_date)

    end_date = query_cfg.get("end_date")
    if end_date:
        clauses.append("date <= ?")
        params.append(end_date)

    sql = base_sql.strip()
    if clauses:
        sql = f"{sql}\nWHERE {' AND '.join(clauses)}"
    sql = f"{sql}\nORDER BY date ASC, ticker ASC"
    return sql, params


def run_query(db_path: Path, sql: str, params: List[Any]) -> pd.DataFrame:
    connection = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, connection, params=params)
    finally:
        connection.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def show_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("Query returned no rows.")
        return

    print(f"Retrieved {len(df)} rows across {df['ticker'].nunique()} tickers.")
    print("\nLatest rows:")
    print(df.sort_values("date").groupby("ticker").tail(3))

    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        print("\nDescriptive statistics (numeric columns):")
        print(numeric_df.describe())
    else:
        print("\nNo numeric columns available for descriptive statistics.")


def save_visualization(df: pd.DataFrame) -> None:
    if df.empty:
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not available; skipping visualization.")
        return

    pivot = (
        df.pivot(index="date", columns="ticker", values="close")
        .sort_index()
        .dropna(how="all")
    )
    if pivot.empty:
        print("Not enough data to plot closing prices.")
        return

    ax = pivot.plot(title="Closing Prices Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    fig = ax.get_figure()
    fig.tight_layout()

    output_path = ROOT_DIR / "close_prices.png"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved line chart to {output_path}")


def main() -> None:
    schema = load_schema()
    query_cfg = schema.get("query", {})
    effective_query_cfg = dict(query_cfg)
    effective_query_cfg["tickers"] = (
        (query_cfg.get("tickers") or DEFAULT_QUERY_TICKERS)
        if query_cfg is not None
        else DEFAULT_QUERY_TICKERS
    )

    table_name = effective_query_cfg.get("table")
    if not table_name:
        raise ValueError("Query configuration must specify 'table'.")
    select_sql_path = QUERIES_DIR / f"select_{table_name}.sql"
    base_sql = load_sql(select_sql_path)
    sql, params = build_query(base_sql, effective_query_cfg)
    db_path = ROOT_DIR / schema["database"]

    print("Running query...")
    df = run_query(db_path, sql, params)
    show_summary(df)
    save_visualization(df)


if __name__ == "__main__":
    main()
