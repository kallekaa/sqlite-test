"""Query the SQLite database and produce exploratory analyses and reports."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    QUERIES_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    ensure_directory,
    get_trading_calendar,
    load_schema,
    load_sql,
)

DEFAULT_QUERY_TICKERS = ["AAPL"]
DEFAULT_WINDOWS: Dict[str, int] = {"1m": 30, "3m": 90, "1y": 365}


def load_windows_from_env() -> Dict[str, int]:
    override = os.getenv("EDA_WINDOWS")
    if not override:
        return DEFAULT_WINDOWS
    windows: Dict[str, int] = {}
    for chunk in override.split(","):
        if ":" not in chunk:
            continue
        label, value = chunk.split(":", 1)
        label = label.strip()
        try:
            days = int(value.strip())
        except ValueError:
            continue
        if label:
            windows[label] = days
    return windows or DEFAULT_WINDOWS


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


def run_query(db_path: str, sql: str, params: List[Any]) -> pd.DataFrame:
    connection = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(sql, connection, params=params)
    finally:
        connection.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def run_coverage_checks(df: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    duplicates = int(df.duplicated(["ticker", "date"]).sum())
    results["duplicate_rows"] = duplicates

    price_cols = ["open", "high", "low", "close", "adj_close"]
    invalid_prices = df[(df[price_cols] <= 0).any(axis=1)]
    results["invalid_price_rows"] = len(invalid_prices)

    negative_volume = int((df["volume"] < 0).sum())
    results["negative_volume_rows"] = negative_volume

    missing_by_ticker: Dict[str, int] = {}
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("date")
        start = group["date"].min().date()
        end = group["date"].max().date()
        expected = pd.Index(get_trading_calendar(start, end).date)
        actual = pd.Index(group["date"].dt.date.unique())
        missing = expected.difference(actual)
        missing_by_ticker[ticker] = len(missing)
    results["missing_business_days"] = missing_by_ticker
    return results


def print_coverage_summary(results: Dict[str, Any]) -> None:
    print("\n=== Coverage Checks ===")
    print(f"Duplicate (ticker, date) rows: {results['duplicate_rows']}")
    print(f"Rows with non-positive price values: {results['invalid_price_rows']}")
    print(f"Rows with negative volume: {results['negative_volume_rows']}")
    print("Missing business days by ticker:")
    for ticker, count in results["missing_business_days"].items():
        print(f"  {ticker}: {count}")


def compute_enriched_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.sort_values(["ticker", "date"]).copy()
    frame["return_1d"] = frame.groupby("ticker")["close"].pct_change()
    frame["roll_max"] = frame.groupby("ticker")["close"].cummax()
    frame["drawdown"] = frame["close"] / frame["roll_max"] - 1
    frame["volume_zscore"] = frame.groupby("ticker")["volume"].transform(
        lambda s: (s - s.rolling(window=20).mean()) / s.rolling(window=20).std()
    )
    frame["volume_zscore"] = frame["volume_zscore"].replace([np.inf, -np.inf], np.nan)
    frame["rolling_vol_20d"] = frame.groupby("ticker")["return_1d"].transform(
        lambda s: s.rolling(window=20).std()
    )
    return frame


def slice_window(frame: pd.DataFrame, days: int) -> pd.DataFrame:
    if frame.empty:
        return frame
    max_date = frame["date"].max()
    cutoff = max_date - timedelta(days=days)
    return frame[frame["date"] >= cutoff].copy()


def save_return_distribution(frame: pd.DataFrame, output_path: Path) -> bool:
    returns = frame["return_1d"].dropna()
    if returns.empty:
        return False

    sorted_returns = np.sort(returns.values)
    count = len(sorted_returns)
    probabilities = (np.arange(1, count + 1) - 0.5) / count
    normal_quantiles = np.array([NormalDist().inv_cdf(p) for p in probabilities])

    fig, (ax_hist, ax_qq) = plt.subplots(1, 2, figsize=(12, 5))
    ax_hist.hist(returns, bins=50, color="#4C72B0", alpha=0.8)
    ax_hist.set_title("Return Distribution")
    ax_hist.set_xlabel("Daily Return")
    ax_hist.set_ylabel("Frequency")

    ax_qq.scatter(normal_quantiles, sorted_returns, s=10, color="#55A868")
    ax_qq.plot(normal_quantiles, normal_quantiles, color="grey", linestyle="--", linewidth=1)
    ax_qq.set_title("Normal Q-Q Plot")
    ax_qq.set_xlabel("Theoretical Quantiles")
    ax_qq.set_ylabel("Empirical Quantiles")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def save_rolling_volatility(frame: pd.DataFrame, output_path: Path) -> bool:
    volatility = frame.pivot_table(
        index="date", columns="ticker", values="rolling_vol_20d"
    )
    volatility = volatility.dropna(how="all")
    if volatility.empty:
        return False

    fig, ax = plt.subplots(figsize=(10, 6))
    volatility.plot(ax=ax)
    ax.set_title("20-Day Rolling Volatility (Std of Returns)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def save_correlation_heatmap(frame: pd.DataFrame, output_path: Path) -> bool:
    returns = frame.pivot_table(index="date", columns="ticker", values="return_1d")
    returns = returns.dropna(axis=0, how="all")
    if returns.empty or returns.shape[1] < 2:
        return False

    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Return Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def save_trend_panels(frame: pd.DataFrame, output_path: Path) -> bool:
    closes = frame.pivot_table(index="date", columns="ticker", values="close")
    returns = frame.pivot_table(index="date", columns="ticker", values="return_1d")
    volume_z = frame.pivot_table(index="date", columns="ticker", values="volume_zscore")

    if closes.empty:
        return False

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    closes.plot(ax=axes[0])
    axes[0].set_title("Adjusted Close")
    axes[0].set_ylabel("Price")

    if not returns.empty:
        returns.plot(ax=axes[1])
        axes[1].set_title("Daily Returns")
        axes[1].set_ylabel("Return")
    else:
        axes[1].set_visible(False)

    if not volume_z.empty:
        volume_z.plot(ax=axes[2])
        axes[2].set_title("Volume Z-Score (20-Day)")
        axes[2].set_ylabel("Z-Score")
        axes[2].set_xlabel("Date")
    else:
        axes[2].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return True


def summarize_drawdowns(frame: pd.DataFrame) -> pd.DataFrame:
    drawdowns = frame[["ticker", "date", "drawdown"]].dropna()
    if drawdowns.empty:
        return pd.DataFrame(columns=["ticker", "date", "drawdown"])
    return drawdowns.nsmallest(5, "drawdown")


def generate_reports(frame: pd.DataFrame, label: str, timestamp: str) -> List[Path]:
    outputs: List[Path] = []
    reports_dir = ensure_directory(REPORTS_DIR)

    paths = {
        "returns": reports_dir / f"{timestamp}_returns_{label}.png",
        "volatility": reports_dir / f"{timestamp}_volatility_{label}.png",
        "correlation": reports_dir / f"{timestamp}_correlation_{label}.png",
        "trends": reports_dir / f"{timestamp}_trends_{label}.png",
    }

    if save_return_distribution(frame, paths["returns"]):
        outputs.append(paths["returns"])
    if save_rolling_volatility(frame, paths["volatility"]):
        outputs.append(paths["volatility"])
    if save_correlation_heatmap(frame, paths["correlation"]):
        outputs.append(paths["correlation"])
    if save_trend_panels(frame, paths["trends"]):
        outputs.append(paths["trends"])

    if outputs:
        print(f"\nSaved EDA visuals for '{label}':")
        for path in outputs:
            print(f"  {path}")

    top_drawdowns = summarize_drawdowns(frame)
    if not top_drawdowns.empty:
        print(f"\nTop drawdowns ({label} window):")
        for _, row in top_drawdowns.iterrows():
            print(f"  {row['ticker']} on {row['date'].date()}: {row['drawdown']:.2%}")

    return outputs


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
    db_path = str((ROOT_DIR / schema["database"]).resolve())

    print("Running query...")
    df = run_query(db_path, sql, params)
    if df.empty:
        print("Query returned no rows.")
        return

    coverage = run_coverage_checks(df)
    print_coverage_summary(coverage)

    enriched = compute_enriched_frame(df)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    windows = load_windows_from_env()
    window_cache: Dict[str, pd.DataFrame] = {"full": enriched}

    # Always generate reports for full dataset
    generate_reports(enriched, "full", timestamp)

    for label, days in windows.items():
        window_cache[label] = slice_window(enriched, days)
        if window_cache[label].empty:
            print(f"\nWindow '{label}' is empty; skipping plots.")
            continue
        generate_reports(window_cache[label], label, timestamp)


if __name__ == "__main__":
    main()
