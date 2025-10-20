"""Shared helpers for working with configuration, SQL, and calendars."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
SQL_DIR = ROOT_DIR / "sql"
MIGRATIONS_DIR = SQL_DIR / "migrations"
QUERIES_DIR = SQL_DIR / "queries"
REPORTS_DIR = ROOT_DIR / "reports"
SCHEMA_PATH = CONFIG_DIR / "schema.json"

US_TRADING_DAY = CustomBusinessDay(calendar=USFederalHolidayCalendar())


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_trading_calendar(start: date, end: date) -> pd.DatetimeIndex:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    calendar = pd.date_range(start=start_ts, end=end_ts, freq=US_TRADING_DAY)
    return calendar.tz_localize("UTC")


def load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_sql(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")
