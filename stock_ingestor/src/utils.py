"""Shared helpers for working with configuration and SQL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT_DIR / "config"
SCHEMA_PATH = CONFIG_DIR / "schema.json"
SQL_DIR = ROOT_DIR / "sql"
MIGRATIONS_DIR = SQL_DIR / "migrations"
QUERIES_DIR = SQL_DIR / "queries"


def load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_sql(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"SQL file not found: {path}")
    return path.read_text(encoding="utf-8")
