"""Initialize the SQLite database using the JSON schema definition."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


ROOT_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT_DIR / "config" / "schema.json"


def load_schema() -> Dict[str, Any]:
    with SCHEMA_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def build_create_statement(table: Dict[str, Any]) -> str:
    columns: List[str] = [
        f"{column['name']} {column['type']}" for column in table["columns"]
    ]
    primary_key = table.get("primary_key")
    if primary_key:
        columns.append(f"PRIMARY KEY ({', '.join(primary_key)})")
    columns_sql = ", ".join(columns)
    return f"CREATE TABLE IF NOT EXISTS {table['name']} ({columns_sql});"


def create_database(schema: Dict[str, Any]) -> None:
    db_path = ROOT_DIR / schema["database"]
    print(f"Creating database at {db_path}")
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON;")
        for table in schema["tables"]:
            statement = build_create_statement(table)
            print(f"Ensuring table {table['name']} exists")
            connection.execute(statement)
        connection.commit()
    finally:
        connection.close()


def main() -> None:
    schema = load_schema()
    create_database(schema)
    print("Database initialization complete.")


if __name__ == "__main__":
    main()
