"""Initialize the SQLite database using SQL migrations defined on disk."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

from utils import MIGRATIONS_DIR, ROOT_DIR, load_schema


def discover_migrations() -> List[Tuple[int, Path]]:
    migrations: List[Tuple[int, Path]] = []
    if not MIGRATIONS_DIR.exists():
        return migrations

    for path in MIGRATIONS_DIR.glob("*.sql"):
        prefix = path.stem.split("_", 1)[0]
        try:
            version = int(prefix)
        except ValueError as exc:
            raise ValueError(
                f"Migration filename must start with an integer version: {path.name}"
            ) from exc
        migrations.append((version, path))

    migrations.sort(key=lambda item: item[0])
    return migrations


def get_current_version(connection: sqlite3.Connection) -> int:
    row = connection.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def apply_migration(
    connection: sqlite3.Connection,
    version: int,
    path: Path,
) -> None:
    print(f"Applying migration {version:04d} ({path.name})")
    sql = path.read_text(encoding="utf-8")
    connection.executescript(sql)
    connection.execute(f"PRAGMA user_version = {version}")
    connection.commit()


def create_database(schema: Dict[str, Any]) -> None:
    db_path = ROOT_DIR / schema["database"]
    print(f"Ensuring database at {db_path}")
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON;")
        current_version = get_current_version(connection)
        migrations = discover_migrations()
        pending = [(version, path) for version, path in migrations if version > current_version]

        if not pending:
            print(f"No migrations to apply (user_version={current_version}).")
            return

        for version, path in pending:
            apply_migration(connection, version, path)

        print(f"Database migrated to version {pending[-1][0]}.")
    finally:
        connection.close()


def main() -> None:
    schema = load_schema()
    create_database(schema)
    print("Database initialization complete.")


if __name__ == "__main__":
    main()
