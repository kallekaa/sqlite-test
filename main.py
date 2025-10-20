import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "stock_ingestor" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import create_db  # type: ignore  # noqa: E402
import ingest_and_store  # type: ignore  # noqa: E402
import feature_engineering  # type: ignore  # noqa: E402
import query_and_show  # type: ignore  # noqa: E402


def main() -> None:
    print("=== Create database ===")
    create_db.main()

    print("\n=== Ingest prices ===")
    ingest_and_store.main()

    print("\n=== Engineer features ===")
    feature_engineering.main()

    print("\n=== Query and show ===")
    query_and_show.main()


if __name__ == "__main__":
    main()
