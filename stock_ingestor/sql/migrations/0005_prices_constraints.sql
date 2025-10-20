-- Rebuild prices table with stronger constraints and additional metadata.
PRAGMA foreign_keys = OFF;

DROP TABLE IF EXISTS prices__new;

CREATE TABLE prices__new (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    open REAL NOT NULL CHECK (open >= 0),
    high REAL NOT NULL CHECK (high >= 0),
    low REAL NOT NULL CHECK (low >= 0),
    close REAL NOT NULL CHECK (close >= 0),
    adj_close REAL NOT NULL CHECK (adj_close >= 0),
    volume INTEGER NOT NULL CHECK (volume >= 0),
    PRIMARY KEY (ticker, date)
);

INSERT INTO prices__new (ticker, date, timestamp_utc, open, high, low, close, adj_close, volume)
SELECT
    ticker,
    date,
    date || 'T00:00:00Z' AS timestamp_utc,
    COALESCE(open, close, adj_close, 0),
    COALESCE(high, close, adj_close, 0),
    COALESCE(low, close, adj_close, 0),
    COALESCE(close, adj_close, open, 0),
    COALESCE(adj_close, close, open, 0),
    MAX(volume, 0)
FROM prices;

DROP TABLE prices;
ALTER TABLE prices__new RENAME TO prices;

CREATE INDEX IF NOT EXISTS idx_prices_date_ticker ON prices (date, ticker);
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date_close ON prices (ticker, date, close);

PRAGMA foreign_keys = ON;
