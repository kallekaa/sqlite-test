-- Table to store engineered lag features derived from price history.
CREATE TABLE IF NOT EXISTS features (
    ticker TEXT NOT NULL,
    date TEXT NOT NULL,
    close_lag_1 REAL,
    close_lag_3 REAL,
    close_lag_5 REAL,
    volume_lag_1 INTEGER,
    volume_lag_3 INTEGER,
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_features_ticker_date
    ON features (ticker, date);
