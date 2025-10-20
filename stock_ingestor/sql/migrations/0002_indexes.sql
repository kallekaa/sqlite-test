-- Additional indexes to accelerate common lookups.
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
    ON prices (ticker, date);

CREATE INDEX IF NOT EXISTS idx_prices_date
    ON prices (date);
