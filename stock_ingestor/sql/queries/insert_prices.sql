INSERT OR REPLACE INTO prices (
    ticker,
    date,
    timestamp_utc,
    open,
    high,
    low,
    close,
    adj_close,
    volume
) VALUES (
    :ticker,
    :date,
    :timestamp_utc,
    :open,
    :high,
    :low,
    :close,
    :adj_close,
    :volume
);
