INSERT OR REPLACE INTO prices (
    ticker,
    date,
    open,
    high,
    low,
    close,
    adj_close,
    volume
) VALUES (
    :ticker,
    :date,
    :open,
    :high,
    :low,
    :close,
    :adj_close,
    :volume
);
