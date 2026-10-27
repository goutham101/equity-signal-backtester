CREATE TABLE IF NOT EXISTS tickers (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    added_on    DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS prices (
    ticker      TEXT REFERENCES tickers(ticker),
    date        DATE NOT NULL,
    adj_close   NUMERIC(14,6) NOT NULL CHECK (adj_close > 0),
    volume      BIGINT CHECK (volume >= 0),
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id      SERIAL PRIMARY KEY,
    run_at      TIMESTAMP DEFAULT NOW(),
    signal_name TEXT,
    config_json JSONB,
    sharpe      NUMERIC,
    ann_return  NUMERIC,
    max_dd      NUMERIC
);
