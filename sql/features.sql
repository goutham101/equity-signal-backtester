-- Daily return and momentum features computed with window functions.
-- All values on date t use closes up to and including close t.
--
-- LAG counts rows, not trading days. This is only correct when a ticker has
-- no missing dates inside its history. clean.py reports any leftover gaps.
CREATE OR REPLACE VIEW daily_features AS
SELECT
    ticker,
    date,
    adj_close,
    adj_close / LAG(adj_close, 1)   OVER w - 1 AS ret_1d,
    adj_close / LAG(adj_close, 5)   OVER w - 1 AS ret_5d,
    LAG(adj_close, 21) OVER w / LAG(adj_close, 252) OVER w - 1 AS mom_12_1
FROM prices
WINDOW w AS (PARTITION BY ticker ORDER BY date);
