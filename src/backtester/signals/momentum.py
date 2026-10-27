"""Momentum 12-1 signal (Skill.md section 7)."""

import pandas as pd


def momentum_12_1(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-month momentum skipping the most recent month. Uses closes up to date t.

    Value at date t = close(t - skip) / close(t - lookback) - 1.
    With the defaults that is the return from ~12 months ago to ~1 month ago.
    The last month is skipped because of short-term reversal.
    Both shifts are positive, so only past closes are used.
    The first `lookback` rows are NaN (warm-up).
    """
    recent_price = prices.shift(skip)
    old_price = prices.shift(lookback)
    return recent_price / old_price - 1
