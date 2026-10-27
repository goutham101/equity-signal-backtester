"""Short-term reversal signal (Skill.md section 7)."""

import pandas as pd


def short_term_reversal(prices: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Minus the past `lookback`-day return. Uses closes t - lookback and t.

    Value at date t = -(close(t) / close(t - lookback) - 1).
    The minus sign makes last week's losers score high, because they
    tend to bounce back. The first `lookback` rows are NaN (warm-up).
    """
    past_return = prices / prices.shift(lookback) - 1
    return -past_return
