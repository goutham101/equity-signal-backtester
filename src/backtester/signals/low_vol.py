"""Low volatility signal (Skill.md section 7)."""

import pandas as pd


def low_volatility(prices: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Minus the rolling std of daily returns. Uses closes t - window to t.

    returns[d] is the close-to-close return from close d-1 to close d.
    The rolling window at date t covers returns[t - window + 1 .. t], so it
    needs closes from t - window up to t. The window only looks backward
    (pandas default, no center=True). The minus sign makes calm stocks
    score high. A window with any missing return gives NaN.
    """
    # fill_method=None: do not forward-fill missing prices before the return.
    returns = prices.pct_change(fill_method=None)
    volatility = returns.rolling(window).std()
    return -volatility
