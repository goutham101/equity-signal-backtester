"""Close-to-close daily returns and forward returns used for IC analysis.

Timing convention (Skill.md section 5.2):
- The signal is decided at close t.
- We trade at close t + exec_lag.
- The forward return for horizon h runs from close t + exec_lag to close
  t + exec_lag + h, and it is stored on row t (the decision date).
"""

import pandas as pd


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close returns. returns[d] = close d / close d-1 - 1.

    The first row is NaN. Missing prices are NOT filled here
    (fill_method=None), so a gap gives NaN instead of a fake zero return.
    """
    return prices.pct_change(fill_method=None)


def forward_returns(prices: pd.DataFrame, horizon: int, exec_lag: int = 1) -> pd.DataFrame:
    """Return from close t+exec_lag to close t+exec_lag+horizon, aligned to date t.

    Uses future closes on purpose: this is the label we compare the signal
    against, never an input to a signal. The last exec_lag + horizon rows
    are NaN because those future closes do not exist yet.

    exec_lag=1 is the default and realistic. exec_lag=0 (trade at the same
    close used for the signal) is optimistic and only for comparison.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if exec_lag < 0:
        raise ValueError("exec_lag cannot be negative")

    start = prices.shift(-exec_lag)
    end = prices.shift(-(exec_lag + horizon))
    return end / start - 1
