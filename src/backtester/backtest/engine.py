"""Backtest engine: apply held weights to daily returns and subtract costs.

Timing convention (Skill.md section 5.2):
- target_weights[t] is decided at close t.
- returns[d] is the close-to-close return from close d-1 to close d.
- Holdings during day d are target_weights.shift(1 + exec_lag).
  With exec_lag=1, a decision at close t is traded at close t+1 and first earns
  the return of day t+2 (close t+1 -> close t+2).
"""

import pandas as pd


def _align_weights(target_weights: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Put target weights on exactly the same dates and tickers as `returns`.

    - Tickers in the weights but not in returns raise an error (we would
      silently lose positions otherwise).
    - Tickers in returns but not in the weights get weight 0.
    - Dates in the weights but not in returns raise an error.
    - Dates in returns but not in the weights carry the last known decision
      forward (ffill, never bfill). Dates before any decision get weight 0.
    """
    extra_tickers = target_weights.columns.difference(returns.columns)
    if len(extra_tickers) > 0:
        raise ValueError(f"weights have tickers missing from returns: {list(extra_tickers)}")

    extra_dates = target_weights.index.difference(returns.index)
    if len(extra_dates) > 0:
        raise ValueError(f"weights have {len(extra_dates)} dates missing from returns")

    aligned = target_weights.reindex(index=returns.index, columns=returns.columns)
    aligned = aligned.ffill().fillna(0.0)
    return aligned


def run_backtest(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    cost_bps: float = 10.0,
    exec_lag: int = 1,
) -> pd.DataFrame:
    """Run a daily backtest and return gross, costs, net and turnover per day.

    target_weights: weights decided at close t (forward-filled between rebalances).
    returns: close-to-close returns, returns[d] = close d-1 -> close d.
    Holdings earn returns starting 1 + exec_lag days after the decision.

    cost_bps is the cost per unit of weight traded (one side). Turnover on day d
    is the total absolute change in held weights from day d-1 to day d.
    exec_lag=0 (trade at the signal's own close) is optimistic; use it only
    as a comparison.
    """
    if exec_lag < 0:
        raise ValueError("exec_lag must be >= 0")

    returns = returns.sort_index()
    weights = _align_weights(target_weights.sort_index(), returns)

    held = weights.shift(1 + exec_lag).fillna(0.0)
    gross = (held * returns.fillna(0.0)).sum(axis=1)

    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * cost_bps / 10_000
    net = gross - costs

    return pd.DataFrame({"gross": gross, "costs": costs, "net": net, "turnover": turnover})


def cost_sensitivity(
    target_weights: pd.DataFrame,
    returns: pd.DataFrame,
    bps_list: list[float],
    exec_lag: int = 1,
) -> dict[float, pd.DataFrame]:
    """Run the same backtest at several cost levels.

    Returns a dict mapping cost_bps -> the run_backtest result frame. Gross
    returns and turnover are identical across entries; only costs and net change.
    Metrics (Sharpe etc.) are computed elsewhere (analysis/metrics.py).
    """
    results = {}
    for bps in bps_list:
        results[bps] = run_backtest(target_weights, returns, cost_bps=bps, exec_lag=exec_lag)
    return results
