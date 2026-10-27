"""Information coefficient (IC) analysis.

IC on a date = Spearman rank correlation, across stocks, between the signal
decided at close t and the forward return from close t+exec_lag to close
t+exec_lag+h (see forward_returns.py). All series here are indexed by the
signal date t.
"""

import numpy as np
import pandas as pd

from backtester.analysis.forward_returns import forward_returns


def daily_ic(signal: pd.DataFrame, fwd_ret: pd.DataFrame, min_stocks: int = 50) -> pd.Series:
    """Spearman IC per date between signal at close t and forward return from close t+1.

    Only stocks with both a signal and a forward return count. Dates with
    fewer than min_stocks such stocks are dropped from the result.
    """
    valid = signal.notna() & fwd_ret.notna()
    counts = valid.sum(axis=1)

    sig_valid = signal.where(valid)
    ret_valid = fwd_ret.where(valid)
    ic = sig_valid.corrwith(ret_valid, axis=1, method="spearman")

    enough_stocks = counts >= min_stocks
    return ic[enough_stocks]


def ic_summary(ic: pd.Series, horizon: int) -> dict:
    """Summary stats for a daily IC series.

    The t-stat uses only every horizon-th date. Forward returns for
    consecutive dates overlap by horizon-1 days, so daily ICs are strongly
    correlated and a t-stat on every day would be inflated. Sampling every
    horizon-th date gives (roughly) independent observations.
    t = mean / std * sqrt(n) on that sample.
    """
    ic = ic.dropna()

    mean_ic = ic.mean()
    std_ic = ic.std()
    if std_ic > 0:
        ic_ir = mean_ic / std_ic
    else:
        ic_ir = np.nan

    non_overlap = ic.iloc[::horizon]
    n_non_overlap = len(non_overlap)
    std_non_overlap = non_overlap.std()
    if n_non_overlap >= 2 and std_non_overlap > 0:
        t_stat = non_overlap.mean() / std_non_overlap * np.sqrt(n_non_overlap)
    else:
        t_stat = np.nan

    if len(ic) > 0:
        pct_positive = (ic > 0).mean()
    else:
        pct_positive = np.nan

    return {
        "mean_ic": mean_ic,
        "ic_std": std_ic,
        "ic_ir": ic_ir,
        "pct_positive": pct_positive,
        "t_stat": t_stat,
        "n_dates": len(ic),
        "n_non_overlap": n_non_overlap,
    }


def rolling_ic(ic: pd.Series, window: int = 252) -> pd.Series:
    """Trailing mean IC over the last `window` IC observations (252 ~ 12 months).

    Backward-looking only (no center=True). The value on date t averages ICs
    of signal dates up to t. Note those ICs use forward returns that end
    after t, so this is a research chart, not a tradable signal.
    """
    return ic.rolling(window, min_periods=window).mean()


def quintile_returns(
    signal: pd.DataFrame,
    fwd_ret: pd.DataFrame,
    quantiles: int = 5,
    min_stocks: int = 50,
) -> pd.Series:
    """Average forward return of each signal quantile (Q1 = lowest signal).

    For each date t: bucket stocks by the signal at close t with pd.qcut,
    then average the forward return (from close t+exec_lag) inside each
    bucket. Finally average each bucket over all dates. Dates with fewer
    than max(min_stocks, quantiles) valid stocks, or where qcut fails
    (too many tied values), are skipped.
    """
    min_count = max(min_stocks, quantiles)
    per_date = {}

    for date in signal.index:
        sig_today = signal.loc[date]
        ret_today = fwd_ret.loc[date]
        both_valid = sig_today.notna() & ret_today.notna()
        if both_valid.sum() < min_count:
            continue

        sig_today = sig_today[both_valid]
        ret_today = ret_today[both_valid]
        try:
            buckets = pd.qcut(sig_today, quantiles, labels=False)
        except ValueError:
            continue

        per_date[date] = ret_today.groupby(buckets).mean()

    labels = [f"Q{i + 1}" for i in range(quantiles)]
    if not per_date:
        return pd.Series(np.nan, index=labels, name="mean_fwd_return")

    table = pd.DataFrame(per_date).T
    table = table.reindex(columns=range(quantiles))
    result = table.mean()
    result.index = labels
    result.name = "mean_fwd_return"
    return result


def ic_table(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    horizons: list[int],
    exec_lag: int = 1,
    min_stocks: int = 50,
) -> pd.DataFrame:
    """IC summary for every signal and horizon, one row per (signal, horizon).

    Forward returns run from close t+exec_lag to close t+exec_lag+horizon.
    """
    rows = []
    for horizon in horizons:
        fwd = forward_returns(prices, horizon=horizon, exec_lag=exec_lag)
        for name, signal in signals.items():
            ic = daily_ic(signal, fwd, min_stocks=min_stocks)
            stats = ic_summary(ic, horizon=horizon)
            row = {"signal": name, "horizon": horizon}
            row.update(stats)
            rows.append(row)

    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["signal", "horizon"]).reset_index(drop=True)
    return table
