"""Portfolio construction: turn a signal into target weights.

Timing: every weight row built here is a *target* decided at close t, using the
signal value at close t only. The engine (engine.py) is the only place that
shifts these weights forward to the days they are actually held.

Simplification: between rebalances we hold the target weights constant
(forward-fill). In reality weights drift as prices move. We ignore that drift.
"""

import pandas as pd

VALID_FREQS = ("monthly", "weekly")
VALID_KINDS = ("long_short", "long_only", "equal_weight")


def rebalance_dates(index: pd.DatetimeIndex, freq: str = "monthly") -> pd.DatetimeIndex:
    """Return the last trading day of each month (or week) found in `index`.

    The dates come from the actual trading index, not the calendar. If a month
    ends on a holiday or weekend, the rebalance date is the last day in the
    index that falls in that month. Weights are decided at the close of these days.

    Note: the last period in the index may be incomplete (data ends mid-month).
    Its last available day is still returned as a rebalance date.
    """
    if freq not in VALID_FREQS:
        raise ValueError(f"freq must be one of {VALID_FREQS}, got {freq!r}")

    # Drop the index name: if it matched the period name, groupby would treat it as a column.
    index = pd.DatetimeIndex(index, name=None).sort_values()
    if freq == "monthly":
        periods = index.to_period("M")
    else:
        periods = index.to_period("W")

    dates = pd.Series(index, index=index)
    last_per_period = dates.groupby(periods).max()
    return pd.DatetimeIndex(last_per_period.values)


def _quantile_buckets(signal_row: pd.Series, quantiles: int) -> pd.Series | None:
    """Assign each stock with a valid signal to a bucket 0 (lowest) .. k-1 (highest).

    Returns None when a safe split is not possible:
    - fewer valid stocks than `quantiles`, or
    - so many tied values that qcut collapses everything into a single bucket.

    When ties create duplicate bin edges, qcut drops the duplicate edges
    (duplicates="drop"). That can leave fewer than `quantiles` buckets, so the
    top and bottom buckets may hold more than 20% of stocks on that date.
    """
    valid = signal_row.dropna()
    if len(valid) < quantiles:
        return None

    buckets = pd.qcut(valid, quantiles, labels=False, duplicates="drop")
    # If every value is tied, qcut can return all NaN buckets. Treat that like
    # a single bucket: there is no top or bottom to trade.
    buckets = buckets.dropna()
    if buckets.nunique() < 2:
        return None
    return buckets


def quintile_long_short_weights(signal_row: pd.Series, quantiles: int = 5) -> pd.Series:
    """Dollar-neutral weights for one date: long the top bucket, short the bottom.

    Uses the signal at a single close t. Long stocks each get +1 / n_long,
    short stocks each get -1 / n_short, everyone else (including NaN signals) 0.
    If a safe split is not possible (see _quantile_buckets), all weights are 0
    for that date, which means "stay in cash".
    """
    weights = pd.Series(0.0, index=signal_row.index)
    buckets = _quantile_buckets(signal_row, quantiles)
    if buckets is None:
        return weights

    top_label = buckets.max()
    bottom_label = buckets.min()
    long_names = buckets.index[buckets == top_label]
    short_names = buckets.index[buckets == bottom_label]

    weights.loc[long_names] = 1.0 / len(long_names)
    weights.loc[short_names] = -1.0 / len(short_names)
    return weights


def long_only_weights(signal_row: pd.Series, quantiles: int = 5) -> pd.Series:
    """Long-only weights for one date: equal weight in the top bucket, summing to 1.

    Uses the signal at a single close t. Same bucket logic and the same
    "all zeros if no safe split" rule as quintile_long_short_weights.
    """
    weights = pd.Series(0.0, index=signal_row.index)
    buckets = _quantile_buckets(signal_row, quantiles)
    if buckets is None:
        return weights

    top_label = buckets.max()
    long_names = buckets.index[buckets == top_label]
    weights.loc[long_names] = 1.0 / len(long_names)
    return weights


def equal_weight_benchmark(row: pd.Series) -> pd.Series:
    """Equal weight in every stock with a valid (non-NaN) value at close t.

    `row` can be a signal row or a price row; only which values are NaN matters.
    Weights sum to 1. If nothing is valid, all weights are 0.
    """
    weights = pd.Series(0.0, index=row.index)
    valid_names = row.index[row.notna()]
    if len(valid_names) == 0:
        return weights
    weights.loc[valid_names] = 1.0 / len(valid_names)
    return weights


def _weights_for_row(signal_row: pd.Series, kind: str, quantiles: int) -> pd.Series:
    """Dispatch one date's signal row to the right weighting function."""
    if kind == "long_short":
        return quintile_long_short_weights(signal_row, quantiles)
    if kind == "long_only":
        return long_only_weights(signal_row, quantiles)
    return equal_weight_benchmark(signal_row)


def build_target_weights(
    signal: pd.DataFrame,
    freq: str = "monthly",
    kind: str = "long_short",
    quantiles: int = 5,
) -> pd.DataFrame:
    """Build a wide DataFrame of target weights decided at each close t.

    Steps:
    1. Find rebalance dates (last trading day of each month/week) in signal.index.
    2. On each rebalance date t, compute weights from the signal at close t only.
    3. Forward-fill those weights until the next rebalance (weight drift ignored).
    4. Dates before the first rebalance get weight 0 (no position yet).

    The result has the same index and columns as `signal`. Row t is a target
    decided at close t; the engine shifts it by 1 + exec_lag days before it
    earns any return.
    """
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")

    signal = signal.sort_index()
    dates = rebalance_dates(signal.index, freq)

    rows = {}
    for date in dates:
        rows[date] = _weights_for_row(signal.loc[date], kind, quantiles)

    rebalance_weights = pd.DataFrame.from_dict(rows, orient="index")
    rebalance_weights = rebalance_weights.reindex(columns=signal.columns)

    # Forward-fill only carries a past decision forward in time, so it is safe.
    # Never back-fill: that would copy a future decision into the past.
    target_weights = rebalance_weights.reindex(signal.index).ffill()
    target_weights = target_weights.fillna(0.0)
    return target_weights
