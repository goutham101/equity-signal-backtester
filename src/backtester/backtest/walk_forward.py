"""Walk-forward fitting of signal combination weights (Skill.md section 10).

Idea: never pick weights using the data we then judge them on.
- Train on 5 calendar years, test on the next calendar year, step 1 year.
- Fit the combination weights on training dates only.
- Apply those fixed weights to the test year only.
- Glue the test years together into one out-of-sample signal.

Timing (Skill.md section 5.2):
- Ranked signals on row t use data up to and including close t.
- The forward return label on row t runs from close t+exec_lag to close
  t+exec_lag+horizon.

Purge rule (why there is a gap between train and test):
A training label on date t uses closes up to t + exec_lag + horizon. If the
last training date sat right next to the test start, its label would use
closes from inside the test period. So we leave a gap of
max(purge_days, horizon + exec_lag) trading days. The spec only asks for
"at least horizon"; horizon + exec_lag is the strictly safe version because
it also counts the execution lag.
"""

import numpy as np
import pandas as pd

from backtester.analysis.forward_returns import forward_returns
from backtester.analysis.ic import daily_ic


def purge_gap(purge_days: int = 21, horizon: int = 21, exec_lag: int = 1) -> int:
    """Number of trading days to leave between the last train date and the first test date.

    A training label on date t ends at close t + exec_lag + horizon, so the
    gap must be at least horizon + exec_lag. purge_days lets you ask for more.
    """
    return max(purge_days, horizon + exec_lag)


def _first_year_is_complete(dates: pd.DatetimeIndex) -> bool:
    """True if the index starts in the first week of January.

    If the data starts mid-year, that first calendar year is partial and
    cannot count as a full training year.
    """
    first = dates[0]
    return first.month == 1 and first.day <= 7


def make_windows(
    dates: pd.DatetimeIndex,
    train_years: int = 5,
    test_years: int = 1,
    purge_days: int = 21,
    horizon: int = 21,
    exec_lag: int = 1,
) -> list[dict]:
    """Rolling train/test windows built from the actual trading dates.

    For each test period (test_years calendar years, stepping test_years at
    a time so test periods never overlap; with the default of 1 this is the
    spec's "step 1 year"):
    - test_start / test_end = first / last trading date in the test years.
    - train_start = first trading date of the calendar year train_years
      before the test period.
    - train_end = the trading date that leaves `gap` full trading days
      strictly between it and test_start, where
      gap = max(purge_days, horizon + exec_lag). So the label on train_end,
      which ends at close train_end + exec_lag + horizon, ends before
      close test_start.

    Windows without a full training period (the data does not reach back
    train_years complete calendar years) are dropped. The last test period
    may be a partial year if the data stops mid-year; it is still out of
    sample, so it is kept.

    Each window is a dict with keys: label, train_start, train_end,
    test_start, test_end.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    if len(dates) == 0:
        return []

    gap = purge_gap(purge_days, horizon, exec_lag)
    years = sorted(set(dates.year))

    # The earliest year we may train on.
    first_full_year = years[0]
    if not _first_year_is_complete(dates):
        first_full_year = years[0] + 1

    windows = []
    test_first_year = first_full_year + train_years
    while test_first_year <= years[-1]:
        test_last_year = test_first_year + test_years - 1
        in_test = (dates.year >= test_first_year) & (dates.year <= test_last_year)
        test_dates = dates[in_test]
        train_dates_start = dates[dates.year == test_first_year - train_years]

        if len(test_dates) == 0 or len(train_dates_start) == 0:
            test_first_year += test_years
            continue

        test_start = test_dates[0]
        test_start_pos = dates.get_loc(test_start)
        train_end_pos = test_start_pos - gap - 1
        train_start = train_dates_start[0]

        if train_end_pos < dates.get_loc(train_start):
            test_first_year += test_years
            continue

        if test_years == 1:
            label = str(test_first_year)
        else:
            label = f"{test_first_year}-{test_last_year}"

        windows.append(
            {
                "label": label,
                "train_start": train_start,
                "train_end": dates[train_end_pos],
                "test_start": test_start,
                "test_end": test_dates[-1],
            }
        )
        test_first_year += test_years

    return windows


def _normalize_weights(raw: pd.Series) -> pd.Series:
    """Clip weights at 0 and scale them to sum to 1.

    Fallback: if no signal has a positive raw weight (all <= 0 or NaN), we
    have no evidence that any signal works, so we use equal weights instead
    of dividing by zero. Equal weight is the no-fitting baseline.
    """
    clipped = raw.fillna(0.0).clip(lower=0.0)
    total = clipped.sum()
    if total <= 0:
        return pd.Series(1.0 / len(raw), index=raw.index)
    return clipped / total


def fit_ic_weights(
    ranked: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    min_stocks: int = 50,
) -> pd.Series:
    """IC-weighted combination weights fit on training dates only.

    For each signal: mean daily Spearman IC between the ranked signal at
    close t and the forward return from close t+exec_lag (fwd_ret row t),
    over t in train_dates. Negative mean IC -> weight 0. Weights are
    normalized to sum to 1. If every mean IC is <= 0 (or NaN), fall back
    to equal weights.

    The caller must make sure train_dates end early enough (see
    make_windows) that the labels on those rows do not reach into the test
    period.
    """
    fwd_train = fwd_ret.reindex(index=train_dates)
    mean_ics = {}
    for name, signal in ranked.items():
        signal_train = signal.reindex(index=train_dates, columns=fwd_ret.columns)
        ic = daily_ic(signal_train, fwd_train, min_stocks=min_stocks)
        mean_ics[name] = ic.mean()

    raw = pd.Series(mean_ics, dtype=float)
    return _normalize_weights(raw)


def fit_regression_weights(
    ranked: dict[str, pd.DataFrame],
    fwd_ret: pd.DataFrame,
    train_dates: pd.DatetimeIndex,
    min_stocks: int = 50,
) -> pd.Series:
    """Pooled-OLS combination weights fit on training dates only.

    Stacks every (date, stock) pair in train_dates where all signals and the
    forward return exist, then regresses the forward return (close
    t+exec_lag to close t+exec_lag+horizon) on the ranked signals at close t,
    with an intercept. The forward return is demeaned per date first, so
    the regression explains which stocks beat the others on that date, not
    whether the whole market went up.

    Dates with fewer than min_stocks complete rows are skipped. Coefficients
    are clipped at 0 and normalized to sum to 1 (equal-weight fallback if
    none are positive), the same as fit_ic_weights.
    """
    names = list(ranked.keys())
    fwd_train = fwd_ret.reindex(index=train_dates)

    # Long format: one row per (date, ticker), one column per signal.
    columns = {}
    for name in names:
        signal_train = ranked[name].reindex(index=train_dates, columns=fwd_ret.columns)
        columns[name] = signal_train.stack(future_stack=True)
    demeaned = fwd_train.sub(fwd_train.mean(axis=1), axis=0)
    columns["fwd"] = demeaned.stack(future_stack=True)
    pooled = pd.DataFrame(columns).dropna()

    # Drop thin dates.
    rows_per_date = pooled.groupby(level=0).size()
    good_dates = rows_per_date[rows_per_date >= min_stocks].index
    pooled = pooled[pooled.index.get_level_values(0).isin(good_dates)]

    if len(pooled) <= len(names):
        raw = pd.Series(np.nan, index=names, dtype=float)
        return _normalize_weights(raw)

    x = pooled[names].to_numpy()
    x = np.column_stack([np.ones(len(x)), x])
    y = pooled["fwd"].to_numpy()
    coefs, _, _, _ = np.linalg.lstsq(x, y, rcond=None)

    raw = pd.Series(coefs[1:], index=names, dtype=float)
    return _normalize_weights(raw)


def _weighted_combination(
    ranked: dict[str, pd.DataFrame],
    weights: pd.Series,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
) -> pd.DataFrame:
    """Weighted sum of ranked signals on the given dates.

    Uses only signal values at close t for row t. Like combine_equal_weight's
    default, a stock needs a value from every signal on that date, otherwise
    the result is NaN (even if that signal's weight is 0).
    """
    total = pd.DataFrame(0.0, index=dates, columns=tickers)
    all_present = pd.DataFrame(True, index=dates, columns=tickers)
    for name, signal in ranked.items():
        part = signal.reindex(index=dates, columns=tickers)
        all_present = all_present & part.notna()
        total = total + weights[name] * part.fillna(0.0)
    return total.where(all_present)


def walk_forward_signal(
    ranked: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    horizon: int = 21,
    exec_lag: int = 1,
    train_years: int = 5,
    test_years: int = 1,
    purge_days: int = 21,
    min_stocks: int = 50,
    method: str = "ic_weighted",
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Out-of-sample combined signal from walk-forward fitted weights.

    Steps:
    1. Forward returns from close t+exec_lag to close t+exec_lag+horizon
       (the training labels).
    2. Windows from make_windows on prices.index.
    3. For each window, fit weights on train_start..train_end only, then
       apply them to the ranked signals (close t) on test_start..test_end.

    Returns:
    - combined: date x ticker, NaN on every date outside a test window.
    - weights: one row per window label, one column per signal.
    - windows: the list from make_windows.

    method is "ic_weighted" (Option A) or "regression" (Option B).
    """
    if method == "ic_weighted":
        fit = fit_ic_weights
    elif method == "regression":
        fit = fit_regression_weights
    else:
        raise ValueError(f"unknown method: {method}")

    dates = prices.index
    tickers = prices.columns
    fwd = forward_returns(prices, horizon=horizon, exec_lag=exec_lag)
    windows = make_windows(
        dates,
        train_years=train_years,
        test_years=test_years,
        purge_days=purge_days,
        horizon=horizon,
        exec_lag=exec_lag,
    )

    combined = pd.DataFrame(np.nan, index=dates, columns=tickers)
    weight_rows = {}
    for window in windows:
        in_train = (dates >= window["train_start"]) & (dates <= window["train_end"])
        in_test = (dates >= window["test_start"]) & (dates <= window["test_end"])
        train_dates = dates[in_train]
        test_dates = dates[in_test]

        weights = fit(ranked, fwd, train_dates, min_stocks=min_stocks)
        weight_rows[window["label"]] = weights
        combined.loc[test_dates] = _weighted_combination(ranked, weights, test_dates, tickers)

    weights_table = pd.DataFrame(weight_rows).T
    weights_table = weights_table.reindex(columns=list(ranked.keys()))
    weights_table.index.name = "window"
    return combined, weights_table, windows
