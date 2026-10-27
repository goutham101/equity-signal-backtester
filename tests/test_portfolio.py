"""Tests for portfolio construction: quintile weights, rebalance dates, forward-fill."""

import numpy as np
import pandas as pd
import pytest

from backtester.backtest.engine import run_backtest
from backtester.backtest.portfolio import (
    build_target_weights,
    equal_weight_benchmark,
    long_only_weights,
    quintile_long_short_weights,
    rebalance_dates,
)


@pytest.fixture
def simple_signal(synthetic_prices):
    """Inline 12-1 momentum (not imported from signals, which is built in parallel)."""
    return synthetic_prices.shift(21) / synthetic_prices.shift(252) - 1


def test_long_short_weights_sum_and_no_overlap():
    row = pd.Series(np.arange(20, dtype=float), index=[f"T{i:02d}" for i in range(20)])
    weights = quintile_long_short_weights(row)

    longs = weights[weights > 0]
    shorts = weights[weights < 0]
    assert longs.sum() == pytest.approx(1.0)
    assert shorts.sum() == pytest.approx(-1.0)
    assert len(longs) == 4
    assert len(shorts) == 4
    assert longs.index.intersection(shorts.index).empty
    # Highest signals are long, lowest are short.
    assert set(longs.index) == {"T16", "T17", "T18", "T19"}
    assert set(shorts.index) == {"T00", "T01", "T02", "T03"}


def test_long_short_weights_ignore_nans():
    row = pd.Series([np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.nan])
    weights = quintile_long_short_weights(row)
    assert weights.iloc[0] == 0.0
    assert weights.iloc[-1] == 0.0
    assert weights[weights > 0].sum() == pytest.approx(1.0)
    assert weights[weights < 0].sum() == pytest.approx(-1.0)


def test_too_few_stocks_gives_zero_weights():
    row = pd.Series([1.0, 2.0, np.nan, np.nan, np.nan, np.nan])
    assert (quintile_long_short_weights(row) == 0.0).all()
    assert (long_only_weights(row) == 0.0).all()


def test_all_tied_signal_gives_zero_weights():
    row = pd.Series([0.5] * 10)
    assert (quintile_long_short_weights(row) == 0.0).all()


def test_long_only_weights_sum_to_one():
    row = pd.Series(np.arange(20, dtype=float))
    weights = long_only_weights(row)
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    assert (weights > 0).sum() == 4


def test_equal_weight_benchmark():
    row = pd.Series([1.0, np.nan, 3.0, 4.0])
    weights = equal_weight_benchmark(row)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.iloc[1] == 0.0
    assert weights.iloc[0] == pytest.approx(1 / 3)


def test_rebalance_dates_handle_missing_month_end():
    # Business days for Jan-Mar 2024, but drop Jan 31 and Feb 29 (pretend holidays).
    index = pd.bdate_range("2024-01-01", "2024-03-29")
    index = index.drop([pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")])

    dates = rebalance_dates(index, "monthly")

    expected = pd.DatetimeIndex(["2024-01-30", "2024-02-28", "2024-03-29"])
    assert list(dates) == list(expected)
    assert all(d in index for d in dates)


def test_rebalance_dates_weekly():
    index = pd.bdate_range("2024-01-01", "2024-01-19")  # Mon Jan 1 .. Fri Jan 19
    index = index.drop(pd.Timestamp("2024-01-12"))  # Friday holiday in week 2

    dates = rebalance_dates(index, "weekly")

    expected = pd.DatetimeIndex(["2024-01-05", "2024-01-11", "2024-01-19"])
    assert list(dates) == list(expected)


def test_rebalance_dates_bad_freq():
    with pytest.raises(ValueError):
        rebalance_dates(pd.bdate_range("2024-01-01", periods=5), "daily")


@pytest.mark.parametrize("freq", ["monthly", "weekly"])
def test_weights_constant_between_rebalances(simple_signal, freq):
    weights = build_target_weights(simple_signal, freq=freq)
    rebal = rebalance_dates(simple_signal.index, freq)

    changes = weights.diff().abs().sum(axis=1)
    change_dates = changes.index[changes > 0]
    # Weights can only change on a rebalance date.
    assert change_dates.isin(rebal).all()
    assert len(change_dates) > 0


def test_weights_on_rebalance_dates_sum_correctly(simple_signal):
    weights = build_target_weights(simple_signal, freq="monthly")
    rebal = rebalance_dates(simple_signal.index, "monthly")
    active = weights.loc[rebal]
    active = active[active.abs().sum(axis=1) > 0]

    assert len(active) > 0
    long_sums = active.where(active > 0, 0.0).sum(axis=1)
    short_sums = active.where(active < 0, 0.0).sum(axis=1)
    np.testing.assert_allclose(long_sums, 1.0)
    np.testing.assert_allclose(short_sums, -1.0)


def test_weights_before_first_signal_are_zero(simple_signal):
    weights = build_target_weights(simple_signal, freq="monthly")
    # Momentum needs 252 days of warm-up, so nothing is held before that.
    assert (weights.iloc[:252] == 0.0).all().all()
    assert not weights.isna().any().any()


def test_long_only_target_weights_sum_to_one(simple_signal):
    weights = build_target_weights(simple_signal, freq="monthly", kind="long_only")
    invested = weights[weights.sum(axis=1) > 0]
    np.testing.assert_allclose(invested.sum(axis=1), 1.0)


def test_turnover_zero_between_rebalances(synthetic_prices, simple_signal):
    returns = synthetic_prices.pct_change()
    weights = build_target_weights(simple_signal, freq="monthly")
    result = run_backtest(weights, returns, cost_bps=10.0, exec_lag=1)

    # Holdings change 2 trading days after each rebalance (shift(2) with exec_lag=1).
    rebal = rebalance_dates(simple_signal.index, "monthly")
    positions = [returns.index.get_loc(d) + 2 for d in rebal]
    trade_days = returns.index[[p for p in positions if p < len(returns.index)]]

    traded = result.index[result["turnover"] > 0]
    assert traded.isin(trade_days).all()
    assert len(traded) > 0
    non_trade = result.loc[~result.index.isin(trade_days), "turnover"]
    assert (non_trade == 0.0).all()


def test_rebalance_dates_with_named_index():
    """Regression: a DatetimeIndex named "date" (as read from Postgres) used to crash groupby."""
    index = pd.bdate_range("2024-01-01", "2024-03-31", name="date")
    dates = rebalance_dates(index, "monthly")
    assert list(dates.strftime("%Y-%m-%d")) == ["2024-01-31", "2024-02-29", "2024-03-29"]
