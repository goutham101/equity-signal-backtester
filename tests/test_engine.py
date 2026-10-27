"""Tests for the backtest engine: timing, costs, alignment, and no look-ahead."""

import numpy as np
import pandas as pd
import pytest

from backtester.backtest.engine import cost_sensitivity, run_backtest
from backtester.backtest.portfolio import build_target_weights


def _one_stock_frames(n_days: int = 10):
    """A single stock 'A' with zero returns everywhere; tests fill in what they need."""
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    weights = pd.DataFrame({"A": 0.0}, index=dates)
    returns = pd.DataFrame({"A": 0.0}, index=dates)
    return weights, returns


@pytest.mark.parametrize("exec_lag", [0, 1, 2])
def test_weight_earns_return_exactly_1_plus_lag_days_later(exec_lag):
    weights, returns = _one_stock_frames()
    t = 3
    weights.iloc[t:] = 1.0  # decided at close t, held from then on

    # Put a nonzero return on every day after t, one at a time, and see when it pays.
    for d in range(len(returns)):
        test_returns = returns.copy()
        test_returns.iloc[d] = 0.05
        result = run_backtest(weights, test_returns, cost_bps=0.0, exec_lag=exec_lag)
        if d >= t + 1 + exec_lag:
            assert result["gross"].iloc[d] == pytest.approx(0.05)
        else:
            assert result["gross"].sum() == 0.0


def test_single_day_weight_earns_only_day_t_plus_2():
    weights, returns = _one_stock_frames()
    t = 3
    weights.iloc[t] = 1.0  # one-day position decided at close t
    returns.iloc[:] = 0.01  # every day has a return

    result = run_backtest(weights, returns, cost_bps=0.0, exec_lag=1)

    expected = pd.Series(0.0, index=returns.index)
    expected.iloc[t + 2] = 0.01
    pd.testing.assert_series_equal(result["gross"], expected, check_names=False)

    result_lag0 = run_backtest(weights, returns, cost_bps=0.0, exec_lag=0)
    expected_lag0 = pd.Series(0.0, index=returns.index)
    expected_lag0.iloc[t + 1] = 0.01
    pd.testing.assert_series_equal(result_lag0["gross"], expected_lag0, check_names=False)


def test_full_flip_costs_two_units():
    weights, returns = _one_stock_frames()
    weights.iloc[:] = 1.0
    weights.iloc[5:] = -1.0  # flip decided at close 5, held from day 7

    result = run_backtest(weights, returns, cost_bps=10.0, exec_lag=1)

    assert result["costs"].iloc[7] == pytest.approx(2 * 0.001)
    assert result["turnover"].iloc[7] == pytest.approx(2.0)


def test_initial_entry_costs_one_unit():
    weights, returns = _one_stock_frames()
    weights.iloc[3:] = 1.0

    result = run_backtest(weights, returns, cost_bps=10.0, exec_lag=1)

    assert result["costs"].iloc[5] == pytest.approx(0.001)
    assert result["costs"].sum() == pytest.approx(0.001)
    assert result["net"].iloc[5] == pytest.approx(-0.001)


def test_net_equals_gross_minus_costs(synthetic_prices):
    returns = synthetic_prices.pct_change()
    signal = synthetic_prices.shift(21) / synthetic_prices.shift(252) - 1
    weights = build_target_weights(signal, freq="monthly")
    result = run_backtest(weights, returns, cost_bps=10.0)

    pd.testing.assert_series_equal(
        result["net"], result["gross"] - result["costs"], check_names=False
    )
    assert list(result.columns) == [
        "gross", "costs", "net", "turnover", "gross_long", "gross_short"
    ]


def test_alignment_reorders_columns_and_fills_missing_tickers():
    dates = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.DataFrame({"A": 0.01, "B": 0.02, "C": 0.03}, index=dates)
    weights = pd.DataFrame({"B": 1.0, "A": 0.0}, index=dates)  # no C, columns reordered

    result = run_backtest(weights, returns, cost_bps=0.0, exec_lag=1)

    assert result["gross"].iloc[2:].tolist() == pytest.approx([0.02] * 4)


def test_alignment_rejects_unknown_tickers():
    dates = pd.bdate_range("2024-01-01", periods=6)
    returns = pd.DataFrame({"A": 0.01}, index=dates)
    weights = pd.DataFrame({"A": 0.5, "Z": 0.5}, index=dates)
    with pytest.raises(ValueError):
        run_backtest(weights, returns)


def test_cost_sensitivity_returns_one_frame_per_level(synthetic_prices):
    returns = synthetic_prices.pct_change()
    signal = synthetic_prices.shift(21) / synthetic_prices.shift(252) - 1
    weights = build_target_weights(signal, freq="monthly")

    results = cost_sensitivity(weights, returns, [0, 10, 50], exec_lag=1)

    assert list(results) == [0, 10, 50]
    pd.testing.assert_series_equal(results[0]["gross"], results[50]["gross"])
    assert (results[0]["costs"] == 0.0).all()
    assert results[50]["net"].sum() < results[10]["net"].sum() < results[0]["net"].sum()


@pytest.mark.parametrize("kind", ["long_short", "long_only", "equal_weight"])
@pytest.mark.parametrize("freq", ["monthly", "weekly"])
def test_target_weights_do_not_use_future_data(synthetic_prices, kind, freq):
    cutoff = synthetic_prices.index[500]

    def signal_fn(prices):
        return prices.shift(21) / prices.shift(252) - 1

    original = build_target_weights(signal_fn(synthetic_prices), freq=freq, kind=kind)

    corrupted = synthetic_prices.copy()
    future = corrupted.index > cutoff
    noise = np.random.default_rng(0).uniform(1, 1000, corrupted.loc[future].shape)
    corrupted.loc[future] = noise
    recomputed = build_target_weights(signal_fn(corrupted), freq=freq, kind=kind)

    pd.testing.assert_frame_equal(original.loc[:cutoff], recomputed.loc[:cutoff])
    # Sanity: corruption really changed something after the cutoff. The
    # equal-weight benchmark only depends on which values are NaN, so skip it.
    if kind != "equal_weight":
        assert not original.loc[cutoff:].iloc[1:].equals(recomputed.loc[cutoff:].iloc[1:])


def test_backtest_does_not_use_future_returns(synthetic_prices):
    cutoff = synthetic_prices.index[500]
    returns = synthetic_prices.pct_change()
    signal = synthetic_prices.shift(21) / synthetic_prices.shift(252) - 1
    weights = build_target_weights(signal, freq="monthly")
    original = run_backtest(weights, returns)

    corrupted_returns = returns.copy()
    corrupted_returns.loc[corrupted_returns.index > cutoff] = 0.5
    recomputed = run_backtest(weights, corrupted_returns)

    pd.testing.assert_frame_equal(original.loc[:cutoff], recomputed.loc[:cutoff])


def test_legs_sum_to_gross(synthetic_prices):
    returns = synthetic_prices.pct_change()
    signal = synthetic_prices.shift(21) / synthetic_prices.shift(252) - 1
    weights = build_target_weights(signal, freq="monthly", kind="long_short")
    result = run_backtest(weights, returns, cost_bps=10.0)

    legs = result["gross_long"] + result["gross_short"]
    pd.testing.assert_series_equal(legs, result["gross"], check_names=False, rtol=0, atol=1e-15)
    # Both legs really trade in a long-short book.
    assert (result["gross_long"] != 0).any()
    assert (result["gross_short"] != 0).any()


def test_long_only_has_no_short_leg(synthetic_prices):
    returns = synthetic_prices.pct_change()
    signal = synthetic_prices.shift(21) / synthetic_prices.shift(252) - 1
    weights = build_target_weights(signal, freq="monthly", kind="long_only")
    result = run_backtest(weights, returns, cost_bps=10.0)

    assert (result["gross_short"] == 0.0).all()
    pd.testing.assert_series_equal(result["gross_long"], result["gross"], check_names=False)


def test_only_short_leg_makes_money():
    # Long A, short B. A goes nowhere, B falls 2% every day, so all profit
    # comes from the short leg.
    dates = pd.bdate_range("2024-01-01", periods=6)
    weights = pd.DataFrame({"A": 1.0, "B": -1.0}, index=dates)
    returns = pd.DataFrame({"A": 0.0, "B": -0.02}, index=dates)

    result = run_backtest(weights, returns, cost_bps=0.0, exec_lag=1)

    # Held from day 2 (decision at close 0, trade at close 1).
    assert (result["gross_long"] == 0.0).all()
    assert result["gross_short"].iloc[:2].tolist() == [0.0, 0.0]
    assert result["gross_short"].iloc[2:].tolist() == pytest.approx([0.02] * 4)
    pd.testing.assert_series_equal(result["gross_short"], result["gross"], check_names=False)
