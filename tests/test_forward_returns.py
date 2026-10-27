"""Forward returns land on the signal date t and start at close t+exec_lag."""

import numpy as np
import pandas as pd
import pytest

from backtester.analysis.forward_returns import daily_returns, forward_returns


@pytest.fixture
def five_day_prices() -> pd.DataFrame:
    """One ticker, five closes: 100, 110, 99, 120, 132."""
    dates = pd.bdate_range("2024-01-01", periods=5)
    return pd.DataFrame({"A": [100.0, 110.0, 99.0, 120.0, 132.0]}, index=dates)


def test_exec_lag_1_horizon_1(five_day_prices):
    fwd = forward_returns(five_day_prices, horizon=1, exec_lag=1)["A"]
    # Row t = close t+2 / close t+1 - 1.
    assert fwd.iloc[0] == pytest.approx(99 / 110 - 1)
    assert fwd.iloc[1] == pytest.approx(120 / 99 - 1)
    assert fwd.iloc[2] == pytest.approx(132 / 120 - 1)
    assert np.isnan(fwd.iloc[3])
    assert np.isnan(fwd.iloc[4])


def test_exec_lag_0_horizon_1(five_day_prices):
    fwd = forward_returns(five_day_prices, horizon=1, exec_lag=0)["A"]
    # Row t = close t+1 / close t - 1.
    assert fwd.iloc[0] == pytest.approx(110 / 100 - 1)
    assert fwd.iloc[1] == pytest.approx(99 / 110 - 1)
    assert fwd.iloc[2] == pytest.approx(120 / 99 - 1)
    assert fwd.iloc[3] == pytest.approx(132 / 120 - 1)
    assert np.isnan(fwd.iloc[4])


def test_exec_lag_1_horizon_2(five_day_prices):
    fwd = forward_returns(five_day_prices, horizon=2, exec_lag=1)["A"]
    # Row t = close t+3 / close t+1 - 1.
    assert fwd.iloc[0] == pytest.approx(120 / 110 - 1)
    assert fwd.iloc[1] == pytest.approx(132 / 99 - 1)
    assert fwd.iloc[2:].isna().all()


def test_index_and_columns_unchanged(five_day_prices):
    fwd = forward_returns(five_day_prices, horizon=1, exec_lag=1)
    pd.testing.assert_index_equal(fwd.index, five_day_prices.index)
    pd.testing.assert_index_equal(fwd.columns, five_day_prices.columns)


def test_daily_returns_close_to_close(five_day_prices):
    rets = daily_returns(five_day_prices)["A"]
    assert np.isnan(rets.iloc[0])
    assert rets.iloc[1] == pytest.approx(0.10)
    assert rets.iloc[2] == pytest.approx(-0.10)


def test_daily_returns_does_not_fill_gaps(five_day_prices):
    prices = five_day_prices.copy()
    prices.iloc[2, 0] = np.nan
    rets = daily_returns(prices)["A"]
    assert np.isnan(rets.iloc[2])
    assert np.isnan(rets.iloc[3])


def test_bad_arguments_raise(five_day_prices):
    with pytest.raises(ValueError):
        forward_returns(five_day_prices, horizon=0)
    with pytest.raises(ValueError):
        forward_returns(five_day_prices, horizon=1, exec_lag=-1)
