"""IC, IC summary, rolling IC, quintile returns, and the IC table."""

import numpy as np
import pandas as pd
import pytest

from backtester.analysis.forward_returns import forward_returns
from backtester.analysis.ic import (
    daily_ic,
    ic_summary,
    ic_table,
    quintile_returns,
    rolling_ic,
)

MIN_STOCKS = 10  # synthetic data has only 20 tickers


@pytest.fixture
def fwd_5d(synthetic_prices) -> pd.DataFrame:
    return forward_returns(synthetic_prices, horizon=5, exec_lag=1)


def test_perfect_signal_has_ic_one(fwd_5d):
    ic = daily_ic(fwd_5d, fwd_5d, min_stocks=MIN_STOCKS)
    assert len(ic) > 0
    assert np.allclose(ic.values, 1.0)


def test_reversed_signal_has_ic_minus_one(fwd_5d):
    ic = daily_ic(-fwd_5d, fwd_5d, min_stocks=MIN_STOCKS)
    assert np.allclose(ic.values, -1.0)


def test_ic_drops_trailing_nan_dates(synthetic_prices, fwd_5d):
    ic = daily_ic(fwd_5d, fwd_5d, min_stocks=MIN_STOCKS)
    # Last 1 + 5 rows have no forward return.
    assert len(ic) == len(synthetic_prices) - 6
    assert ic.index[-1] == synthetic_prices.index[-7]


def test_min_stocks_filter(fwd_5d):
    signal = fwd_5d.copy()
    thin_date = signal.index[10]
    signal.loc[thin_date, signal.columns[:15]] = np.nan  # only 5 stocks left

    ic = daily_ic(signal, fwd_5d, min_stocks=MIN_STOCKS)
    assert thin_date not in ic.index
    assert signal.index[11] in ic.index

    ic_loose = daily_ic(signal, fwd_5d, min_stocks=5)
    assert thin_date in ic_loose.index


def test_t_stat_uses_non_overlapping_dates():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=100)
    ic = pd.Series(rng.normal(0.03, 0.1, size=100), index=dates)

    stats = ic_summary(ic, horizon=21)

    sample = ic.iloc[[0, 21, 42, 63, 84]]
    expected_t = sample.mean() / sample.std() * np.sqrt(5)
    assert stats["n_non_overlap"] == 5
    assert stats["n_dates"] == 100
    assert stats["t_stat"] == pytest.approx(expected_t)


def test_ic_summary_basic_stats():
    ic = pd.Series([0.1, -0.1, 0.2, 0.0])
    stats = ic_summary(ic, horizon=1)
    assert stats["mean_ic"] == pytest.approx(0.05)
    assert stats["ic_std"] == pytest.approx(ic.std())
    assert stats["ic_ir"] == pytest.approx(0.05 / ic.std())
    assert stats["pct_positive"] == pytest.approx(0.5)
    assert stats["n_non_overlap"] == 4


def test_rolling_ic_is_trailing_mean():
    ic = pd.Series([1.0, 2.0, 3.0, 4.0])
    rolled = rolling_ic(ic, window=2)
    assert np.isnan(rolled.iloc[0])
    assert rolled.iloc[1] == pytest.approx(1.5)
    assert rolled.iloc[3] == pytest.approx(3.5)


def test_quintile_returns_increase_for_perfect_signal(fwd_5d):
    q = quintile_returns(fwd_5d, fwd_5d, quantiles=5, min_stocks=MIN_STOCKS)
    assert list(q.index) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert q.is_monotonic_increasing
    assert q["Q5"] > q["Q1"]


def test_quintile_returns_skips_thin_dates(fwd_5d):
    q = quintile_returns(fwd_5d, fwd_5d, quantiles=5, min_stocks=50)
    assert q.isna().all()


def test_ic_table_shape(synthetic_prices):
    signals = {"perfect_ish": synthetic_prices.shift(-6) / synthetic_prices.shift(-1) - 1}
    table = ic_table(signals, synthetic_prices, horizons=[5, 21], exec_lag=1,
                     min_stocks=MIN_STOCKS)
    assert len(table) == 2
    assert set(table["horizon"]) == {5, 21}
    for col in ["mean_ic", "ic_std", "ic_ir", "pct_positive", "t_stat"]:
        assert col in table.columns
    five_day = table[table["horizon"] == 5].iloc[0]
    assert five_day["mean_ic"] == pytest.approx(1.0)
