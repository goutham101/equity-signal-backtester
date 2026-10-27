"""IC, IC summary, rolling IC, quintile returns, and the IC table."""

import numpy as np
import pandas as pd
import pytest

from backtester.analysis.forward_returns import forward_returns
from backtester.analysis.ic import (
    daily_ic,
    ic_summary,
    ic_table,
    newey_west_tstat,
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


def test_newey_west_zero_lags_is_plain_tstat():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0.03, 0.1, size=200))
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_tstat(x, lags=0) == pytest.approx(plain)


def test_newey_west_shrinks_t_for_overlapping_data():
    """Overlapping sums are autocorrelated, so the plain t-stat is too big."""
    rng = np.random.default_rng(1)
    shocks = pd.Series(rng.normal(0.02, 1.0, size=3000))
    overlapping = shocks.rolling(21).sum().dropna()
    plain = newey_west_tstat(overlapping, lags=0)
    corrected = newey_west_tstat(overlapping, lags=21)
    assert abs(corrected) < abs(plain) / 2


def test_newey_west_random_ic_is_near_zero():
    """Pure-noise ICs give a t-stat inside the usual range most of the time."""
    hits = 0
    for seed in range(200):
        noise = pd.Series(np.random.default_rng(seed).normal(0, 0.05, size=1000))
        if abs(newey_west_tstat(noise, lags=21)) < 1.96:
            hits += 1
    assert hits >= 180  # about 95% expected


def test_ic_summary_uses_newey_west():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=100)
    ic = pd.Series(rng.normal(0.03, 0.1, size=100), index=dates)
    stats = ic_summary(ic, horizon=21)
    assert stats["n_dates"] == 100
    assert stats["t_stat"] == pytest.approx(newey_west_tstat(ic, lags=21))


def test_ic_summary_basic_stats():
    ic = pd.Series([0.1, -0.1, 0.2, 0.0])
    stats = ic_summary(ic, horizon=1)
    assert stats["mean_ic"] == pytest.approx(0.05)
    assert stats["ic_std"] == pytest.approx(ic.std())
    assert stats["ic_ir"] == pytest.approx(0.05 / ic.std())
    assert stats["pct_positive"] == pytest.approx(0.5)


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
