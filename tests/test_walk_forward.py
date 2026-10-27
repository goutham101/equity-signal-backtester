"""Tests for walk-forward windows, weight fitting, and the out-of-sample signal."""

import numpy as np
import pandas as pd
import pytest

from backtester.analysis.forward_returns import forward_returns
from backtester.backtest.walk_forward import (
    fit_ic_weights,
    fit_regression_weights,
    make_windows,
    walk_forward_signal,
)
from backtester.signals.transforms import cs_rank

HORIZON = 21
EXEC_LAG = 1
MIN_STOCKS = 10


@pytest.fixture
def long_prices() -> pd.DataFrame:
    """Geometric random walk: 30 tickers x 10 calendar years (2010-2019), fixed seed."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2010-01-01", "2019-12-31")
    tickers = [f"L{i:02d}" for i in range(30)]
    daily_returns = rng.normal(0.0003, 0.02, size=(len(dates), len(tickers)))
    prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def price_signals(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Three simple backward-looking ranked signals built only from closes up to t."""
    return {
        "mom": cs_rank(prices.pct_change(63, fill_method=None)),
        "rev": cs_rank(-prices.pct_change(5, fill_method=None)),
        "lowvol": cs_rank(-prices.pct_change(fill_method=None).rolling(21).std()),
    }


def noise_signal(prices: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Random ranked signal with no relation to returns."""
    rng = np.random.default_rng(seed)
    values = rng.random(prices.shape)
    return cs_rank(pd.DataFrame(values, index=prices.index, columns=prices.columns))


# ---------- windows ----------


def test_windows_cover_expected_years(long_prices):
    windows = make_windows(long_prices.index, horizon=HORIZON, exec_lag=EXEC_LAG)
    labels = [w["label"] for w in windows]
    assert labels == ["2015", "2016", "2017", "2018", "2019"]
    assert windows[0]["train_start"] == pd.Timestamp("2010-01-01")
    assert windows[0]["test_start"] == pd.Timestamp("2015-01-01")
    assert windows[0]["test_end"] == pd.Timestamp("2015-12-31")


def test_windows_no_overlap_and_purge_gap(long_prices):
    dates = long_prices.index
    windows = make_windows(dates, purge_days=5, horizon=HORIZON, exec_lag=EXEC_LAG)
    assert len(windows) > 0

    for w in windows:
        assert w["train_start"] < w["train_end"] < w["test_start"] <= w["test_end"]
        gap = dates.get_loc(w["test_start"]) - dates.get_loc(w["train_end"])
        assert gap >= HORIZON + EXEC_LAG
        # The label on the last training date must end before the test starts.
        label_end_pos = dates.get_loc(w["train_end"]) + EXEC_LAG + HORIZON
        assert label_end_pos < dates.get_loc(w["test_start"])

    for earlier, later in zip(windows[:-1], windows[1:], strict=True):
        assert earlier["test_end"] < later["test_start"]


def test_windows_purge_days_can_widen_gap(long_prices):
    dates = long_prices.index
    windows = make_windows(dates, purge_days=60, horizon=HORIZON, exec_lag=EXEC_LAG)
    for w in windows:
        gap = dates.get_loc(w["test_start"]) - dates.get_loc(w["train_end"])
        assert gap >= 60


def test_windows_drop_partial_first_year():
    dates = pd.bdate_range("2010-06-15", "2017-12-31")
    windows = make_windows(dates)
    # 2010 is partial, so the first full train period is 2011-2015.
    assert [w["label"] for w in windows] == ["2016", "2017"]
    assert windows[0]["train_start"].year == 2011


def test_windows_empty_when_too_short(synthetic_prices):
    # About 3 years of data: no full 5-year train period.
    assert make_windows(synthetic_prices.index) == []


# ---------- weight fitting ----------


def test_ic_weights_sum_to_one_and_drop_negative(long_prices):
    fwd = forward_returns(long_prices, horizon=HORIZON, exec_lag=EXEC_LAG)
    ranked = {
        "good": cs_rank(fwd),
        "bad": cs_rank(-fwd),
        "noise": noise_signal(long_prices, seed=1),
    }
    train_dates = long_prices.index[:1000]
    weights = fit_ic_weights(ranked, fwd, train_dates, min_stocks=MIN_STOCKS)

    assert weights.sum() == pytest.approx(1.0)
    assert weights["bad"] == 0.0
    assert (weights >= 0).all()


def test_ic_weights_all_negative_falls_back_to_equal(long_prices):
    fwd = forward_returns(long_prices, horizon=HORIZON, exec_lag=EXEC_LAG)
    ranked = {"a": cs_rank(-fwd), "b": cs_rank(-fwd * 2)}
    train_dates = long_prices.index[:1000]
    weights = fit_ic_weights(ranked, fwd, train_dates, min_stocks=MIN_STOCKS)
    assert weights["a"] == pytest.approx(0.5)
    assert weights["b"] == pytest.approx(0.5)


@pytest.mark.parametrize("fit", [fit_ic_weights, fit_regression_weights])
def test_planted_signal_gets_almost_all_weight(long_prices, fit):
    fwd = forward_returns(long_prices, horizon=HORIZON, exec_lag=EXEC_LAG)
    ranked = {
        "planted": cs_rank(fwd),
        "noise1": noise_signal(long_prices, seed=1),
        "noise2": noise_signal(long_prices, seed=2),
    }
    train_dates = long_prices.index[:1250]
    weights = fit(ranked, fwd, train_dates, min_stocks=MIN_STOCKS)

    assert weights.sum() == pytest.approx(1.0)
    assert weights["planted"] > 0.9


def test_regression_weights_all_negative_falls_back_to_equal(long_prices):
    fwd = forward_returns(long_prices, horizon=HORIZON, exec_lag=EXEC_LAG)
    ranked = {"a": cs_rank(-fwd), "b": cs_rank(-fwd)}
    weights = fit_regression_weights(ranked, fwd, long_prices.index[:1000], min_stocks=MIN_STOCKS)
    assert weights.tolist() == pytest.approx([0.5, 0.5])


# ---------- walk-forward signal ----------


@pytest.mark.parametrize("method", ["ic_weighted", "regression"])
def test_signal_nan_outside_test_windows(long_prices, method):
    ranked = price_signals(long_prices)
    combined, weights, windows = walk_forward_signal(
        ranked, long_prices, min_stocks=MIN_STOCKS, method=method
    )

    in_test = pd.Series(False, index=long_prices.index)
    for w in windows:
        in_test[(in_test.index >= w["test_start"]) & (in_test.index <= w["test_end"])] = True

    assert combined.loc[~in_test.to_numpy()].isna().all().all()
    assert combined.loc[in_test.to_numpy()].notna().any().any()

    assert list(weights.index) == [w["label"] for w in windows]
    assert list(weights.columns) == list(ranked.keys())
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)


def test_signal_matches_manual_weighted_sum(long_prices):
    ranked = price_signals(long_prices)
    combined, weights, windows = walk_forward_signal(ranked, long_prices, min_stocks=MIN_STOCKS)

    w = windows[2]
    date = long_prices.index[long_prices.index >= w["test_start"]][10]
    expected = sum(weights.loc[w["label"], name] * ranked[name].loc[date] for name in ranked)
    pd.testing.assert_series_equal(combined.loc[date], expected, check_names=False)


def test_signal_needs_all_signals_present(long_prices):
    ranked = price_signals(long_prices)
    ranked["mom"] = ranked["mom"].copy()
    ranked["mom"].loc["2016-03-01":"2016-03-10", "L00"] = np.nan
    combined, _, _ = walk_forward_signal(ranked, long_prices, min_stocks=MIN_STOCKS)
    assert combined.loc["2016-03-01":"2016-03-10", "L00"].isna().all()
    assert combined.loc["2016-03-01":"2016-03-10", "L01"].notna().all()


# ---------- no look-ahead (the key test) ----------


@pytest.mark.parametrize("method", ["ic_weighted", "regression"])
def test_walk_forward_does_not_use_future_data(long_prices, method):
    """Corrupt every close from a test_start onward.

    The weights for that window (and all earlier windows) and the combined
    signal on every date before test_start must not change.
    """
    original_ranked = price_signals(long_prices)
    original, original_weights, windows = walk_forward_signal(
        original_ranked, long_prices, min_stocks=MIN_STOCKS, method=method
    )

    target = windows[2]
    cutoff = target["test_start"]

    corrupted = long_prices.copy()
    future = corrupted.index >= cutoff
    rng = np.random.default_rng(0)
    corrupted.loc[future] = rng.uniform(1, 1000, corrupted.loc[future].shape)

    corrupted_ranked = price_signals(corrupted)
    recomputed, recomputed_weights, _ = walk_forward_signal(
        corrupted_ranked, corrupted, min_stocks=MIN_STOCKS, method=method
    )

    labels_up_to_target = [w["label"] for w in windows[:3]]
    pd.testing.assert_frame_equal(
        original_weights.loc[labels_up_to_target],
        recomputed_weights.loc[labels_up_to_target],
    )
    before = original.index < cutoff
    pd.testing.assert_frame_equal(original.loc[before], recomputed.loc[before])


def test_corruption_inside_label_window_does_change_weights(long_prices):
    """Sanity check that the no-look-ahead test is not vacuous.

    Corrupting closes right after train_end (inside the purge gap, which the
    last training labels do use) should change that window's weights.
    """
    fwd = forward_returns(long_prices, horizon=HORIZON, exec_lag=EXEC_LAG)
    # Two noisy versions of the label, both with positive IC, so neither
    # weight is clipped to 0 and a change in the labels shows up.
    rng = np.random.default_rng(3)
    noise_a = rng.normal(0, 0.05, fwd.shape)
    noise_b = rng.normal(0, 0.10, fwd.shape)
    ranked = {"a": cs_rank(fwd + noise_a), "b": cs_rank(fwd + noise_b)}
    w = make_windows(long_prices.index)[2]
    dates = long_prices.index
    train_dates = dates[(dates >= w["train_start"]) & (dates <= w["train_end"])]
    original = fit_ic_weights(ranked, fwd, train_dates, min_stocks=MIN_STOCKS)

    corrupted = long_prices.copy()
    after_train = corrupted.index > w["train_end"]
    rng = np.random.default_rng(0)
    corrupted.loc[after_train] = rng.uniform(1, 1000, corrupted.loc[after_train].shape)
    fwd_corrupted = forward_returns(corrupted, horizon=HORIZON, exec_lag=EXEC_LAG)
    changed = fit_ic_weights(ranked, fwd_corrupted, train_dates, min_stocks=MIN_STOCKS)

    assert not np.allclose(original.to_numpy(), changed.to_numpy())
