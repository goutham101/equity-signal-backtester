"""Unit tests for signals and cross-sectional transforms, on tiny hand-made data."""

import numpy as np
import pandas as pd
import pytest

from backtester.signals import (
    combine_equal_weight,
    cs_rank,
    cs_zscore,
    low_volatility,
    mask_thin_dates,
    momentum_12_1,
    random_signal,
    short_term_reversal,
)


def make_frame(values: dict[str, list[float]]) -> pd.DataFrame:
    """Small wide DataFrame on business days."""
    n_days = len(next(iter(values.values())))
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    return pd.DataFrame(values, index=dates, dtype=float)


# ---------- momentum ----------

def test_momentum_hand_computed():
    # Prices 10, 11, ..., 19. lookback=4, skip=1:
    # value at row i = price[i-1] / price[i-4] - 1.
    prices = make_frame({"A": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]})
    result = momentum_12_1(prices, lookback=4, skip=1)

    assert result["A"].iloc[:4].isna().all()
    # Row 4: price[3] / price[0] - 1 = 13 / 10 - 1 = 0.3
    assert result["A"].iloc[4] == pytest.approx(0.3)
    # Row 9: price[8] / price[5] - 1 = 18 / 15 - 1 = 0.2
    assert result["A"].iloc[9] == pytest.approx(0.2)


def test_momentum_default_windows(synthetic_prices):
    result = momentum_12_1(synthetic_prices)
    assert result.iloc[:252].isna().all().all()
    row = 300
    expected = synthetic_prices.iloc[row - 21] / synthetic_prices.iloc[row - 252] - 1
    pd.testing.assert_series_equal(result.iloc[row], expected, check_names=False)


# ---------- reversal ----------

def test_reversal_hand_computed():
    prices = make_frame({"A": [100, 101, 102, 103, 104, 110, 90]})
    result = short_term_reversal(prices, lookback=5)

    assert result["A"].iloc[:5].isna().all()
    # Row 5: -(110 / 100 - 1) = -0.10 (a winner gets a low score)
    assert result["A"].iloc[5] == pytest.approx(-0.10)
    # Row 6: -(90 / 101 - 1) is positive (a loser gets a high score)
    assert result["A"].iloc[6] == pytest.approx(-(90 / 101 - 1))


# ---------- low volatility ----------

def test_low_vol_hand_computed():
    # Returns: NaN, +10%, -10%, +10%, -10% (prices 100, 110, 99, 108.9, 98.01)
    prices = make_frame({"A": [100, 110, 99, 108.9, 98.01]})
    result = low_volatility(prices, window=3)

    # Window of 3 returns needs rows 1..3, so rows 0..2 are NaN.
    assert result["A"].iloc[:3].isna().all()
    expected = -np.std([0.10, -0.10, 0.10], ddof=1)
    assert result["A"].iloc[3] == pytest.approx(expected)
    assert result["A"].iloc[4] == pytest.approx(-np.std([-0.10, 0.10, -0.10], ddof=1))


def test_low_vol_calm_stock_scores_higher():
    prices = make_frame({
        "calm": [100, 101, 100, 101, 100, 101],
        "wild": [100, 120, 90, 125, 85, 130],
    })
    result = low_volatility(prices, window=3)
    assert (result["calm"].iloc[3:] > result["wild"].iloc[3:]).all()


def test_low_vol_does_not_fill_missing_prices():
    prices = make_frame({"A": [100, 101, np.nan, 103, 104, 105, 106]})
    result = low_volatility(prices, window=2)
    # Returns at rows 2 and 3 are NaN, so windows touching them are NaN.
    assert result["A"].iloc[:5].isna().all()
    assert result["A"].iloc[5:].notna().all()


# ---------- cs_rank ----------

def test_cs_rank_values_and_nans():
    signal = make_frame({
        "A": [1.0, 5.0],
        "B": [2.0, np.nan],
        "C": [3.0, 4.0],
        "D": [4.0, 3.0],
    })
    ranked = cs_rank(signal)

    assert ranked.iloc[0].tolist() == pytest.approx([0.25, 0.5, 0.75, 1.0])
    # Row 1 has 3 valid values: D=3 -> 1/3, C=4 -> 2/3, A=5 -> 1.
    assert np.isnan(ranked.iloc[1]["B"])
    assert ranked.iloc[1][["A", "C", "D"]].tolist() == pytest.approx([1.0, 2 / 3, 1 / 3])

    valid = ranked.stack().dropna()
    assert (valid > 0).all() and (valid <= 1).all()


# ---------- cs_zscore ----------

def test_cs_zscore_row_mean_zero(synthetic_prices):
    signal = momentum_12_1(synthetic_prices)
    z = cs_zscore(signal, clip=np.inf)
    valid_rows = z.dropna(how="all")
    assert len(valid_rows) > 0
    assert np.allclose(valid_rows.mean(axis=1), 0.0, atol=1e-10)
    assert np.allclose(valid_rows.std(axis=1), 1.0)


def test_cs_zscore_is_clipped():
    # One outlier among 20 stocks: its unclipped z-score is 19 / sqrt(20), about 4.25.
    values = {f"T{i}": [0.0] for i in range(19)}
    values["OUT"] = [100.0]
    signal = make_frame(values)

    unclipped = cs_zscore(signal, clip=np.inf)
    assert unclipped["OUT"].iloc[0] > 3.0

    clipped = cs_zscore(signal, clip=3.0)
    assert clipped["OUT"].iloc[0] == pytest.approx(3.0)
    assert (clipped.abs() <= 3.0).all().all()


def test_cs_zscore_constant_row_is_nan():
    signal = make_frame({"A": [1.0], "B": [1.0], "C": [1.0]})
    assert cs_zscore(signal).isna().all().all()


# ---------- mask_thin_dates ----------

def test_mask_thin_dates():
    signal = make_frame({
        "A": [1.0, 1.0, np.nan],
        "B": [2.0, np.nan, np.nan],
        "C": [3.0, np.nan, 5.0],
    })
    masked = mask_thin_dates(signal, min_stocks=2)

    pd.testing.assert_series_equal(masked.iloc[0], signal.iloc[0])
    assert masked.iloc[1].isna().all()
    assert masked.iloc[2].isna().all()
    # Input is not changed in place.
    assert signal.iloc[1]["A"] == 1.0


# ---------- combine_equal_weight ----------

def test_combine_equal_weight_requires_all_by_default():
    a = make_frame({"X": [0.2, 0.4], "Y": [1.0, np.nan]})
    b = make_frame({"X": [0.6, 0.8], "Y": [0.5, 0.5]})
    combined = combine_equal_weight({"a": a, "b": b})

    assert combined["X"].tolist() == pytest.approx([0.4, 0.6])
    assert combined["Y"].iloc[0] == pytest.approx(0.75)
    assert np.isnan(combined["Y"].iloc[1])


def test_combine_equal_weight_average_of_available():
    a = make_frame({"X": [0.2], "Y": [np.nan]})
    b = make_frame({"X": [0.6], "Y": [0.5]})
    combined = combine_equal_weight([a, b], min_signals=1)

    assert combined["X"].iloc[0] == pytest.approx(0.4)
    assert combined["Y"].iloc[0] == pytest.approx(0.5)


# ---------- random baseline ----------

def test_random_signal_shape_and_nans():
    prices = make_frame({"A": [1.0, np.nan, 3.0], "B": [1.0, 2.0, 3.0]})
    signal = random_signal(prices, seed=1)

    assert signal.shape == prices.shape
    assert np.isnan(signal["A"].iloc[1])
    assert signal.drop(index=prices.index[1])["A"].notna().all()
    pd.testing.assert_frame_equal(signal, random_signal(prices, seed=1))


def test_random_signal_ic_is_near_zero(synthetic_prices):
    # 5-day forward return from close t+1 to close t+6 (exec_lag = 1), aligned to t.
    start = synthetic_prices.shift(-1)
    end = synthetic_prices.shift(-6)
    forward = end / start - 1

    mean_ics = []
    for seed in range(50):
        signal = random_signal(synthetic_prices, seed=seed)
        daily_ic = signal.corrwith(forward, axis=1, method="spearman")
        mean_ics.append(daily_ic.mean())

    mean_ics = np.array(mean_ics)
    # Each seed: ~800 daily ICs on 20 stocks, std of the mean is about 0.008.
    assert np.abs(mean_ics).max() < 0.05
    assert abs(mean_ics.mean()) < 0.01
