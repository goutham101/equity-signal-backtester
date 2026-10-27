"""No-look-ahead test (Skill.md section 15.1). The most important test.

Idea: replace every price after a cutoff date with random junk. If a signal
only uses data up to close t, its values up to the cutoff cannot change.
"""

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

CUTOFF_ROW = 500


def corrupt_after_cutoff(prices: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Copy of prices with every row after the cutoff replaced by random values."""
    corrupted = prices.copy()
    future = corrupted.index > cutoff
    rng = np.random.default_rng(0)
    corrupted.loc[future] = rng.uniform(1, 1000, corrupted.loc[future].shape)
    return corrupted


def assert_no_lookahead(signal_func, prices: pd.DataFrame) -> None:
    """Fail if signal_func's output up to the cutoff depends on later prices."""
    cutoff = prices.index[CUTOFF_ROW]
    original = signal_func(prices)
    recomputed = signal_func(corrupt_after_cutoff(prices, cutoff))
    pd.testing.assert_frame_equal(original.loc[:cutoff], recomputed.loc[:cutoff])


def combined_signal(prices: pd.DataFrame) -> pd.DataFrame:
    """Baseline combination: equal-weight average of the 3 ranked signals."""
    ranked = {
        "momentum": cs_rank(momentum_12_1(prices)),
        "reversal": cs_rank(short_term_reversal(prices)),
        "low_vol": cs_rank(low_volatility(prices)),
    }
    return combine_equal_weight(ranked)


SIGNAL_FUNCS = {
    "momentum_12_1": momentum_12_1,
    "short_term_reversal": short_term_reversal,
    "low_volatility": low_volatility,
    "random_signal": lambda p: random_signal(p, seed=7),
}

TRANSFORMS = {
    "cs_rank": cs_rank,
    "cs_zscore": cs_zscore,
    "mask_thin_dates": lambda s: mask_thin_dates(s, min_stocks=15),
}


@pytest.mark.parametrize("name", list(SIGNAL_FUNCS))
def test_signals_do_not_use_future_data(synthetic_prices, name):
    assert_no_lookahead(SIGNAL_FUNCS[name], synthetic_prices)


@pytest.mark.parametrize("signal_name", list(SIGNAL_FUNCS))
@pytest.mark.parametrize("transform_name", list(TRANSFORMS))
def test_transforms_do_not_use_future_data(synthetic_prices, signal_name, transform_name):
    signal_func = SIGNAL_FUNCS[signal_name]
    transform = TRANSFORMS[transform_name]
    assert_no_lookahead(lambda p: transform(signal_func(p)), synthetic_prices)


def test_combined_signal_does_not_use_future_data(synthetic_prices):
    assert_no_lookahead(combined_signal, synthetic_prices)


def test_combined_signal_average_of_available_does_not_use_future_data(synthetic_prices):
    def combined_any(prices):
        ranked = [cs_rank(momentum_12_1(prices)), cs_rank(short_term_reversal(prices))]
        return combine_equal_weight(ranked, min_signals=1)

    assert_no_lookahead(combined_any, synthetic_prices)


def test_check_has_real_values_before_cutoff(synthetic_prices):
    # Guard against a vacuous pass: the compared window must hold real numbers.
    cutoff = synthetic_prices.index[CUTOFF_ROW]
    assert combined_signal(synthetic_prices).loc[:cutoff].notna().any().any()


# ---------- the test must be able to catch a leak ----------

def leaky_next_day(prices: pd.DataFrame) -> pd.DataFrame:
    """Deliberately wrong: uses close t+1 at date t."""
    return prices.shift(-1)


def leaky_full_period_zscore(prices: pd.DataFrame) -> pd.DataFrame:
    """Deliberately wrong: normalises with the mean and std over ALL dates."""
    signal = momentum_12_1(prices)
    return (signal - signal.mean()) / signal.std()


def leaky_centered_window(prices: pd.DataFrame) -> pd.DataFrame:
    """Deliberately wrong: a centered rolling window looks forward."""
    return prices.rolling(5, center=True).mean()


@pytest.mark.parametrize(
    "leaky_func", [leaky_next_day, leaky_full_period_zscore, leaky_centered_window]
)
def test_lookahead_check_catches_leaks(synthetic_prices, leaky_func):
    with pytest.raises(AssertionError):
        assert_no_lookahead(leaky_func, synthetic_prices)
