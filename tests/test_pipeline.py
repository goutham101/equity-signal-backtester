"""Tests for the shared pipeline helpers (synthetic data, no database)."""

import copy

import pandas as pd
import pytest

from backtester import pipeline
from backtester.config import load_config


@pytest.fixture
def small_cfg() -> dict:
    """The project config with a small min_stocks so 20 synthetic tickers are enough."""
    cfg = copy.deepcopy(load_config())
    cfg["signals"]["min_stocks"] = 5
    return cfg


def test_build_signals_has_all_keys(synthetic_prices, small_cfg):
    signals = pipeline.build_signals(synthetic_prices, small_cfg)
    assert list(signals) == pipeline.SIGNAL_NAMES


def test_build_signals_values_are_ranks_or_nan(synthetic_prices, small_cfg):
    signals = pipeline.build_signals(synthetic_prices, small_cfg)
    for name, frame in signals.items():
        values = frame.stack().dropna()
        assert len(values) > 0, name
        assert (values > 0).all(), name
        assert (values <= 1).all(), name


def test_build_signals_masks_thin_dates(synthetic_prices, small_cfg):
    prices = synthetic_prices.copy()
    # Leave only 3 stocks with a price on one date: below min_stocks=5.
    thin_date = prices.index[600]
    prices.loc[thin_date, prices.columns[3:]] = float("nan")
    signals = pipeline.build_signals(prices, small_cfg)
    assert signals["random"].loc[thin_date].isna().all()


def test_random_signal_reproducible_by_seed(synthetic_prices, small_cfg):
    first = pipeline.build_signals(synthetic_prices, small_cfg)["random"]
    second = pipeline.build_signals(synthetic_prices, small_cfg)["random"]
    pd.testing.assert_frame_equal(first, second)

    other_cfg = copy.deepcopy(small_cfg)
    other_cfg["seed"] = small_cfg["seed"] + 1
    different = pipeline.build_signals(synthetic_prices, other_cfg)["random"]
    assert not first.equals(different)


def test_save_and_load_signals_round_trip(synthetic_prices, small_cfg, tmp_path):
    signals = pipeline.build_signals(synthetic_prices, small_cfg)
    pipeline.save_signals(signals, tmp_path, small_cfg)
    loaded = pipeline.load_signals(tmp_path, small_cfg)
    for name in pipeline.SIGNAL_NAMES:
        pd.testing.assert_frame_equal(signals[name], loaded[name], check_freq=False)


def test_load_signals_rejects_stale_config(synthetic_prices, small_cfg, tmp_path):
    signals = pipeline.build_signals(synthetic_prices, small_cfg)
    pipeline.save_signals(signals, tmp_path, small_cfg)
    changed = copy.deepcopy(small_cfg)
    changed["signals"]["min_stocks"] = 6
    with pytest.raises(ValueError):
        pipeline.load_signals(tmp_path, changed)


def test_update_results_md_keeps_other_sections(tmp_path):
    path = tmp_path / "results.md"
    pipeline.update_results_md(path, {"IC": "ic table", "Backtest": "old"})
    pipeline.update_results_md(path, {"Backtest": "new"})
    sections = pipeline.read_results_sections(path)
    assert sections == {"IC": "ic table", "Backtest": "new"}
