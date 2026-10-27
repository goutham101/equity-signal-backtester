"""Phase 0 placeholder tests: config loads and the fixture has the right shape."""

from backtester.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["backtest"]["exec_lag"] == 1
    assert cfg["analysis"]["ic_horizons"] == [5, 21, 63]


def test_synthetic_prices_shape(synthetic_prices):
    assert synthetic_prices.shape == (800, 20)
    assert synthetic_prices.index.is_monotonic_increasing
    assert (synthetic_prices > 0).all().all()
