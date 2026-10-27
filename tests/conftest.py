"""Shared test fixtures. Synthetic data only, so tests need no internet or database."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """Geometric random walk: 20 tickers x 800 business days, fixed seed."""
    rng = np.random.default_rng(42)
    n_days, n_tickers = 800, 20
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    daily_returns = rng.normal(0.0003, 0.02, size=(n_days, n_tickers))
    prices = 100 * np.exp(np.cumsum(daily_returns, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)
