"""Cached loaders for the precomputed files in dashboard/data/.

The app only reads these files (docs/dashboard_data_contract.md). It never
touches Postgres, yfinance, or data/processed/.

Every cached function takes the data folder as an argument, so changing
DASHBOARD_DATA_DIR (tests do this) never returns stale cached data.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from components import REPO_ROOT

DEFAULT_DATA_DIR = REPO_ROOT / "dashboard" / "data"

REQUIRED_FILES = [
    "daily_returns.parquet",
    "ic_daily.parquet",
    "ic_summary.parquet",
    "quintile_returns.parquet",
    "walk_forward_weights.parquet",
    "walk_forward_windows.parquet",
    "summary.parquet",
    "metadata.json",
]


def data_dir() -> Path:
    """Folder with the exported files: $DASHBOARD_DATA_DIR or dashboard/data."""
    return Path(os.environ.get("DASHBOARD_DATA_DIR", DEFAULT_DATA_DIR))


def missing_files() -> list[str]:
    """Names of required files that don't exist in the data folder."""
    folder = data_dir()
    return [name for name in REQUIRED_FILES if not (folder / name).exists()]


@st.cache_data(show_spinner=False)
def _read_parquet(folder: str, name: str) -> pd.DataFrame:
    """Read one Parquet file from the data folder."""
    return pd.read_parquet(Path(folder) / name)


@st.cache_data(show_spinner=False)
def _read_metadata(folder: str) -> dict:
    """Read metadata.json from the data folder."""
    return json.loads((Path(folder) / "metadata.json").read_text())


@st.cache_data(show_spinner=False)
def _read_series(
    folder: str, signal: str, portfolio: str, rebalance: str, period: str
) -> pd.DataFrame:
    """One return series from daily_returns.parquet, indexed by date."""
    df = _read_parquet(folder, "daily_returns.parquet")
    mask = (
        (df["signal"] == signal)
        & (df["portfolio"] == portfolio)
        & (df["rebalance"] == rebalance)
        & (df["period"] == period)
    )
    columns = ["date", "gross", "turnover", "gross_long", "gross_short"]
    frame = df.loc[mask, columns].sort_values("date").set_index("date")
    frame.index.name = "date"
    return frame


def daily_returns() -> pd.DataFrame:
    """All daily return rows (long format)."""
    return _read_parquet(str(data_dir()), "daily_returns.parquet")


def series(signal: str, portfolio: str, rebalance: str, period: str) -> pd.DataFrame:
    """Daily gross, turnover, and leg returns for one series. Empty if it doesn't exist."""
    return _read_series(str(data_dir()), signal, portfolio, rebalance, period)


def ic_daily() -> pd.DataFrame:
    """Daily IC per signal and horizon."""
    return _read_parquet(str(data_dir()), "ic_daily.parquet")


def ic_summary() -> pd.DataFrame:
    """IC summary stats per signal and horizon (Newey-West t-stats)."""
    return _read_parquet(str(data_dir()), "ic_summary.parquet")


def quintile_returns() -> pd.DataFrame:
    """Average forward return per signal, horizon, and quintile."""
    return _read_parquet(str(data_dir()), "quintile_returns.parquet")


def walk_forward_weights() -> pd.DataFrame:
    """Fitted signal weights per walk-forward window."""
    return _read_parquet(str(data_dir()), "walk_forward_weights.parquet")


def walk_forward_windows() -> pd.DataFrame:
    """Train and test dates per walk-forward window."""
    return _read_parquet(str(data_dir()), "walk_forward_windows.parquet")


def summary() -> pd.DataFrame:
    """Overview headline table (out-of-sample, long-short, monthly, 10 bps)."""
    return _read_parquet(str(data_dir()), "summary.parquet")


def metadata() -> dict:
    """Universe, dates, findings, limitations, links, and test status."""
    return _read_metadata(str(data_dir()))
