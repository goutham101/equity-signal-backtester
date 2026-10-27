"""Current S&P 500 constituents (ticker, name, sector).

This list is today's membership, so it has survivorship bias: companies that
were removed or went bankrupt are missing. That is accepted for v1.
"""

from pathlib import Path

import pandas as pd

from backtester.config import ROOT

CONSTITUENTS_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
)
RAW_DIR = ROOT / "data" / "raw"
CACHE_PATH = RAW_DIR / "sp500_constituents.csv"


def to_yfinance_ticker(symbol: str) -> str:
    """Convert a listing symbol to Yahoo's format (BRK.B -> BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def fetch_constituents(cache_path: Path = CACHE_PATH, refresh: bool = False) -> pd.DataFrame:
    """Read the raw constituents CSV, from the local cache if it exists."""
    if cache_path.exists() and not refresh:
        return pd.read_csv(cache_path)

    raw = pd.read_csv(CONSTITUENTS_URL)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(cache_path, index=False)
    return raw


def load_universe(cache_path: Path = CACHE_PATH, refresh: bool = False) -> pd.DataFrame:
    """Return current S&P 500 members with columns ticker, name, sector.

    Tickers are already converted to yfinance format and sorted.
    """
    raw = fetch_constituents(cache_path, refresh)

    universe = pd.DataFrame(
        {
            "ticker": raw["Symbol"].astype(str).map(to_yfinance_ticker),
            "name": raw["Security"],
            "sector": raw["GICS Sector"],
        }
    )
    universe = universe.drop_duplicates(subset="ticker")
    universe = universe.sort_values("ticker").reset_index(drop=True)
    return universe
