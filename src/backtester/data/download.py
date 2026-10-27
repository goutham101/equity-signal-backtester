"""Download daily adjusted prices and volume from yfinance.

Each ticker is cached as its own CSV in data/raw/prices/, so a rerun only
downloads tickers that are not cached yet. Pass refresh=True to re-download.
"""

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from backtester.config import ROOT

logger = logging.getLogger(__name__)

PRICE_CACHE_DIR = ROOT / "data" / "raw" / "prices"
FAILURES_PATH = ROOT / "data" / "raw" / "download_failures.csv"


def cache_path(ticker: str, cache_dir: Path = PRICE_CACHE_DIR) -> Path:
    """Path of the cached CSV for one ticker."""
    return cache_dir / f"{ticker}.csv"


def read_cached(ticker: str, cache_dir: Path = PRICE_CACHE_DIR) -> pd.DataFrame:
    """Read one ticker's cached CSV (index=date, columns adj_close, volume)."""
    df = pd.read_csv(cache_path(ticker, cache_dir), index_col="date", parse_dates=["date"])
    return df.sort_index()


def write_cached(ticker: str, df: pd.DataFrame, cache_dir: Path = PRICE_CACHE_DIR) -> None:
    """Save one ticker's adj_close and volume to its cache CSV."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path(ticker, cache_dir), index_label="date")


def download_batch(
    tickers: list[str], start: str, end: str | None, retries: int = 3, wait: float = 5.0
) -> pd.DataFrame | None:
    """Call yfinance for one batch of tickers, retrying on errors.

    Uses auto_adjust=True, so "Close" is the split- and dividend-adjusted close.
    Returns the raw yfinance frame (columns are a (field, ticker) MultiIndex),
    or None if every attempt failed.
    """
    for attempt in range(1, retries + 1):
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                auto_adjust=True,
                group_by="column",
                progress=False,
                threads=True,
            )
            if raw is not None and not raw.empty:
                return raw
            logger.warning("Empty result for batch (attempt %d/%d)", attempt, retries)
        except Exception as exc:  # yfinance raises many different error types
            logger.warning("Batch failed (attempt %d/%d): %s", attempt, retries, exc)
        time.sleep(wait * attempt)
    return None


def split_batch(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Split a yfinance batch into one (adj_close, volume) frame per ticker.

    Tickers with no valid close at all are left out (they count as failures).
    """
    per_ticker = {}
    for ticker in tickers:
        if ticker not in raw["Close"].columns:
            continue
        df = pd.DataFrame(
            {
                "adj_close": raw["Close"][ticker],
                "volume": raw["Volume"][ticker],
            }
        )
        df = df.dropna(subset=["adj_close"])
        if df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index.name = "date"
        per_ticker[ticker] = df
    return per_ticker


def download_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    batch_size: int = 50,
    sleep_seconds: float = 2.0,
    refresh: bool = False,
    cache_dir: Path = PRICE_CACHE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Download (or load from cache) daily adjusted close and volume.

    Prices are adjusted closes (auto_adjust=True). Returns wide frames
    (index=date, columns=ticker) for adj close and volume, plus the list of
    tickers that failed to download.
    """
    to_download = []
    for ticker in tickers:
        if refresh or not cache_path(ticker, cache_dir).exists():
            to_download.append(ticker)
    print(f"{len(tickers) - len(to_download)} tickers cached, {len(to_download)} to download")

    failures = []
    for i in range(0, len(to_download), batch_size):
        batch = to_download[i : i + batch_size]
        print(f"Downloading batch {i // batch_size + 1}: {len(batch)} tickers")
        raw = download_batch(batch, start, end)
        if raw is None:
            failures.extend(batch)
            continue

        per_ticker = split_batch(raw, batch)
        for ticker, df in per_ticker.items():
            write_cached(ticker, df, cache_dir)
        for ticker in batch:
            if ticker not in per_ticker:
                failures.append(ticker)

        # Be polite to the API between batches.
        if i + batch_size < len(to_download):
            time.sleep(sleep_seconds)

    if failures:
        logger.warning("Failed tickers: %s", failures)
        save_failures(failures)

    prices, volume = load_cached_wide(tickers, start, end, cache_dir)
    return prices, volume, failures


def save_failures(failures: list[str], path: Path = FAILURES_PATH) -> None:
    """Write the failed tickers to a CSV so they can be inspected later."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": failures}).to_csv(path, index=False)


def load_cached_wide(
    tickers: list[str], start: str, end: str | None, cache_dir: Path = PRICE_CACHE_DIR
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine cached per-ticker CSVs into wide adj close and volume frames.

    Only dates between start and end (inclusive) are kept. Index is sorted.
    """
    closes = {}
    volumes = {}
    for ticker in tickers:
        if not cache_path(ticker, cache_dir).exists():
            continue
        df = read_cached(ticker, cache_dir)
        closes[ticker] = df["adj_close"]
        volumes[ticker] = df["volume"]

    prices = pd.DataFrame(closes).sort_index()
    volume = pd.DataFrame(volumes).sort_index()

    prices = prices.loc[start:end]
    volume = volume.loc[start:end]
    # Use one datetime unit everywhere so frames from different sources line up.
    prices.index = pd.DatetimeIndex(prices.index).as_unit("ns")
    volume.index = pd.DatetimeIndex(volume.index).as_unit("ns")
    prices.index.name = "date"
    volume.index.name = "date"
    return prices.astype(float), volume.astype(float)
