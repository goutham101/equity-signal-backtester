"""Phase 1: universe -> download -> clean -> load into Postgres -> features view.

Usage:
    .venv/bin/python scripts/01_download.py            # full S&P 500
    .venv/bin/python scripts/01_download.py --limit 10  # first 10 tickers only
"""

import argparse
import logging
import warnings

import sqlalchemy as sa

from backtester.config import load_config
from backtester.data import db
from backtester.data.clean import clean_prices, print_report
from backtester.data.download import download_prices
from backtester.data.universe import load_universe


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="only use the first N tickers")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache, re-download")
    return parser.parse_args()


def main() -> None:
    """Run the full Phase 1 data pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # yfinance triggers many pandas deprecation warnings that are not about our code.
    warnings.filterwarnings("ignore", module="yfinance")
    args = parse_args()
    cfg = load_config()["data"]

    # 1. Universe
    universe = load_universe(refresh=args.refresh)
    if args.limit is not None:
        universe = universe.head(args.limit)
    tickers = universe["ticker"].tolist()
    print(f"Universe: {len(tickers)} tickers")

    # 2. Download (cached in data/raw/prices/)
    prices, volume, failures = download_prices(
        tickers, start=cfg["start_date"], end=cfg["end_date"], refresh=args.refresh
    )
    print(f"Downloaded: {prices.shape[1]} tickers x {prices.shape[0]} dates")
    print(f"Failed downloads: {len(failures)} {failures}")

    # 3. Clean
    result = clean_prices(
        prices,
        volume,
        min_history_days=cfg["min_history_days"],
        max_ffill_days=cfg["max_ffill_days"],
    )
    print()
    print_report(result)

    # 4. Load into Postgres (tickers first, because prices references tickers)
    engine = db.get_engine()
    kept = universe[universe["ticker"].isin(result.prices.columns)]
    n_tickers = db.upsert_tickers(engine, kept)
    long = db.prices_to_long(result.prices, result.volume)
    n_rows = db.upsert_prices(engine, long)
    print(f"\nUpserted {n_tickers} tickers and {n_rows} price rows")

    # 5. Features view
    db.run_sql_file(engine)
    with engine.connect() as conn:
        total_rows = conn.execute(sa.text("SELECT COUNT(*) FROM prices")).scalar_one()
        total_tickers = conn.execute(
            sa.text("SELECT COUNT(DISTINCT ticker) FROM prices")
        ).scalar_one()
        n_features = conn.execute(
            sa.text("SELECT COUNT(*) FROM daily_features WHERE mom_12_1 IS NOT NULL")
        ).scalar_one()
    print(f"prices table now has {total_rows} rows across {total_tickers} tickers")
    print(f"daily_features rows with mom_12_1: {n_features}")


if __name__ == "__main__":
    main()
