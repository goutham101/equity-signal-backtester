"""Phase 2: load prices from Postgres -> build ranked signals -> save as Parquet.

Usage:
    .venv/bin/python scripts/02_build_signals.py           # real run
    .venv/bin/python scripts/02_build_signals.py --smoke   # tiny-DB check, writes data/smoke/
"""

import argparse

from backtester import pipeline
from backtester.config import load_config
from backtester.data import db


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    pipeline.add_smoke_flag(parser)
    return parser.parse_args()


def print_coverage(signals: dict) -> None:
    """Print first valid date, number of dates and average stocks per date for each signal."""
    print()
    print(f"{'signal':<14} {'first valid date':<18} {'dates':>6} {'avg stocks/date':>16}")
    for name, frame in signals.items():
        cov = pipeline.signal_coverage(frame)
        print(f"{name:<14} {str(cov['first_valid_date']):<18} "
              f"{cov['n_dates']:>6} {cov['avg_stocks']:>16.1f}")
        if cov["n_dates"] == 0:
            pipeline.warn(f"{name} has no valid dates at all.")


def main() -> None:
    """Build every ranked signal and save it to disk."""
    args = parse_args()
    cfg = pipeline.apply_smoke(load_config(), args.smoke)
    paths = pipeline.output_paths(args.smoke)
    pipeline.print_mode(args.smoke, cfg)

    engine = db.get_engine()
    prices = pipeline.load_prices(engine, cfg)

    signals = pipeline.build_signals(prices, cfg)
    print_coverage(signals)

    pipeline.save_signals(signals, paths["signals"], cfg)
    print(f"\nSaved {len(signals)} signals to {paths['signals']}")


if __name__ == "__main__":
    main()
