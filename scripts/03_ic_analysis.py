"""Phase 3: IC table, rolling IC chart and quintile charts for every signal.

Reads the signals saved by 02_build_signals.py and prices from Postgres.
Forward returns run from close t+exec_lag to close t+exec_lag+h (exec_lag
from config, default 1), aligned to the signal date t.

Usage:
    .venv/bin/python scripts/03_ic_analysis.py           # real run, writes reports/
    .venv/bin/python scripts/03_ic_analysis.py --smoke   # tiny-DB check, writes data/smoke/
"""

import argparse

import pandas as pd

from backtester import pipeline
from backtester.analysis.forward_returns import forward_returns
from backtester.analysis.ic import daily_ic, ic_table, quintile_returns, rolling_ic
from backtester.config import load_config
from backtester.data import db
from backtester.reporting import plots

# Horizon used for the rolling IC and quintile charts (about one month).
CHART_HORIZON = 21
ROLLING_WINDOW = 252


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    pipeline.add_smoke_flag(parser)
    return parser.parse_args()


def ic_rows(table: pd.DataFrame) -> dict[str, dict]:
    """Turn the ic_table frame into {"signal h=21": {stat: value}} rows for markdown."""
    rows = {}
    for record in table.to_dict(orient="records"):
        name = f"{record['signal']} h={record['horizon']}"
        stats = {key: value for key, value in record.items() if key not in ("signal", "horizon")}
        rows[name] = stats
    return rows


def quintile_rows(quintiles: dict[str, pd.Series]) -> dict[str, dict]:
    """Turn {signal: quintile series} into markdown rows, plus the Q5 - Q1 spread."""
    rows = {}
    for name, series in quintiles.items():
        row = series.to_dict()
        row["Q5-Q1"] = series.iloc[-1] - series.iloc[0]
        rows[name] = row
    return rows


def main() -> None:
    """Run the IC analysis and write figures plus the IC section of results.md."""
    args = parse_args()
    cfg = pipeline.apply_smoke(load_config(), args.smoke)
    paths = pipeline.output_paths(args.smoke)
    pipeline.print_mode(args.smoke, cfg)

    min_stocks = cfg["signals"]["min_stocks"]
    exec_lag = cfg["backtest"]["exec_lag"]
    quantiles = cfg["backtest"]["quantiles"]
    horizons = cfg["analysis"]["ic_horizons"]

    engine = db.get_engine()
    prices = pipeline.load_prices(engine, cfg)
    signals = pipeline.load_signals(paths["signals"], cfg)

    # 1. IC table for every signal x horizon.
    table = ic_table(signals, prices, horizons, exec_lag=exec_lag, min_stocks=min_stocks)
    print()
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    n_flags = pipeline.check_ic_red_flags(table)

    # 2. Rolling 12-month IC and quintile returns at the chart horizon.
    fwd = forward_returns(prices, horizon=CHART_HORIZON, exec_lag=exec_lag)
    rolling = {}
    quintiles = {}
    for name, signal in signals.items():
        ic = daily_ic(signal, fwd, min_stocks=min_stocks)
        rolling[name] = rolling_ic(ic, window=ROLLING_WINDOW)
        quintiles[name] = quintile_returns(signal, fwd, quantiles=quantiles, min_stocks=min_stocks)

    figures = paths["figures"]
    rolling_path = plots.plot_rolling_ic(
        rolling,
        figures / "rolling_ic_21d.png",
        title=f"Rolling 12-month mean IC ({CHART_HORIZON}-day forward return)",
    )
    print(f"Saved {rolling_path}")
    for name, series in quintiles.items():
        quintile_path = plots.plot_quintile_returns(
            series,
            figures / f"quintiles_21d_{name}.png",
            title=f"{name}: mean {CHART_HORIZON}-day forward return by quintile",
            horizon=CHART_HORIZON,
        )
        print(f"Saved {quintile_path}")

    print("\nQuintile mean forward returns (21d):")
    print(pd.DataFrame(quintiles).T.to_string(float_format=lambda x: f"{x:.4%}"))

    # 3. Write the IC sections of results.md (other sections are kept).
    note = (
        f"Spearman IC between the ranked signal at close t and the forward return "
        f"from close t+{exec_lag} to close t+{exec_lag}+h. t-stat is Newey-West "
        f"with h lags (corrects for overlapping forward returns). "
        f"Dates with fewer than {min_stocks} stocks are skipped."
    )
    quintile_note = (
        f"Average {CHART_HORIZON}-day forward return per signal quintile "
        "(Q1 = lowest signal), averaged over all dates."
    )
    sections = {
        "Signal IC": note + "\n\n" + pipeline.table_markdown(ic_rows(table)),
        "Quintile forward returns (21d)": (
            quintile_note + "\n\n" + pipeline.table_markdown(quintile_rows(quintiles))
        ),
    }
    results_path = pipeline.update_results_md(paths["results_md"], sections)
    print(f"\nUpdated {results_path}")
    print(f"IC red flags: {n_flags}")


if __name__ == "__main__":
    main()
