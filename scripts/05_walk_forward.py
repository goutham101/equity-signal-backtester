"""Phase 5: walk-forward combination of the three signals, out-of-sample only.

For each window: fit signal weights on 5 years of training data, skip a purge
gap, then apply those weights to the next test year. Concatenating the test
years gives one out-of-sample (OOS) signal.

To compare fairly, every other signal (single signals, equal-weight combo,
random) is cut to the same test dates. So every row in the table covers
exactly the same OOS period.

Timing: weights for a test year use only training labels that end before the
test year starts (see walk_forward.make_windows). Target weights are decided
at close t and first earn the return of day t+1+exec_lag.

Usage:
    .venv/bin/python scripts/05_walk_forward.py           # real run, logs to backtest_runs
    .venv/bin/python scripts/05_walk_forward.py --smoke   # tiny-DB check, data/smoke/, no logging
"""

import argparse

import pandas as pd

from backtester import pipeline
from backtester.analysis.forward_returns import daily_returns
from backtester.analysis.metrics import summary
from backtester.backtest.engine import run_backtest
from backtester.backtest.portfolio import build_target_weights
from backtester.backtest.walk_forward import walk_forward_signal
from backtester.config import load_config
from backtester.data import db
from backtester.reporting import plots

# Fit on 21-day forward returns, the same horizon as the monthly rebalance.
FIT_HORIZON = 21
FREQ = "monthly"
KIND = "long_short"


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    pipeline.add_smoke_flag(parser)
    return parser.parse_args()


def oos_dates(windows: list[dict], index: pd.DatetimeIndex) -> pd.Series:
    """True on dates inside any test window, False everywhere else."""
    in_test = pd.Series(False, index=index)
    for w in windows:
        in_test.loc[w["test_start"]:w["test_end"]] = True
    return in_test


def cut_to_oos(signal: pd.DataFrame, in_test: pd.Series) -> pd.DataFrame:
    """Blank a signal outside the test windows so it covers the OOS period only."""
    cut = signal.copy()
    cut.loc[~in_test.reindex(cut.index, fill_value=False), :] = float("nan")
    return cut


def windows_table(windows: list[dict]) -> pd.DataFrame:
    """Windows as a DataFrame, one row per test year."""
    table = pd.DataFrame(windows).set_index("label")
    return table[["train_start", "train_end", "test_start", "test_end"]]


def main() -> None:
    """Fit walk-forward weights, backtest OOS, compare, save, log."""
    args = parse_args()
    cfg = pipeline.apply_smoke(load_config(), args.smoke)
    paths = pipeline.output_paths(args.smoke)
    pipeline.print_mode(args.smoke, cfg)

    wf = cfg["walk_forward"]
    bt = cfg["backtest"]
    min_stocks = cfg["signals"]["min_stocks"]

    engine = db.get_engine()
    prices = pipeline.load_prices(engine, cfg)
    signals = pipeline.load_signals(paths["signals"], cfg)
    returns = daily_returns(prices)

    ranked = {name: signals[name] for name in pipeline.COMBO_INPUTS}
    combined, weights, windows = walk_forward_signal(
        ranked,
        prices,
        horizon=FIT_HORIZON,
        exec_lag=bt["exec_lag"],
        train_years=wf["train_years"],
        test_years=wf["test_years"],
        purge_days=wf["purge_days"],
        min_stocks=min_stocks,
        method=wf["method"],
    )
    if not windows:
        raise SystemExit("No walk-forward windows: not enough history for one train period.")

    print(f"\n{len(windows)} windows, method={wf['method']}, horizon={FIT_HORIZON}d")
    print(windows_table(windows).to_string())
    print("\nFitted weights per window:")
    print(weights.round(3).to_string())

    in_test = oos_dates(windows, prices.index)
    candidates = {"walk_forward": combined}
    for name in pipeline.SIGNAL_NAMES:
        candidates[name] = cut_to_oos(signals[name], in_test)

    rows = {}
    frames = {}
    for name, signal in candidates.items():
        target = build_target_weights(signal, FREQ, KIND, bt["quantiles"])
        pipeline.check_weight_sums(target, KIND, f"OOS {name}")
        results = run_backtest(target, returns, cost_bps=bt["cost_bps"], exec_lag=bt["exec_lag"])
        stats = summary(results)
        pipeline.check_backtest_red_flags(f"OOS {name}", stats, results)
        rows[name] = stats
        frames[name] = results

    # Save for the dashboard export (Phase 7).
    paths["backtests"].mkdir(parents=True, exist_ok=True)
    frames["walk_forward"].to_parquet(paths["backtests"] / f"walk_forward__{KIND}__{FREQ}.parquet")
    weights.to_parquet(paths["backtests"] / "walk_forward_weights.parquet")
    windows_table(windows).to_parquet(paths["backtests"] / "walk_forward_windows.parquet")

    plots.plot_equity_curve(frames["walk_forward"], paths["figures"] / "equity_walk_forward.png")

    first_test = windows[0]["test_start"].date()
    last_test = windows[-1]["test_end"].date()
    note = (
        f"Out-of-sample only: {first_test} to {last_test}, {len(windows)} test years. "
        f"Every row covers the same dates. Long-short quintiles, {FREQ} rebalance, "
        f"{bt['cost_bps']} bps costs, exec_lag={bt['exec_lag']}. "
        f"Weights fit with method `{wf['method']}` on {FIT_HORIZON}-day forward returns."
    )
    weight_rows = {label: row.to_dict() for label, row in weights.iterrows()}
    sections = {
        "Walk-forward (out-of-sample)": note + "\n\n" + pipeline.table_markdown(rows),
        "Walk-forward fitted weights": (
            "Weight per signal for each test year, fit on that window's training data only.\n\n"
            + pipeline.table_markdown(weight_rows)
        ),
    }
    results_path = pipeline.update_results_md(paths["results_md"], sections)
    print(f"\nWrote {results_path}")

    print("\nOOS net Sharpe:")
    for name, stats in rows.items():
        print(f"  {name:13s} {stats['sharpe_net']:+.2f}")

    wf_stats = rows["walk_forward"]
    run = {
        "signal_name": "walk_forward",
        "config": {
            "run": {"signal": "walk_forward", "portfolio": KIND, "rebalance": FREQ,
                    "cost_bps": bt["cost_bps"], "exec_lag": bt["exec_lag"],
                    "method": wf["method"], "fit_horizon": FIT_HORIZON},
            "full_config": cfg,
        },
        "sharpe": wf_stats["sharpe_net"],
        "ann_return": wf_stats["ann_return_net"],
        "max_dd": wf_stats["max_drawdown_net"],
    }
    pipeline.log_runs(engine, [run], args.smoke)


if __name__ == "__main__":
    main()
