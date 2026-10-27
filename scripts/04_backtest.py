"""Phase 4: quintile backtests for every signal, gross and net of costs.

For each signal and each rebalance frequency (monthly, weekly):
- long-short: long the top quintile (+1), short the bottom quintile (-1)
- long-only: top quintile only, reported as excess over an equal-weight
  benchmark of all stocks with a price (same rebalance dates, same costs)
- cost sensitivity of the long-short book over config cost_sensitivity
- exec_lag=0 comparison of the long-short book. exec_lag=0 trades at the
  same close the signal uses, which is OPTIMISTIC and not tradable.

Timing: target weights are decided at close t; the engine holds them from
close t+exec_lag, so they first earn the return of day t+1+exec_lag.

Usage:
    .venv/bin/python scripts/04_backtest.py           # real run, logs to backtest_runs
    .venv/bin/python scripts/04_backtest.py --smoke   # tiny-DB check, data/smoke/, no logging
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtester import pipeline
from backtester.analysis.forward_returns import daily_returns
from backtester.analysis.metrics import annual_return, sharpe_ratio, summary
from backtester.backtest.engine import cost_sensitivity, run_backtest
from backtester.backtest.portfolio import build_target_weights
from backtester.config import load_config
from backtester.data import db
from backtester.reporting import plots

FREQS = ["monthly", "weekly"]

# Warn if gross Sharpe moves by more than this between exec_lag 1 and 0.
# Some change is expected (reversal most of all); a huge one is suspicious.
LAG_SHARPE_CHANGE_WARN = 1.0


def parse_args() -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    pipeline.add_smoke_flag(parser)
    return parser.parse_args()


def benchmark_universe(prices: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    """Prices, blanked out on dates where the signal has no values at all.

    The equal-weight benchmark holds every stock with a price at close t.
    Blanking the warm-up dates keeps the benchmark in cash while the
    long-only book is in cash, so the excess return compares like with like.
    """
    active = signal.notna().any(axis=1)
    universe = prices.copy()
    universe.loc[~active, :] = np.nan
    return universe


def save_frame(results: pd.DataFrame, folder: Path, name: str, kind: str, freq: str) -> Path:
    """Save one daily backtest frame as folder/<signal>__<portfolio>__<freq>.parquet.

    Columns are kept exactly as run_backtest returns them.
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}__{kind}__{freq}.parquet"
    results.to_parquet(path)
    return path


def excess_stats(strategy: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    """Annual excess return and its Sharpe (information ratio) vs the benchmark, net."""
    excess = strategy["net"] - benchmark["net"]
    return {
        "bench_ann_return_net": annual_return(benchmark["net"]),
        "excess_ann_return_net": annual_return(excess),
        "excess_sharpe_net": sharpe_ratio(excess),
    }


def run_record(name: str, kind: str, freq: str, stats: dict, cfg: dict) -> dict:
    """One row for backtest_runs: the signal, the full config and headline net metrics."""
    bt_cfg = cfg["backtest"]
    run_info = {
        "signal": name,
        "portfolio": kind,
        "rebalance": freq,
        "cost_bps": bt_cfg["cost_bps"],
        "exec_lag": bt_cfg["exec_lag"],
        "quantiles": bt_cfg["quantiles"],
    }
    if "excess_ann_return_net" in stats:
        run_info["excess_ann_return_net"] = stats["excess_ann_return_net"]
    return {
        "signal_name": name,
        "config": {"run": run_info, "full_config": cfg},
        "sharpe": stats["sharpe_net"],
        "ann_return": stats["ann_return_net"],
        "max_dd": stats["max_drawdown_net"],
    }


def backtest_signal(name, signal, prices, returns, freq, cfg, folder) -> dict:
    """Run every backtest for one signal and one rebalance frequency.

    Returns a dict with the result frames, metric rows and run records.
    """
    bt_cfg = cfg["backtest"]
    cost_bps = bt_cfg["cost_bps"]
    exec_lag = bt_cfg["exec_lag"]
    quantiles = bt_cfg["quantiles"]
    label = f"{name} {freq}"

    # Long-short
    ls_weights = build_target_weights(signal, freq=freq, kind="long_short", quantiles=quantiles)
    pipeline.check_weight_sums(ls_weights, "long_short", f"{label} long-short")
    ls = run_backtest(ls_weights, returns, cost_bps=cost_bps, exec_lag=exec_lag)
    ls_stats = summary(ls)
    pipeline.check_backtest_red_flags(f"{label} long-short", ls_stats, ls)
    save_frame(ls, folder, name, "long_short", freq)

    # Long-only and its equal-weight benchmark
    lo_weights = build_target_weights(signal, freq=freq, kind="long_only", quantiles=quantiles)
    pipeline.check_weight_sums(lo_weights, "long_only", f"{label} long-only")
    lo = run_backtest(lo_weights, returns, cost_bps=cost_bps, exec_lag=exec_lag)

    universe = benchmark_universe(prices, signal)
    bench_weights = build_target_weights(universe, freq=freq, kind="equal_weight")
    bench = run_backtest(bench_weights, returns, cost_bps=cost_bps, exec_lag=exec_lag)

    lo_stats = summary(lo)
    lo_stats.update(excess_stats(lo, bench))
    pipeline.check_backtest_red_flags(f"{label} long-only", lo_stats, lo)
    save_frame(lo, folder, name, "long_only", freq)
    save_frame(bench, folder, name, "benchmark", freq)

    # Cost sensitivity of the long-short book
    by_cost = cost_sensitivity(ls_weights, returns, bt_cfg["cost_sensitivity"], exec_lag=exec_lag)
    sharpe_by_cost = {}
    for bps, frame in by_cost.items():
        sharpe_by_cost[bps] = sharpe_ratio(frame["net"])

    # exec_lag = 0 comparison (OPTIMISTIC)
    lag0 = run_backtest(ls_weights, returns, cost_bps=cost_bps, exec_lag=0)
    lag0_stats = summary(lag0)
    lag_row = {
        f"sharpe_gross_lag{exec_lag}": ls_stats["sharpe_gross"],
        "sharpe_gross_lag0_OPTIMISTIC": lag0_stats["sharpe_gross"],
        f"sharpe_net_lag{exec_lag}": ls_stats["sharpe_net"],
        "sharpe_net_lag0_OPTIMISTIC": lag0_stats["sharpe_net"],
    }
    change = lag0_stats["sharpe_gross"] - ls_stats["sharpe_gross"]
    if np.isfinite(change) and abs(change) > LAG_SHARPE_CHANGE_WARN:
        pipeline.warn(f"{label} long-short: gross Sharpe changes by {change:+.2f} from "
                      f"exec_lag={exec_lag} to 0. Expected a bit for reversal; a huge "
                      "change means the signal uses same-day info.")

    return {
        "long_short": ls,
        "ls_stats": ls_stats,
        "lo_stats": lo_stats,
        "sharpe_by_cost": sharpe_by_cost,
        "lag_row": lag_row,
        "runs": [
            run_record(name, "long_short", freq, ls_stats, cfg),
            run_record(name, "long_only", freq, lo_stats, cfg),
        ],
    }


def save_figures(freq: str, outputs: dict, figures: Path) -> None:
    """Equity curve per signal, one drawdown chart and one cost chart for this frequency."""
    for name, out in outputs.items():
        plots.plot_equity_curve(
            out["long_short"],
            figures / f"equity_{name}_long_short_{freq}.png",
            title=f"{name}: long-short quintile, {freq} rebalance (growth of $1)",
        )

    net_returns = {name: out["long_short"]["net"] for name, out in outputs.items()}
    plots.plot_drawdown(
        net_returns,
        figures / f"drawdown_long_short_{freq}.png",
        title=f"Drawdown, long-short net of costs, {freq} rebalance",
    )

    sharpe_table = pd.DataFrame({name: out["sharpe_by_cost"] for name, out in outputs.items()})
    plots.plot_cost_sensitivity(
        sharpe_table,
        figures / f"cost_sensitivity_long_short_{freq}.png",
        title=f"Net Sharpe vs transaction cost, long-short, {freq} rebalance",
    )
    print(f"Saved {freq} figures to {figures}")


def print_table(title: str, rows: dict[str, dict]) -> None:
    """Print a metrics table to the terminal."""
    print(f"\n{title}")
    print(pd.DataFrame(rows).T.to_string(float_format=lambda x: f"{x:.4f}"))


def build_sections(all_outputs: dict, cfg: dict) -> dict[str, str]:
    """Markdown sections for results.md: metrics, cost sensitivity, exec_lag comparison."""
    bt_cfg = cfg["backtest"]
    sections = {}
    note = (
        f"Quintile portfolios, cost {bt_cfg['cost_bps']} bps per unit traded, "
        f"exec_lag={bt_cfg['exec_lag']}. Gross and net of costs. Short borrow costs "
        "ignored. Weight drift between rebalances ignored. Risk-free rate ignored."
    )
    lag_rows = {}
    for freq, outputs in all_outputs.items():
        ls_rows = {name: out["ls_stats"] for name, out in outputs.items()}
        lo_rows = {name: out["lo_stats"] for name, out in outputs.items()}
        cost_rows = {}
        for name, out in outputs.items():
            cost_rows[name] = {f"{bps} bps": value for bps, value in out["sharpe_by_cost"].items()}
            lag_rows[f"{name} {freq}"] = out["lag_row"]

        sections[f"Backtest: long-short ({freq})"] = (
            note + "\n\n" + pipeline.table_markdown(ls_rows)
        )
        sections[f"Backtest: long-only vs equal-weight ({freq})"] = (
            "Top quintile only. excess_* = long-only net minus the equal-weight "
            "benchmark net (all stocks with a price, same rebalance dates and costs).\n\n"
            + pipeline.table_markdown(lo_rows)
        )
        sections[f"Cost sensitivity: long-short net Sharpe ({freq})"] = (
            pipeline.table_markdown(cost_rows)
        )

    sections["Execution lag comparison (exec_lag=0 is OPTIMISTIC)"] = (
        "exec_lag=0 trades at the same close the signal uses. That is not "
        "tradable in practice, so these numbers are an optimistic upper bound "
        "for comparison only.\n\n" + pipeline.table_markdown(lag_rows)
    )
    return sections


def main() -> None:
    """Run all backtests, save frames, figures and tables, and log the runs."""
    args = parse_args()
    cfg = pipeline.apply_smoke(load_config(), args.smoke)
    paths = pipeline.output_paths(args.smoke)
    pipeline.print_mode(args.smoke, cfg)

    engine = db.get_engine()
    prices = pipeline.load_prices(engine, cfg)
    signals = pipeline.load_signals(paths["signals"], cfg)

    returns = daily_returns(prices)
    pipeline.check_extreme_returns(returns)

    all_outputs = {}
    runs = []
    for freq in FREQS:
        outputs = {}
        for name, signal in signals.items():
            out = backtest_signal(name, signal, prices, returns, freq, cfg, paths["backtests"])
            outputs[name] = out
            runs.extend(out["runs"])
        all_outputs[freq] = outputs

        print_table(f"Long-short ({freq})", {n: o["ls_stats"] for n, o in outputs.items()})
        print_table(f"Long-only ({freq})", {n: o["lo_stats"] for n, o in outputs.items()})
        print_table(f"Cost sensitivity, net Sharpe ({freq})",
                    {n: o["sharpe_by_cost"] for n, o in outputs.items()})
        save_figures(freq, outputs, paths["figures"])

    lag_rows = {}
    for freq, outputs in all_outputs.items():
        for name, out in outputs.items():
            lag_rows[f"{name} {freq}"] = out["lag_row"]
    print_table("Execution lag comparison (exec_lag=0 is OPTIMISTIC)", lag_rows)

    print(f"\nSaved daily backtest frames to {paths['backtests']}")
    results_path = pipeline.update_results_md(paths["results_md"], build_sections(all_outputs, cfg))
    print(f"Updated {results_path}")

    pipeline.log_runs(engine, runs, args.smoke)


if __name__ == "__main__":
    main()
