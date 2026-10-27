"""Phase 7: export everything the Streamlit dashboard needs to dashboard/data/.

The app only reads these files (see docs/dashboard_data_contract.md). It
never touches Postgres or data/processed/.

Inputs:
- prices from Postgres, ranked signals from data/processed/signals/
- full-period engine frames from data/processed/backtests/ (written by script 04)
- the out-of-sample frames are recomputed with pipeline.run_oos_comparison,
  the same function script 05 uses. It is deterministic.

This script never logs to backtest_runs and never re-runs scripts 04/05:
that would add duplicate runs and inflate the variations count it exports.

Float columns are stored as float64 (the files are only a few MB, so no
need to shrink them).

Usage:
    .venv/bin/python scripts/06_export_dashboard_data.py
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa

from backtester import pipeline
from backtester.analysis.forward_returns import forward_returns
from backtester.analysis.ic import daily_ic, ic_summary, quintile_returns
from backtester.backtest.walk_forward import windows_table
from backtester.config import ROOT, load_config
from backtester.data import db
from backtester.reporting import export

OUT_DIR = ROOT / "dashboard" / "data"
FULL_FREQS = ["monthly", "weekly"]
FULL_KINDS = ["long_short", "long_only"]
# Each signal has its own benchmark (in cash during that signal's warm-up).
# The momentum one is the canonical benchmark for the dashboard.
BENCHMARK_SOURCE = "momentum"

UNIVERSE_LABELS = {"sp500_current": "Current S&P 500 constituents"}
LINKS = {
    "github": "https://github.com/goutham101/equity-signal-backtester",
    "readme": "https://github.com/goutham101/equity-signal-backtester#readme",
    "linkedin": None,
}


def now_iso() -> str:
    """Current UTC time as an ISO string, to the second."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def count_backtest_runs(engine: sa.Engine) -> int:
    """Number of rows in backtest_runs (= variations tried)."""
    with engine.connect() as conn:
        return int(conn.execute(sa.text("SELECT COUNT(*) FROM backtest_runs")).scalar_one())


def load_frame(folder: Path, signal: str, kind: str, freq: str) -> pd.DataFrame:
    """Read one saved engine frame, e.g. momentum__long_short__monthly.parquet."""
    path = folder / f"{signal}__{kind}__{freq}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run scripts/04_backtest.py first.")
    return pd.read_parquet(path)


def full_period_returns(folder: Path) -> pd.DataFrame:
    """Long rows for every saved full-period frame, plus the benchmark."""
    parts = []
    for freq in FULL_FREQS:
        for name in pipeline.SIGNAL_NAMES:
            for kind in FULL_KINDS:
                frame = load_frame(folder, name, kind, freq)
                parts.append(export.frame_to_long(frame, name, kind, freq, "full"))

        bench = load_frame(folder, BENCHMARK_SOURCE, "benchmark", freq)
        parts.append(export.frame_to_long(bench, "benchmark", "long_only", freq, "full"))
    return pd.concat(parts, ignore_index=True)


def oos_returns(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Long rows for the out-of-sample frames (long-short, monthly only)."""
    parts = []
    for name, frame in frames.items():
        parts.append(
            export.frame_to_long(frame, name, pipeline.OOS_KIND, pipeline.OOS_FREQ, "oos")
        )
    return pd.concat(parts, ignore_index=True)


def check_saved_walk_forward(folder: Path, recomputed: pd.DataFrame) -> None:
    """Warn if the recomputed walk-forward frame differs from the one script 05 saved."""
    path = folder / f"walk_forward__{pipeline.OOS_KIND}__{pipeline.OOS_FREQ}.parquet"
    if not path.exists():
        pipeline.warn(f"{path} missing, cannot cross-check the recomputed walk-forward frame.")
        return
    saved = pd.read_parquet(path)
    same_shape = saved.shape == recomputed.shape
    if same_shape and np.allclose(saved.to_numpy(), recomputed.to_numpy(), equal_nan=True):
        print("Recomputed walk-forward frame matches the one saved by script 05.")
    else:
        pipeline.warn("Recomputed walk-forward frame differs from the saved one. "
                      "Signals or prices changed since script 05 ran?")


def ic_outputs(signals, prices, cfg) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Daily IC, IC summary and quintile returns for every signal and horizon."""
    min_stocks = cfg["signals"]["min_stocks"]
    exec_lag = cfg["backtest"]["exec_lag"]
    quantiles = cfg["backtest"]["quantiles"]

    ic_parts = []
    summary_rows = []
    quintile_parts = []
    for horizon in cfg["analysis"]["ic_horizons"]:
        fwd = forward_returns(prices, horizon=horizon, exec_lag=exec_lag)
        for name in pipeline.SIGNAL_NAMES:
            signal = signals[name]
            ic = daily_ic(signal, fwd, min_stocks=min_stocks)
            ic_parts.append(export.ic_to_long(ic, name, horizon))

            row = {"signal": name, "horizon": horizon}
            row.update(ic_summary(ic, horizon=horizon))
            summary_rows.append(row)

            quintiles = quintile_returns(signal, fwd, quantiles=quantiles, min_stocks=min_stocks)
            quintile_parts.append(export.quintiles_to_long(quintiles, name, horizon))
        print(f"IC done for h={horizon}")

    ic_daily = pd.concat(ic_parts, ignore_index=True)
    ic_table = pd.DataFrame(summary_rows)
    quintile_table = pd.concat(quintile_parts, ignore_index=True)
    return ic_daily, ic_table, quintile_table


def summary_table(stats: dict[str, dict], ic_table: pd.DataFrame) -> pd.DataFrame:
    """Overview headline table: OOS, long-short, monthly, config cost. One row per signal."""
    ic21 = ic_table[ic_table["horizon"] == 21].set_index("signal")["mean_ic"]
    rows = []
    for name, signal_stats in stats.items():
        mean_ic = ic21.get(name, np.nan)
        rows.append(export.summary_row(name, signal_stats, mean_ic))
    return pd.DataFrame(rows)


def run_lookahead_test() -> dict:
    """Run tests/test_no_lookahead.py now and report passing/failing and the pass count."""
    pytest_bin = Path(sys.executable).parent / "pytest"
    result = subprocess.run(
        [str(pytest_bin), "tests/test_no_lookahead.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    status = "passing" if result.returncode == 0 else "failing"
    if status == "failing":
        print(output)
        pipeline.warn("tests/test_no_lookahead.py is FAILING. The dashboard will say so.")
    return {"status": status, "checked_at": now_iso(), "tests": export.parse_pytest_passed(output)}


def build_metadata(prices, windows, cfg, variations: int, lookahead: dict) -> dict:
    """metadata.json contents (see the contract)."""
    readme = (ROOT / "README.md").read_text()
    bt = cfg["backtest"]
    universe = cfg["data"]["universe"]
    return {
        "universe": UNIVERSE_LABELS.get(universe, universe),
        "n_stocks": int(prices.shape[1]),
        "start_date": str(prices.index.min().date()),
        "end_date": str(prices.index.max().date()),
        "oos_start": str(windows[0]["test_start"].date()),
        "oos_end": str(windows[-1]["test_end"].date()),
        "last_updated": now_iso(),
        "variations_tried": variations,
        "default_cost_bps": bt["cost_bps"],
        "exec_lag": bt["exec_lag"],
        "quantiles": bt["quantiles"],
        "walk_forward_method": cfg["walk_forward"]["method"],
        "lookahead_test": lookahead,
        "key_findings": export.readme_bullets(readme, "Key findings"),
        "limitations": export.readme_bullets(readme, "Limitations"),
        "links": LINKS,
    }


def save(frame: pd.DataFrame, name: str) -> None:
    """Write one parquet file to dashboard/data/ and print its size."""
    path = OUT_DIR / name
    frame.to_parquet(path, index=False)
    print(f"  {name:32s} {len(frame):>8,d} rows  {path.stat().st_size / 1024:8.1f} KB")


def main() -> None:
    """Build every dashboard file and write it to dashboard/data/."""
    cfg = load_config()
    paths = pipeline.output_paths(smoke=False)
    pipeline.print_mode(False, cfg)

    engine = db.get_engine()
    runs_before = count_backtest_runs(engine)
    prices = pipeline.load_prices(engine, cfg)
    signals = pipeline.load_signals(paths["signals"], cfg)

    print("\nRecomputing the out-of-sample comparison (no logging)...")
    oos = pipeline.run_oos_comparison(signals, prices, cfg)
    check_saved_walk_forward(paths["backtests"], oos["frames"]["walk_forward"])

    returns = pd.concat(
        [full_period_returns(paths["backtests"]), oos_returns(oos["frames"])],
        ignore_index=True,
    )

    print("\nComputing IC tables...")
    ic_daily, ic_table, quintile_table = ic_outputs(signals, prices, cfg)
    summary = summary_table(oos["stats"], ic_table)

    print("\nRunning the look-ahead test...")
    lookahead = run_lookahead_test()

    runs_after = count_backtest_runs(engine)
    if runs_after != runs_before:
        pipeline.warn(f"backtest_runs changed during export ({runs_before} -> {runs_after}).")
    metadata = build_metadata(prices, oos["windows"], cfg, runs_after, lookahead)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting to {OUT_DIR}:")
    save(returns, "daily_returns.parquet")
    save(ic_daily, "ic_daily.parquet")
    save(ic_table, "ic_summary.parquet")
    save(quintile_table, "quintile_returns.parquet")
    save(export.weights_to_long(oos["weights"]), "walk_forward_weights.parquet")
    save(export.windows_to_frame(windows_table(oos["windows"])), "walk_forward_windows.parquet")
    save(summary, "summary.parquet")

    metadata_path = OUT_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"  metadata.json  variations_tried={runs_after}, "
          f"lookahead={lookahead['status']} ({lookahead['tests']} tests)")

    total = sum(p.stat().st_size for p in OUT_DIR.iterdir() if p.is_file())
    print(f"\nTotal size of {OUT_DIR.name}/: {total / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
