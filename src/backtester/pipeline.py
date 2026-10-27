"""Shared helpers for the run scripts (scripts/02, 03, 04).

Every script builds prices, signals and output folders the same way, with
all parameters taken from config.yaml. Keeping this in one place means the
IC analysis and the backtest always see exactly the same signals.

Smoke mode (--smoke) is for checking that the scripts run on a tiny
database. It lowers min_stocks, writes everything under data/smoke/
instead of reports/ and data/processed/, and never logs to backtest_runs.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa

from backtester.analysis.forward_returns import daily_returns
from backtester.analysis.metrics import results_to_markdown, summary, write_results_md
from backtester.backtest.engine import run_backtest
from backtester.backtest.portfolio import build_target_weights
from backtester.backtest.walk_forward import cut_to_oos, oos_dates, walk_forward_signal
from backtester.config import ROOT
from backtester.data import db
from backtester.signals import (
    combine_equal_weight,
    cs_rank,
    low_volatility,
    mask_thin_dates,
    momentum_12_1,
    random_signal,
    short_term_reversal,
)

# The three real signals that go into the equal-weight combination.
COMBO_INPUTS = ["momentum", "reversal", "low_vol"]
SIGNAL_NAMES = ["momentum", "reversal", "low_vol", "random", "equal_weight"]

# min_stocks used in smoke mode, so the scripts run on a 10-ticker database.
SMOKE_MIN_STOCKS = 4

# Spec section 13 red-flag thresholds.
MAX_PLAUSIBLE_IC = 0.10
MAX_PLAUSIBLE_SHARPE = 2.0
WEIGHT_SUM_TOL = 1e-9

SIGNALS_CONFIG_FILE = "signals_config.json"

# Walk-forward out-of-sample comparison (scripts 05 and 06).
# Fit on 21-day forward returns, the same horizon as the monthly rebalance.
OOS_FIT_HORIZON = 21
OOS_FREQ = "monthly"
OOS_KIND = "long_short"


# ---------- command line, config and folders ----------


def add_smoke_flag(parser: argparse.ArgumentParser) -> None:
    """Add the --smoke option shared by every run script."""
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"quick check on a tiny DB: min_stocks={SMOKE_MIN_STOCKS}, outputs go to "
            "data/smoke/, nothing is logged to backtest_runs"
        ),
    )


def apply_smoke(cfg: dict, smoke: bool) -> dict:
    """Return a copy of the config, with min_stocks lowered if smoke is True."""
    cfg = copy.deepcopy(cfg)
    if smoke:
        cfg["signals"]["min_stocks"] = SMOKE_MIN_STOCKS
    return cfg


def output_paths(smoke: bool) -> dict[str, Path]:
    """Folders and files the scripts write to.

    Real run: signals and backtest frames in data/processed/, figures and
    results.md in reports/. Smoke run: everything under data/smoke/.
    """
    if smoke:
        base = ROOT / "data" / "smoke"
        figures = base / "figures"
        results_md = base / "results.md"
        processed = base / "processed"
    else:
        figures = ROOT / "reports" / "figures"
        results_md = ROOT / "reports" / "results.md"
        processed = ROOT / "data" / "processed"

    return {
        "signals": processed / "signals",
        "backtests": processed / "backtests",
        "figures": figures,
        "results_md": results_md,
    }


def print_mode(smoke: bool, cfg: dict) -> None:
    """Print one line saying whether this is a smoke run or a real run."""
    min_stocks = cfg["signals"]["min_stocks"]
    if smoke:
        print(f"SMOKE RUN (min_stocks={min_stocks}). Outputs go to data/smoke/. "
              "Nothing is logged to backtest_runs. These numbers are NOT results.")
    else:
        print(f"Real run (min_stocks={min_stocks}).")


# ---------- prices ----------


def load_prices(engine: sa.Engine, cfg: dict) -> pd.DataFrame:
    """Read adjusted closes from Postgres as a wide frame, using the config dates."""
    data_cfg = cfg["data"]
    prices = db.read_prices(engine, start=data_cfg["start_date"], end=data_cfg["end_date"])
    print(f"Loaded prices: {prices.shape[1]} tickers x {prices.shape[0]} dates "
          f"({prices.index.min().date()} to {prices.index.max().date()})")
    return prices


# ---------- signals ----------


def rank_and_mask(raw: pd.DataFrame, min_stocks: int) -> pd.DataFrame:
    """Cross-sectional rank on each date, then blank out dates with too few stocks.

    Both steps use only the row at close t, so no future data comes in.
    """
    ranked = cs_rank(raw)
    return mask_thin_dates(ranked, min_stocks)


def build_signals(prices: pd.DataFrame, cfg: dict) -> dict[str, pd.DataFrame]:
    """Build every ranked signal from prices, all parameters from the config.

    Each signal at date t uses closes up to and including close t. Values are
    percentile ranks in (0, 1] or NaN. Dates with fewer than min_stocks valid
    stocks are all NaN.

    Keys: momentum, reversal, low_vol, random, equal_weight.
    equal_weight is the average of the three ranked real signals (random is
    left out), ranked again so it is on the same (0, 1] scale.
    """
    sig_cfg = cfg["signals"]
    min_stocks = sig_cfg["min_stocks"]

    raw = {
        "momentum": momentum_12_1(
            prices,
            lookback=sig_cfg["momentum"]["lookback"],
            skip=sig_cfg["momentum"]["skip"],
        ),
        "reversal": short_term_reversal(prices, lookback=sig_cfg["reversal"]["lookback"]),
        "low_vol": low_volatility(prices, window=sig_cfg["low_vol"]["window"]),
        "random": random_signal(prices, seed=cfg["seed"]),
    }

    signals = {}
    for name, frame in raw.items():
        signals[name] = rank_and_mask(frame, min_stocks)

    combo_parts = {name: signals[name] for name in COMBO_INPUTS}
    combo = combine_equal_weight(combo_parts)
    signals["equal_weight"] = rank_and_mask(combo, min_stocks)
    return signals


def signal_coverage(signal: pd.DataFrame) -> dict:
    """First date with any value, number of such dates, and average stocks per date."""
    counts = signal.notna().sum(axis=1)
    active = counts[counts > 0]
    if len(active) == 0:
        return {"first_valid_date": None, "n_dates": 0, "avg_stocks": 0.0}
    return {
        "first_valid_date": active.index[0].date(),
        "n_dates": len(active),
        "avg_stocks": float(active.mean()),
    }


def _signals_config(cfg: dict) -> dict:
    """The part of the config that decides what the signals look like."""
    return {
        "signals": cfg["signals"],
        "seed": cfg["seed"],
        "data": cfg["data"],
    }


def save_signals(signals: dict[str, pd.DataFrame], folder: Path, cfg: dict) -> None:
    """Write each signal to folder/<name>.parquet, plus the config used to build them."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in signals.items():
        frame.to_parquet(folder / f"{name}.parquet")

    config_used = json.dumps(_signals_config(cfg), indent=2, default=str)
    (folder / SIGNALS_CONFIG_FILE).write_text(config_used)


def load_signals(folder: Path, cfg: dict) -> dict[str, pd.DataFrame]:
    """Read the signals saved by save_signals.

    Raises an error if the files are missing or were built with a different
    signal config than the current one, so stale signals never get used.
    """
    folder = Path(folder)
    config_file = folder / SIGNALS_CONFIG_FILE
    if not config_file.exists():
        raise FileNotFoundError(
            f"No saved signals in {folder}. Run scripts/02_build_signals.py first "
            "(with the same --smoke setting)."
        )

    saved_config = json.loads(config_file.read_text())
    current_config = json.loads(json.dumps(_signals_config(cfg), default=str))
    if saved_config != current_config:
        raise ValueError(
            f"Signals in {folder} were built with a different config. "
            "Re-run scripts/02_build_signals.py."
        )

    signals = {}
    for name in SIGNAL_NAMES:
        signals[name] = pd.read_parquet(folder / f"{name}.parquet")
    return signals


# ---------- walk-forward out-of-sample comparison ----------


def run_oos_comparison(
    signals: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    cfg: dict,
) -> dict:
    """Walk-forward combo vs every other signal, on the same out-of-sample dates.

    1. Fit walk-forward weights on the COMBO_INPUTS (walk_forward_signal),
       giving one combined signal that is NaN outside the test years.
    2. Cut every signal in SIGNAL_NAMES to the same test dates.
    3. Backtest each one long-short, monthly, with the config costs and exec_lag.

    Deterministic, and never logs to backtest_runs (the caller decides that).
    Returns a dict with:
    - frames: {name: run_backtest output}, walk_forward first
    - stats: {name: metrics.summary of that frame}
    - weights: fitted weights, one row per test year
    - windows: the window dicts from make_windows
    """
    wf = cfg["walk_forward"]
    bt = cfg["backtest"]

    ranked = {name: signals[name] for name in COMBO_INPUTS}
    combined, weights, windows = walk_forward_signal(
        ranked,
        prices,
        horizon=OOS_FIT_HORIZON,
        exec_lag=bt["exec_lag"],
        train_years=wf["train_years"],
        test_years=wf["test_years"],
        purge_days=wf["purge_days"],
        min_stocks=cfg["signals"]["min_stocks"],
        method=wf["method"],
    )
    if not windows:
        raise ValueError("No walk-forward windows: not enough history for one train period.")

    in_test = oos_dates(windows, prices.index)
    candidates = {"walk_forward": combined}
    for name in SIGNAL_NAMES:
        candidates[name] = cut_to_oos(signals[name], in_test)

    returns = daily_returns(prices)
    frames = {}
    stats = {}
    for name, signal in candidates.items():
        target = build_target_weights(signal, OOS_FREQ, OOS_KIND, bt["quantiles"])
        check_weight_sums(target, OOS_KIND, f"OOS {name}")
        results = run_backtest(target, returns, cost_bps=bt["cost_bps"], exec_lag=bt["exec_lag"])
        stats[name] = summary(results)
        check_backtest_red_flags(f"OOS {name}", stats[name], results)
        frames[name] = results

    return {"frames": frames, "stats": stats, "weights": weights, "windows": windows}


# ---------- results.md ----------


def read_results_sections(path: Path) -> dict[str, str]:
    """Parse an existing results.md into {heading: markdown text}, in file order."""
    path = Path(path)
    if not path.exists():
        return {}

    sections = {}
    heading = None
    lines = []
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            if heading is not None:
                sections[heading] = "\n".join(lines).strip()
            heading = line[3:].strip()
            lines = []
        elif heading is not None:
            lines.append(line)
    if heading is not None:
        sections[heading] = "\n".join(lines).strip()
    return sections


def update_results_md(path: Path, new_sections: dict[str, str]) -> Path:
    """Replace (or add) some sections of results.md and keep all the others.

    Script 03 writes the IC sections and script 04 the backtest sections,
    so neither script wipes out the other's tables.
    """
    sections = read_results_sections(path)
    for heading, text in new_sections.items():
        sections[heading] = text
    return write_results_md(sections, path)


def table_markdown(rows: dict[str, dict]) -> str:
    """Render {row_name: {column: value}} as a markdown table."""
    return results_to_markdown(rows)


# ---------- red flags (Skill.md section 13) ----------


def warn(message: str) -> None:
    """Print a loud red-flag warning line."""
    print(f"!!! WARNING: {message}")


def check_ic_red_flags(ic_df: pd.DataFrame) -> int:
    """Warn for every (signal, horizon) with |mean IC| above 0.10. Returns the count."""
    count = 0
    for row in ic_df.itertuples(index=False):
        if np.isfinite(row.mean_ic) and abs(row.mean_ic) > MAX_PLAUSIBLE_IC:
            warn(f"{row.signal} h={row.horizon}: mean IC {row.mean_ic:.4f} has "
                 f"|IC| > {MAX_PLAUSIBLE_IC}. Likely look-ahead bias, check shifts.")
            count += 1
    return count


def check_weight_sums(weights: pd.DataFrame, kind: str, label: str) -> int:
    """Warn if the long side does not sum to 1 (or the short side to -1).

    Checks every date with at least one position. All-zero rows mean
    "in cash" (warm-up or no safe split), which is allowed.
    Returns the number of bad dates.
    """
    long_sum = weights.clip(lower=0).sum(axis=1)
    short_sum = weights.clip(upper=0).sum(axis=1)
    invested = weights.abs().sum(axis=1) > 0

    long_bad = invested & ((long_sum - 1.0).abs() > WEIGHT_SUM_TOL)
    bad = long_bad
    if kind == "long_short":
        short_bad = invested & ((short_sum + 1.0).abs() > WEIGHT_SUM_TOL)
        bad = long_bad | short_bad

    n_bad = int(bad.sum())
    if n_bad > 0:
        warn(f"{label}: {n_bad} dates where long side != 1"
             + (" or short side != -1" if kind == "long_short" else "")
             + ". Quintile/NaN handling bug?")
    return n_bad


def check_backtest_red_flags(label: str, stats: dict, results: pd.DataFrame) -> int:
    """Warn on |Sharpe| > 2 (gross or net) and on zero total turnover. Returns the count."""
    count = 0
    for key in ["sharpe_gross", "sharpe_net"]:
        value = stats[key]
        if np.isfinite(value) and abs(value) > MAX_PLAUSIBLE_SHARPE:
            warn(f"{label}: {key} = {value:.2f} has |Sharpe| > {MAX_PLAUSIBLE_SHARPE}. "
                 "Look-ahead bias or a cost bug?")
            count += 1

    if results["turnover"].sum() == 0:
        warn(f"{label}: turnover is 0 on every day. Weight forward-fill or rebalance bug?")
        count += 1
    return count


def check_extreme_returns(returns: pd.DataFrame) -> int:
    """Warn if any daily return is -100% or worse, or +100% or more. Returns the count."""
    extreme = returns.abs() >= 1.0
    n_extreme = int(extreme.sum().sum())
    if n_extreme > 0:
        tickers = list(returns.columns[extreme.any(axis=0)])
        warn(f"{n_extreme} daily returns of +-100% or more ({tickers}). "
             "Bad data or a missing split adjustment?")
    return n_extreme


# ---------- backtest_runs logging ----------


def _clean_number(value) -> float | None:
    """Turn a numpy number into a plain float, or None if it is NaN/inf."""
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value


def log_runs(engine: sa.Engine, runs: list[dict], smoke: bool) -> int:
    """Log each run to backtest_runs. Smoke runs are never logged.

    Each run dict needs signal_name, config, sharpe, ann_return, max_dd.
    Returns how many rows were written.
    """
    if smoke:
        print(f"Smoke run: NOT logging {len(runs)} runs to backtest_runs.")
        return 0

    for run in runs:
        db.log_backtest_run(
            engine,
            signal_name=run["signal_name"],
            config=run["config"],
            sharpe=_clean_number(run["sharpe"]),
            ann_return=_clean_number(run["ann_return"]),
            max_dd=_clean_number(run["max_dd"]),
        )
    print(f"Logged {len(runs)} runs to backtest_runs.")
    return len(runs)
