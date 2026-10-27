"""Pure helpers for the dashboard export (scripts/06_export_dashboard_data.py).

Everything here works on plain DataFrames and strings, with no database or
files, so it can be unit-tested on synthetic data. The file layout is in
docs/dashboard_data_contract.md.
"""

import re

import numpy as np
import pandas as pd

from backtester.analysis.metrics import trim_to_active

# Engine columns the dashboard needs. Net is NOT stored: the app recomputes
# it from gross and turnover so the cost slider works.
RETURN_COLUMNS = ["gross", "turnover", "gross_long", "gross_short"]
KEY_COLUMNS = ["date", "signal", "portfolio", "rebalance", "period"]


def frame_to_long(
    results: pd.DataFrame,
    signal: str,
    portfolio: str,
    rebalance: str,
    period: str,
) -> pd.DataFrame:
    """One engine frame -> long rows: date, signal, portfolio, rebalance, period, returns.

    Warm-up rows before the first trade are dropped first (trim_to_active),
    so the series starts on the first day with a position.
    """
    active = trim_to_active(results)
    long = active[RETURN_COLUMNS].copy()
    long.index = pd.DatetimeIndex(long.index).astype("datetime64[ns]")
    long.index.name = "date"
    long = long.reset_index()
    long.insert(1, "signal", signal)
    long.insert(2, "portfolio", portfolio)
    long.insert(3, "rebalance", rebalance)
    long.insert(4, "period", period)
    return long


def ic_to_long(ic: pd.Series, signal: str, horizon: int) -> pd.DataFrame:
    """Daily IC series (index = signal date) -> rows: date, signal, horizon, ic."""
    long = pd.DataFrame({"ic": ic.astype(float)})
    long.index = pd.DatetimeIndex(long.index).astype("datetime64[ns]")
    long.index.name = "date"
    long = long.reset_index()
    long.insert(1, "signal", signal)
    long.insert(2, "horizon", horizon)
    return long


def quintiles_to_long(quintiles: pd.Series, signal: str, horizon: int) -> pd.DataFrame:
    """quintile_returns output (index Q1..Qn) -> rows: signal, horizon, quintile, avg_fwd_return.

    quintile is an int, 1 = lowest signal.
    """
    rows = []
    for label, value in quintiles.items():
        rows.append(
            {
                "signal": signal,
                "horizon": horizon,
                "quintile": int(str(label).lstrip("Q")),
                "avg_fwd_return": float(value),
            }
        )
    return pd.DataFrame(rows)


def weights_to_long(weights: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward weights (rows = window, columns = signal) -> rows: window, signal, weight."""
    rows = []
    for window, row in weights.iterrows():
        for signal, weight in row.items():
            rows.append({"window": str(window), "signal": signal, "weight": float(weight)})
    return pd.DataFrame(rows)


def windows_to_frame(table: pd.DataFrame) -> pd.DataFrame:
    """Windows table (index = label) -> flat frame with a window column and four dates."""
    frame = table.copy()
    for column in ["train_start", "train_end", "test_start", "test_end"]:
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    frame.index = frame.index.astype(str)
    frame.index.name = "window"
    return frame.reset_index()


def readme_bullets(text: str, heading: str) -> list[str]:
    """Bullet points under '## <heading>' in a markdown text, markdown kept.

    The section ends at the next '## ' heading. Each bullet loses its leading
    '- '. A non-empty line right after a bullet line is a wrapped
    continuation and is joined to that bullet with a space. A blank line
    ends the bullet.
    Raises ValueError if the heading is missing, so a renamed README section
    fails loudly instead of exporting an empty list.
    """
    lines = text.splitlines()
    target = f"## {heading}"
    start = None
    for i, line in enumerate(lines):
        if line.strip() == target:
            start = i + 1
            break
    if start is None:
        raise ValueError(f"README has no '{target}' section")

    bullets = []
    in_bullet = False
    for line in lines[start:]:
        if line.startswith("## "):
            break
        stripped = line.strip()
        if stripped.startswith("- "):
            bullets.append(stripped[2:].strip())
            in_bullet = True
        elif stripped and in_bullet:
            bullets[-1] = bullets[-1] + " " + stripped
        else:
            # A blank line ends the bullet; a later paragraph is not part of it.
            in_bullet = False
    return bullets


def parse_pytest_passed(output: str) -> int:
    """Number of passed tests from pytest's summary line (e.g. '22 passed in 1.2s'). 0 if none."""
    match = re.search(r"(\d+) passed", output)
    if match is None:
        return 0
    return int(match.group(1))


def summary_row(signal: str, stats: dict, mean_ic_21d: float) -> dict:
    """One row of summary.parquet from a metrics.summary dict."""
    return {
        "signal": signal,
        "sharpe_net": stats["sharpe_net"],
        "ann_return_net": stats["ann_return_net"],
        "ann_vol_net": stats["ann_vol_net"],
        "max_drawdown_net": stats["max_drawdown_net"],
        "mean_ic_21d": mean_ic_21d if mean_ic_21d is not None else np.nan,
        "ann_turnover": stats["ann_turnover"],
    }
