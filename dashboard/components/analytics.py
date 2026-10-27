"""Turn loaded series into the numbers the pages show.

Every formula comes from backtester.analysis.metrics, so the dashboard
matches the README and reports/results.md. This module only filters,
loops, and arranges the results into tables.
"""

import pandas as pd

import components  # noqa: F401  (puts src/ on sys.path)
from backtester.analysis import metrics
from components import formatting as fmt

# Rolling windows: 252 trading days is one year.
ROLLING_WINDOW = metrics.TRADING_DAYS


def with_cost(frame: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Add net and costs columns at a given cost, in the shape metrics.summary expects."""
    out = frame.copy()
    out["net"] = metrics.net_returns_at_cost(frame, cost_bps)
    out["costs"] = out["gross"] - out["net"]
    return out


def gross_net_metrics(frame: pd.DataFrame, cost_bps: float) -> dict[str, tuple[float, float]]:
    """Headline metrics as {name: (gross value, net value)}.

    metrics.summary gives the net side; the gross side uses the same
    metrics functions on the gross column.
    """
    costed = with_cost(frame, cost_bps)
    stats = metrics.summary(costed)
    gross = metrics.trim_to_active(costed)["gross"]
    turnover = stats["ann_turnover"]
    return {
        "sharpe": (stats["sharpe_gross"], stats["sharpe_net"]),
        "ann_return": (stats["ann_return_gross"], stats["ann_return_net"]),
        "ann_vol": (metrics.annual_volatility(gross), stats["ann_vol_net"]),
        "max_drawdown": (metrics.max_drawdown(gross), stats["max_drawdown_net"]),
        "turnover": (turnover, turnover),
        "hit_rate": (metrics.hit_rate(gross), stats["hit_rate_net"]),
    }


def leg_returns(frame: pd.DataFrame) -> dict[str, float]:
    """Annual gross return of the long and short legs (before costs)."""
    stats = metrics.summary(with_cost(frame, 0))
    return {
        "Long leg": stats.get("ann_return_long", float("nan")),
        "Short leg": stats.get("ann_return_short", float("nan")),
    }


def net_sharpe(frame: pd.DataFrame, cost_bps: float) -> float:
    """Net Sharpe at a given cost, warm-up rows dropped first."""
    return metrics.summary(with_cost(frame, cost_bps))["sharpe_net"]


def cost_curve(frames: dict[str, pd.DataFrame], costs: range) -> pd.DataFrame:
    """Net Sharpe for each signal (columns) at each cost level (index, bps)."""
    table = {}
    for signal, frame in frames.items():
        active = metrics.trim_to_active(frame)
        table[signal] = [
            metrics.sharpe_ratio(metrics.net_returns_at_cost(active, bps)) for bps in costs
        ]
    curve = pd.DataFrame(table, index=list(costs))
    curve.index.name = "cost_bps"
    return curve


def breakeven_table(frames: dict[str, pd.DataFrame], cost_bps: float) -> pd.DataFrame:
    """Breakeven cost, gross Sharpe, and net Sharpe at the current cost per signal."""
    rows = []
    for signal, frame in frames.items():
        stats = metrics.summary(with_cost(frame, cost_bps))
        rows.append(
            {
                "signal": signal,
                "breakeven_bps": stats["breakeven_bps"],
                "sharpe_gross": stats["sharpe_gross"],
                "sharpe_net": stats["sharpe_net"],
                "ann_turnover": stats["ann_turnover"],
            }
        )
    return pd.DataFrame(rows)


def summary_table(frames: dict[str, pd.DataFrame], cost_bps: float) -> pd.DataFrame:
    """metrics.summary for each signal at one cost level, one row per signal."""
    rows = []
    for signal, frame in frames.items():
        stats = metrics.summary(with_cost(frame, cost_bps))
        rows.append({"signal": signal, **stats})
    return pd.DataFrame(rows)


def net_equity(frame: pd.DataFrame, cost_bps: float) -> pd.Series:
    """Growth of $1 after costs."""
    return metrics.equity_curve(metrics.net_returns_at_cost(frame, cost_bps))


def monthly_grid(returns: pd.Series) -> pd.DataFrame:
    """Calendar-month returns as a years x months table (1..12)."""
    monthly = metrics.monthly_returns(returns)
    grid = pd.DataFrame(
        {"year": monthly.index.year, "month": monthly.index.month, "ret": monthly.values}
    )
    return grid.pivot(index="year", columns="month", values="ret")


def rolling_ic(ic: pd.Series, window: int = ROLLING_WINDOW) -> pd.Series:
    """Trailing mean of daily IC over `window` observations (backward-looking)."""
    return ic.sort_index().rolling(window).mean()


def walk_forward_takeaway(table: pd.DataFrame, cost_bps: float) -> str:
    """One plain sentence: did fitting beat equal weighting and the best single signal?

    table: output of summary_table on the out-of-sample frames.
    """
    sharpe = table.set_index("signal")["sharpe_net"]
    if "walk_forward" not in sharpe or "equal_weight" not in sharpe:
        return "Not enough out-of-sample data to compare the walk-forward combo."

    wf = sharpe["walk_forward"]
    ew = sharpe["equal_weight"]
    singles = sharpe.reindex(fmt.SINGLE_SIGNALS).dropna()

    vs_ew = "beat" if wf > ew else "did not beat"
    sentence = (
        f"Out of sample at {fmt.fmt_bps(cost_bps)}, the walk-forward combo had a net "
        f"Sharpe of {fmt.fmt_sharpe(wf)}. It {vs_ew} the equal-weight combo "
        f"({fmt.fmt_sharpe(ew)})"
    )
    if singles.empty:
        return sentence + "."

    best = singles.idxmax()
    vs_best = "beat" if wf > singles[best] else "lost to"
    return (
        sentence + f" and {vs_best} the best single signal, "
        f"{fmt.signal_label(best)} ({fmt.fmt_sharpe(singles[best])})."
    )
