"""Charts for the results write-up (Skill.md section 11).

Each function takes data and an output path, draws one chart with a title,
axis labels with units and a legend, saves a PNG, and closes the figure.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from backtester.analysis.metrics import drawdown_series, equity_curve  # noqa: E402


def _save(fig, path: str | Path) -> Path:
    """Save a figure as PNG (creating the folder) and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_equity_curve(results: pd.DataFrame, path: str | Path, title: str = "Equity curve") -> Path:
    """Growth of $1, gross vs net of costs. results has 'gross' and 'net' daily returns.

    Equity at close d compounds returns through close d.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(equity_curve(results["gross"]), label="Gross")
    ax.plot(equity_curve(results["net"]), label="Net of costs")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(r"Growth of \$1")  # \$ so matplotlib does not read $...$ as math
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_drawdown(net_returns: dict[str, pd.Series], path: str | Path,
                  title: str = "Drawdown (net)") -> Path:
    """Drawdown from running peak, in percent, for one or more net return series."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, returns in net_returns.items():
        ax.plot(drawdown_series(returns) * 100, label=name)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_rolling_ic(rolling_ics: dict[str, pd.Series], path: str | Path,
                    title: str = "Rolling 12-month mean IC") -> Path:
    """Rolling mean IC per signal, indexed by signal date t."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, series in rolling_ics.items():
        ax.plot(series, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Signal date")
    ax.set_ylabel("Mean Spearman IC (unitless)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)


def plot_quintile_returns(quintiles: pd.Series, path: str | Path,
                          title: str = "Average forward return by quintile",
                          horizon: int | None = None) -> Path:
    """Bar chart of mean forward return per quintile (Q1 = lowest signal), in percent."""
    fig, ax = plt.subplots(figsize=(7, 4))
    label = "Mean forward return"
    if horizon is not None:
        label = f"Mean {horizon}-day forward return"
    ax.bar(quintiles.index.astype(str), quintiles.values * 100, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Signal quintile (Q1 = lowest)")
    ax.set_ylabel("Forward return (%)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    return _save(fig, path)


def plot_cost_sensitivity(sharpe_by_cost: pd.DataFrame, path: str | Path,
                          title: str = "Net Sharpe vs transaction cost") -> Path:
    """Net Sharpe vs cost. Index = cost in bps, one column per signal."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in sharpe_by_cost.columns:
        ax.plot(sharpe_by_cost.index, sharpe_by_cost[name], marker="o", label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Cost per unit traded (bps)")
    ax.set_ylabel("Net Sharpe ratio (annualized)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, path)
