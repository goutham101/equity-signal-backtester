"""Performance metrics for daily strategy return series (Skill.md section 11).

Inputs are daily returns indexed by date, where the value on day d is the
return earned from close d-1 to close d (the backtest engine output).
Annualization uses 252 trading days. The risk-free rate is ignored: the
main strategy is dollar-neutral long-short, so it is roughly self-financing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252

# A standard deviation below this counts as zero (floating point noise on a
# constant series is around 1e-19, not exactly 0).
ZERO_STD_TOL = 1e-12


def annual_return(returns: pd.Series) -> float:
    """Annualized arithmetic return: mean daily return * 252."""
    return returns.mean() * TRADING_DAYS


def annual_volatility(returns: pd.Series) -> float:
    """Annualized volatility: daily std * sqrt(252)."""
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series) -> float:
    """Annualized return / annualized volatility, risk-free rate ignored.

    If volatility is zero (for example a constant return series) the ratio
    is undefined, so this returns NaN instead of dividing by zero or
    returning infinity.
    """
    vol = annual_volatility(returns)
    if not np.isfinite(vol) or vol < ZERO_STD_TOL:
        return np.nan
    return annual_return(returns) / vol


def equity_curve(returns: pd.Series, start: float = 1.0) -> pd.Series:
    """Compounded equity at each close: start * cumprod(1 + daily return)."""
    return start * (1 + returns.fillna(0.0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Drawdown at each close: equity / running peak of equity - 1 (0 or negative)."""
    equity = equity_curve(returns)
    return equity / equity.cummax() - 1


def max_drawdown_from_equity(equity: pd.Series) -> float:
    """Largest peak-to-trough drop of an equity series, as a negative fraction."""
    drawdown = equity / equity.cummax() - 1
    return drawdown.min()


def max_drawdown(returns: pd.Series) -> float:
    """Max drawdown of the equity built from daily returns, as a negative fraction."""
    return max_drawdown_from_equity(equity_curve(returns))


def calmar_ratio(returns: pd.Series) -> float:
    """Annualized return / abs(max drawdown). NaN if there was no drawdown."""
    mdd = max_drawdown(returns)
    if not np.isfinite(mdd) or mdd == 0:
        return np.nan
    return annual_return(returns) / abs(mdd)


def annual_turnover(turnover: pd.Series) -> float:
    """Average annual turnover: mean daily turnover * 252.

    Daily turnover is sum of |change in held weight| across stocks, so a
    full flip of a 2x gross long-short book counts as 4.
    """
    return turnover.mean() * TRADING_DAYS


def hit_rate(returns: pd.Series) -> float:
    """Fraction of days with a strictly positive return (NaN days excluded)."""
    clean = returns.dropna()
    if len(clean) == 0:
        return np.nan
    return (clean > 0).mean()


def monthly_returns(returns: pd.Series) -> pd.Series:
    """Compound daily returns into calendar-month returns, one value per month end."""
    growth = 1 + returns.fillna(0.0)
    return growth.resample("ME").prod() - 1


def worst_month(returns: pd.Series) -> float:
    """Lowest compounded calendar-month return."""
    return monthly_returns(returns).min()


def summary(results: pd.DataFrame) -> dict:
    """Headline metrics for one backtest.

    results: backtest engine output with columns gross, costs, net, turnover,
    indexed by date (row d = return from close d-1 to close d).
    Return/risk metrics use net returns unless the key says gross.
    """
    gross = results["gross"]
    net = results["net"]
    return {
        "ann_return_gross": annual_return(gross),
        "ann_return_net": annual_return(net),
        "ann_vol_net": annual_volatility(net),
        "sharpe_gross": sharpe_ratio(gross),
        "sharpe_net": sharpe_ratio(net),
        "max_drawdown_net": max_drawdown(net),
        "calmar_net": calmar_ratio(net),
        "ann_turnover": annual_turnover(results["turnover"]),
        "ann_cost": annual_return(results["costs"]),
        "hit_rate_net": hit_rate(net),
        "worst_month_net": worst_month(net),
    }


def _format_cell(value) -> str:
    """Format one table cell: floats to 4 decimals, NaN as 'nan'."""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        return f"{value:.4f}"
    return str(value)


def results_to_markdown(rows: dict[str, dict]) -> str:
    """Render {row_name: {column: value}} as a markdown table.

    Columns are the union of all keys, in first-seen order.
    """
    columns = []
    for metrics in rows.values():
        for key in metrics:
            if key not in columns:
                columns.append(key)

    header = "| name | " + " | ".join(columns) + " |"
    divider = "|---|" + "---|" * len(columns)
    lines = [header, divider]
    for name, metrics in rows.items():
        cells = [_format_cell(metrics.get(col, "")) for col in columns]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_results_md(
    sections: dict[str, str | dict[str, dict]],
    path: str | Path = "reports/results.md",
) -> Path:
    """Write reports/results.md with one '## heading' per section.

    Each section value is either ready-made markdown text or a rows dict
    that gets rendered with results_to_markdown.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parts = ["# Results", "", "Auto-generated. Do not edit by hand.", ""]
    for heading, content in sections.items():
        parts.append(f"## {heading}")
        parts.append("")
        if isinstance(content, dict):
            parts.append(results_to_markdown(content))
        else:
            parts.append(content)
        parts.append("")

    path.write_text("\n".join(parts))
    return path
