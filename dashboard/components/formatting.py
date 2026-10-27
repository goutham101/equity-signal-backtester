"""Signal labels, colors, and number formatting. No Streamlit here, so it's easy to test."""

import math

# Keys, labels, and colors come from docs/dashboard_data_contract.md.
SIGNAL_LABELS = {
    "momentum": "Momentum",
    "reversal": "Reversal",
    "low_vol": "Low Vol",
    "equal_weight": "Equal-Weight Combo",
    "walk_forward": "Walk-Forward Combo",
    "random": "Random Baseline",
    "benchmark": "Equal-Weight Benchmark (all stocks)",
}

SIGNAL_COLORS = {
    "momentum": "#6366F1",
    "reversal": "#F59E0B",
    "low_vol": "#10B981",
    "equal_weight": "#A855F7",
    "walk_forward": "#F43F5E",
    "random": "#94A3B8",
    "benchmark": "#0EA5E9",
}

# Order of the sidebar signal selector (spec).
SELECTOR_SIGNALS = [
    "momentum",
    "reversal",
    "low_vol",
    "equal_weight",
    "walk_forward",
    "random",
]

# The three single signals (used for "best single signal" comparisons).
SINGLE_SIGNALS = ["momentum", "reversal", "low_vol"]

# Signals that exist for the full period (everything except walk_forward).
FULL_PERIOD_SIGNALS = ["momentum", "reversal", "low_vol", "equal_weight", "random"]

PORTFOLIO_LABELS = {"long_short": "Long-short", "long_only": "Long-only"}
REBALANCE_LABELS = {"monthly": "Monthly", "weekly": "Weekly"}


def signal_label(key: str) -> str:
    """Display label for a signal key (falls back to the key itself)."""
    return SIGNAL_LABELS.get(key, key)


def signal_color(key: str) -> str:
    """Fixed chart color for a signal key (gray if unknown)."""
    return SIGNAL_COLORS.get(key, "#7f7f7f")


def _missing(value) -> bool:
    """True for None or NaN."""
    if value is None:
        return True
    try:
        return math.isnan(value)
    except TypeError:
        return False


def fmt_pct(value, decimals: int = 1) -> str:
    """Decimal to percent: 0.124 -> '12.4%'. Missing -> 'n/a'."""
    if _missing(value):
        return "n/a"
    return f"{value * 100:.{decimals}f}%"


def fmt_sharpe(value) -> str:
    """Sharpe ratio to 2 decimals: 0.1312 -> '0.13'."""
    if _missing(value):
        return "n/a"
    return f"{value:.2f}"


def fmt_bps(value, decimals: int = 0) -> str:
    """Basis points: 10 -> '10 bps'."""
    if _missing(value):
        return "n/a"
    return f"{value:.{decimals}f} bps"


def fmt_ic(value) -> str:
    """IC to 4 decimals (ICs are small, 0.01 is already meaningful)."""
    if _missing(value):
        return "n/a"
    return f"{value:.4f}"


def fmt_turnover(value) -> str:
    """Annual turnover as a multiple of capital: 11.3 -> '11.3×'."""
    if _missing(value):
        return "n/a"
    return f"{value:.1f}×"


def fmt_number(value, decimals: int = 2) -> str:
    """Plain number with fixed decimals (t-stats, IC IR)."""
    if _missing(value):
        return "n/a"
    return f"{value:.{decimals}f}"
