"""Page setup and the shared sidebar (same controls on every page).

Selections are shared across pages through st.session_state. Streamlit
drops a widget's state when you switch pages, so each widget copies its
value into a plain session key ("_keep_<key>") on change, and the value is
restored from there before the widget is drawn on the next page.
"""

from dataclasses import dataclass
from datetime import datetime

import streamlit as st

from components import formatting as fmt
from components import loaders, style

APP_TITLE = "Equity Signal Backtester"
APP_TAGLINE = (
    "Would three classic stock signals have made money on the S&P 500 "
    "after trading costs? All results are precomputed."
)
FOOTER = "Educational project. Not investment advice."

# Walk-Forward Combo only exists for this one combination (data contract).
WALK_FORWARD_COMBO = ("long_short", "monthly")


@dataclass(frozen=True)
class Selection:
    """The sidebar choices for the current run."""

    signal: str
    portfolio: str
    rebalance: str
    cost_bps: int

    @property
    def walk_forward_unavailable(self) -> bool:
        """True when Walk-Forward Combo is picked with a combination that doesn't exist."""
        combo = (self.portfolio, self.rebalance)
        return self.signal == "walk_forward" and combo != WALK_FORWARD_COMBO

    @property
    def period(self) -> str:
        """Walk-forward only has out-of-sample data; everything else uses the full period."""
        return "oos" if self.signal == "walk_forward" else "full"


def _keep(key: str) -> None:
    """Callback: copy a widget's value into a session key that survives page switches."""
    st.session_state[f"_keep_{key}"] = st.session_state[key]


def _restore(key: str, default) -> None:
    """Put the remembered value (or the default) back before the widget is drawn."""
    st.session_state[key] = st.session_state.get(f"_keep_{key}", default)


def setup_page(title: str, icon: str) -> Selection:
    """Page config, data check, and sidebar. Call first on every page."""
    st.set_page_config(page_title=f"{title} · {APP_TITLE}", page_icon=icon, layout="wide")
    style.inject_css()

    missing = loaders.missing_files()
    if missing:
        st.error(
            f"Dashboard data not found in `{loaders.data_dir()}`. "
            f"Missing: {', '.join(missing)}. "
            "Run `python scripts/06_export_dashboard_data.py` first."
        )
        st.stop()

    return render_sidebar(loaders.metadata())


def _controls(default_cost: int) -> Selection:
    """The four global controls."""
    _restore("signal", "momentum")
    signal = st.selectbox(
        "Signal",
        options=fmt.SELECTOR_SIGNALS,
        format_func=fmt.signal_label,
        key="signal",
        on_change=_keep,
        args=("signal",),
        help="Which strategy to look at. Random Baseline is the no-skill comparison.",
    )
    _restore("portfolio", "long_short")
    portfolio = st.radio(
        "Portfolio type",
        options=list(fmt.PORTFOLIO_LABELS),
        format_func=fmt.PORTFOLIO_LABELS.get,
        key="portfolio",
        on_change=_keep,
        args=("portfolio",),
        horizontal=True,
        help="Long-short buys the top 20% and shorts the bottom 20%. "
        "Long-only just holds the top 20%.",
    )
    _restore("rebalance", "monthly")
    rebalance = st.radio(
        "Rebalance",
        options=list(fmt.REBALANCE_LABELS),
        format_func=fmt.REBALANCE_LABELS.get,
        key="rebalance",
        on_change=_keep,
        args=("rebalance",),
        horizontal=True,
        help="How often the portfolio is rebuilt from the latest signal.",
    )
    _restore("cost_bps", default_cost)
    cost = st.slider(
        "Transaction cost",
        min_value=0,
        max_value=50,
        step=1,
        format="%d bps",
        key="cost_bps",
        on_change=_keep,
        args=("cost_bps",),
        help="Cost per unit of weight traded, in basis points (1 bps = 0.01%). "
        "Net return = gross - turnover × bps / 10,000.",
    )
    return Selection(signal=signal, portfolio=portfolio, rebalance=rebalance, cost_bps=cost)


def _date(text: str | None) -> str:
    """ISO date or timestamp -> 'YYYY-MM-DD' ('n/a' if missing)."""
    if not text:
        return "n/a"
    try:
        return datetime.fromisoformat(text).date().isoformat()
    except ValueError:
        return text


def _data_info(meta: dict) -> None:
    """Universe, date range, stock count, last updated."""
    st.markdown("**Data**")
    st.caption(
        f"{meta.get('universe', 'n/a')}  \n"
        f"{_date(meta.get('start_date'))} to {_date(meta.get('end_date'))}  \n"
        f"{meta.get('n_stocks', 'n/a')} stocks  \n"
        f"Last updated {_date(meta.get('last_updated'))}"
    )


def _links(meta: dict) -> None:
    """GitHub, README, and LinkedIn links. Links that are null are hidden."""
    links = meta.get("links") or {}
    names = [("github", "GitHub repo"), ("readme", "README"), ("linkedin", "LinkedIn")]
    parts = [f"[{label}]({links[key]})" for key, label in names if links.get(key)]
    if parts:
        st.markdown(" · ".join(parts))


def render_sidebar(meta: dict) -> Selection:
    """Title, controls, data info, links, and the disclaimer."""
    with st.sidebar:
        style.brand(APP_TITLE, "S&P 500 · 2010–2026")
        st.caption(APP_TAGLINE)
        selection = _controls(int(meta.get("default_cost_bps", 10)))
        st.divider()
        _data_info(meta)
        _links(meta)
        st.caption(FOOTER)
    return selection


def oos_years(meta: dict) -> str:
    """Out-of-sample span as years, e.g. '2015–2026'."""
    start = str(meta.get("oos_start") or "")[:4]
    end = str(meta.get("oos_end") or "")[:4]
    return f"{start}–{end}" if start and end else "test years"


def walk_forward_notice() -> None:
    """Explain why Walk-Forward Combo can't be shown for the current choice."""
    years = oos_years(loaders.metadata())
    st.info(
        f"The Walk-Forward Combo only exists out-of-sample ({years}), long-short, "
        "monthly rebalance. Its weights are refit each year, so there is no "
        "long-only or weekly version. Pick **Long-short** and **Monthly** in the "
        "sidebar to see it."
    )
