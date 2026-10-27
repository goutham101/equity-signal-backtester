"""Overview page: the headline results a first-time visitor sees.

Run from the repo root: streamlit run dashboard/app.py
"""

import pandas as pd
import streamlit as st
from components import analytics, charts, loaders, sidebar
from components import formatting as fmt

sidebar.setup_page("Overview", "📈")
meta = loaders.metadata()
headline_bps = int(meta.get("default_cost_bps", 10))
oos = sidebar.oos_years(meta)

st.title("Equity Signal Backtester")
st.markdown(
    "If I had traded on a signal in the past, using only what I could have known "
    "at the time, would it have made money after trading costs?"
)

# Headline table, straight from summary.parquet.
st.subheader(
    f"Headline results: out-of-sample ({oos}), long-short, monthly, {fmt.fmt_bps(headline_bps)}"
)
st.caption(
    f"Fixed at {fmt.fmt_bps(headline_bps)} and does not change with the sidebar "
    "controls, so the headline can't be cherry-picked. Use the other pages to explore."
)

table = loaders.summary().copy()
display = pd.DataFrame(
    {
        "Signal": table["signal"].map(fmt.signal_label),
        "Net Sharpe": table["sharpe_net"].map(fmt.fmt_sharpe),
        "Annual return": table["ann_return_net"].map(fmt.fmt_pct),
        "Volatility": table["ann_vol_net"].map(fmt.fmt_pct),
        "Max drawdown": table["max_drawdown_net"].map(fmt.fmt_pct),
        "Mean IC (21d)": table["mean_ic_21d"].map(fmt.fmt_ic),
        "Annual turnover": table["ann_turnover"].map(fmt.fmt_turnover),
    }
)
st.dataframe(
    display,
    hide_index=True,
    column_config={
        "Net Sharpe": st.column_config.TextColumn(
            help="Annual return divided by annual volatility, after costs."
        ),
        "Annual return": st.column_config.TextColumn(help="Average daily net return × 252."),
        "Volatility": st.column_config.TextColumn(
            help="Standard deviation of daily net returns × √252."
        ),
        "Max drawdown": st.column_config.TextColumn(
            help="Largest drop from a previous peak in the net equity curve."
        ),
        "Mean IC (21d)": st.column_config.TextColumn(
            help="Average daily rank correlation between the signal and the next "
            "21-day return, full period. Not defined for the walk-forward combo."
        ),
        "Annual turnover": st.column_config.TextColumn(
            help="How many times the portfolio's capital is traded per year."
        ),
    },
)

# Out-of-sample equity curves at the headline cost.
lines = []
for signal in ["walk_forward", "equal_weight", "random"]:
    frame = loaders.series(signal, "long_short", "monthly", "oos")
    if frame.empty:
        continue
    lines.append(
        {
            "series": analytics.net_equity(frame, headline_bps),
            "name": fmt.signal_label(signal),
            "color": fmt.signal_color(signal),
        }
    )
if lines:
    charts.show(
        charts.equity_chart(
            lines,
            f"Out-of-sample equity ({oos}), long-short, monthly, "
            f"net of {fmt.fmt_bps(headline_bps)}",
        )
    )

left, right = st.columns([3, 2])
with left:
    st.subheader("Key findings")
    findings = meta.get("key_findings") or []
    if findings:
        st.markdown("\n".join(f"- {item}" for item in findings))
    else:
        st.caption("No findings exported yet.")

with right:
    with st.container(border=True):
        st.markdown("**How to read this**")
        st.markdown(
            "**Sharpe ratio** is return per unit of risk: annual return divided by "
            "annual volatility. Above 1 is strong, around 0 means no edge, below 0 "
            "lost money.\n\n"
            "**IC** (information coefficient) measures whether higher-ranked stocks "
            "really went on to earn more. It runs from -1 to 1, and for real signals "
            "even 0.02 to 0.05 is meaningful."
        )
