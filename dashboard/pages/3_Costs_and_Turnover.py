"""Costs & Turnover: how much trading eats into each signal."""

import pandas as pd
import streamlit as st
from components import analytics, charts, loaders, sidebar, style
from components import formatting as fmt

selection = sidebar.setup_page("Costs & Turnover", "💸")
style.hero(
    "Costs & Turnover",
    "The same trades at different costs. Watch how fast a high-turnover signal falls apart.",
    kicker="Where strategies die",
)

portfolio, rebalance, bps = selection.portfolio, selection.rebalance, selection.cost_bps
st.caption(
    f"Full period, {fmt.PORTFOLIO_LABELS[portfolio].lower()}, {rebalance}. "
    "The Walk-Forward Combo only exists out-of-sample, so it's on the Walk-Forward page."
)

frames = {}
for signal in fmt.FULL_PERIOD_SIGNALS:
    frame = loaders.series(signal, portfolio, rebalance, "full")
    if not frame.empty:
        frames[signal] = frame
if not frames:
    st.info("No data for this combination.")
    st.stop()

# Net Sharpe vs cost, recomputed live from gross and turnover.
curve = analytics.cost_curve(frames, range(0, 51))
charts.show(charts.cost_curve_chart(curve, bps, "Net Sharpe vs trading cost"))

# Breakeven table.
style.section("Breakeven cost")
breakeven = analytics.breakeven_table(frames, bps)


def _breakeven_text(value: float) -> str:
    """Breakeven bps, with a plain note when the signal loses money before costs."""
    if value == 0:
        return "0 bps (loses before costs)"
    return fmt.fmt_bps(value, 1)


st.dataframe(
    pd.DataFrame(
        {
            "Signal": breakeven["signal"].map(fmt.signal_label),
            "Breakeven cost": breakeven["breakeven_bps"].map(_breakeven_text),
            "Gross Sharpe": breakeven["sharpe_gross"].map(fmt.fmt_sharpe),
            f"Net Sharpe at {fmt.fmt_bps(bps)}": breakeven["sharpe_net"].map(fmt.fmt_sharpe),
        }
    ),
    hide_index=True,
    column_config={
        "Breakeven cost": st.column_config.TextColumn(
            help="The cost per unit traded at which net Sharpe hits zero."
        ),
        "Gross Sharpe": st.column_config.TextColumn(help="Sharpe before any trading costs."),
        f"Net Sharpe at {fmt.fmt_bps(bps)}": st.column_config.TextColumn(
            help="Sharpe after costs at the sidebar cost level."
        ),
    },
)

# Turnover bars.
charts.show(
    charts.bar_chart(
        [fmt.signal_label(s) for s in breakeven["signal"]],
        list(breakeven["ann_turnover"]),
        [fmt.signal_color(s) for s in breakeven["signal"]],
        "Annual turnover (times capital traded per year)",
        "Annual turnover",
        text=[fmt.fmt_turnover(v) for v in breakeven["ann_turnover"]],
    )
)

# Monthly vs weekly for the sidebar signal.
signal = selection.signal
style.section(f"Monthly vs weekly: {fmt.signal_label(signal)}")
if signal == "walk_forward":
    sidebar.walk_forward_notice()
    st.stop()

rows = {}
for freq in ["monthly", "weekly"]:
    frame = loaders.series(signal, portfolio, freq, "full")
    if frame.empty:
        continue
    stats = analytics.gross_net_metrics(frame, bps)["sharpe"]
    rows[fmt.REBALANCE_LABELS[freq]] = {
        "Gross Sharpe": stats[0],
        f"Net Sharpe ({fmt.fmt_bps(bps)})": stats[1],
    }
comparison = pd.DataFrame(rows).T
if comparison.empty:
    st.info("No data for this combination.")
else:
    fig = charts.grouped_bar_chart(comparison, "Same signal, two rebalance frequencies", "Sharpe")
    charts.show(charts.with_zero_line(fig))
    st.caption("Weekly rebalancing usually trades more, so costs take a bigger bite.")
