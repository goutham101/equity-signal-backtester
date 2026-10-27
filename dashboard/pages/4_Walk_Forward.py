"""Walk-Forward: fit signal weights on past data only, then test on the next year."""

import pandas as pd
import streamlit as st
from components import analytics, charts, loaders, sidebar, style
from components import formatting as fmt

selection = sidebar.setup_page("Walk-Forward", ":material/timeline:")
meta = loaders.metadata()
bps = selection.cost_bps
oos = sidebar.oos_years(meta)

style.page_header(
    "Walk-Forward",
    "Fit signal weights on 5 years, test on the next one, repeat. Only test-year results count.",
    eyebrow="Out-of-sample",
)
st.caption(
    f"Out-of-sample ({oos}), long-short, monthly: the only combination the walk-forward "
    f"combo exists for, so the portfolio and rebalance controls don't apply here. "
    f"Costs follow the sidebar ({fmt.fmt_bps(bps)})."
)

# Timeline of train / purge / test windows.
windows = loaders.walk_forward_windows()
charts.show(charts.timeline_chart(windows, "Train, purge gap, and test window for each year"))
st.caption(
    "Each window trains on the previous years, skips a short purge gap so training "
    "labels can't reach into the test year, then tests on the next year."
)

# Fitted weights per window.
weights = loaders.walk_forward_weights()
charts.show(charts.weights_chart(weights, "Fitted signal weights per test window"))

# Out-of-sample comparison: table + equity curves at the sidebar cost.
style.section(f"Out-of-sample comparison ({oos}), net of {fmt.fmt_bps(bps)}")
order = ["walk_forward", "equal_weight", *fmt.SINGLE_SIGNALS, "random"]
frames = {}
for signal in order:
    frame = loaders.series(signal, "long_short", "monthly", "oos")
    if not frame.empty:
        frames[signal] = frame

table = analytics.summary_table(frames, bps)
st.markdown(f"**{analytics.walk_forward_takeaway(table, bps)}**")

st.dataframe(
    pd.DataFrame(
        {
            "Signal": table["signal"].map(fmt.signal_label),
            "Net Sharpe": table["sharpe_net"].map(fmt.fmt_sharpe),
            "Gross Sharpe": table["sharpe_gross"].map(fmt.fmt_sharpe),
            "Annual return (net)": table["ann_return_net"].map(fmt.fmt_pct),
            "Max drawdown (net)": table["max_drawdown_net"].map(fmt.fmt_pct),
            "Annual turnover": table["ann_turnover"].map(fmt.fmt_turnover),
        }
    ),
    hide_index=True,
    column_config={
        "Net Sharpe": st.column_config.TextColumn(
            help="Annual return divided by annual volatility, after costs."
        ),
        "Gross Sharpe": st.column_config.TextColumn(help="Sharpe before trading costs."),
        "Annual return (net)": st.column_config.TextColumn(help="Average daily net return × 252."),
        "Max drawdown (net)": st.column_config.TextColumn(
            help="Largest drop from a previous peak in the net equity curve."
        ),
        "Annual turnover": st.column_config.TextColumn(
            help="Times the capital is traded per year."
        ),
    },
)

lines = [
    {
        "series": analytics.net_equity(frame, bps),
        "name": fmt.signal_label(signal),
        "color": fmt.signal_color(signal),
        "dash": "solid" if signal in ("walk_forward", "equal_weight") else "dot",
    }
    for signal, frame in frames.items()
]
charts.show(charts.equity_chart(lines, f"Out-of-sample equity ({oos}), net of {fmt.fmt_bps(bps)}"))
