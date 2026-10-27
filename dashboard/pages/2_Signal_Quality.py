"""Signal Quality: does each signal rank stocks in the right order (IC)?"""

import pandas as pd
import streamlit as st
from components import analytics, charts, loaders, sidebar, style
from components import formatting as fmt

selection = sidebar.setup_page("Signal Quality", ":material/insights:")
style.page_header(
    "Signal Quality",
    "Does the signal rank tomorrow's winners above the losers? "
    "Measured before any portfolio or cost.",
    eyebrow="Information coefficient",
)
st.caption(
    "IC is the rank correlation between today's signal and the forward return that "
    "starts tomorrow (close t+1). It doesn't depend on portfolio type, rebalance, or costs."
)

horizon = st.radio(
    "Forward return horizon",
    options=[5, 21, 63],
    index=1,
    format_func=lambda h: f"{h} days",
    horizontal=True,
    key="ic_horizon",
    help="How far ahead the forward return looks, in trading days.",
)

# Summary table for every signal.
summary = loaders.ic_summary()
summary = summary[summary["horizon"] == horizon]
table = pd.DataFrame(
    {
        "Signal": summary["signal"].map(fmt.signal_label),
        "Mean IC": summary["mean_ic"].map(fmt.fmt_ic),
        "IC std": summary["ic_std"].map(fmt.fmt_ic),
        "IC IR": summary["ic_ir"].map(fmt.fmt_number),
        "% positive": summary["pct_positive"].map(fmt.fmt_pct),
        "t-stat (Newey-West)": summary["t_stat"].map(fmt.fmt_number),
        "Dates": summary["n_dates"].astype(int),
    }
)
st.dataframe(
    table,
    hide_index=True,
    column_config={
        "Mean IC": st.column_config.TextColumn(
            help="Average daily IC. Above 0 means higher-ranked stocks earned more."
        ),
        "IC std": st.column_config.TextColumn(help="How much the daily IC moves around."),
        "IC IR": st.column_config.TextColumn(
            help="Mean IC divided by IC std: how consistent the signal is."
        ),
        "% positive": st.column_config.TextColumn(help="Share of days with IC above 0."),
        "t-stat (Newey-West)": st.column_config.TextColumn(
            help=f"t-stat of mean IC with Newey-West errors ({horizon} lags), because "
            "overlapping forward returns make neighbouring days correlated. "
            "Roughly, above 2 is significant."
        ),
        "Dates": st.column_config.NumberColumn(help="Number of dates with a valid IC."),
    },
)

# Rolling 12-month IC, all signals.
ic = loaders.ic_daily()
ic = ic[ic["horizon"] == horizon]
lines = []
for signal in fmt.FULL_PERIOD_SIGNALS:
    daily = ic[ic["signal"] == signal].set_index("date")["ic"]
    if daily.empty:
        continue
    lines.append(
        {
            "series": analytics.rolling_ic(daily),
            "name": fmt.signal_label(signal),
            "color": fmt.signal_color(signal),
        }
    )
fig = charts.line_chart(lines, f"Rolling 12-month IC ({horizon}-day horizon)", "Mean IC")
charts.show(charts.with_zero_line(fig))
st.caption("Each point is the mean of the last 252 daily ICs, so it only uses past data.")

# Per-signal charts for the sidebar signal.
signal = selection.signal
label = fmt.signal_label(signal)
style.section(f"{label}: quintiles and IC distribution")
signal_ic = ic[ic["signal"] == signal]["ic"]
if signal_ic.empty:
    st.info(
        f"IC isn't defined for {label}: its weights change every year, so there is no "
        "single signal to rank stocks by. Pick another signal in the sidebar."
    )
    st.stop()

color = fmt.signal_color(signal)
quintiles = loaders.quintile_returns()
quintiles = quintiles[
    (quintiles["signal"] == signal) & (quintiles["horizon"] == horizon)
].sort_values("quintile")

left, right = st.columns(2)
with left:
    fig = charts.bar_chart(
        [f"Q{q}" for q in quintiles["quintile"]],
        list(quintiles["avg_fwd_return"]),
        [color] * len(quintiles),
        f"Average {horizon}-day forward return by quintile",
        "Average forward return",
        y_format=".2%",
        text=[fmt.fmt_pct(v, 2) for v in quintiles["avg_fwd_return"]],
    )
    fig.update_xaxes(title="Q1 = lowest signal, Q5 = highest")
    charts.show(fig)
    st.caption("A useful signal climbs from Q1 to Q5.")
with right:
    charts.show(
        charts.histogram(
            signal_ic, signal_ic.mean(), f"Daily IC distribution ({horizon}-day horizon)", color
        )
    )
