"""Signal Explorer: one strategy in detail, driven by the sidebar controls."""

import streamlit as st
from components import analytics, charts, loaders, sidebar, style
from components import formatting as fmt
from components.analytics import metrics  # imported via components so src/ is on sys.path

selection = sidebar.setup_page("Signal Explorer", "🔍")
style.hero(
    "Signal Explorer",
    "Equity, drawdowns, rolling Sharpe and monthly returns for any signal. "
    "Every number reacts to the sidebar.",
    kicker="Explore",
)

if selection.walk_forward_unavailable:
    sidebar.walk_forward_notice()
    st.stop()

signal, period, bps = selection.signal, selection.period, selection.cost_bps
color = fmt.signal_color(signal)
frame = loaders.series(signal, selection.portfolio, selection.rebalance, period)
if frame.empty:
    st.info("No data for this combination.")
    st.stop()

oos_tag = "Out-of-sample: " if period == "oos" else ""
combo = (
    f"{fmt.signal_label(signal)}, {fmt.PORTFOLIO_LABELS[selection.portfolio].lower()}, "
    f"{selection.rebalance}"
)
st.caption(f"{oos_tag}{combo}. Net returns use {fmt.fmt_bps(bps)} from the sidebar.")

# Date range: allowed on this page only (spec). Defaults to the full history.
years = sorted(frame.index.year.unique())
if len(years) > 1:
    start_year, end_year = st.select_slider(
        "Years shown",
        options=years,
        value=(years[0], years[-1]),
        key=f"years_{signal}_{selection.portfolio}_{selection.rebalance}",
        help="Zoom into a sub-period. The Overview headline always uses the full test period.",
    )
    frame = frame[(frame.index.year >= start_year) & (frame.index.year <= end_year)]

# Metric cards: net value, gross value underneath.
CARDS = [
    (
        "sharpe",
        "Sharpe",
        fmt.fmt_sharpe,
        "Annual return divided by annual volatility. Higher is better.",
    ),
    ("ann_return", "Annual return", fmt.fmt_pct, "Average daily return × 252."),
    ("ann_vol", "Volatility", fmt.fmt_pct, "Standard deviation of daily returns × √252."),
    (
        "max_drawdown",
        "Max drawdown",
        fmt.fmt_pct,
        "Largest drop from a previous peak of the equity curve.",
    ),
    (
        "turnover",
        "Turnover",
        fmt.fmt_turnover,
        "Times the capital is traded per year. Same before and after costs.",
    ),
    ("hit_rate", "Hit rate", fmt.fmt_pct, "Share of days with a positive return."),
]
values = analytics.gross_net_metrics(frame, bps)
# Two rows of three cards, so the numbers have room.
columns = st.columns(3) + st.columns(3)
for column, (key, label, formatter, tip) in zip(columns, CARDS, strict=True):
    gross, net = values[key]
    column.metric(
        label if key == "turnover" else f"{label} (net)",
        formatter(net),
        delta=f"gross {formatter(gross)}",
        delta_color="off",
        delta_arrow="off",
        help=tip,
    )

# Equity: gross vs net, optional overlays.
show_random = signal != "random" and st.toggle(
    "Overlay random baseline",
    value=False,
    help="Adds the random signal (same portfolio and costs) as a no-skill comparison.",
)
show_benchmark = selection.portfolio == "long_only" and st.toggle(
    "Overlay equal-weight benchmark",
    value=True,
    help="Equal-weight portfolio of all stocks, the natural yardstick for long-only.",
)

lines = [
    {
        "series": metrics.equity_curve(frame["gross"]),
        "name": "Gross",
        "color": color,
        "dash": "dot",
    },
    {
        "series": analytics.net_equity(frame, bps),
        "name": f"Net of {fmt.fmt_bps(bps)}",
        "color": color,
    },
]
overlays = []
if show_random:
    overlays.append(("random", selection.portfolio))
if show_benchmark:
    overlays.append(("benchmark", "long_only"))
for other, portfolio in overlays:
    other_frame = loaders.series(other, portfolio, selection.rebalance, period)
    other_frame = other_frame.loc[frame.index.min() : frame.index.max()]
    if not other_frame.empty:
        lines.append(
            {
                "series": analytics.net_equity(other_frame, bps),
                "name": f"{fmt.signal_label(other)} (net)",
                "color": fmt.signal_color(other),
            }
        )
charts.show(charts.equity_chart(lines, f"{oos_tag}Equity curve, gross vs net"))

net = metrics.net_returns_at_cost(frame, bps)
left, right = st.columns(2)
with left:
    charts.show(
        charts.drawdown_chart(metrics.drawdown_series(net), f"{oos_tag}Drawdown (net)", color)
    )
with right:
    rolling = metrics.rolling_sharpe(net)
    fig = charts.line_chart(
        [{"series": rolling, "name": "Rolling Sharpe", "color": color}],
        f"{oos_tag}Rolling 12-month Sharpe (net)",
        "Sharpe",
    )
    charts.show(charts.with_zero_line(fig))

charts.show(charts.monthly_heatmap(analytics.monthly_grid(net), f"{oos_tag}Monthly net returns"))

# Long vs short legs (long-short only).
style.section("Which side made the money?")
if selection.portfolio != "long_short":
    st.caption("The leg breakdown only applies to long-short portfolios.")
else:
    legs = analytics.leg_returns(frame)
    leg_colors = ["#17becf", "#e377c2"]  # not used by any signal
    left, right = st.columns([1, 2])
    with left:
        for (name, value), column in zip(legs.items(), st.columns(2), strict=True):
            column.metric(
                f"{name} (gross)",
                fmt.fmt_pct(value),
                help="Annual return this side contributed before costs. "
                "For the short leg, positive means the shorted stocks fell.",
            )
    with right:
        leg_lines = [
            {
                "series": metrics.equity_curve(frame["gross_long"]),
                "name": "Long leg",
                "color": leg_colors[0],
            },
            {
                "series": metrics.equity_curve(frame["gross_short"]),
                "name": "Short leg",
                "color": leg_colors[1],
            },
        ]
        charts.show(charts.equity_chart(leg_lines, f"{oos_tag}Cumulative gross return by leg"))
