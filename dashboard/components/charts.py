"""Reusable Plotly charts.

Charts are drawn with st.plotly_chart(..., theme="streamlit"), so fonts,
backgrounds, and gridlines follow the light or dark Streamlit theme. Only
series colors are fixed (one color per signal).
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import formatting as fmt

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]  # fmt: skip

# Colors for walk-forward timeline phases.
PHASE_COLORS = {"Train": "#1f77b4", "Purge gap": "#bbbbbb", "Test (out-of-sample)": "#d62728"}


def show(fig: go.Figure) -> None:
    """Render a figure with the Streamlit theme (works in light and dark mode)."""
    st.plotly_chart(fig, theme="streamlit", width="stretch")


def _layout(fig: go.Figure, title: str, y_title: str, y_format: str | None = None) -> go.Figure:
    """Shared layout: title, axis labels, legend on top, unified hover."""
    fig.update_layout(
        title=title,
        yaxis_title=y_title,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"t": 80, "b": 40},
    )
    if y_format:
        fig.update_yaxes(tickformat=y_format)
    return fig


def line_chart(
    lines: list[dict], title: str, y_title: str, y_format: str | None = None
) -> go.Figure:
    """Line chart from a list of {"series", "name", "color", optional "dash"} dicts."""
    fig = go.Figure()
    for line in lines:
        data = line["series"].dropna()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                name=line["name"],
                mode="lines",
                line={"color": line["color"], "dash": line.get("dash", "solid"), "width": 2},
            )
        )
    return _layout(fig, title, y_title, y_format)


def equity_chart(lines: list[dict], title: str) -> go.Figure:
    """Growth of $1 for each line, log scale so early and late moves compare fairly."""
    fig = line_chart(lines, title, "Growth of $1 (log scale)")
    fig.update_yaxes(type="log")
    return fig


def drawdown_chart(drawdown: pd.Series, title: str, color: str) -> go.Figure:
    """Drawdown from the running peak, shaded down to zero."""
    fig = go.Figure(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line={"color": color, "width": 1},
        )
    )
    return _layout(fig, title, "Drawdown", ".0%")


def with_zero_line(fig: go.Figure) -> go.Figure:
    """Add a dashed horizontal line at zero."""
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.7)
    return fig


def monthly_heatmap(grid: pd.DataFrame, title: str) -> go.Figure:
    """Years x months heatmap of monthly returns, red below zero and green above."""
    grid = grid.reindex(columns=range(1, 13))
    text = grid.map(lambda v: "" if pd.isna(v) else f"{v * 100:.1f}%")
    limit = float(grid.abs().max().max()) if grid.notna().any().any() else 0.01
    fig = go.Figure(
        go.Heatmap(
            z=grid.values,
            x=MONTH_NAMES,
            y=[str(year) for year in grid.index],
            text=text.values,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            zmin=-limit,
            zmax=limit,
            colorbar={"tickformat": ".0%"},
            hovertemplate="%{y} %{x}: %{text}<extra></extra>",
        )
    )
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_layout(title=title, height=max(300, 28 * len(grid) + 120))
    return fig


def bar_chart(
    labels: list[str],
    values: list[float],
    colors: list[str],
    title: str,
    y_title: str,
    y_format: str | None = None,
    text: list[str] | None = None,
) -> go.Figure:
    """Simple vertical bar chart with one color per bar."""
    fig = go.Figure(
        go.Bar(x=labels, y=values, marker_color=colors, text=text, textposition="outside")
    )
    fig.update_layout(showlegend=False)
    return _layout(fig, title, y_title, y_format)


def grouped_bar_chart(table: pd.DataFrame, title: str, y_title: str) -> go.Figure:
    """Grouped bars: index = x categories, one bar group per column."""
    fig = go.Figure()
    for column in table.columns:
        fig.add_trace(go.Bar(x=list(table.index), y=table[column].values, name=str(column)))
    fig.update_layout(barmode="group")
    return _layout(fig, title, y_title)


def histogram(values: pd.Series, mean: float, title: str, color: str) -> go.Figure:
    """Histogram of values with a vertical line at the mean."""
    fig = go.Figure(go.Histogram(x=values.dropna(), nbinsx=60, marker_color=color, opacity=0.8))
    fig.add_vline(
        x=mean,
        line_dash="dash",
        line_color="gray",
        line_width=2,
        annotation_text=f"mean {mean:.4f}",
        annotation_position="top right",
    )
    fig.add_vline(x=0, line_color="gray", line_dash="dot", opacity=0.4)
    fig.update_layout(
        title=title,
        xaxis_title="Daily IC",
        yaxis_title="Number of days",
        showlegend=False,
        bargap=0.05,
    )
    return fig


def cost_curve_chart(curve: pd.DataFrame, current_bps: float, title: str) -> go.Figure:
    """Net Sharpe vs cost, one line per signal, with the current slider cost marked."""
    lines = [
        {
            "series": curve[signal],
            "name": fmt.signal_label(signal),
            "color": fmt.signal_color(signal),
        }
        for signal in curve.columns
    ]
    fig = line_chart(lines, title, "Net Sharpe")
    fig.update_xaxes(title="Cost per unit traded (bps)")
    fig.add_vline(
        x=current_bps, line_dash="dot", line_color="gray", annotation_text=fmt.fmt_bps(current_bps)
    )
    return with_zero_line(fig)


def timeline_chart(windows: pd.DataFrame, title: str) -> go.Figure:
    """Horizontal bars per window: train period, purge gap, then the test year."""
    phases = [
        ("Train", "train_start", "train_end"),
        ("Purge gap", "train_end", "test_start"),
        ("Test (out-of-sample)", "test_start", "test_end"),
    ]
    fig = go.Figure()
    for name, start_col, end_col in phases:
        start = pd.to_datetime(windows[start_col])
        end = pd.to_datetime(windows[end_col])
        # Plotly date bars: base is the start, x is the length in milliseconds.
        length_ms = (end - start).dt.total_seconds() * 1000
        fig.add_trace(
            go.Bar(
                y=windows["window"].astype(str),
                x=length_ms,
                base=start,
                orientation="h",
                name=name,
                marker_color=PHASE_COLORS[name],
                customdata=list(
                    zip(start.dt.date.astype(str), end.dt.date.astype(str), strict=True)
                ),
                hovertemplate=name + ": %{customdata[0]} to %{customdata[1]}<extra></extra>",
            )
        )
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed", title="Test window")
    fig.update_layout(
        title=title,
        barmode="overlay",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        height=max(300, 32 * len(windows) + 140),
    )
    return fig


def weights_chart(weights: pd.DataFrame, title: str) -> go.Figure:
    """Stacked bars of each signal's fitted weight in each window."""
    fig = go.Figure()
    for signal, group in weights.groupby("signal", sort=False):
        fig.add_trace(
            go.Bar(
                x=group["window"].astype(str),
                y=group["weight"],
                name=fmt.signal_label(signal),
                marker_color=fmt.signal_color(signal),
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Test window", type="category")
    return _layout(fig, title, "Weight", ".0%")
