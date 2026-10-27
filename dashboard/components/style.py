"""Look and feel: design tokens, global CSS, page header, and section headings.

Design system: design-system/equity-signal-backtester/MASTER.md, generated with
the ui-ux-pro-max skill ("Data-Dense Dashboard": navy and slate surfaces,
amber highlights, Fira Sans for text and Fira Code for numbers).

Colors are CSS variables with a light set and a dark set. The dark set kicks
in through prefers-color-scheme, which matches Streamlit because the app
follows the system theme (.streamlit/config.toml defines both). Every
text/background pair is at least 4.5:1, and data colors at least 3:1.
"""

import html

import streamlit as st

_CSS = """
<style>
:root {
    --ebt-surface: #FFFFFF;
    --ebt-surface-2: #F1F5F9;
    --ebt-border: #E2E8F0;
    --ebt-border-strong: #CBD5E1;
    --ebt-text: #0F172A;
    --ebt-muted: #64748B;
    --ebt-primary: #1E40AF;
    --ebt-primary-soft: rgba(30, 64, 175, 0.08);
    --ebt-accent: #B45309;
    --ebt-accent-soft: rgba(217, 119, 6, 0.12);
    --ebt-up: #047857;
    --ebt-down: #B91C1C;
    --ebt-mono: 'Fira Code', ui-monospace, SFMono-Regular, Menlo, monospace;
    --ebt-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
}
@media (prefers-color-scheme: dark) {
    :root {
        --ebt-surface: #1E293B;
        --ebt-surface-2: #172033;
        --ebt-border: #334155;
        --ebt-border-strong: #475569;
        --ebt-text: #E2E8F0;
        --ebt-muted: #94A3B8;
        --ebt-primary: #60A5FA;
        --ebt-primary-soft: rgba(96, 165, 250, 0.12);
        --ebt-accent: #F59E0B;
        --ebt-accent-soft: rgba(245, 158, 11, 0.14);
        --ebt-up: #34D399;
        --ebt-down: #F87171;
        --ebt-shadow: 0 1px 2px rgba(0, 0, 0, 0.35);
    }
}

/* Dense layout: 8px rhythm, wide but bounded content. */
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1360px; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stVerticalBlock"] { gap: 0.75rem; }

/* Page header: eyebrow, title, one-line summary, status pills. No gradient. */
.ebt-header {
    border-bottom: 1px solid var(--ebt-border);
    padding: 4px 0 16px 0; margin: 0 0 8px 0;
}
.ebt-eyebrow {
    font-family: var(--ebt-mono); font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--ebt-accent);
    margin-bottom: 6px;
}
.ebt-header h1 {
    font-size: 1.75rem !important; font-weight: 600 !important; letter-spacing: -0.01em;
    margin: 0 0 4px 0 !important; padding: 0 !important; line-height: 1.2;
}
.ebt-header p { color: var(--ebt-muted); font-size: 0.95rem; margin: 0; max-width: 820px;
    line-height: 1.5; }
.ebt-pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.ebt-pill {
    display: inline-flex; align-items: baseline; gap: 6px;
    background: var(--ebt-surface); border: 1px solid var(--ebt-border);
    border-radius: 6px; padding: 4px 10px; font-size: 0.78rem;
}
.ebt-pill .k { color: var(--ebt-muted); text-transform: uppercase; letter-spacing: 0.06em;
    font-size: 0.68rem; font-weight: 600; }
.ebt-pill .v { font-family: var(--ebt-mono); font-weight: 500;
    font-variant-numeric: tabular-nums; }

/* Section heading: small square marker, title, optional amber badge. */
.ebt-section { margin: 22px 0 2px 0; display: flex; align-items: center; gap: 8px; }
.ebt-section .mark { width: 8px; height: 8px; border-radius: 2px; background: var(--ebt-primary); }
.ebt-section .title { font-size: 1rem; font-weight: 600; }
.ebt-section-caption { color: var(--ebt-muted); font-size: 0.85rem; margin: 2px 0 6px 16px;
    line-height: 1.5; }
.ebt-badge {
    font-family: var(--ebt-mono); font-size: 0.7rem; font-weight: 500;
    color: var(--ebt-accent); background: var(--ebt-accent-soft);
    border: 1px solid var(--ebt-accent-soft); border-radius: 4px; padding: 1px 7px;
}

/* KPI tiles: compact, bordered, numbers in Fira Code with tabular figures. */
[data-testid="stMetric"] {
    background: var(--ebt-surface); border: 1px solid var(--ebt-border);
    border-radius: 8px; padding: 12px 14px; box-shadow: var(--ebt-shadow);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important; font-weight: 600 !important; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--ebt-muted);
}
[data-testid="stMetricValue"] {
    font-family: var(--ebt-mono); font-size: clamp(1.25rem, 1.8vw, 1.6rem) !important;
    font-weight: 600 !important; font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
}
[data-testid="stMetricDelta"] { font-family: var(--ebt-mono); font-size: 0.75rem !important; }

/* Charts and tables: flat bordered panels. */
[data-testid="stPlotlyChart"] {
    background: var(--ebt-surface); border: 1px solid var(--ebt-border);
    border-radius: 8px; padding: 8px 10px; box-sizing: border-box; box-shadow: var(--ebt-shadow);
}
[data-testid="stDataFrame"] {
    border: 1px solid var(--ebt-border); border-radius: 8px; overflow: hidden;
}
[data-testid="stAlert"] { border-radius: 8px; }

/* Hover states only for motion-tolerant users (150 ms, colour only, no layout shift). */
@media (prefers-reduced-motion: no-preference) {
    [data-testid="stMetric"], [data-testid="stPlotlyChart"] {
        transition: border-color 150ms ease-out;
    }
}
[data-testid="stMetric"]:hover, [data-testid="stPlotlyChart"]:hover {
    border-color: var(--ebt-border-strong);
}

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--ebt-border); }
[data-testid="stSidebarNav"] a { border-radius: 6px; font-weight: 500; }
.ebt-brand { display: flex; align-items: center; gap: 10px; margin: 2px 0 6px 0; }
.ebt-logo {
    width: 32px; height: 32px; border-radius: 6px; flex-shrink: 0;
    background: var(--ebt-primary); color: #FFFFFF;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--ebt-mono); font-weight: 600; font-size: 0.8rem;
}
.ebt-brand-name { font-weight: 600; font-size: 0.98rem; line-height: 1.15; }
.ebt-brand-sub { font-family: var(--ebt-mono); font-size: 0.7rem; color: var(--ebt-muted); }

/* Inline code reads as data. */
code { font-family: var(--ebt-mono) !important; }
</style>
"""


def inject_css() -> None:
    """Add the global stylesheet. Call once per page run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(
    title: str,
    subtitle: str = "",
    eyebrow: str = "",
    pills: list[tuple[str, str]] | None = None,
) -> None:
    """Compact page header: eyebrow, title, summary, and key/value status pills.

    pills: list of (label, value), e.g. [("Universe", "498 stocks")].
    All text is escaped.
    """
    parts = ['<div class="ebt-header">']
    if eyebrow:
        parts.append(f'<div class="ebt-eyebrow">{html.escape(eyebrow)}</div>')
    parts.append(f"<h1>{html.escape(title)}</h1>")
    if subtitle:
        parts.append(f"<p>{html.escape(subtitle)}</p>")
    if pills:
        items = "".join(
            f'<span class="ebt-pill"><span class="k">{html.escape(k)}</span>'
            f'<span class="v">{html.escape(v)}</span></span>'
            for k, v in pills
        )
        parts.append(f'<div class="ebt-pills">{items}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title: str, caption: str = "", badge: str = "") -> None:
    """Section heading with a small marker, optional amber badge and caption."""
    badge_html = f'<span class="ebt-badge">{html.escape(badge)}</span>' if badge else ""
    st.markdown(
        f'<div class="ebt-section"><span class="mark"></span>'
        f'<span class="title">{html.escape(title)}</span>{badge_html}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(
            f'<div class="ebt-section-caption">{html.escape(caption)}</div>',
            unsafe_allow_html=True,
        )


def brand(name: str, tagline: str) -> None:
    """Monogram plus app name, for the top of the sidebar."""
    st.markdown(
        f'<div class="ebt-brand"><div class="ebt-logo">ESB</div><div>'
        f'<div class="ebt-brand-name">{html.escape(name)}</div>'
        f'<div class="ebt-brand-sub">{html.escape(tagline)}</div></div></div>',
        unsafe_allow_html=True,
    )
