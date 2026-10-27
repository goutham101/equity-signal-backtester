"""Look and feel: global CSS, the page hero banner, and section headings.

Colors are mostly semi-transparent (rgba) so the same CSS reads well on
Streamlit's light and dark themes. Only the hero banner has a fixed
background, and its text is always white on it.
"""

import html

import streamlit as st

ACCENT = "#4F46E5"  # indigo-600

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Set the font on containers and let it inherit. Targeting every element would
   also override the icon font, which turns icons into their names as text. */
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
button, input, select, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1280px; }
[data-testid="stHeader"] { background: transparent; }

/* Headings */
h1, h2, h3 { letter-spacing: -0.02em; font-weight: 700 !important; }
h3 { font-size: 1.15rem !important; margin-top: 1.6rem !important; }

/* Hero banner */
.ebt-hero {
    position: relative; overflow: hidden;
    background: linear-gradient(120deg, #4338CA 0%, #6366F1 45%, #8B5CF6 75%, #EC4899 120%);
    border-radius: 22px; padding: 30px 34px 28px; margin: 0 0 26px 0; color: #fff;
    box-shadow: 0 18px 40px -18px rgba(79, 70, 229, 0.55);
}
.ebt-hero::after {
    content: ""; position: absolute; right: -60px; top: -80px; width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(255,255,255,0.22) 0%, rgba(255,255,255,0) 65%);
}
.ebt-kicker {
    text-transform: uppercase; letter-spacing: 0.14em; font-size: 0.72rem; font-weight: 600;
    opacity: 0.85; margin-bottom: 6px;
}
.ebt-hero h1 {
    color: #fff !important; font-size: 2.1rem !important; font-weight: 800 !important;
    margin: 0 0 6px 0 !important; padding: 0 !important; line-height: 1.15;
}
.ebt-hero p { color: rgba(255,255,255,0.9); font-size: 1.02rem; margin: 0; max-width: 760px; }
.ebt-chips { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
.ebt-chip {
    background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.28);
    border-radius: 999px; padding: 4px 12px; font-size: 0.8rem; font-weight: 500;
    backdrop-filter: blur(6px);
}

/* Section heading with a short accent bar */
.ebt-section { margin: 30px 0 4px 0; display: flex; align-items: center; gap: 10px; }
.ebt-section span.bar { width: 4px; height: 20px; border-radius: 4px; background: #6366F1; }
.ebt-section span.title { font-size: 1.12rem; font-weight: 700; letter-spacing: -0.01em; }
.ebt-section-caption { opacity: 0.65; font-size: 0.88rem; margin: 2px 0 12px 14px; }

/* Metric tiles as cards */
[data-testid="stMetric"] {
    background: rgba(127, 127, 127, 0.04);
    border: 1px solid rgba(127, 127, 127, 0.16);
    border-radius: 16px; padding: 16px 18px 14px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 8px 24px -12px rgba(16, 24, 40, 0.12);
    transition: transform .15s ease, box-shadow .15s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 2px 4px rgba(16, 24, 40, 0.06), 0 14px 30px -12px rgba(79, 70, 229, 0.28);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important; font-weight: 600 !important; text-transform: uppercase;
    letter-spacing: 0.06em; opacity: 0.7;
}
[data-testid="stMetricValue"] { font-size: clamp(1.4rem, 2vw, 1.9rem) !important;
    font-weight: 750 !important;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; font-weight: 500; }

[data-testid="stHorizontalBlock"] { margin-bottom: 4px; }

/* Charts and tables in cards */
[data-testid="stPlotlyChart"] {
    background: rgba(127, 127, 127, 0.03);
    border: 1px solid rgba(127, 127, 127, 0.14);
    border-radius: 18px; padding: 10px 12px; box-sizing: border-box;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 10px 28px -16px rgba(16, 24, 40, 0.14);
}
/* No padding on tables: the grid is sized to the full width and would overflow. */
[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 10px 28px -16px rgba(16, 24, 40, 0.14);
}

/* Bordered containers (info boxes etc.) */
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px !important; }
[data-testid="stAlert"] { border-radius: 14px; }

/* Sidebar */
[data-testid="stSidebar"] { border-right: 1px solid rgba(127, 127, 127, 0.14); }
[data-testid="stSidebarNav"] a { border-radius: 10px; font-weight: 500; }
.ebt-brand { display: flex; align-items: center; gap: 10px; margin: 2px 0 6px 0; }
.ebt-logo {
    width: 34px; height: 34px; border-radius: 10px; flex-shrink: 0;
    background: linear-gradient(135deg, #4F46E5, #8B5CF6 60%, #EC4899);
    display: flex; align-items: center; justify-content: center;
    color: #fff; font-weight: 800; font-size: 0.95rem;
    box-shadow: 0 6px 16px -6px rgba(99, 102, 241, 0.7);
}
.ebt-brand-name { font-weight: 700; font-size: 1rem; line-height: 1.15; }
.ebt-brand-sub { font-size: 0.75rem; opacity: 0.6; }

/* Small badge, e.g. "Out-of-sample" */
.ebt-badge {
    display: inline-block; background: rgba(99, 102, 241, 0.12); color: #6366F1;
    border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 999px;
    padding: 2px 10px; font-size: 0.75rem; font-weight: 600; margin-left: 6px;
}
</style>
"""


def inject_css() -> None:
    """Add the global stylesheet. Call once per page run."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str = "", kicker: str = "", chips: list[str] | None = None) -> None:
    """Gradient banner at the top of a page. All text is escaped."""
    chip_html = "".join(f'<span class="ebt-chip">{html.escape(c)}</span>' for c in chips or [])
    parts = ['<div class="ebt-hero">']
    if kicker:
        parts.append(f'<div class="ebt-kicker">{html.escape(kicker)}</div>')
    parts.append(f"<h1>{html.escape(title)}</h1>")
    if subtitle:
        parts.append(f"<p>{html.escape(subtitle)}</p>")
    if chip_html:
        parts.append(f'<div class="ebt-chips">{chip_html}</div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


def section(title: str, caption: str = "", badge: str = "") -> None:
    """Section heading with an accent bar, optional badge and caption."""
    badge_html = f'<span class="ebt-badge">{html.escape(badge)}</span>' if badge else ""
    st.markdown(
        f'<div class="ebt-section"><span class="bar"></span>'
        f'<span class="title">{html.escape(title)}</span>{badge_html}</div>',
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(
            f'<div class="ebt-section-caption">{html.escape(caption)}</div>',
            unsafe_allow_html=True,
        )


def brand(name: str, tagline: str) -> None:
    """Logo square plus app name, for the top of the sidebar."""
    st.markdown(
        f'<div class="ebt-brand"><div class="ebt-logo">ES</div><div>'
        f'<div class="ebt-brand-name">{html.escape(name)}</div>'
        f'<div class="ebt-brand-sub">{html.escape(tagline)}</div></div></div>',
        unsafe_allow_html=True,
    )
