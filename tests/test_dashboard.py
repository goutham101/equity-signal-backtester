"""Smoke tests for the Streamlit dashboard, using synthetic contract-shaped data.

AppTest runs each page in-process, so DASHBOARD_DATA_DIR points the app at a
temp folder built by tests/dashboard_fixtures.py.
"""

import math
import sys
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from dashboard_fixtures import build_dashboard_data  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard"
APP = str(DASHBOARD_DIR / "app.py")
PAGES = [
    "pages/1_Signal_Explorer.py",
    "pages/2_Signal_Quality.py",
    "pages/3_Costs_and_Turnover.py",
    "pages/4_Walk_Forward.py",
    "pages/5_Methodology.py",
]
TIMEOUT = 60

# Same import path the app uses (Streamlit puts dashboard/ on sys.path).
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from components import formatting as fmt  # noqa: E402


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    """Synthetic dashboard data, built once for this module."""
    return build_dashboard_data(tmp_path_factory.mktemp("dashboard_data"))


@pytest.fixture
def app(data_dir, monkeypatch) -> AppTest:
    """An AppTest for the Overview page, pointed at the synthetic data."""
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(data_dir))
    return AppTest.from_file(APP, default_timeout=TIMEOUT)


def _run_page(app: AppTest, page: str | None = None) -> AppTest:
    """Run the app (or switch to a page and run it) and fail on any exception."""
    if page is not None:
        app.switch_page(page)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _info_text(app: AppTest) -> str:
    """All st.info messages on the current page, joined."""
    return " ".join(str(el.value) for el in app.info)


# --- Pages load ---


def test_overview_loads(app):
    _run_page(app)
    assert any("out-of-sample" in str(h.value).lower() for h in app.subheader)
    assert len(app.dataframe) == 1


@pytest.mark.parametrize("page", PAGES)
def test_every_page_loads(app, page):
    _run_page(app)
    _run_page(app, page)


def test_missing_data_shows_error(monkeypatch, tmp_path):
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(tmp_path / "nothing_here"))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert any("06_export_dashboard_data" in str(e.value) for e in at.error)


# --- Sidebar controls ---


def test_walk_forward_weekly_shows_info(app):
    _run_page(app, PAGES[0])
    app.sidebar.selectbox(key="signal").set_value("walk_forward")
    app.sidebar.radio(key="rebalance").set_value("weekly")
    _run_page(app)
    assert "only exists out-of-sample" in _info_text(app)


def test_walk_forward_long_short_monthly_shows_data(app):
    _run_page(app, PAGES[0])
    app.sidebar.selectbox(key="signal").set_value("walk_forward")
    _run_page(app)
    assert "only exists out-of-sample" not in _info_text(app)
    assert len(app.metric) >= 6


def test_cost_slider_changes_net_sharpe(app):
    _run_page(app, PAGES[0])
    before = app.metric[0].value
    app.sidebar.slider(key="cost_bps").set_value(50)
    _run_page(app)
    assert app.metric[0].value != before


@pytest.mark.parametrize("page", PAGES)
def test_pages_with_long_only_weekly_and_high_cost(app, page):
    _run_page(app)
    app.sidebar.radio(key="portfolio").set_value("long_only")
    app.sidebar.radio(key="rebalance").set_value("weekly")
    app.sidebar.slider(key="cost_bps").set_value(35)
    _run_page(app)
    _run_page(app, page)


def test_ic_page_walk_forward_and_horizons(app):
    _run_page(app, PAGES[1])
    for horizon in [5, 63]:
        app.radio(key="ic_horizon").set_value(horizon)
        _run_page(app)
    app.sidebar.selectbox(key="signal").set_value("walk_forward")
    _run_page(app)
    assert "isn't defined" in _info_text(app)


def test_explorer_toggles_and_year_range(app):
    _run_page(app, PAGES[0])
    for toggle in app.toggle:
        toggle.set_value(not toggle.value)
    _run_page(app)
    years = app.select_slider[0]
    lo, hi = years.value
    years.set_value((lo + 1, hi))
    _run_page(app)


def test_selection_survives_page_switch(app):
    _run_page(app)
    app.sidebar.slider(key="cost_bps").set_value(25)
    app.sidebar.selectbox(key="signal").set_value("reversal")
    _run_page(app)
    _run_page(app, PAGES[2])
    assert app.sidebar.slider(key="cost_bps").value == 25
    assert app.sidebar.selectbox(key="signal").value == "reversal"


def test_failing_lookahead_and_linkedin(monkeypatch, tmp_path):
    folder = build_dashboard_data(
        tmp_path, lookahead_status="failing", linkedin="https://www.linkedin.com/in/example"
    )
    monkeypatch.setenv("DASHBOARD_DATA_DIR", str(folder))
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    _run_page(at, PAGES[4])
    assert any("Failing" in str(e.value) for e in at.error)
    assert any("LinkedIn" in str(m.value) for m in at.sidebar.markdown)


def test_linkedin_hidden_when_null(app):
    _run_page(app)
    assert not any("LinkedIn" in str(m.value) for m in app.sidebar.markdown)


# --- Formatting helpers ---


def test_fmt_pct():
    assert fmt.fmt_pct(0.124) == "12.4%"
    assert fmt.fmt_pct(-0.51) == "-51.0%"
    assert fmt.fmt_pct(0.01234, 2) == "1.23%"
    assert fmt.fmt_pct(math.nan) == "n/a"
    assert fmt.fmt_pct(None) == "n/a"


def test_fmt_sharpe():
    assert fmt.fmt_sharpe(0.1312) == "0.13"
    assert fmt.fmt_sharpe(-1.2912) == "-1.29"
    assert fmt.fmt_sharpe(float("nan")) == "n/a"


def test_fmt_bps():
    assert fmt.fmt_bps(10) == "10 bps"
    assert fmt.fmt_bps(3.24, 1) == "3.2 bps"
    assert fmt.fmt_bps(math.nan) == "n/a"


def test_fmt_ic_turnover_number():
    assert fmt.fmt_ic(0.01041) == "0.0104"
    assert fmt.fmt_turnover(11.34) == "11.3×"
    assert fmt.fmt_number(3.184) == "3.18"
    assert fmt.fmt_turnover(None) == "n/a"


def test_signal_labels_and_colors_match_contract():
    assert fmt.signal_label("low_vol") == "Low Vol"
    assert fmt.signal_label("walk_forward") == "Walk-Forward Combo"
    assert fmt.signal_color("momentum") == "#1f77b4"
    assert fmt.signal_color("benchmark") == "#bcbd22"
    assert len(set(fmt.SIGNAL_COLORS.values())) == len(fmt.SIGNAL_COLORS)
    assert [fmt.signal_label(s) for s in fmt.SELECTOR_SIGNALS] == [
        "Momentum",
        "Reversal",
        "Low Vol",
        "Equal-Weight Combo",
        "Walk-Forward Combo",
        "Random Baseline",
    ]
