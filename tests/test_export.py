"""Tests for the dashboard export helpers and the shared out-of-sample cut. Synthetic data only."""

import numpy as np
import pandas as pd
import pytest

from backtester.backtest.walk_forward import cut_to_oos, make_windows, oos_dates, windows_table
from backtester.reporting import export

README = """# Title

Intro text.

## Key findings

- **Bold one.** First finding with `code`.
- Second finding
  wrapped onto a second line.

Some trailing paragraph.

## Limitations

- **Survivorship bias.** Today's members only.
- No borrow costs.

## What I'd do next

- Not a limitation.
"""


def engine_frame() -> pd.DataFrame:
    """Small engine-like frame: 3 warm-up days with no trades, then 4 active days."""
    dates = pd.bdate_range("2020-01-01", periods=7)
    return pd.DataFrame(
        {
            "gross": [0, 0, 0, 0.01, -0.02, 0.0, 0.03],
            "costs": [0, 0, 0, 0.0002, 0, 0, 0],
            "net": [0, 0, 0, 0.0098, -0.02, 0.0, 0.03],
            "turnover": [0, 0, 0, 2.0, 0, 0, 0],
            "gross_long": [0, 0, 0, 0.02, -0.01, 0.01, 0.02],
            "gross_short": [0, 0, 0, -0.01, -0.01, -0.01, 0.01],
        },
        index=pd.DatetimeIndex(dates, name="date"),
    )


# ---------- README parsing ----------


def test_readme_bullets_key_findings_keep_markdown():
    bullets = export.readme_bullets(README, "Key findings")
    assert bullets == [
        "**Bold one.** First finding with `code`.",
        "Second finding wrapped onto a second line.",
    ]


def test_readme_bullets_stop_at_next_section():
    bullets = export.readme_bullets(README, "Limitations")
    assert bullets == ["**Survivorship bias.** Today's members only.", "No borrow costs."]


def test_readme_bullets_missing_section_raises():
    with pytest.raises(ValueError):
        export.readme_bullets(README, "Nope")


def test_parse_pytest_passed():
    assert export.parse_pytest_passed("......\n22 passed in 1.23s\n") == 22
    assert export.parse_pytest_passed("1 failed, 21 passed in 2s") == 21
    assert export.parse_pytest_passed("no tests ran") == 0


# ---------- long-format reshaping ----------


def test_frame_to_long_trims_warmup_and_drops_net():
    long = export.frame_to_long(engine_frame(), "momentum", "long_short", "monthly", "full")
    assert list(long.columns) == export.KEY_COLUMNS + export.RETURN_COLUMNS
    assert len(long) == 4
    assert long["date"].iloc[0] == pd.Timestamp("2020-01-06")
    assert long["date"].dtype == "datetime64[ns]"
    assert (long["signal"] == "momentum").all()
    assert (long["period"] == "full").all()
    assert long["gross"].tolist() == [0.01, -0.02, 0.0, 0.03]


def test_frame_to_long_never_trades_is_empty():
    frame = engine_frame()
    frame["turnover"] = 0.0
    long = export.frame_to_long(frame, "random", "long_only", "weekly", "full")
    assert len(long) == 0


def test_ic_to_long():
    ic = pd.Series([0.1, -0.05], index=pd.bdate_range("2021-01-01", periods=2))
    long = export.ic_to_long(ic, "reversal", 21)
    assert list(long.columns) == ["date", "signal", "horizon", "ic"]
    assert long["ic"].tolist() == [0.1, -0.05]
    assert (long["horizon"] == 21).all()


def test_quintiles_to_long_int_quintiles():
    q = pd.Series([0.01, 0.02, 0.03], index=["Q1", "Q2", "Q3"])
    long = export.quintiles_to_long(q, "low_vol", 5)
    assert long["quintile"].tolist() == [1, 2, 3]
    assert long["avg_fwd_return"].tolist() == [0.01, 0.02, 0.03]


def test_weights_to_long():
    weights = pd.DataFrame(
        {"momentum": [0.7, 0.0], "reversal": [0.3, 1.0]},
        index=pd.Index(["2015", "2016"], name="window"),
    )
    long = export.weights_to_long(weights)
    assert len(long) == 4
    row = long[(long["window"] == "2016") & (long["signal"] == "reversal")]
    assert row["weight"].iloc[0] == 1.0


def test_summary_row_nan_ic_for_walk_forward():
    stats = {"sharpe_net": -0.2, "ann_return_net": -0.03, "ann_vol_net": 0.15,
             "max_drawdown_net": -0.5, "ann_turnover": 30.0, "sharpe_gross": 0.0}
    row = export.summary_row("walk_forward", stats, np.nan)
    assert np.isnan(row["mean_ic_21d"])
    assert row["sharpe_net"] == -0.2
    assert "sharpe_gross" not in row


# ---------- out-of-sample cut ----------


@pytest.fixture
def years_index() -> pd.DatetimeIndex:
    """Business days over 8 calendar years, 2010-2017."""
    return pd.bdate_range("2010-01-01", "2017-12-31")


def test_oos_dates_cover_only_test_windows(years_index):
    windows = make_windows(years_index, train_years=5, test_years=1)
    in_test = oos_dates(windows, years_index)
    # Test years are 2015, 2016, 2017.
    assert not in_test[years_index.year < 2015].any()
    assert in_test[years_index.year >= 2015].all()


def test_cut_to_oos_blanks_outside_and_keeps_inside(years_index):
    rng = np.random.default_rng(0)
    signal = pd.DataFrame(rng.random((len(years_index), 4)), index=years_index,
                          columns=["A", "B", "C", "D"])
    windows = make_windows(years_index, train_years=5, test_years=1)
    cut = cut_to_oos(signal, oos_dates(windows, years_index))

    before = years_index < windows[0]["test_start"]
    assert cut[before].isna().all().all()
    pd.testing.assert_frame_equal(cut[~before], signal[~before])
    # Original is not modified.
    assert signal.notna().all().all()


def test_windows_table_to_frame(years_index):
    windows = make_windows(years_index, train_years=5, test_years=1)
    frame = export.windows_to_frame(windows_table(windows))
    assert list(frame.columns) == ["window", "train_start", "train_end", "test_start", "test_end"]
    assert frame["window"].tolist() == ["2015", "2016", "2017"]
    assert frame["test_start"].dtype == "datetime64[ns]"
