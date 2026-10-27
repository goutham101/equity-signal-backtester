"""Phase 1 tests: cleaning checks, wide/long round trip, and SQL vs pandas momentum."""

import numpy as np
import pandas as pd
import pytest

from backtester.data import clean, db
from backtester.data.universe import to_yfinance_ticker

# ---------- fixtures ----------


@pytest.fixture
def small_prices() -> pd.DataFrame:
    """3 tickers x 10 days with hand-placed gaps and one big move."""
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = {
        # gap of 2 days (rows 3, 4)
        "AAA": [10, 11, 12, np.nan, np.nan, 13, 14, 15, 16, 17],
        # gap of 4 days (rows 2-5), longer than the ffill limit of 2 below
        "BBB": [20, 21, np.nan, np.nan, np.nan, np.nan, 22, 23, 24, 25],
        # lists late (rows 0-3 missing) and has one +100% day
        "CCC": [np.nan, np.nan, np.nan, np.nan, 5, 5, 10, 10, 10, 10],
    }
    return pd.DataFrame(data, index=dates, dtype=float)


@pytest.fixture
def small_volume(small_prices) -> pd.DataFrame:
    """Constant volume where a price exists."""
    return small_prices.notna().astype(float) * 1000


# ---------- cleaning ----------


def test_ffill_respects_limit(small_prices):
    filled, n_filled = clean.forward_fill_limited(small_prices, max_days=2)

    # AAA's 2-day gap is fully filled with the last close before it.
    assert filled["AAA"].iloc[3] == 12
    assert filled["AAA"].iloc[4] == 12
    # BBB's 4-day gap: only the first 2 days are filled.
    assert filled["BBB"].iloc[2] == 21
    assert filled["BBB"].iloc[3] == 21
    assert np.isnan(filled["BBB"].iloc[4])
    assert np.isnan(filled["BBB"].iloc[5])
    # 2 (AAA) + 2 (BBB) cells filled
    assert n_filled == 4


def test_no_backfill(small_prices):
    filled, _ = clean.forward_fill_limited(small_prices, max_days=5)
    # CCC has no price before row 4, so rows 0-3 must stay NaN.
    assert filled["CCC"].iloc[:4].isna().all()


def test_ffill_never_uses_future_prices(small_prices):
    """Changing future prices must not change filled values in the past."""
    filled, _ = clean.forward_fill_limited(small_prices, max_days=5)
    corrupted = small_prices.copy()
    corrupted.iloc[6:] = 999.0
    filled_corrupted, _ = clean.forward_fill_limited(corrupted, max_days=5)
    pd.testing.assert_frame_equal(filled.iloc[:6], filled_corrupted.iloc[:6])


def test_big_moves_flagged_not_deleted(small_prices, small_volume):
    flagged = clean.flag_big_moves(small_prices, threshold=0.5)
    assert len(flagged) == 1
    assert flagged.iloc[0]["ticker"] == "CCC"
    assert flagged.iloc[0]["ret"] == pytest.approx(1.0)

    result = clean.clean_prices(small_prices, small_volume, min_history_days=1, max_ffill_days=2)
    assert result.report["big_moves_flagged"] == 1
    # The big move is still in the cleaned data.
    assert result.prices["CCC"].iloc[6] == 10


def test_big_move_counts_on_random_walk(synthetic_prices):
    prices = synthetic_prices.copy()
    prices.iloc[100, 0] = prices.iloc[99, 0] * 3.0  # +200%, then about -67% the next day
    prices.iloc[300, 5] = prices.iloc[299, 5] * 0.3  # -70%, then about +233% the next day
    flagged = clean.flag_big_moves(prices, threshold=0.5)
    # Each spike creates the jump and the move back.
    assert len(flagged) == 4
    assert set(flagged["ticker"]) == {"T00", "T05"}


def test_min_history_drop(small_prices, small_volume):
    # CCC has 6 real closes, AAA 8, BBB 6. With min 7, BBB and CCC drop.
    result = clean.clean_prices(small_prices, small_volume, min_history_days=7, max_ffill_days=2)
    assert result.dropped_tickers == ["BBB", "CCC"]
    assert result.report["tickers_dropped_short_history"] == 2
    assert list(result.prices.columns) == ["AAA"]
    assert list(result.volume.columns) == ["AAA"]


def test_min_history_counts_real_closes_not_filled(small_prices):
    """History is measured before forward-filling, so filled days don't count."""
    kept, dropped = clean.drop_short_history(small_prices, min_days=8)
    assert dropped == ["BBB", "CCC"]
    assert list(kept.columns) == ["AAA"]


def test_gap_report(small_prices, small_volume):
    gaps = clean.find_gaps(small_prices).set_index("ticker")
    assert gaps.loc["AAA", "longest_gap"] == 2
    assert gaps.loc["BBB", "longest_gap"] == 4
    # CCC's late listing is not a gap.
    assert gaps.loc["CCC", "missing_days"] == 0

    result = clean.clean_prices(small_prices, small_volume, min_history_days=1, max_ffill_days=2)
    assert result.report["long_gap_tickers"] == ["BBB"]
    assert result.report["missing_days_after_ffill"] == 2
    assert result.report["tickers_with_gaps_after_ffill"] == 1


def test_nonpositive_prices_removed_and_counted(small_prices, small_volume):
    prices = small_prices.copy()
    prices.iloc[0, 0] = 0.0
    result = clean.clean_prices(prices, small_volume, min_history_days=1, max_ffill_days=2)
    assert result.report["nonpositive_prices_removed"] == 1
    assert np.isnan(result.prices["AAA"].iloc[0])


def test_liquidity_mask_uses_past_only(synthetic_prices):
    rng = np.random.default_rng(1)
    volume = pd.DataFrame(
        rng.uniform(1e5, 1e6, synthetic_prices.shape),
        index=synthetic_prices.index,
        columns=synthetic_prices.columns,
    )
    liquid = clean.liquidity_mask(synthetic_prices, volume, window=60)
    # Not enough history before day 60.
    assert not liquid.iloc[:59].any().any()

    cutoff = 400
    corrupted = volume.copy()
    corrupted.iloc[cutoff + 1 :] = 1.0
    liquid_corrupted = clean.liquidity_mask(synthetic_prices, corrupted, window=60)
    pd.testing.assert_frame_equal(liquid.iloc[: cutoff + 1], liquid_corrupted.iloc[: cutoff + 1])


def test_yfinance_ticker_format():
    assert to_yfinance_ticker("BRK.B") == "BRK-B"
    assert to_yfinance_ticker("aapl") == "AAPL"


# ---------- wide <-> long ----------


def test_wide_long_round_trip(synthetic_prices):
    wide = synthetic_prices.iloc[:50, :5].copy()
    wide.iloc[3, 2] = np.nan
    long = db.wide_to_long(wide, "adj_close")
    assert list(long.columns) == ["ticker", "date", "adj_close"]
    assert len(long) == 50 * 5 - 1

    back = db.long_to_wide(long, "adj_close")
    pd.testing.assert_frame_equal(
        back, wide, check_freq=False, check_names=False, check_index_type=False
    )


# ---------- SQL vs pandas (needs Postgres) ----------

TEST_TICKERS = ["TEST_A", "TEST_B"]


@pytest.fixture
def engine():
    """Postgres engine, or skip if the database is not reachable."""
    try:
        eng = db.get_engine()
        with eng.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except Exception as exc:
        pytest.skip(f"Postgres not reachable: {exc}")
    yield eng
    eng.dispose()


@pytest.mark.db
def test_sql_momentum_matches_pandas(engine, synthetic_prices):
    # Round to 6 decimals because the prices column is NUMERIC(14,6).
    prices = synthetic_prices.iloc[:400, :2].round(6)
    prices.columns = TEST_TICKERS
    volume = pd.DataFrame(1000.0, index=prices.index, columns=prices.columns)
    meta = pd.DataFrame({"ticker": TEST_TICKERS, "name": "test", "sector": "test"})

    try:
        db.delete_tickers(engine, TEST_TICKERS)
        db.upsert_tickers(engine, meta)
        db.upsert_prices(engine, db.prices_to_long(prices, volume))
        db.run_sql_file(engine)

        # Read back prices as wide and check the round trip through Postgres.
        stored = db.read_prices(engine, tickers=TEST_TICKERS)
        pd.testing.assert_frame_equal(
            stored, prices, check_freq=False, check_names=False, check_index_type=False
        )

        features = db.read_features(engine, tickers=TEST_TICKERS)
        sql_mom = db.long_to_wide(features, "mom_12_1")
        pandas_mom = prices.shift(21) / prices.shift(252) - 1

        assert sql_mom.notna().sum().sum() == pandas_mom.notna().sum().sum()
        pd.testing.assert_frame_equal(
            sql_mom,
            pandas_mom,
            check_freq=False,
            check_names=False,
            check_index_type=False,
            rtol=1e-9,
        )
    finally:
        db.delete_tickers(engine, TEST_TICKERS)


def test_blank_known_bad_history():
    """Prices before the known-good date are blanked; later prices are kept."""
    from backtester.data.clean import blank_known_bad_history

    dates = pd.bdate_range("2016-06-29", periods=6)
    prices = pd.DataFrame({"BAD": [40.0, 41.0, 42.0, 67.0, 68.0, 69.0], "OK": 1.0}, index=dates)
    volume = prices * 0 + 100
    known = {"BAD": (str(dates[3].date()), "test"), "MISSING": ("2016-01-01", "not in data")}
    new_prices, new_volume, n_blanked = blank_known_bad_history(prices, volume, known)
    assert n_blanked == 3
    assert new_prices["BAD"].iloc[:3].isna().all()
    assert new_prices["BAD"].iloc[3:].tolist() == [67.0, 68.0, 69.0]
    assert new_prices["OK"].notna().all()
    assert new_volume["BAD"].iloc[:3].isna().all()
