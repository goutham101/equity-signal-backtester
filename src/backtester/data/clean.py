"""Data quality checks for wide price data (index=date, columns=ticker).

Every check reports counts. Nothing is dropped silently, and nothing is
back-filled (back-filling copies future prices into the past).
"""

from dataclasses import dataclass, field

import pandas as pd

# Tickers whose adjusted prices are wrong before a date, found by inspecting
# flagged big moves by hand. Prices before the date are blanked, so the
# ticker's history starts on that date. We don't try to invent the true return.
KNOWN_BAD_HISTORY = {
    # Fortive spin-off: Yahoo's adjustment turns the real drop into a fake
    # +61% jump (41.94 -> 67.61) on 2016-07-05.
    "DHR": ("2016-07-05", "bad Fortive spin-off adjustment in Yahoo data"),
}


@dataclass
class CleanResult:
    """Cleaned data plus everything the checks found."""

    prices: pd.DataFrame
    volume: pd.DataFrame
    report: dict
    big_moves: pd.DataFrame
    gaps: pd.DataFrame
    liquid: pd.DataFrame | None = None
    dropped_tickers: list[str] = field(default_factory=list)


def remove_nonpositive(prices: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Set prices <= 0 to NaN (they are bad data). Returns prices and the count."""
    bad = prices <= 0
    count = int(bad.sum().sum())
    return prices.mask(bad), count


def flag_big_moves(prices: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """List every daily return with |return| > threshold. Nothing is deleted.

    Return on date d = close d / previous valid close - 1 (close d-1 -> close d).
    Output columns: date, ticker, ret.
    """
    returns = prices.pct_change(fill_method=None)
    big = returns.where(returns.abs() > threshold)

    flagged = big.stack().dropna().reset_index()
    flagged.columns = ["date", "ticker", "ret"]
    return flagged.sort_values(["ticker", "date"]).reset_index(drop=True)


def longest_gap(series: pd.Series) -> int:
    """Longest run of missing values between a ticker's first and last price."""
    first = series.first_valid_index()
    last = series.last_valid_index()
    if first is None:
        return 0

    missing = series.loc[first:last].isna()
    longest = 0
    current = 0
    for is_missing in missing:
        if is_missing:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def find_gaps(prices: pd.DataFrame) -> pd.DataFrame:
    """Per ticker: missing days inside its price history and the longest gap.

    Only counts dates between the ticker's first and last valid close, so a
    stock that listed in 2015 is not "missing" 2010-2014.
    """
    rows = []
    for ticker in prices.columns:
        series = prices[ticker]
        first = series.first_valid_index()
        last = series.last_valid_index()
        if first is None:
            inside_missing = 0
        else:
            inside_missing = int(series.loc[first:last].isna().sum())
        rows.append(
            {"ticker": ticker, "missing_days": inside_missing, "longest_gap": longest_gap(series)}
        )
    return pd.DataFrame(rows, columns=["ticker", "missing_days", "longest_gap"])


def forward_fill_limited(prices: pd.DataFrame, max_days: int) -> tuple[pd.DataFrame, int]:
    """Forward-fill missing prices for at most max_days in a row. Never back-fills.

    A filled value on date d is the last close before d, so it only uses the past.
    Returns filled prices and the number of cells filled.
    """
    filled = prices.ffill(limit=max_days)
    n_filled = int((filled.notna() & prices.isna()).sum().sum())
    return filled, n_filled


def history_lengths(prices: pd.DataFrame) -> pd.Series:
    """Number of real (not filled) closes per ticker."""
    return prices.notna().sum()


def drop_short_history(prices: pd.DataFrame, min_days: int) -> tuple[pd.DataFrame, list[str]]:
    """Drop tickers with fewer than min_days real closes. Returns kept prices and dropped list."""
    lengths = history_lengths(prices)
    dropped = sorted(lengths[lengths < min_days].index.tolist())
    return prices.drop(columns=dropped), dropped


def liquidity_mask(
    prices: pd.DataFrame, volume: pd.DataFrame, window: int = 60, bottom_pct: float = 0.10
) -> pd.DataFrame:
    """True where a stock is liquid enough to trade on that date.

    Dollar volume = close d * volume d. The average uses the window ending at
    close t (past data only, no centering). On each date, stocks in the bottom
    bottom_pct of average dollar volume across tickers are marked False.
    Dates without a full window are False.
    """
    dollar_volume = prices * volume.reindex_like(prices)
    avg_dollar_volume = dollar_volume.rolling(window, min_periods=window).mean()
    pct_rank = avg_dollar_volume.rank(axis=1, pct=True)
    return pct_rank > bottom_pct


def blank_known_bad_history(
    prices: pd.DataFrame, volume: pd.DataFrame, known: dict = KNOWN_BAD_HISTORY
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Blank prices and volume before each ticker's known-good start date.

    Returns the new prices, new volume, and how many price cells were blanked.
    """
    prices = prices.copy()
    volume = volume.copy()
    n_blanked = 0
    for ticker, (good_from, _reason) in known.items():
        if ticker not in prices.columns:
            continue
        before = prices.index < pd.Timestamp(good_from)
        n_blanked += int(prices.loc[before, ticker].notna().sum())
        prices.loc[before, ticker] = float("nan")
        volume.loc[before, ticker] = float("nan")
    return prices, volume, n_blanked


def clean_prices(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    min_history_days: int = 504,
    max_ffill_days: int = 5,
    big_move_threshold: float = 0.5,
    use_liquidity_filter: bool = False,
) -> CleanResult:
    """Run every check from spec 6.5 and return cleaned data plus a report.

    Order: remove bad prices -> blank known-bad history -> flag big moves and gaps on raw data ->
    drop short histories -> forward-fill (limited) -> count leftover gaps.
    Uses closes only up to each date (ffill only, never bfill).
    """
    prices = prices.sort_index()
    volume = volume.sort_index().reindex_like(prices)
    report = {
        "tickers_in": prices.shape[1],
        "dates_in": prices.shape[0],
        "price_cells_in": int(prices.notna().sum().sum()),
    }

    prices, n_nonpositive = remove_nonpositive(prices)
    report["nonpositive_prices_removed"] = n_nonpositive

    prices, volume, n_blanked = blank_known_bad_history(prices, volume)
    report["known_bad_history_cells_blanked"] = n_blanked
    report["known_bad_history_tickers"] = sorted(KNOWN_BAD_HISTORY)

    big_moves = flag_big_moves(prices, big_move_threshold)
    report["big_moves_flagged"] = len(big_moves)
    report["tickers_with_big_moves"] = int(big_moves["ticker"].nunique())

    gaps = find_gaps(prices)
    long_gap_tickers = gaps.loc[gaps["longest_gap"] > max_ffill_days, "ticker"].tolist()
    report["missing_days_inside_history"] = int(gaps["missing_days"].sum())
    report["tickers_with_long_gaps"] = len(long_gap_tickers)
    report["long_gap_tickers"] = long_gap_tickers

    prices, dropped = drop_short_history(prices, min_history_days)
    volume = volume[prices.columns]
    report["tickers_dropped_short_history"] = len(dropped)
    report["dropped_tickers"] = dropped

    prices, n_filled = forward_fill_limited(prices, max_ffill_days)
    report["cells_forward_filled"] = n_filled

    # Gaps longer than max_ffill_days are still NaN. SQL LAG counts rows, so
    # these tickers make LAG-based features in daily_features misaligned.
    leftover = find_gaps(prices)
    report["tickers_with_gaps_after_ffill"] = int((leftover["missing_days"] > 0).sum())
    report["missing_days_after_ffill"] = int(leftover["missing_days"].sum())

    liquid = None
    if use_liquidity_filter:
        liquid = liquidity_mask(prices, volume)
        report["avg_illiquid_per_date"] = float((~liquid & prices.notna()).sum(axis=1).mean())

    report["tickers_out"] = prices.shape[1]
    report["price_cells_out"] = int(prices.notna().sum().sum())

    return CleanResult(
        prices=prices,
        volume=volume,
        report=report,
        big_moves=big_moves,
        gaps=gaps,
        liquid=liquid,
        dropped_tickers=dropped,
    )


def print_report(result: CleanResult, max_rows: int = 20) -> None:
    """Print the cleaning report and the first few flagged big moves."""
    print("Cleaning report")
    print("-" * 40)
    for key, value in result.report.items():
        print(f"{key:32s} {value}")

    if len(result.big_moves) > 0:
        print(f"\nDaily moves over the threshold (first {max_rows}):")
        print(result.big_moves.head(max_rows).to_string(index=False))
