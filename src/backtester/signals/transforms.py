"""Cross-sectional transforms and signal combination (Skill.md section 7.1, 7.2).

Every function here works across tickers on one date at a time.
Nothing uses statistics over the whole time period, so the value at
date t depends only on signal values at date t.
"""

import numpy as np
import pandas as pd


def cs_rank(signal: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank across tickers for each date, in (0, 1]. NaNs stay NaN.

    Uses only the signal row at date t, so it uses the same closes as the signal.
    """
    return signal.rank(axis=1, pct=True)


def cs_zscore(signal: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Z-score across tickers for each date, clipped to [-clip, clip].

    Uses only the signal row at date t, so it uses the same closes as the signal.
    Rows where every value is equal (std = 0) become NaN instead of inf.
    """
    mean = signal.mean(axis=1)
    std = signal.std(axis=1)
    std = std.replace(0.0, np.nan)
    z = signal.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-clip, clip)


def mask_thin_dates(signal: pd.DataFrame, min_stocks: int) -> pd.DataFrame:
    """Set a whole date to NaN if it has fewer than `min_stocks` valid values.

    Uses only the signal row at date t. Ranks and quintiles on a handful of
    stocks are noisy, so those dates are skipped.
    """
    valid_count = signal.notna().sum(axis=1)
    too_thin = valid_count < min_stocks
    masked = signal.copy()
    masked.loc[too_thin, :] = np.nan
    return masked


def combine_equal_weight(
    ranked_signals: dict[str, pd.DataFrame] | list[pd.DataFrame],
    min_signals: int | None = None,
) -> pd.DataFrame:
    """Equal-weight average of ranked signals for each date and ticker.

    Uses only the signal values at date t, so it uses the same closes as
    the input signals.

    NaN handling: by default (min_signals=None) a stock needs a value from
    every signal on that date, otherwise the result is NaN. This keeps the
    combined score meaning the same thing for every stock, and avoids a
    stock getting a score from one signal while another signal is still
    warming up. Pass a smaller min_signals to average whatever is available
    when at least that many signals have a value.

    Inputs should already be ranked (e.g. with cs_rank) so they share a scale.
    """
    if isinstance(ranked_signals, dict):
        frames = list(ranked_signals.values())
    else:
        frames = list(ranked_signals)
    if len(frames) == 0:
        raise ValueError("need at least one signal to combine")
    if min_signals is None:
        min_signals = len(frames)

    # Line all signals up on the same dates and tickers.
    index = frames[0].index
    columns = frames[0].columns
    for frame in frames[1:]:
        index = index.union(frame.index)
        columns = columns.union(frame.columns)
    aligned = [frame.reindex(index=index, columns=columns) for frame in frames]

    total = pd.DataFrame(0.0, index=index, columns=columns)
    count = pd.DataFrame(0, index=index, columns=columns)
    for frame in aligned:
        total = total + frame.fillna(0.0)
        count = count + frame.notna().astype(int)

    average = total / count
    return average.where(count >= min_signals)
