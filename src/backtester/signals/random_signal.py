"""Random baseline signal (Skill.md section 13)."""

import numpy as np
import pandas as pd


def random_signal(prices: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Uniform random score per date and ticker. Uses no price values at all.

    Only the shape of `prices` and where it is NaN are used: the value at
    date t is NaN where close t is NaN. This signal has no information,
    so its IC and Sharpe should be close to 0. If they are not, the
    evaluation code has a bug.
    """
    rng = np.random.default_rng(seed)
    values = rng.uniform(0.0, 1.0, size=prices.shape)
    signal = pd.DataFrame(values, index=prices.index, columns=prices.columns)
    return signal.where(prices.notna())
