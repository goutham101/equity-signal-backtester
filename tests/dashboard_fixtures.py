"""Synthetic dashboard data shaped like docs/dashboard_data_contract.md.

Test-only. Writes small random files into a folder so the Streamlit app can
be loaded without running the real export script. Numbers are fake.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtester.analysis import metrics

FULL_SIGNALS = ["momentum", "reversal", "low_vol", "equal_weight", "random"]
OOS_SIGNALS = ["walk_forward", *FULL_SIGNALS]
HORIZONS = [5, 21, 63]

START = "2012-01-02"
END = "2016-12-30"
OOS_START = "2014-01-02"


def _rebalance_days(dates: pd.DatetimeIndex, rebalance: str) -> pd.Series:
    """True on the last trading day of each month (or week)."""
    key = dates.to_period("M" if rebalance == "monthly" else "W")
    last = pd.Series(dates, index=dates).groupby(key).transform("max")
    return pd.Series(dates == last.values, index=dates)


def _one_series(rng, dates, signal, portfolio, rebalance, period) -> pd.DataFrame:
    """Random daily gross returns, legs, and turnover for one series."""
    n = len(dates)
    long_leg = rng.normal(0.0004, 0.008, n)
    short_leg = rng.normal(-0.0001, 0.008, n) if portfolio == "long_short" else np.zeros(n)
    turnover = np.where(_rebalance_days(dates, rebalance), rng.uniform(0.3, 1.5, n), 0.0)
    return pd.DataFrame(
        {
            "date": dates,
            "signal": signal,
            "portfolio": portfolio,
            "rebalance": rebalance,
            "period": period,
            "gross": long_leg + short_leg,
            "turnover": turnover,
            "gross_long": long_leg,
            "gross_short": short_leg,
        }
    )


def _daily_returns(rng, dates) -> pd.DataFrame:
    """daily_returns.parquet: full-period grid, benchmark, and the OOS slice."""
    parts = []
    for signal in FULL_SIGNALS:
        for portfolio in ["long_short", "long_only"]:
            for rebalance in ["monthly", "weekly"]:
                parts.append(_one_series(rng, dates, signal, portfolio, rebalance, "full"))
    for rebalance in ["monthly", "weekly"]:
        parts.append(_one_series(rng, dates, "benchmark", "long_only", rebalance, "full"))
    oos_dates = dates[dates >= OOS_START]
    for signal in OOS_SIGNALS:
        parts.append(_one_series(rng, oos_dates, signal, "long_short", "monthly", "oos"))
    return pd.concat(parts, ignore_index=True)


def _ic_daily(rng, dates) -> pd.DataFrame:
    """ic_daily.parquet: one IC per date, signal, horizon."""
    parts = []
    for signal in FULL_SIGNALS:
        for horizon in HORIZONS:
            parts.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "signal": signal,
                        "horizon": horizon,
                        "ic": rng.normal(0.01, 0.1, len(dates)),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True)


def _ic_summary(ic: pd.DataFrame) -> pd.DataFrame:
    """ic_summary.parquet: simple stats (t-stat here is plain, not Newey-West)."""
    grouped = ic.groupby(["signal", "horizon"])["ic"]
    out = grouped.agg(mean_ic="mean", ic_std="std", n_dates="count").reset_index()
    out["ic_ir"] = out["mean_ic"] / out["ic_std"]
    out["pct_positive"] = grouped.apply(lambda s: (s > 0).mean()).values
    out["t_stat"] = out["ic_ir"] * np.sqrt(out["n_dates"])
    return out


def _quintiles(rng) -> pd.DataFrame:
    """quintile_returns.parquet."""
    rows = [
        {
            "signal": s,
            "horizon": h,
            "quintile": q,
            "avg_fwd_return": 0.001 * q * h / 21 + rng.normal(0, 0.001),
        }
        for s in FULL_SIGNALS
        for h in HORIZONS
        for q in range(1, 6)
    ]
    return pd.DataFrame(rows)


def _walk_forward(rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """walk_forward_windows.parquet and walk_forward_weights.parquet."""
    windows, weights = [], []
    for year in [2014, 2015, 2016]:
        test_start = pd.Timestamp(f"{year}-01-02")
        windows.append(
            {
                "window": str(year),
                "train_start": pd.Timestamp(f"{year - 5}-01-03"),
                "train_end": test_start - pd.Timedelta(days=32),
                "test_start": test_start,
                "test_end": pd.Timestamp(f"{year}-12-30"),
            }
        )
        raw = rng.uniform(0, 1, 3)
        raw[2] = 0.0
        for signal, w in zip(["momentum", "reversal", "low_vol"], raw / raw.sum(), strict=True):
            weights.append({"window": str(year), "signal": signal, "weight": w})
    return pd.DataFrame(windows), pd.DataFrame(weights)


def _summary(daily: pd.DataFrame, ic: pd.DataFrame) -> pd.DataFrame:
    """summary.parquet, computed with metrics.summary like the real export."""
    rows = []
    mean_ic = ic[ic["horizon"] == 21].groupby("signal")["ic"].mean()
    for signal in OOS_SIGNALS:
        frame = daily[(daily["signal"] == signal) & (daily["period"] == "oos")]
        frame = frame.set_index("date")
        frame = frame.assign(net=metrics.net_returns_at_cost(frame, 10))
        frame = frame.assign(costs=frame["gross"] - frame["net"])
        stats = metrics.summary(frame)
        rows.append(
            {
                "signal": signal,
                "sharpe_net": stats["sharpe_net"],
                "ann_return_net": stats["ann_return_net"],
                "ann_vol_net": stats["ann_vol_net"],
                "max_drawdown_net": stats["max_drawdown_net"],
                "mean_ic_21d": mean_ic.get(signal, np.nan),
                "ann_turnover": stats["ann_turnover"],
            }
        )
    return pd.DataFrame(rows)


def _metadata(lookahead_status: str, linkedin: str | None) -> dict:
    """metadata.json contents."""
    return {
        "universe": "Synthetic test universe",
        "n_stocks": 20,
        "start_date": START,
        "end_date": END,
        "oos_start": OOS_START,
        "oos_end": END,
        "last_updated": "2026-10-27T12:00:00",
        "variations_tried": 21,
        "default_cost_bps": 10,
        "exec_lag": 1,
        "quantiles": 5,
        "walk_forward_method": "ic_weighted",
        "lookahead_test": {
            "status": lookahead_status,
            "checked_at": "2026-10-27T12:00:00",
            "tests": 22,
        },
        "key_findings": ["**Fake finding one.** Synthetic data.", "Fake finding two."],
        "limitations": ["**Synthetic.** Not real data."],
        "links": {
            "github": "https://github.com/goutham101/equity-signal-backtester",
            "readme": "https://github.com/goutham101/equity-signal-backtester#readme",
            "linkedin": linkedin,
        },
    }


def build_dashboard_data(
    folder: Path, seed: int = 0, lookahead_status: str = "passing", linkedin: str | None = None
) -> Path:
    """Write every contract file into `folder` and return it."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(START, END)

    daily = _daily_returns(rng, dates)
    ic = _ic_daily(rng, dates)
    windows, weights = _walk_forward(rng)

    daily.to_parquet(folder / "daily_returns.parquet", index=False)
    ic.to_parquet(folder / "ic_daily.parquet", index=False)
    _ic_summary(ic).to_parquet(folder / "ic_summary.parquet", index=False)
    _quintiles(rng).to_parquet(folder / "quintile_returns.parquet", index=False)
    weights.to_parquet(folder / "walk_forward_weights.parquet", index=False)
    windows.to_parquet(folder / "walk_forward_windows.parquet", index=False)
    _summary(daily, ic).to_parquet(folder / "summary.parquet", index=False)
    meta = _metadata(lookahead_status, linkedin)
    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
    return folder


if __name__ == "__main__":
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dashboard_fixture_data")
    print(build_dashboard_data(target))
