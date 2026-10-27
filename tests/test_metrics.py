"""Performance metrics, markdown output, and plot smoke tests."""

import numpy as np
import pandas as pd
import pytest

from backtester.analysis import metrics
from backtester.backtest.engine import run_backtest
from backtester.backtest.portfolio import build_target_weights
from backtester.reporting import plots


def test_max_drawdown_known_series():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert metrics.max_drawdown_from_equity(equity) == pytest.approx(-0.5)

    returns = equity.pct_change().fillna(0.0)
    assert metrics.max_drawdown(returns) == pytest.approx(-0.5)


def test_sharpe_constant_returns_does_not_crash():
    returns = pd.Series([0.001] * 100)
    result = metrics.sharpe_ratio(returns)
    assert np.isnan(result)


def test_sharpe_known_value():
    returns = pd.Series([0.01, -0.01, 0.02, 0.0])
    expected = returns.mean() * 252 / (returns.std() * np.sqrt(252))
    assert metrics.sharpe_ratio(returns) == pytest.approx(expected)


def test_annual_return_and_vol():
    returns = pd.Series([0.001] * 252)
    assert metrics.annual_return(returns) == pytest.approx(0.252)
    returns = pd.Series([0.01, -0.01] * 50)
    assert metrics.annual_volatility(returns) == pytest.approx(returns.std() * np.sqrt(252))


def test_hit_rate():
    returns = pd.Series([0.01, -0.01, 0.0, 0.02])
    assert metrics.hit_rate(returns) == pytest.approx(0.5)


def test_worst_month():
    jan = pd.bdate_range("2024-01-01", "2024-01-31")
    feb = pd.bdate_range("2024-02-01", "2024-02-29")
    returns = pd.Series(0.0, index=jan.append(feb))
    returns[jan[0]] = 0.10
    returns[feb[0]] = -0.10
    returns[feb[1]] = -0.10  # Feb compounds to 0.9 * 0.9 - 1 = -0.19
    assert metrics.worst_month(returns) == pytest.approx(-0.19)


def test_calmar_and_turnover():
    returns = pd.Series([0.01, -0.02, 0.01, 0.01])
    expected = metrics.annual_return(returns) / abs(metrics.max_drawdown(returns))
    assert metrics.calmar_ratio(returns) == pytest.approx(expected)
    assert metrics.annual_turnover(pd.Series([0.1, 0.0, 0.1, 0.0])) == pytest.approx(0.05 * 252)


@pytest.fixture
def fake_results() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=300)
    gross = pd.Series(rng.normal(0.0005, 0.01, 300), index=dates)
    turnover = pd.Series(0.0, index=dates)
    turnover.iloc[::21] = 2.0
    costs = turnover * 10 / 10_000
    return pd.DataFrame(
        {"gross": gross, "costs": costs, "net": gross - costs, "turnover": turnover}
    )


def test_summary_has_gross_and_net_sharpe(fake_results):
    out = metrics.summary(fake_results)
    assert "sharpe_gross" in out and "sharpe_net" in out
    assert out["sharpe_gross"] > out["sharpe_net"]
    assert out["max_drawdown_net"] <= 0


def test_markdown_table_and_file(tmp_path, fake_results):
    rows = {"momentum": metrics.summary(fake_results), "random": {"sharpe_net": np.nan}}
    table = metrics.results_to_markdown(rows)
    assert table.startswith("| name | ann_return_gross")
    assert "| random |" in table
    assert "nan" in table

    path = metrics.write_results_md({"Backtest": rows, "Notes": "Some text."},
                                    path=tmp_path / "results.md")
    text = path.read_text()
    assert "## Backtest" in text
    assert "## Notes" in text
    assert "Some text." in text


def test_plots_write_png(tmp_path, fake_results):
    ic = pd.Series(np.linspace(-0.05, 0.05, 300), index=fake_results.index)
    quintiles = pd.Series([-0.01, 0.0, 0.002, 0.005, 0.01],
                          index=["Q1", "Q2", "Q3", "Q4", "Q5"])
    cost_table = pd.DataFrame({"momentum": [1.0, 0.8, 0.5, 0.1, -0.5]},
                              index=[0, 5, 10, 20, 50])

    paths = [
        plots.plot_equity_curve(fake_results, tmp_path / "equity.png"),
        plots.plot_drawdown({"momentum": fake_results["net"]}, tmp_path / "dd.png"),
        plots.plot_rolling_ic({"momentum": ic.rolling(20).mean()}, tmp_path / "ic.png"),
        plots.plot_quintile_returns(quintiles, tmp_path / "q.png", horizon=21),
        plots.plot_cost_sensitivity(cost_table, tmp_path / "cost.png"),
    ]
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 0


def _momentum_backtest(prices: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Monthly long-short 12-1 momentum backtest on synthetic prices."""
    returns = prices.pct_change()
    signal = prices.shift(21) / prices.shift(252) - 1
    weights = build_target_weights(signal, freq="monthly")
    return run_backtest(weights, returns, cost_bps=cost_bps)


def test_net_returns_at_cost_matches_engine(synthetic_prices):
    base = _momentum_backtest(synthetic_prices, cost_bps=0.0)
    for bps in [0.0, 5.0, 10.0, 50.0]:
        engine_result = _momentum_backtest(synthetic_prices, cost_bps=bps)
        live_net = metrics.net_returns_at_cost(base, bps)
        pd.testing.assert_series_equal(live_net, engine_result["net"], check_names=False)


def test_breakeven_cost_gives_zero_mean_net():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=500)
    gross = pd.Series(rng.normal(0.0008, 0.01, 500), index=dates)
    turnover = pd.Series(0.0, index=dates)
    turnover.iloc[::5] = 1.5
    results = pd.DataFrame({"gross": gross, "turnover": turnover})

    bps = metrics.breakeven_cost_bps(results)
    expected = gross.mean() / turnover.mean() * 10_000
    assert bps == pytest.approx(expected)
    assert bps > 0

    net = metrics.net_returns_at_cost(results, bps)
    assert net.mean() == pytest.approx(0.0, abs=1e-15)
    assert metrics.sharpe_ratio(net) == pytest.approx(0.0, abs=1e-10)
    # A bit more cost makes it lose, a bit less makes it win.
    assert metrics.sharpe_ratio(metrics.net_returns_at_cost(results, bps + 1)) < 0
    assert metrics.sharpe_ratio(metrics.net_returns_at_cost(results, bps - 1)) > 0


def test_breakeven_on_real_engine_output(synthetic_prices):
    # On this random walk, momentum loses before costs, so its breakeven is 0.
    momentum = _momentum_backtest(synthetic_prices, cost_bps=10.0)
    assert momentum["gross"].mean() < 0
    assert metrics.breakeven_cost_bps(momentum) == 0.0

    # The flipped signal makes money before costs: rerun at its breakeven cost.
    returns = synthetic_prices.pct_change()
    signal = -(synthetic_prices.shift(21) / synthetic_prices.shift(252) - 1)
    weights = build_target_weights(signal, freq="monthly")
    flipped = run_backtest(weights, returns, cost_bps=10.0)
    bps = metrics.breakeven_cost_bps(flipped)
    assert bps > 0

    at_breakeven = run_backtest(weights, returns, cost_bps=bps)
    assert at_breakeven["net"].mean() == pytest.approx(0.0, abs=1e-15)
    assert metrics.sharpe_ratio(at_breakeven["net"]) == pytest.approx(0.0, abs=1e-10)


def test_breakeven_zero_turnover_is_nan():
    dates = pd.bdate_range("2020-01-01", periods=10)
    results = pd.DataFrame({"gross": 0.001, "turnover": 0.0}, index=dates)
    assert np.isnan(metrics.breakeven_cost_bps(results))


def test_breakeven_losing_before_costs_is_zero():
    dates = pd.bdate_range("2020-01-01", periods=10)
    results = pd.DataFrame({"gross": -0.001, "turnover": 0.5}, index=dates)
    assert metrics.breakeven_cost_bps(results) == 0.0


def test_rolling_sharpe_known_series():
    returns = pd.Series([0.01, -0.01, 0.02, 0.0, 0.03, -0.02])
    window = 4
    result = metrics.rolling_sharpe(returns, window=window)

    assert result.iloc[: window - 1].isna().all()
    for end in range(window - 1, len(returns)):
        chunk = returns.iloc[end - window + 1 : end + 1]
        assert result.iloc[end] == pytest.approx(metrics.sharpe_ratio(chunk))


def test_rolling_sharpe_constant_window_is_nan():
    returns = pd.Series([0.001] * 5 + [0.01, -0.01, 0.02])
    result = metrics.rolling_sharpe(returns, window=3)
    assert result.iloc[2:5].isna().all()  # windows of constant returns
    assert np.isfinite(result.iloc[7])


def test_rolling_sharpe_does_not_use_future_data():
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2020-01-01", periods=400)
    returns = pd.Series(rng.normal(0.0005, 0.01, 400), index=dates)
    cutoff = dates[300]
    original = metrics.rolling_sharpe(returns, window=60)

    corrupted = returns.copy()
    corrupted[corrupted.index > cutoff] = 0.5
    recomputed = metrics.rolling_sharpe(corrupted, window=60)

    pd.testing.assert_series_equal(original.loc[:cutoff], recomputed.loc[:cutoff])
    assert not original.loc[cutoff:].iloc[1:].equals(recomputed.loc[cutoff:].iloc[1:])


def test_summary_adds_breakeven_and_legs(synthetic_prices):
    result = _momentum_backtest(synthetic_prices, cost_bps=10.0)
    out = metrics.summary(result)
    assert out["breakeven_bps"] == pytest.approx(metrics.breakeven_cost_bps(result))
    legs_total = out["ann_return_long"] + out["ann_return_short"]
    assert legs_total == pytest.approx(out["ann_return_gross"])


def test_summary_works_without_leg_columns(fake_results):
    out = metrics.summary(fake_results)
    assert "ann_return_long" not in out
    assert "ann_return_short" not in out
    assert "breakeven_bps" in out


def test_trim_to_active_drops_warmup_rows():
    """Zero rows before the first trade must not dilute annual return."""
    dates = pd.bdate_range("2020-01-01", periods=6)
    results = pd.DataFrame(
        {
            "gross": [0.0, 0.0, 0.0, 0.01, 0.01, 0.01],
            "costs": [0.0, 0.0, 0.0, 0.001, 0.0, 0.0],
            "net": [0.0, 0.0, 0.0, 0.009, 0.01, 0.01],
            "turnover": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        },
        index=dates,
    )
    trimmed = metrics.trim_to_active(results)
    assert trimmed.index[0] == dates[3]
    assert len(trimmed) == 3
    assert metrics.summary(results)["ann_return_gross"] == pytest.approx(0.01 * 252)


def test_trim_to_active_never_trades():
    dates = pd.bdate_range("2020-01-01", periods=3)
    zeros = pd.DataFrame(0.0, index=dates, columns=["gross", "costs", "net", "turnover"])
    assert metrics.trim_to_active(zeros).empty
