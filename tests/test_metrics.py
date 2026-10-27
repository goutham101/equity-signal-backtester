"""Performance metrics, markdown output, and plot smoke tests."""

import numpy as np
import pandas as pd
import pytest

from backtester.analysis import metrics
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
