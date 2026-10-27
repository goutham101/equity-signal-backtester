---
name: equity-signal-backtester
description: Use when building, extending, debugging, testing, or documenting the Equity Signal Backtester project (Python, pandas, PostgreSQL). Covers data ingestion, signal construction, IC analysis, long-short portfolio backtesting, transaction costs, walk-forward testing, performance metrics, bias prevention, tests, and the README.
---

# Equity Signal Backtester

## 1. Purpose of this project

A portfolio project for **Goutham** (UConn CS + Analytics minor, sophomore) to show readiness for **data science / quant research internships at hedge funds** (target example: Schonfeld 2027 Data Science Intern).

The project answers one question:

> "If I had traded on this signal in the past, using only information available at the time, would it have made money after trading costs?"

What recruiters should see:
- Working with **financial time-series data** in Python, pandas, and SQL
- Building **features/signals** and validating them with **backtesting**
- Statistical rigor: **IC analysis, walk-forward testing, honest reporting**
- Engineering discipline: **PostgreSQL, Docker, pytest, CI**
- Understanding of the 3 classic backtest traps: **look-ahead bias, survivorship bias, overfitting**

The goal is **not** a profitable strategy. The goal is a **correct, honest, well-explained research pipeline**. Weak or negative results are fine if they are real and explained.

---

## 2. How Claude should work on this project

Goutham is learning finance concepts through this project and must be able to **explain every line in an interview**. So:

1. **Explain before building.** Before each phase, give a short plain-English explanation of the finance concept (2–5 sentences, no jargon dumps). After writing code, explain the key lines, especially anything involving `shift`, alignment, or rolling windows.
2. **Keep code simple and readable.** Plain pandas and NumPy. No clever one-liners that hide logic. Short functions with clear names and docstrings.
3. **Do not use backtesting frameworks.** No `vectorbt`, `backtrader`, `zipline`, `bt`, `quantstats`, or similar. The whole point is that Goutham built the engine himself.
4. **Ask before major design changes** (new universe, new data source, changing the timing convention, adding big dependencies).
5. **Never fabricate or "improve" results.** Run the code and report actual numbers. Never tune parameters on test data to make results look better.
6. **Flag suspicious results loudly.** If a result looks too good (see Section 13), stop and hunt for a bug before moving on.
7. **Verify by running.** Run tests and scripts after changes. Don't claim something works without running it.
8. **Match Goutham's writing style** in README and comments: direct, clear, short sentences, active voice, student voice, no hype.
9. **Work in phases** (Section 14). Finish and verify one phase before starting the next.

---

## 3. Tech stack

| Layer | Tool | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Data wrangling | pandas, NumPy | Core of the project |
| Stats | SciPy | Spearman correlation, t-stats |
| Regression (walk-forward weights) | scikit-learn or NumPy `lstsq` | Keep it simple |
| Database | PostgreSQL 16 | Run in Docker |
| DB access | SQLAlchemy + psycopg | |
| Containers | Docker + docker-compose | Postgres only; honestly adds Docker to the resume |
| Data source | `yfinance` (primary) | Unofficial; check it still works. Backups: Stooq, Tiingo free tier |
| Plots | matplotlib | Save PNGs to `reports/figures/` |
| Testing | pytest | Use synthetic data so tests don't need the internet or DB |
| CI | GitHub Actions | Lint + tests on every push |
| Lint/format | ruff | |
| Config | one `config.yaml` | All parameters live here, not hardcoded |

---

## 4. Repository structure

```
equity-signal-backtester/
├── config.yaml                  # all parameters (dates, universe, costs, lookbacks)
├── docker-compose.yml           # Postgres service
├── requirements.txt
├── README.md
├── .github/workflows/ci.yml
├── sql/
│   ├── schema.sql               # table definitions
│   └── features.sql             # returns/momentum via window functions
├── src/backtester/
│   ├── __init__.py
│   ├── config.py                # load config.yaml
│   ├── data/
│   │   ├── universe.py          # list of tickers
│   │   ├── download.py          # pull prices from yfinance
│   │   ├── clean.py             # data quality checks
│   │   └── db.py                # read/write Postgres
│   ├── signals/
│   │   ├── momentum.py
│   │   ├── reversal.py
│   │   ├── low_vol.py
│   │   └── transforms.py        # cross-sectional rank / z-score
│   ├── analysis/
│   │   ├── forward_returns.py
│   │   ├── ic.py                # information coefficient
│   │   └── metrics.py           # Sharpe, drawdown, turnover, etc.
│   ├── backtest/
│   │   ├── portfolio.py         # quintile long-short weights
│   │   ├── engine.py            # apply weights, costs, returns
│   │   └── walk_forward.py      # rolling train/test combination
│   └── reporting/
│       └── plots.py
├── scripts/
│   ├── 01_download.py
│   ├── 02_build_signals.py
│   ├── 03_ic_analysis.py
│   ├── 04_backtest.py
│   └── 05_walk_forward.py
├── notebooks/
│   └── results.ipynb            # one clean notebook showing final results
├── reports/
│   ├── figures/
│   └── results.md               # auto-generated tables
└── tests/
    ├── conftest.py              # synthetic price fixtures
    ├── test_signals.py
    ├── test_no_lookahead.py     # the most important test
    ├── test_forward_returns.py
    ├── test_portfolio.py
    ├── test_engine.py
    ├── test_metrics.py
    └── test_walk_forward.py
```

---

## 5. Data conventions (follow these everywhere)

### 5.1 Shape
- **Wide format** for all computation: `index = date` (DatetimeIndex, sorted ascending), `columns = ticker`, values = float.
- **Long format** (`ticker, date, value`) only for storage in Postgres.
- Convert with `pivot` / `melt`. Always sort the index after pivoting.

### 5.2 Timing convention (critical)

```
Day t close:    compute signal using data up to and including close t
Day t+1 close:  enter the new positions (execution lag = 1 day)
Day t+2 onward: positions earn returns
```

- `returns[d]` = close-to-close return from close `d-1` to close `d`.
- `target_weights[t]` = weights decided at close `t`.
- Holdings during day `d` = `target_weights.shift(1 + exec_lag)`.
  - With `exec_lag = 1` (default): `held = target_weights.shift(2)`.
  - `exec_lag = 0` (trade at the same close used for the signal) is unrealistic. Only use it for a comparison and label it as optimistic.
- Forward returns used for IC must match: for horizon `h`, measure from close `t+1` to close `t+1+h`.

```python
def forward_returns(prices: pd.DataFrame, horizon: int, exec_lag: int = 1) -> pd.DataFrame:
    """Return from close t+exec_lag to close t+exec_lag+horizon, aligned to date t."""
    start = prices.shift(-exec_lag)
    end = prices.shift(-(exec_lag + horizon))
    return end / start - 1
```

**Every function that touches timing must say in its docstring which close it uses.**

### 5.3 Trading calendar
- 252 trading days per year.
- 21 trading days ≈ 1 month, 5 ≈ 1 week, 63 ≈ 1 quarter.
- Rebalance on the **last trading day of each month** (derive from the actual index, not calendar dates).

---

## 6. Data ingestion (Phase 1)

### 6.1 Universe
Default: **current S&P 500 constituents**. This has **survivorship bias** (companies that went bankrupt or got removed are missing). That's accepted for v1 and must be documented in the README's Limitations section.

Optional alternatives (ask Goutham before switching):
- **ETF universe** (~50–100 sector, country, bond, and commodity ETFs). Much less survivorship bias, multiple asset classes.
- **Historical S&P 500 membership** lists (available on GitHub). Fixes membership, but delisted stocks' prices are hard to get for free.

### 6.2 Download
- 10–15 years of daily data (e.g., 2010-01-01 to latest).
- Use **adjusted close** (`auto_adjust=True` in yfinance) so splits and dividends don't create fake jumps.
- Download in batches; sleep between batches; retry on failure; log failures.
- Cache raw downloads to `data/raw/` (gitignored) so reruns don't hit the API.

### 6.3 Schema (`sql/schema.sql`)

```sql
CREATE TABLE IF NOT EXISTS tickers (
    ticker      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    added_on    DATE DEFAULT CURRENT_DATE
);

CREATE TABLE IF NOT EXISTS prices (
    ticker      TEXT REFERENCES tickers(ticker),
    date        DATE NOT NULL,
    adj_close   NUMERIC(14,6) NOT NULL CHECK (adj_close > 0),
    volume      BIGINT CHECK (volume >= 0),
    PRIMARY KEY (ticker, date)
);

CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id      SERIAL PRIMARY KEY,
    run_at      TIMESTAMP DEFAULT NOW(),
    signal_name TEXT,
    config_json JSONB,
    sharpe      NUMERIC,
    ann_return  NUMERIC,
    max_dd      NUMERIC
);
```

`backtest_runs` logs every run with its config, so results are reproducible and you can show you didn't cherry-pick.

### 6.4 Features in SQL (`sql/features.sql`)
Show SQL skill by computing returns and momentum with window functions. Cross-check against the pandas version in a test.

```sql
CREATE OR REPLACE VIEW daily_features AS
SELECT
    ticker,
    date,
    adj_close,
    adj_close / LAG(adj_close, 1)   OVER w - 1 AS ret_1d,
    adj_close / LAG(adj_close, 5)   OVER w - 1 AS ret_5d,
    LAG(adj_close, 21) OVER w / LAG(adj_close, 252) OVER w - 1 AS mom_12_1
FROM prices
WINDOW w AS (PARTITION BY ticker ORDER BY date);
```

Note: `LAG` counts rows, not trading days. That's only correct if each ticker has no missing dates. The cleaning step must check this.

### 6.5 Data cleaning (`clean.py`)
Report counts for every check (don't silently drop data):
- Drop tickers with fewer than 2 years of history.
- Flag daily returns with |return| > 50%. Print them; inspect a few manually. Don't auto-delete.
- Forward-fill **at most 5 days** of missing prices. Never back-fill (back-filling is look-ahead).
- Flag tickers with long gaps.
- Optional liquidity filter: exclude stocks whose 60-day average dollar volume (price × volume) is in the bottom 10% **as of each date** (rolling, past data only).

---

## 7. Signals (Phase 2)

All signals are computed at close `t` using data up to close `t` only. **Higher signal value = expected higher future return.**

| Signal | Formula (wide DataFrame) | Warm-up | Idea |
|---|---|---|---|
| Momentum 12-1 | `prices.shift(21) / prices.shift(252) - 1` | 252 days | Winners over the past year keep winning. Skip the last month because of short-term reversal. |
| Short-term reversal | `-(prices / prices.shift(5) - 1)` | 5 days | Last week's losers tend to bounce back. |
| Low volatility | `-returns.rolling(60).std()` | 60 days | Calmer stocks have historically had better risk-adjusted returns. |

```python
def momentum_12_1(prices: pd.DataFrame, lookback: int = 252, skip: int = 21) -> pd.DataFrame:
    """12-month momentum skipping the most recent month. Uses closes up to date t."""
    return prices.shift(skip) / prices.shift(lookback) - 1
```

### 7.1 Cross-sectional transforms (`transforms.py`)
Raw signal values change scale over time, so transform **across stocks on each date**:

```python
def cs_rank(signal: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank across tickers for each date (0 to 1). NaNs stay NaN."""
    return signal.rank(axis=1, pct=True)

def cs_zscore(signal: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Z-score across tickers for each date, clipped to limit outliers."""
    mean = signal.mean(axis=1)
    std = signal.std(axis=1)
    z = signal.sub(mean, axis=0).div(std, axis=0)
    return z.clip(-clip, clip)
```

Rules:
- Cross-sectional transforms (across tickers on one date) are safe.
- **Never** normalize using statistics over the whole time period (e.g., `signal.mean()` over all dates). That leaks future information.
- Skip any date with fewer than `min_stocks` (e.g., 50) valid values.

### 7.2 Combined signal
Equal-weight average of the 3 ranked signals is the baseline. Walk-forward combination (Section 10) is the upgrade.

---

## 8. Signal evaluation with IC (Phase 3)

The **information coefficient** = Spearman rank correlation, on each date, between the signal and the forward return.

```python
def daily_ic(signal: pd.DataFrame, fwd_ret: pd.DataFrame, min_stocks: int = 50) -> pd.Series:
    """Spearman IC per date between signal at t and forward return from t+1."""
    valid = signal.notna() & fwd_ret.notna()
    counts = valid.sum(axis=1)
    ic = signal.where(valid).corrwith(fwd_ret.where(valid), axis=1, method="spearman")
    return ic[counts >= min_stocks]
```

Report for each signal and horizon (5, 21, 63 days):
- **Mean IC**
- **IC standard deviation**
- **IC IR** = mean / std
- **% of dates with positive IC**
- **t-stat**: compute it on **non-overlapping dates only** (e.g., sample every 21st date for 21-day horizon). Overlapping forward returns make daily ICs highly correlated, which inflates t-stats if you use every day.
- **Rolling 12-month mean IC chart** (does the signal decay over time?)
- **Quintile spread chart**: average forward return of each quintile (Q1–Q5). A good signal shows returns increasing from Q1 to Q5.

Expected magnitudes: mean IC of **0.01–0.05** is realistic. Anything above ~0.10 on these simple signals almost certainly means a bug.

---

## 9. Portfolio backtest (Phase 4)

### 9.1 Portfolio construction (`portfolio.py`)
On each rebalance date:
1. Take the ranked signal for that date (drop NaNs).
2. Split into quintiles with `pd.qcut(..., 5, labels=False)`.
3. **Long** Q5 (top 20%): each weight = `+1 / n_long`.
4. **Short** Q1 (bottom 20%): each weight = `-1 / n_short`.
5. Everyone else = 0.

So long side sums to +1, short side sums to −1 (dollar-neutral, gross exposure 2). The portfolio return is the long-minus-short spread.

Also build a **long-only** version: Q5 equal-weight vs. an equal-weight benchmark of all stocks. Report excess return.

Between rebalances, hold target weights constant (forward-fill target weights from rebalance dates). Simplification: this ignores weight drift. Document it.

### 9.2 Engine (`engine.py`)

```python
def run_backtest(target_weights: pd.DataFrame, returns: pd.DataFrame,
                 cost_bps: float = 10.0, exec_lag: int = 1) -> pd.DataFrame:
    """
    target_weights: weights decided at close t (forward-filled between rebalances).
    returns: close-to-close returns, returns[d] = close d-1 -> close d.
    Holdings earn returns starting 1 + exec_lag days after the decision.
    """
    held = target_weights.shift(1 + exec_lag).fillna(0.0)
    gross = (held * returns.fillna(0.0)).sum(axis=1)

    turnover = held.diff().abs().sum(axis=1).fillna(0.0)
    costs = turnover * cost_bps / 10_000
    net = gross - costs

    return pd.DataFrame({"gross": gross, "costs": costs, "net": net, "turnover": turnover})
```

Cost notes:
- `cost_bps` is the cost per unit of weight traded (one side). 10 bps is a reasonable baseline for liquid large caps.
- Run a **cost sensitivity table**: 0, 5, 10, 20, 50 bps.
- Short borrow costs are ignored in v1. Document it.

### 9.3 Rebalance frequency comparison
Run monthly (default) and weekly. Short-term reversal usually needs frequent rebalancing and gets destroyed by costs. That's a key finding to write about.

---

## 10. Walk-forward testing (Phase 5)

Use walk-forward whenever anything is **chosen** from data (combination weights, lookback lengths, thresholds).

### 10.1 Setup
- **Rolling windows**: train 5 years, test the next 1 year, step forward 1 year.
- **Gap (purge)**: leave a gap of at least `horizon` trading days between the end of training and the start of testing. Training labels (forward returns) near the end of the train window overlap the test period; the gap prevents that leak.

```
Train 2012–2016 | gap | Test 2017
Train 2013–2017 | gap | Test 2018
Train 2014–2018 | gap | Test 2019
...
```

### 10.2 What to fit in training
Pick **one** simple method and explain it:
- **Option A (recommended): IC-weighted.** Weight each signal by its mean IC in the training window (set negative-IC signals to 0, normalize weights to sum to 1).
- **Option B: Linear regression.** Pooled regression of forward returns on the 3 ranked signals over training dates. Use coefficients as weights.

Apply the fitted weights to the test year only. Concatenate all test years into one **out-of-sample** return series. **Only out-of-sample results go in the headline results table.**

### 10.3 Compare
- Equal-weight combination (no fitting)
- Walk-forward combination
- Each single signal

If walk-forward doesn't beat equal-weight, say so. That's a common and honest result.

---

## 11. Metrics (`metrics.py`)

| Metric | Formula |
|---|---|
| Annualized return | `net.mean() * 252` |
| Annualized volatility | `net.std() * sqrt(252)` |
| Sharpe ratio | ann. return / ann. vol (risk-free rate ignored for long-short; mention it) |
| Max drawdown | `equity = (1 + net).cumprod(); (equity / equity.cummax() - 1).min()` |
| Calmar ratio | ann. return / abs(max drawdown) |
| Avg annual turnover | `turnover.mean() * 252` |
| Hit rate | % of days with positive net return |
| Worst month | min of monthly compounded returns |

Write a `summary(results) -> dict` function and a function that renders a markdown table into `reports/results.md`.

### Charts (`plots.py`)
1. Equity curve, gross vs. net, per signal
2. Drawdown chart
3. Rolling 12-month IC
4. Quintile average forward returns (bar chart)
5. Cost sensitivity (Sharpe vs. cost bps)

Every chart: title, axis labels with units, legend, saved as PNG.

---

## 12. Bias prevention rules (hard rules)

### Look-ahead bias
- [ ] Every signal at date `t` uses only data up to close `t`.
- [ ] Holdings use `target_weights.shift(1 + exec_lag)`.
- [ ] Forward returns start at `t + exec_lag`, never at `t`.
- [ ] No full-period normalization, mean, std, or min/max.
- [ ] No back-filling (`bfill`) anywhere.
- [ ] Rolling windows only look backward (pandas default; never use `center=True`).
- [ ] Universe filters (liquidity, etc.) use only past data at each date.
- [ ] `test_no_lookahead.py` passes.

### Survivorship bias
- [ ] Universe choice and its bias are documented in README Limitations.

### Overfitting
- [ ] All parameter choices are fixed before looking at test results, or chosen via walk-forward.
- [ ] Every backtest run is logged in `backtest_runs` with its config.
- [ ] The README reports how many signal/parameter variations were tried in total.

---

## 13. Sanity checks and red flags

Stop and debug if any of these happen:

| Red flag | Likely cause |
|---|---|
| Mean IC > 0.10 | Look-ahead bias (missing shift) |
| Sharpe > 2 on simple signals | Look-ahead bias or a cost bug |
| Equity curve is a nearly straight line | Look-ahead bias |
| Results change a lot when `exec_lag` goes from 0 to 1 | Signal depends on same-day info (expected a bit for reversal; huge changes are suspicious) |
| Turnover is 0 or huge | Weight forward-fill or rebalance-date bug |
| Long side weights don't sum to 1 | Quintile/NaN handling bug |
| Daily returns of ±100% | Bad data, missing split adjustment |

Positive sanity checks:
- Momentum should show positive long-run IC with a big crash around 2009 (momentum crash) if data goes that far back, and weak periods in some recent years.
- A **random signal** (e.g., `np.random.rand` per date) must give IC ≈ 0 and Sharpe ≈ 0 before costs. Build this as a baseline and include it in results.

---

## 14. Phases and acceptance criteria

### Phase 0: Setup
- Repo structure, `requirements.txt`, `config.yaml`, `docker-compose.yml`, CI workflow, ruff.
- **Done when:** `docker compose up` starts Postgres, `pytest` runs (even with 1 placeholder test), CI is green.

### Phase 1: Data
- Universe, download, clean, load into Postgres, SQL features view.
- **Done when:** prices table loaded, cleaning report printed, SQL momentum matches pandas momentum in a test.

### Phase 2: Signals
- 3 signals + cross-sectional transforms + random baseline.
- **Done when:** signal unit tests and no-look-ahead test pass.

### Phase 3: IC analysis
- IC table (3 horizons), rolling IC chart, quintile chart.
- **Done when:** `reports/results.md` has the IC table and figures are saved.

### Phase 4: Backtest
- Quintile long-short and long-only, costs, monthly vs. weekly, cost sensitivity.
- **Done when:** engine tests pass and the results table shows gross and net metrics for every signal.

### Phase 5: Walk-forward
- Rolling train/test with purge gap, out-of-sample combined signal.
- **Done when:** walk-forward tests pass and out-of-sample results are in the table.

### Phase 6: Write-up
- README, clean notebook, final figures.
- **Done when:** someone could clone, run, and reproduce the results by following the README.

---

## 15. Tests (pytest, synthetic data)

Use synthetic prices from `conftest.py` (e.g., geometric random walk for 20 tickers × 800 days, fixed seed). Tests must not need internet or Postgres, except the SQL-vs-pandas test, which is marked `@pytest.mark.db` and skipped in CI if no DB is available.

### 15.1 The no-look-ahead test (most important)

```python
def test_signals_do_not_use_future_data(synthetic_prices):
    cutoff = synthetic_prices.index[500]

    original = momentum_12_1(synthetic_prices)

    corrupted = synthetic_prices.copy()
    future = corrupted.index > cutoff
    corrupted.loc[future] = np.random.default_rng(0).uniform(1, 1000, corrupted.loc[future].shape)

    recomputed = momentum_12_1(corrupted)

    pd.testing.assert_frame_equal(original.loc[:cutoff], recomputed.loc[:cutoff])
```

Parametrize it over **every signal function and transform**, and run the same idea on the portfolio weights.

### 15.2 Other tests
- **Signals:** hand-computed momentum on a tiny known price series matches the function.
- **Forward returns:** on a 5-day hand-made series, values land on the correct dates.
- **Engine timing:** a weight decided at day `t` earns exactly the return of day `t+2` (with `exec_lag=1`). Build a series where only one day has a nonzero return and check where profit appears.
- **Costs:** a single full flip (from +1 to −1 in one stock) with 10 bps creates cost = 2 × 0.001.
- **Portfolio:** long weights sum to +1, short to −1, no stock is both long and short.
- **Metrics:** max drawdown of a known series (e.g., 100 → 120 → 60 → 90) equals −50%. Sharpe of a constant positive return series is handled (zero std) without crashing.
- **Random baseline:** across many seeds, mean IC of a random signal is close to 0.
- **Walk-forward:** test windows never overlap training windows, and the gap is at least `horizon` days.

---

## 16. Config (`config.yaml`)

```yaml
data:
  start_date: "2010-01-01"
  end_date: null            # null = latest available
  universe: "sp500_current"
  min_history_days: 504
  max_ffill_days: 5

signals:
  momentum: {lookback: 252, skip: 21}
  reversal: {lookback: 5}
  low_vol: {window: 60}
  min_stocks: 50

backtest:
  exec_lag: 1
  rebalance: "monthly"      # monthly | weekly
  quantiles: 5
  cost_bps: 10
  cost_sensitivity: [0, 5, 10, 20, 50]

analysis:
  ic_horizons: [5, 21, 63]

walk_forward:
  train_years: 5
  test_years: 1
  method: "ic_weighted"     # ic_weighted | regression
  purge_days: 21

seed: 42
```

---

## 17. README requirements

Sections, in order:
1. **What this is** (2–3 sentences, plain English)
2. **Key results** (one table: each signal, gross Sharpe, net Sharpe, max drawdown, mean IC, turnover) + equity curve image
3. **Key findings** (3–5 bullets of real insights, e.g., "Reversal had the highest gross Sharpe but lost money after 10 bps costs because of ~X× higher turnover")
4. **Method** (data, signals, timing convention, portfolio construction, costs, walk-forward)
5. **How I prevented look-ahead bias** (explain the timing convention and the corruption test)
6. **Limitations** (survivorship bias, simple cost model, no borrow costs, no weight drift, no risk model/sector neutrality, yfinance data quality)
7. **What I'd do next** (sector neutralization, risk model, more realistic costs, point-in-time universe, more signals)
8. **How to run** (exact commands from clone to results)
9. **Project structure**

Style: direct, short sentences, no hype words ("powerful", "cutting-edge", "robust"). Numbers must match `reports/results.md` exactly.

---

## 18. Resume bullet template

Fill in only with real numbers from the final results:

> **Equity Signal Backtester** | *Python, pandas, PostgreSQL, SQL, Docker, pytest, GitHub Actions*
> - Built a backtesting pipeline over [N] stocks and [Y] years of daily price data stored in PostgreSQL, testing momentum, short-term reversal, and low-volatility signals with long-short quintile portfolios
> - Evaluated signals out-of-sample with walk-forward testing, information coefficient, and Sharpe ratio after [Z] bps transaction costs; found [real finding]
> - Prevented look-ahead bias with a pytest suite that verifies corrupting future prices never changes past signals or positions

---

## 19. Interview prep (Claude should help Goutham practice these)

- Walk me through your timing convention. Why `shift(2)`?
- What is look-ahead bias? How did you test for it?
- What is survivorship bias? How does it affect your results, and in which direction?
- Why rank signals cross-sectionally instead of using raw values?
- What's IC, and what's a good value? Why compute the t-stat on non-overlapping dates?
- Why skip the most recent month in momentum?
- Why did reversal look good before costs and bad after?
- What's the difference between in-sample and out-of-sample? Why the purge gap?
- How would you know if your result is just luck or overfitting?
- How many variations did you try? How does that affect how much you trust the best one?
- What would you add with more time? (sector neutrality, risk model, better costs, point-in-time data)
- How would this pipeline change for futures or fixed income?

When practicing, Claude asks one question at a time, waits for Goutham's answer, then gives direct feedback on what was right, what was missing, and a tighter version of the answer.
