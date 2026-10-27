# Dashboard spec (Streamlit)

Written by Goutham, 2026-10-27. Build after the real results exist (Phases 3–5).

## Core design rule

The dashboard reads precomputed results. It never downloads data or runs the full backtest live.

- Add a script `scripts/06_export_dashboard_data.py` that saves all results to small Parquet files in `dashboard/data/`.
- The app just loads those files. That makes it fast, free to host, and reproducible.
- It also solves a deployment problem: Streamlit Cloud can't reach the Postgres running in Docker on the laptop.

The one exception is the cost slider, which can be live because it's cheap math: `net = gross - turnover × bps / 10,000`. Gross returns and turnover are already stored, so it updates instantly.

## Sidebar (on every page)

- Project title + one-line description
- Global controls: signal (Momentum, Reversal, Low Vol, Equal-Weight Combo, Walk-Forward Combo, Random Baseline), portfolio type (long-short / long-only), rebalance frequency (monthly / weekly), transaction cost slider (0–50 bps, default 10)
- Data info: universe, date range, number of stocks, last updated
- Links: GitHub repo, README, LinkedIn
- Small note: "Educational project. Not investment advice."

## Page 1: Overview

The page a recruiter sees first. No controls needed.

- Headline results table: every signal with net Sharpe, annual return, volatility, max drawdown, mean IC, and turnover. Label it clearly as out-of-sample, 10 bps costs.
- Equity curve: the walk-forward combo vs. equal-weight combo vs. random baseline.
- Key findings: 3–5 bullets copied from the README (real findings only).
- "How to read this" box: 2–3 sentences explaining Sharpe and IC in plain English.

## Page 2: Signal Explorer

Driven by the sidebar controls.

- Metric cards across the top: Sharpe, annual return, volatility, max drawdown, turnover, hit rate. Each one shows gross vs. net.
- Equity curve: gross vs. net on the same chart, so the cost impact is obvious.
- Drawdown chart: shaded area below zero.
- Rolling 12-month Sharpe: shows when the signal worked and when it stopped.
- Monthly returns heatmap: years as rows, months as columns, green/red cells.
- Random baseline overlay toggle: puts the random signal on the equity chart for comparison.
- Long vs. short leg breakdown (long-short only): shows which side made the money.

## Page 3: Signal Quality (IC)

- Horizon selector: 5, 21, or 63 days
- IC summary table: mean IC, IC std, IC IR, % positive, t-stat (non-overlapping) for each signal
- Rolling 12-month IC chart: does the signal decay over time?
- Quintile bar chart: average forward return for Q1 through Q5. A good signal climbs from left to right.
- IC histogram: distribution of daily ICs, with the mean marked

## Page 4: Costs & Turnover

Usually where reversal falls apart.

- Cost sensitivity chart: net Sharpe (y) vs. cost in bps (x), one line per signal
- Breakeven cost table: the cost level where each signal's net Sharpe hits zero
- Turnover comparison bar chart: annual turnover per signal
- Monthly vs. weekly comparison: same signal, both frequencies, gross and net Sharpe side by side

## Page 5: Walk-Forward

- Timeline diagram: horizontal bars showing each train window, purge gap, and test year
- Fitted weights chart: stacked bar of each signal's weight in each window
- Out-of-sample comparison: walk-forward combo vs. equal-weight combo vs. single signals (table + equity curve)
- One sentence of honest takeaway, e.g. whether fitting actually beat equal weighting

## Page 6: Methodology & Bias Checks

- Timing diagram: close t → signal computed, close t+1 → trade, t+2 onward → earn returns
- Look-ahead test explanation: how the corruption test works, with a "✅ passing" status pulled from the last CI run or hardcoded at export time
- Survivorship bias note: what the universe is and how it biases results
- Limitations list: same as README
- Count of variations tried (from the `backtest_runs` table)

## Tech details

| Piece | Choice | Why |
|---|---|---|
| Framework | Streamlit (multi-page app) | All Python, fast to build |
| Charts | Plotly | Hover tooltips, zoom, looks clean |
| Data | Parquet files in `dashboard/data/` | Small, fast, no database needed |
| Caching | `@st.cache_data` on every loader | Pages switch instantly |
| Hosting | Streamlit Community Cloud | Free, gives a live link |
| Testing | Streamlit's AppTest | Smoke test that every page loads without errors, run in CI |

Files to export:

- `daily_returns.parquet`: date, signal, portfolio type, rebalance, gross, turnover
- `ic_daily.parquet`: date, signal, horizon, IC
- `quintile_returns.parquet`: signal, horizon, quintile, avg forward return
- `walk_forward_weights.parquet`: window, signal, weight
- `summary.parquet`: headline metrics table
- `metadata.json`: universe, date range, stock count, last updated, variations tried

Folder structure:

```
dashboard/
├── app.py                  # Overview page
├── pages/
│   ├── 1_Signal_Explorer.py
│   ├── 2_Signal_Quality.py
│   ├── 3_Costs_and_Turnover.py
│   ├── 4_Walk_Forward.py
│   └── 5_Methodology.py
├── components/
│   ├── loaders.py          # cached data loading
│   ├── charts.py           # reusable Plotly functions
│   └── sidebar.py          # shared controls
└── data/                   # exported Parquet files
```

Reuse the metrics code: import `metrics.py` from the main package instead of rewriting formulas in the dashboard, so the numbers always match the README.

## Design details

- One fixed color per signal across every chart
- Format everything: percentages as 12.4%, Sharpe to 2 decimals, bps as "10 bps"
- Tooltips on every metric using Streamlit's `help=` parameter, one sentence each
- Light and dark mode both readable
- Clear "Out-of-sample" labels on any chart that uses walk-forward results

## What NOT to include

- Live data downloads or live backtests (slow, fragile, breaks on Streamlit Cloud)
- "Today's top picks" or buy/sell lists (looks like stock tips, not research)
- A "find best parameters" button (it demonstrates overfitting)
- Login or user accounts
- A free date range picker on the headline results (invites cherry-picking). If wanted, only on the Signal Explorer page.

## Time estimate

| Task | Time |
|---|---|
| Export script + loaders | 2–3 hours |
| Overview + Signal Explorer | 3–4 hours |
| IC, Costs, Walk-Forward pages | 3–4 hours |
| Methodology page + polish | 2 hours |
| Deploy + AppTest in CI | 1–2 hours |
