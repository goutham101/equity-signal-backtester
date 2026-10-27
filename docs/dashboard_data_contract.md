# Dashboard data contract

The files `scripts/06_export_dashboard_data.py` writes to `dashboard/data/`, and the
only files the Streamlit app reads. The design is in `docs/dashboard_spec.md`.

Rules:
- The app never touches Postgres, yfinance, or `data/processed/`. Only `dashboard/data/`.
- Everything is long format. Dates are `datetime64[ns]`, no timezone.
- Returns are daily decimals (0.01 = 1%). Turnover is the sum of absolute weight changes that day.
- Net returns are NOT stored. The app computes them with
  `backtester.analysis.metrics.net_returns_at_cost(frame, bps)` = `gross - turnover * bps / 10_000`.
  That's what makes the cost slider live.

## Signal keys and labels

| key | label | color |
|---|---|---|
| `momentum` | Momentum | `#6366F1` |
| `reversal` | Reversal | `#F59E0B` |
| `low_vol` | Low Vol | `#10B981` |
| `equal_weight` | Equal-Weight Combo | `#A855F7` |
| `walk_forward` | Walk-Forward Combo | `#F43F5E` |
| `random` | Random Baseline | `#94A3B8` |
| `benchmark` | Equal-Weight Benchmark (all stocks) | `#0EA5E9` |

`benchmark` only shows up as the comparison for long-only.

## Files

### `daily_returns.parquet`
One row per date × series.

| column | type | notes |
|---|---|---|
| `date` | datetime | |
| `signal` | str | key from the table above |
| `portfolio` | str | `long_short` or `long_only` (`benchmark` rows use `long_only`) |
| `rebalance` | str | `monthly` or `weekly` |
| `period` | str | `full` (2010–2026) or `oos` (walk-forward test dates only) |
| `gross` | float | engine `gross` |
| `turnover` | float | engine `turnover` |
| `gross_long` | float | engine `gross_long` |
| `gross_short` | float | engine `gross_short` |

What exists:
- `period=full`: momentum, reversal, low_vol, equal_weight, random × long_short/long_only × monthly/weekly, plus benchmark (signal=`benchmark`, portfolio=`long_only`) × monthly/weekly.
  There's one benchmark per signal in `data/processed` (each one sits in cash during that signal's warm-up). Export the momentum one as the canonical benchmark.
- `period=oos`: walk_forward, momentum, reversal, low_vol, equal_weight, random × long_short × monthly only. Every signal is cut to the same test dates, exactly as in `scripts/05_walk_forward.py`.
- Walk-Forward Combo exists **only** as oos / long_short / monthly. The app must say so when other options are picked. Don't fake it.
- Rows are already trimmed to start at the first trade (`metrics.trim_to_active`).

### `ic_daily.parquet`
`date`, `signal`, `horizon` (5, 21, 63), `ic`. Signals: momentum, reversal, low_vol, equal_weight, random. `min_stocks` filter from config already applied.

### `ic_summary.parquet`
`signal`, `horizon`, `mean_ic`, `ic_std`, `ic_ir`, `pct_positive`, `t_stat` (Newey-West, h lags), `n_dates`. Same numbers as `reports/results.md`.

### `quintile_returns.parquet`
`signal`, `horizon`, `quintile` (1–5, 1 = lowest signal), `avg_fwd_return`.

### `walk_forward_weights.parquet`
`window` (test-year label str), `signal`, `weight`.

### `walk_forward_windows.parquet`
`window`, `train_start`, `train_end`, `test_start`, `test_end`.

### `summary.parquet`
The Overview headline table: **out-of-sample period, long-short, monthly, 10 bps**. One row per signal: walk_forward, momentum, reversal, low_vol, equal_weight, random.
Columns: `signal`, `sharpe_net`, `ann_return_net`, `ann_vol_net`, `max_drawdown_net`, `mean_ic_21d` (full-period 21d mean IC; NaN for walk_forward), `ann_turnover`. Computed with `metrics.summary` so the numbers match `reports/results.md`.

### `metadata.json`
```json
{
  "universe": "Current S&P 500 constituents",
  "n_stocks": 498,
  "start_date": "2010-01-04",
  "end_date": "2026-10-26",
  "oos_start": "2015-01-02",
  "oos_end": "2026-10-26",
  "last_updated": "<ISO timestamp of export>",
  "variations_tried": 21,
  "default_cost_bps": 10,
  "exec_lag": 1,
  "quantiles": 5,
  "walk_forward_method": "ic_weighted",
  "lookahead_test": {"status": "passing", "checked_at": "<ISO>", "tests": 22},
  "key_findings": ["...bullets from README '## Key findings'..."],
  "limitations": ["...bullets from README '## Limitations'..."],
  "links": {"github": "https://github.com/goutham101/equity-signal-backtester",
            "readme": "https://github.com/goutham101/equity-signal-backtester#readme",
            "linkedin": null}
}
```
- `variations_tried` = row count of `backtest_runs`.
- `lookahead_test` comes from actually running `pytest tests/test_no_lookahead.py` at export time. If it fails, status = `failing`, and the export prints a loud warning.
- `key_findings` and `limitations` are parsed from README.md so there's one source of truth. Keep the markdown (bold etc.); strip the leading `- `.
- `linkedin` stays null until Goutham gives the URL. The app hides the link when it's null.
