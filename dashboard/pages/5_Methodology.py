"""Methodology & bias checks: timing, look-ahead test, survivorship, limitations."""

from datetime import datetime

import streamlit as st
from components import loaders, sidebar, style

sidebar.setup_page("Methodology", ":material/science:")
meta = loaders.metadata()

style.page_header(
    "Methodology & Bias Checks",
    "Timing, the look-ahead test, survivorship bias, and everything this project doesn't do.",
    eyebrow="How I kept it honest",
)

# Timing diagram.
style.section("Timing: no peeking at the future")
lag = meta.get("exec_lag", 1)
steps = [
    ("Close t", "Compute the signal from prices up to and including today's close."),
    (f"Close t+{lag}", "Trade into the new positions, one day later."),
    (f"t+{lag + 1} onward", "The positions start earning returns."),
]
for column, (when, what) in zip(st.columns(len(steps)), steps, strict=True):
    with column.container(border=True):
        st.markdown(f"**{when}**")
        st.write(what)
st.caption(
    f"In code, holdings are `target_weights.shift({lag + 1})`: one shift because the "
    "return on day d runs from close d-1 to close d, plus the execution lag."
)

# Look-ahead corruption test.
style.section("Look-ahead test")
test = meta.get("lookahead_test") or {}
status = test.get("status", "unknown")
checked = test.get("checked_at")
checked_text = ""
if checked:
    try:
        checked_text = datetime.fromisoformat(checked).strftime("%Y-%m-%d %H:%M")
    except ValueError:
        checked_text = checked
detail = f"{test.get('tests', 'n/a')} tests, checked {checked_text or 'n/a'} at export time."
if status == "passing":
    st.success(f"✅ Passing. {detail}")
elif status == "failing":
    st.error(f"❌ Failing. {detail} Don't trust these results until it passes.")
else:
    st.warning(f"Status unknown. {detail}")
st.markdown(
    "`tests/test_no_lookahead.py` takes a price history, replaces every price after some "
    "date with random garbage, and recomputes. Everything before that date must come out "
    "identical. If a signal secretly used future prices, the garbage would leak back and "
    "the test would fail. To check the test itself, it is also fed three deliberately "
    "leaky functions (`shift(-1)`, a full-period z-score, a centred rolling window), and "
    "all three get caught."
)

# Survivorship bias.
style.section("Survivorship bias")
st.markdown(
    f"The universe is **{meta.get('universe', 'n/a')}** ({meta.get('n_stocks', 'n/a')} "
    "stocks). Companies that went bust or were dropped from the index are missing, and "
    "stocks added after big run-ups are included for their whole history. That inflates "
    "long returns and probably hurts low volatility, since volatile winners are "
    "over-represented."
)

# Limitations.
style.section("Limitations")
limitations = meta.get("limitations") or []
if limitations:
    st.markdown("\n".join(f"- {item}" for item in limitations))
else:
    st.caption("No limitations exported yet.")

# Variations tried.
style.section("How many things I tried")
left, right = st.columns([1, 3])
left.metric(
    "Variations tried",
    meta.get("variations_tried", "n/a"),
    help="Number of backtest runs logged in the backtest_runs table. "
    "More tries means more chances of a lucky result.",
)
right.markdown(
    "Every lookback, the quintile count, and the cost level were fixed in `config.yaml` "
    "before seeing results. The dashboard has no parameter search on purpose: picking "
    "the best of many tries is how backtests overfit."
)
