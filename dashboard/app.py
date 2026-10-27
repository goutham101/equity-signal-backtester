"""Dashboard entry point: page navigation only.

Run from the repo root: streamlit run dashboard/app.py
st.navigation gives each page a proper name and icon in the sidebar
(otherwise the first page would be called "app").
"""

import streamlit as st

PAGES = [
    st.Page(
        "pages/0_Overview.py", title="Overview", icon=":material/space_dashboard:", default=True
    ),
    st.Page("pages/1_Signal_Explorer.py", title="Signal Explorer", icon=":material/query_stats:"),
    st.Page("pages/2_Signal_Quality.py", title="Signal Quality", icon=":material/insights:"),
    st.Page("pages/3_Costs_and_Turnover.py", title="Costs & Turnover", icon=":material/payments:"),
    st.Page("pages/4_Walk_Forward.py", title="Walk-Forward", icon=":material/timeline:"),
    st.Page("pages/5_Methodology.py", title="Methodology", icon=":material/science:"),
]

st.navigation(PAGES).run()
