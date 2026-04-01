"""
CFA Level II Study App — interactive theory + charts.
Run with:  streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="CFA Level II Study Guide",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from topics import (
    ethics, quant, economics, fsa, corporate,
    equity, fixed_income, derivatives, alternatives,
    portfolio, risk_management,
)

# ── Navigation ────────────────────────────────────────────────────────────────
# Weight tiers from CFA Institute 2026 curriculum
HIGH_WEIGHT = {
    "Ethics & Professional Standards":  ethics,
    "Financial Statement Analysis":     fsa,
    "Equity Valuation":                 equity,
    "Fixed Income":                     fixed_income,
    "Portfolio Management":             portfolio,
}
MID_WEIGHT = {
    "Quantitative Methods":             quant,
    "Economics":                        economics,
    "Corporate Issuers":                corporate,
    "Derivatives":                      derivatives,
    "Alternative Investments":          alternatives,
}
OTHER = {
    "Risk Management":                  risk_management,
}

ALL_TOPICS = {"🏠 Home": None, **HIGH_WEIGHT, **MID_WEIGHT, **OTHER}

with st.sidebar:
    st.title("CFA Level II")
    st.caption("Study Guide · Nov 2026 Exam")
    st.markdown("---")

    st.markdown("**Core topics (10–15% each)**")
    choice = st.radio(
        "nav", list(ALL_TOPICS.keys()),
        label_visibility="collapsed",
        format_func=lambda x: x,
    )
    st.markdown("---")
    # Quick weight reference
    with st.expander("Topic weights"):
        st.markdown("""
| Topic | Weight |
|---|---|
| Ethics | 10–15% |
| FSA | 10–15% |
| Equity | 10–15% |
| Fixed Income | 10–15% |
| Portfolio Mgmt | 10–15% |
| Quant Methods | 5–10% |
| Economics | 5–10% |
| Corporate | 5–10% |
| Derivatives | 5–10% |
| Alternatives | 5–10% |
""")

if choice == "🏠 Home":
    st.title("CFA Level II Study Guide")
    st.markdown("""
Welcome to your interactive CFA Level II prep portal. Every topic below includes
theory with Level II exam tips, interactive charts, and formula references.

### How to use this app

1. Select a topic in the sidebar
2. Read the theory overview — pay attention to the **Exam tip** callouts
3. Use the sliders to build intuition by seeing how parameters change the charts
4. Cross-reference your formula sheet with the equations shown

### Topics and approximate exam weights

| Tier | Topics |
|---|---|
| **10–15% each** | Ethics, Financial Statement Analysis, Equity Valuation, Fixed Income, Portfolio Management |
| **5–10% each** | Quantitative Methods, Economics, Corporate Issuers, Derivatives, Alternative Investments |

> **Level II vs Level I:** The exam tests *application and analysis*, not just recall.
> Every question is a vignette (mini case study) with 6 questions. Practice linking
> theory to realistic scenarios — that's what this app is designed to help with.
""")

    col1, col2, col3 = st.columns(3)
    col1.info("**Exam date:** November 2026")
    col2.info("**Recommended hours:** 300–350")
    col3.info("**Format:** 2 × 44-question vignette sessions")

else:
    module = ALL_TOPICS[choice]
    module.render()
