"""Alternative Investments — private equity (LBO/VC), hedge funds, real assets."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ── helpers ───────────────────────────────────────────────────────────────────

def lbo_returns(entry_ev, entry_ebitda_mult, exit_ebitda_mult,
                ebitda_growth, debt_pct, holding_years, mgmt_fee_pct=0.02):
    """
    Compute LBO IRR and MOIC.
    Returns: equity_invested, exit_equity, irr, moic, irr_sources breakdown.
    """
    entry_ebitda  = entry_ev / entry_ebitda_mult
    entry_debt    = entry_ev * debt_pct
    equity_in     = entry_ev - entry_debt

    exit_ebitda   = entry_ebitda * (1 + ebitda_growth) ** holding_years
    exit_ev       = exit_ebitda * exit_ebitda_mult

    # Simplified: assume 50% debt paydown over holding period
    remaining_debt = entry_debt * 0.5
    exit_equity    = max(exit_ev - remaining_debt, 0)

    # Management fees (rough drag on equity)
    fee_drag = equity_in * mgmt_fee_pct * holding_years
    net_equity_out = exit_equity - fee_drag

    moic = net_equity_out / equity_in if equity_in > 0 else 0
    irr  = moic ** (1 / holding_years) - 1 if moic > 0 else -1.0

    # IRR attribution (simplified decomposition)
    ebitda_growth_contribution = (exit_ebitda_mult / entry_ebitda_mult) ** 0 * \
                                  ((1 + ebitda_growth) ** holding_years - 1)
    multiple_expansion = (exit_ebitda_mult - entry_ebitda_mult) / entry_ebitda_mult
    debt_paydown_contrib = (entry_debt - remaining_debt) / equity_in

    return dict(
        equity_in=equity_in,
        exit_equity=net_equity_out,
        irr=irr,
        moic=moic,
        ebitda_growth_contrib=ebitda_growth_contribution,
        multiple_expansion=multiple_expansion,
        debt_paydown_contrib=debt_paydown_contrib,
    )


def j_curve(equity_in, irr, holding_years, fee_pct=0.02, n_quarters=None):
    """Simulate PE fund J-curve: early negative NAV due to fees + slow deployment."""
    if n_quarters is None:
        n_quarters = holding_years * 4 + 8
    quarters = np.arange(n_quarters)
    nav = np.zeros(n_quarters)
    cum_invested = np.zeros(n_quarters)
    # Deploy capital over first 3 years
    deploy_quarters = min(12, n_quarters)
    for q in range(n_quarters):
        deployed_pct = min(q / deploy_quarters, 1.0)
        # Fees on committed capital
        fees = equity_in * fee_pct / 4
        # NAV = deployed capital growing at IRR minus fees
        deployed = equity_in * deployed_pct
        growth   = deployed * ((1 + irr) ** (q / 4) - 1)
        nav[q]   = deployed + growth - fees * q
        cum_invested[q] = deployed
    return quarters, nav, cum_invested


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Alternative Investments")

    tab1, tab2, tab3 = st.tabs([
        "Private Equity — LBO & J-Curve",
        "Hedge Fund Strategies",
        "Real Assets — REITs",
    ])

    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
## Private Equity — LBO Mechanics

A **Leveraged Buyout (LBO)** acquires a company using mostly debt (typically 60–70%
of enterprise value), with the equity sponsor contributing the rest.

### Value creation in an LBO

$$\\text{IRR} \\approx \\left(\\frac{\\text{Exit Equity}}{\\text{Entry Equity}}\\right)^{1/T} - 1$$

The three levers of LBO returns:

| Lever | Mechanism | Exam focus |
|---|---|---|
| **EBITDA growth** | Revenue growth, margin expansion, operational improvements | Sustainable vs financial engineering |
| **Multiple expansion** | Buy cheap, sell expensive | Market timing risk |
| **Debt paydown** | Free cash flow reduces debt → equity grows | Requires stable cash flows |

### MOIC vs IRR

**MOIC** (Multiple on Invested Capital) = Exit Equity / Entry Equity — simple,
ignores time. **IRR** accounts for time — the same MOIC achieved faster has a
higher IRR. Both metrics are reported because MOIC captures magnitude, IRR captures speed.

> **Exam tip:** A high MOIC over a long holding period may have a lower IRR than
> a moderate MOIC over a short period. Always consider both.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            entry_ev    = st.slider("Entry Enterprise Value ($M)", 100, 5000, 1000, 50)
            entry_mult  = st.slider("Entry EV/EBITDA multiple", 4.0, 16.0, 8.0, 0.5)
            exit_mult   = st.slider("Exit EV/EBITDA multiple", 4.0, 18.0, 9.0, 0.5)
            ebitda_g    = st.slider("EBITDA growth rate (%/yr)", -5.0, 25.0, 10.0, 0.5) / 100
            debt_pct    = st.slider("Debt / EV at entry (%)", 30, 80, 60, 5) / 100
            hold_yrs    = st.slider("Holding period (years)", 3, 10, 5)

        res = lbo_returns(entry_ev, entry_mult, exit_mult, ebitda_g, debt_pct, hold_yrs)

        # IRR sources waterfall
        labels_wf = [
            "Entry Equity", "EBITDA Growth", "Multiple Expansion",
            "Debt Paydown", "Fees & Costs", "Exit Equity",
        ]
        values_wf = [
            res["equity_in"],
            res["equity_in"] * res["ebitda_growth_contrib"],
            res["equity_in"] * res["multiple_expansion"],
            res["equity_in"] * res["debt_paydown_contrib"],
            -(res["equity_in"] * 0.05 * hold_yrs),   # rough fees
            0,
        ]
        measures_wf = ["absolute", "relative", "relative", "relative", "relative", "total"]

        # J-curve
        q_arr, nav_arr, cum_inv = j_curve(res["equity_in"], res["irr"], hold_yrs)
        irr_over_time = [
            ((max(nav_arr[q], 0.01) / res["equity_in"]) ** (4 / max(q, 1)) - 1) * 100
            if q > 0 else -100
            for q in q_arr
        ]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["LBO Value Bridge", "J-Curve: NAV & Running IRR"],
        )

        fig.add_trace(go.Waterfall(
            orientation="v",
            measure=measures_wf,
            x=labels_wf,
            y=values_wf,
            decreasing=dict(marker_color="tomato"),
            increasing=dict(marker_color="mediumseagreen"),
            totals=dict(marker_color="steelblue"),
            connector=dict(line=dict(color="gray", width=1)),
            text=[f"${v:+,.0f}M" if m == "relative" else f"${v:,.0f}M"
                  for v, m in zip(values_wf, measures_wf)],
            textposition="outside",
        ), row=1, col=1)
        fig.update_yaxes(title_text="$M", row=1, col=1)

        fig.add_trace(go.Scatter(
            x=q_arr / 4, y=nav_arr,
            mode="lines", line=dict(color="steelblue", width=2),
            name="NAV ($M)",
        ), row=1, col=2)
        fig.add_hline(y=0, line_color="black", line_dash="dash", row=1, col=2)
        fig.add_trace(go.Scatter(
            x=q_arr / 4, y=irr_over_time,
            mode="lines", line=dict(color="crimson", width=1.5, dash="dash"),
            name="Running IRR (%)", yaxis="y4",
        ), row=1, col=2)

        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="NAV ($M)", row=1, col=2)
        fig.update_layout(height=470, showlegend=True)

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Entry Equity", f"${res['equity_in']:,.0f}M")
            c2.metric("Exit Equity", f"${res['exit_equity']:,.0f}M")
            c3.metric("MOIC", f"{res['moic']:.2f}x")
            irr_color = "normal" if res['irr'] > 0.15 else "off"
            st.metric("IRR", f"{res['irr']*100:.1f}%",
                      delta="Strong (>15%)" if res['irr'] > 0.15 else "Weak (<15%)",
                      delta_color=irr_color)
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
## Hedge Fund Strategies

Hedge funds use a wide range of strategies with distinct risk-return profiles.
Many have **non-normal return distributions** — the Sharpe ratio alone can be misleading.

### Key strategies

| Strategy | Mechanism | Risk profile |
|---|---|---|
| **Long/Short Equity** | Long undervalued, short overvalued stocks | Market exposure + idiosyncratic |
| **Global Macro** | Directional bets on currencies, rates, commodities | High volatility, fat tails |
| **Event-Driven** | M&A arbitrage, distressed debt, spin-offs | Left-tail risk at deal break |
| **Relative Value** | Fixed income arb, convertible arb | Leverage-dependent; liquidity risk |
| **Managed Futures (CTA)** | Momentum across asset classes | Positive skew, crisis alpha |

### Why Sharpe ratio misleads

Strategies like **merger arbitrage** and **short volatility** have positive Sharpe
ratios most of the time — but crash during tail events. The **Sortino ratio** (downside
deviation only) and **skewness/kurtosis** are more informative for these strategies.

> **Exam tip:** Know that 2-and-20 fee structures (2% AUM + 20% performance) create
> strong incentive alignment but also risk-taking incentives. High-water marks protect
> investors from paying twice for the same gains.
""")

        # Strategy data (approximate annualised, stylised)
        strategies = {
            "Long/Short Equity":      dict(ret=12, vol=12, sharpe=0.9, skew=-0.3, kurt=3.5, drawdown=18),
            "Global Macro":           dict(ret=10, vol=15, sharpe=0.6, skew=0.1,  kurt=4.0, drawdown=25),
            "Merger Arbitrage":       dict(ret=7,  vol=5,  sharpe=1.1, skew=-2.5, kurt=8.0, drawdown=8),
            "Fixed Income Arb":       dict(ret=8,  vol=6,  sharpe=1.0, skew=-1.8, kurt=6.5, drawdown=15),
            "Managed Futures (CTA)":  dict(ret=8,  vol=14, sharpe=0.5, skew=0.8,  kurt=3.0, drawdown=20),
            "Distressed Debt":        dict(ret=13, vol=13, sharpe=0.85,skew=-1.2, kurt=5.5, drawdown=30),
            "Equity (S&P 500)":       dict(ret=10, vol=16, sharpe=0.5, skew=-0.5, kurt=3.5, drawdown=34),
            "Bonds (Aggregate)":      dict(ret=4,  vol=5,  sharpe=0.5, skew=0.1,  kurt=2.8, drawdown=10),
        }

        col1, col2 = st.columns([1, 2])
        with col1:
            view = st.selectbox("Chart view", [
                "Return vs Volatility (Sharpe)",
                "Sharpe vs Max Drawdown",
                "Skewness vs Excess Kurtosis",
            ])
            show_labels = st.checkbox("Show labels", value=True)

        names  = list(strategies.keys())
        vals   = list(strategies.values())
        colors = ["steelblue"] * (len(names) - 2) + ["crimson", "mediumseagreen"]

        if view == "Return vs Volatility (Sharpe)":
            xs = [v["vol"] for v in vals]
            ys = [v["ret"] for v in vals]
            xt, yt = "Volatility (% ann.)", "Return (% ann.)"
            # CML-like line
            rf_approx = 3
            x_cml = np.linspace(0, max(xs) * 1.1, 100)
            max_sh = max(v["sharpe"] for v in vals)
            y_cml  = rf_approx + max_sh * x_cml
        elif view == "Sharpe vs Max Drawdown":
            xs = [v["drawdown"] for v in vals]
            ys = [v["sharpe"] for v in vals]
            xt, yt = "Max Drawdown (%)", "Sharpe Ratio"
        else:
            xs = [v["skew"] for v in vals]
            ys = [v["kurt"] - 3 for v in vals]   # excess kurtosis
            xt, yt = "Skewness", "Excess Kurtosis (fat tails)"

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text" if show_labels else "markers",
            marker=dict(color=colors, size=14, opacity=0.85,
                        line=dict(width=1, color="black")),
            text=names if show_labels else None,
            textposition="top center",
            hovertemplate="%{text}<br>x=%{x:.1f}  y=%{y:.2f}<extra></extra>",
        ))

        if view == "Return vs Volatility (Sharpe)" and 'y_cml' in dir():
            fig2.add_trace(go.Scatter(
                x=x_cml, y=y_cml,
                mode="lines", line=dict(color="gray", dash="dot", width=1.5),
                name=f"Best Sharpe={max_sh:.2f}",
            ))

        if view == "Skewness vs Excess Kurtosis":
            fig2.add_vline(x=0, line_color="gray", line_dash="dash")
            fig2.add_hline(y=0, line_color="gray", line_dash="dash")
            fig2.add_annotation(x=-2.5, y=5,
                text="Crash risk<br>(neg. skew, fat tail)",
                font=dict(color="crimson"), showarrow=False)
            fig2.add_annotation(x=0.8, y=-0.5,
                text="Positive skew<br>(crisis alpha)",
                font=dict(color="steelblue"), showarrow=False)

        fig2.update_layout(
            title=view, xaxis_title=xt, yaxis_title=yt,
            height=520, showlegend=False,
        )

        with col2:
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Note: All figures are stylised approximations for educational purposes.")

    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
## Real Assets — REIT Valuation

**REITs** (Real Estate Investment Trusts) must distribute ≥ 90% of taxable income.
Because of heavy depreciation charges, GAAP net income understates cash flow.

### FFO vs EPS

$$FFO = \\text{Net Income} + \\text{Depreciation} - \\text{Gains on Property Sales}$$

$$AFFO = FFO - \\text{Recurring Capex (maintenance)}$$

**FFO** is the primary REIT earnings metric. AFFO is more conservative and
closer to true distributable cash flow.

### Valuation approaches

| Method | Formula | Notes |
|---|---|---|
| **NAV** | (NOI / Cap Rate) − Net Debt | Most fundamentally driven |
| **P/FFO** | Price / FFO per share | REIT equivalent of P/E |
| **P/AFFO** | Price / AFFO per share | More conservative; preferred for dividend sustainability |
| **Dividend Yield** | DPS / Price | Relevant because of 90% distribution requirement |

> **Exam tip:** Use NAV when property values are clearly observable (e.g., office in
> active market). Use P/FFO for cross-REIT comparisons. Know that depreciation
> artificially depresses REIT net income — always adjust.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            noi          = st.slider("Net Operating Income — NOI ($M)", 10, 500, 80, 5)
            cap_rate     = st.slider("Cap rate (%)", 3.0, 10.0, 5.5, 0.25) / 100
            net_debt     = st.slider("Net debt ($M)", 0, 2000, 400, 25)
            shares_reit  = st.slider("Shares outstanding (M)", 10, 500, 100, 10)
            depreciation = st.slider("D&A expense ($M)", 5, 200, 50, 5)
            gaap_income  = st.slider("GAAP Net Income ($M)", -20, 200, 20, 5)
            prop_gains   = st.slider("Gains on property sales ($M)", 0, 100, 5, 5)
            maint_capex  = st.slider("Maintenance capex ($M)", 5, 100, 20, 5)
            price_ps     = st.slider("Current share price ($)", 5, 100, 35, 1)

        gross_value    = noi / cap_rate
        nav            = gross_value - net_debt
        nav_per_share  = nav / shares_reit

        ffo            = gaap_income + depreciation - prop_gains
        affo           = ffo - maint_capex
        ffo_per_share  = ffo / shares_reit
        affo_per_share = affo / shares_reit

        p_ffo   = price_ps / ffo_per_share if ffo_per_share > 0 else np.nan
        p_affo  = price_ps / affo_per_share if affo_per_share > 0 else np.nan
        premium = (price_ps - nav_per_share) / nav_per_share * 100

        fig3 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Earnings Quality Bridge (per share)", "Valuation Multiples vs NAV"],
        )

        # Earnings bridge
        eps     = gaap_income / shares_reit
        ffo_ps  = ffo_per_share
        affo_ps = affo_per_share

        bridge_labels = ["GAAP EPS", "+ D&A/share", "− Property Gains/share",
                         "= FFO/share", "− Maint. CapEx/share", "= AFFO/share"]
        bridge_values = [
            eps,
            depreciation / shares_reit,
            -prop_gains / shares_reit,
            0,
            -maint_capex / shares_reit,
            0,
        ]
        bridge_measures = ["absolute","relative","relative","total","relative","total"]

        fig3.add_trace(go.Waterfall(
            orientation="v",
            measure=bridge_measures,
            x=bridge_labels,
            y=bridge_values,
            decreasing=dict(marker_color="tomato"),
            increasing=dict(marker_color="mediumseagreen"),
            totals=dict(marker_color="steelblue"),
            text=[f"${v:+.2f}" if m == "relative" else f"${v:.2f}"
                  for v, m in zip(bridge_values, bridge_measures)],
            textposition="outside",
            connector=dict(line=dict(color="gray", width=1)),
        ), row=1, col=1)
        fig3.update_yaxes(title_text="$ per share", row=1, col=1)

        # Multiples
        mult_names = ["P/FFO", "P/AFFO", "Price/NAV"]
        mult_vals  = [p_ffo, p_affo, price_ps / nav_per_share]
        mult_colors = [
            "steelblue" if not np.isnan(p_ffo) else "gray",
            "steelblue" if not np.isnan(p_affo) else "gray",
            "mediumseagreen" if premium < 10 else "darkorange",
        ]
        safe_vals = [v if np.isfinite(v) else 0 for v in mult_vals]
        fig3.add_trace(go.Bar(
            x=mult_names, y=safe_vals, marker_color=mult_colors,
            text=[f"{v:.1f}x" if np.isfinite(v) else "N/A" for v in mult_vals],
            textposition="auto", name="Multiple",
        ), row=1, col=2)
        fig3.update_yaxes(title_text="Multiple (x)", row=1, col=2)

        fig3.update_layout(height=440, showlegend=False)

        with col2:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("NAV/share", f"${nav_per_share:.2f}")
            c2.metric("FFO/share", f"${ffo_per_share:.2f}")
            c3.metric("AFFO/share", f"${affo_per_share:.2f}")
            c4.metric("Price/NAV", f"{price_ps/nav_per_share:.2f}x",
                      delta=f"{premium:+.1f}% to NAV",
                      delta_color="off")
            st.plotly_chart(fig3, use_container_width=True)
