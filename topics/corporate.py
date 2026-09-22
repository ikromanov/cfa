"""Corporate Issuers — capital structure (MM theorems), dividends & buybacks, ESG."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ── helpers ───────────────────────────────────────────────────────────────────

def firm_value_mm(vu, debt, tax_rate, distress_cost_k, distress_slope):
    """
    Trade-off theory:
    VL = VU + PV(Tax Shield) - PV(Financial Distress Costs)
    Distress costs modelled as exponential beyond a threshold.
    """
    tax_shield  = tax_rate * debt
    # Distress cost rises non-linearly with leverage
    equity_approx = max(vu + tax_shield - debt, 1)
    leverage_ratio = debt / (debt + equity_approx)
    distress = distress_cost_k * np.exp(distress_slope * leverage_ratio) - distress_cost_k
    return vu + tax_shield - distress


def buyback_vs_dividend(earnings, shares, share_price, cash_returned):
    """Compare EPS and price per share under dividend vs. buyback."""
    # Dividend scenario
    div_per_share   = cash_returned / shares
    div_shares      = shares          # shares unchanged
    div_price_after = share_price - div_per_share   # price drops by dividend (ex-div)
    div_eps         = earnings / div_shares

    # Buyback scenario
    shares_repurchased = cash_returned / share_price
    buyback_shares     = shares - shares_repurchased
    buyback_price      = share_price   # theoretically unchanged (MM)
    buyback_eps        = earnings / buyback_shares

    return {
        "Dividend": {"EPS": div_eps, "Shares": div_shares, "Price": div_price_after},
        "Buyback":  {"EPS": buyback_eps, "Shares": buyback_shares, "Price": buyback_price},
    }


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Corporate Issuers")

    tab1, tab2, tab3 = st.tabs([
        "Capital Structure — MM & Trade-off",
        "Dividends vs Buybacks",
        "ESG Analysis",
    ])

    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
## Capital Structure — From MM to Trade-off Theory

### Modigliani-Miller (1958) — No Taxes

**Proposition I:** In a frictionless world, firm value is independent of capital structure:
$$V_L = V_U$$

**Proposition II:** Cost of equity rises linearly with leverage to exactly offset the benefit
of cheaper debt — WACC is constant:
$$r_e = r_0 + (r_0 - r_d) \\cdot \\frac{D}{E}$$

### MM with Corporate Taxes (1963)

Interest is tax-deductible → debt creates a **tax shield**:
$$V_L = V_U + T_c \\cdot D$$

Implication: firms should use *maximum* debt. But this is clearly not optimal in practice.

### Trade-off Theory

Adds **financial distress costs** (direct: bankruptcy fees; indirect: lost customers, talent):
$$V_L = V_U + PV(\\text{Tax Shield}) - PV(\\text{Distress Costs})$$

**Optimal capital structure** maximises $V_L$. The optimal D/E balances the marginal tax
benefit of additional debt against the marginal increase in expected distress costs.

### Pecking Order Theory

Firms prefer: retained earnings → debt → equity (asymmetric information about firm quality).
This predicts no stable optimal leverage — instead, leverage reflects past financing decisions.

> **Exam tip:** MM I (no taxes) → value is independent of structure.
> MM II (no taxes) → cost of equity rises, WACC is constant.
> MM with taxes → use all debt. Trade-off → optimal leverage exists.
> Pecking order → no target, but equity is a last resort.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            vu          = st.slider("Unlevered firm value V_U ($M)", 100, 2000, 500, 50)
            tax_rate    = st.slider("Corporate tax rate (%)", 0, 50, 25, 1) / 100
            r0          = st.slider("Unlevered cost of equity r₀ (%)", 5, 20, 10, 1) / 100
            rd          = st.slider("Cost of debt r_d (%)", 2, 15, 5, 1) / 100
            distress_k  = st.slider("Distress cost base ($M)", 0, 200, 30, 5)
            distress_sl = st.slider("Distress cost slope (severity)", 1.0, 8.0, 3.5, 0.5)

        debt_range = np.linspace(0, vu * 1.5, 300)

        vl_no_tax   = np.full_like(debt_range, vu)                          # MM I no tax
        vl_with_tax = vu + tax_rate * debt_range                            # MM with tax
        vl_tradeoff = np.array([firm_value_mm(vu, d, tax_rate, distress_k, distress_sl)
                                for d in debt_range])

        # WACC under MM no-tax (constant) and MM with tax (declining then rising)
        equity_range = np.maximum(vl_tradeoff - debt_range, 1)
        weight_d     = debt_range / np.maximum(vl_tradeoff, 1)
        re_mm        = r0 + (r0 - rd) * debt_range / np.maximum(equity_range, 1)
        wacc_tradeoff = re_mm * (1 - weight_d) + rd * (1 - tax_rate) * weight_d

        # Optimal point
        opt_idx   = int(np.argmax(vl_tradeoff))
        opt_debt  = debt_range[opt_idx]
        opt_vl    = vl_tradeoff[opt_idx]
        opt_lever = opt_debt / opt_vl * 100

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Firm Value vs Debt", "WACC vs Leverage"],
        )

        fig.add_trace(go.Scatter(
            x=debt_range, y=vl_no_tax,
            mode="lines", line=dict(color="gray", width=1.5, dash="dot"),
            name="MM I (no taxes)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=debt_range, y=vl_with_tax,
            mode="lines", line=dict(color="steelblue", width=2, dash="dash"),
            name="MM (with taxes)",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=debt_range, y=vl_tradeoff,
            mode="lines", line=dict(color="crimson", width=2.5),
            name="Trade-off theory",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[opt_debt], y=[opt_vl],
            mode="markers+text",
            marker=dict(size=14, color="gold", symbol="star", line=dict(width=1, color="black")),
            text=[f"Optimal<br>D=${opt_debt:.0f}M<br>D/V={opt_lever:.0f}%"],
            textposition="top right",
            name="Optimal capital structure",
        ), row=1, col=1)

        fig.update_xaxes(title_text="Debt ($M)", row=1, col=1)
        fig.update_yaxes(title_text="Firm Value ($M)", row=1, col=1)

        # WACC
        leverage_pct = debt_range / np.maximum(vl_tradeoff, 1) * 100
        fig.add_trace(go.Scatter(
            x=leverage_pct, y=wacc_tradeoff * 100,
            mode="lines", line=dict(color="crimson", width=2.5),
            name="WACC (trade-off)",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=leverage_pct, y=re_mm * 100,
            mode="lines", line=dict(color="steelblue", width=1.5, dash="dash"),
            name="Cost of equity r_e",
        ), row=1, col=2)
        fig.add_hline(y=rd * 100, line_color="gray", line_dash="dot",
                      annotation_text=f"r_d(after-tax)={rd*(1-tax_rate)*100:.1f}%",
                      row=1, col=2)
        opt_wacc_pct = leverage_pct[opt_idx]
        fig.add_vline(x=opt_wacc_pct, line_dash="dot", line_color="gold",
                      annotation_text="Min WACC", row=1, col=2)

        fig.update_xaxes(title_text="Debt / Firm Value (%)", row=1, col=2)
        fig.update_yaxes(title_text="Cost of capital (%)", row=1, col=2)
        fig.update_layout(height=470, showlegend=True,
                          legend=dict(orientation="h", yanchor="bottom", y=-0.3))

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Optimal Debt", f"${opt_debt:.0f}M")
            c2.metric("Optimal D/V", f"{opt_lever:.1f}%")
            c3.metric("Max Firm Value", f"${opt_vl:.0f}M",
                      delta=f"+${opt_vl - vu:.0f}M vs unlevered")
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
## Dividends vs Share Repurchases

In a Modigliani-Miller world (no taxes, no transaction costs, no signaling), dividends
and buybacks are **equivalent** in terms of shareholder wealth. But in practice:

| | Dividends | Buybacks |
|---|---|---|
| **Tax** | Often taxed as ordinary income | Capital gains (usually lower rate) |
| **Flexibility** | Cutting dividends signals distress | Can be reduced without stigma |
| **EPS effect** | No change (more shares, same earnings) | EPS rises mechanically (fewer shares) |
| **Signaling** | Initiation signals sustained earnings | May signal management believes stock is cheap |
| **Best use** | Mature companies with stable cash flows | Companies with volatile cash flows |

> **Exam tip:** In a vignette, if a company does a buyback, EPS rises — but this is
> purely mechanical (fewer shares). Price-per-share is *unchanged* in the MM world
> because the remaining shares are worth exactly what was paid out.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            earnings      = st.slider("Annual earnings ($M)", 10, 500, 100, 10)
            shares        = st.slider("Shares outstanding (M)", 10, 500, 100, 10)
            share_price   = st.slider("Current share price ($)", 5, 200, 50, 5)
            cash_to_ret   = st.slider("Cash to return ($M)", 1, 200, 30, 5)

        result = buyback_vs_dividend(earnings, shares, share_price, cash_to_ret)

        # Timeline comparison
        quarters = 8
        q = list(range(quarters))

        # Dividend: price gradually recovers between ex-div dates (simple model)
        div_prices  = [share_price - result["Dividend"]["EPS"] / 4 * (i % 4 == 0) for i in q]
        buyback_prices = [share_price] * quarters   # unchanged in MM

        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["EPS & Price Comparison", "Wealth per Share"],
        )

        methods   = ["Dividend", "Buyback"]
        eps_vals  = [result[m]["EPS"] for m in methods]
        price_vals = [result[m]["Price"] for m in methods]

        fig2.add_trace(go.Bar(
            x=methods, y=eps_vals, name="EPS ($)",
            marker_color=["steelblue", "darkorange"],
            text=[f"${v:.2f}" for v in eps_vals], textposition="auto",
        ), row=1, col=1)
        fig2.add_trace(go.Bar(
            x=methods, y=price_vals, name="Price after ($)",
            marker_color=["steelblue", "darkorange"], opacity=0.4,
            text=[f"${v:.2f}" for v in price_vals], textposition="auto",
            yaxis="y2",
        ), row=1, col=1)

        # Wealth = price + cumulative dividends
        div_per_sh = cash_to_ret / shares
        wealth_div = result["Dividend"]["Price"] + div_per_sh    # price + dividend received
        wealth_bk  = result["Buyback"]["Price"]                   # same total wealth

        fig2.add_trace(go.Bar(
            x=["Dividend", "Buyback"],
            y=[wealth_div, wealth_bk],
            name="Total wealth/share",
            marker_color=["steelblue", "darkorange"],
            text=[f"${wealth_div:.2f}", f"${wealth_bk:.2f}"],
            textposition="auto",
        ), row=1, col=2)

        fig2.update_xaxes(title_text="Method", row=1, col=1)
        fig2.update_yaxes(title_text="EPS ($)", row=1, col=1)
        fig2.update_xaxes(title_text="Method", row=1, col=2)
        fig2.update_yaxes(title_text="$ per share", row=1, col=2)
        fig2.update_layout(height=400, showlegend=True, barmode="group")

        with col2:
            c1, c2 = st.columns(2)
            c1.metric("Dividend EPS", f"${result['Dividend']['EPS']:.2f}")
            c1.metric("Dividend Price (ex-div)", f"${result['Dividend']['Price']:.2f}")
            c2.metric("Buyback EPS", f"${result['Buyback']['EPS']:.2f}",
                      delta=f"+{(result['Buyback']['EPS']/result['Dividend']['EPS']-1)*100:.1f}% vs dividend")
            c2.metric("Buyback Price", f"${result['Buyback']['Price']:.2f}")
            st.plotly_chart(fig2, use_container_width=True)
            st.info("Total shareholder wealth is identical under both methods (MM world). "
                    "EPS is higher post-buyback, but this reflects fewer shares, not more value.")

    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
## ESG Integration in Corporate Analysis

ESG (Environmental, Social, Governance) factors can be **financially material risks**
that affect a company's cash flows, cost of capital, and terminal value in a DCF.

### ESG channels into valuation

| ESG Factor | Financial impact | DCF channel |
|---|---|---|
| **Carbon price / climate risk** | Higher operating costs, stranded assets | Lower FCFF, higher WACC |
| **Water stress** | Supply chain disruption, regulatory costs | Revenue risk |
| **Board independence** | Better governance → lower agency costs | Lower WACC (governance premium) |
| **Supply chain labour** | Reputational, regulatory risk | Revenue + cost risk |
| **Disclosure quality** | Reduced information asymmetry | Lower cost of equity |

### Tornado chart: ESG impact on DCF value

The chart below shows each ESG factor's *potential impact* on the enterprise value
given your assumptions about materiality. This is a **sensitivity analysis**, not a
forecast — it shows which ESG risks matter most.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            base_val   = st.slider("Base DCF enterprise value ($M)", 100, 5000, 1000, 100)
            carbon_wt  = st.slider("Carbon risk materiality (%)", 0, 30, 12, 1)
            water_wt   = st.slider("Water stress materiality (%)", 0, 20, 5, 1)
            gov_wt     = st.slider("Governance quality benefit (%)", 0, 15, 8, 1)
            supply_wt  = st.slider("Supply chain risk (%)", 0, 20, 6, 1)
            disclose_wt = st.slider("Disclosure quality benefit (%)", 0, 10, 4, 1)

        factors = [
            ("Carbon / Climate Risk",      -carbon_wt,  "tomato"),
            ("Water & Resource Stress",    -water_wt,   "darkorange"),
            ("Governance Quality",         +gov_wt,     "steelblue"),
            ("Supply Chain Exposure",      -supply_wt,  "coral"),
            ("Disclosure & Transparency",  +disclose_wt,"mediumseagreen"),
        ]
        factors_sorted = sorted(factors, key=lambda x: abs(x[1]), reverse=True)

        labels  = [f[0] for f in factors_sorted]
        impacts = [base_val * f[1] / 100 for f in factors_sorted]
        colors  = [f[2] for f in factors_sorted]
        net_impact = sum(impacts)
        adj_value  = base_val + net_impact

        fig3 = go.Figure(go.Bar(
            y=labels, x=impacts,
            orientation="h",
            marker_color=colors,
            text=[f"${v:+,.0f}M ({v/base_val*100:+.1f}%)" for v in impacts],
            textposition="auto",
        ))
        fig3.add_vline(x=0, line_color="black", line_width=1.5)
        fig3.update_layout(
            title=f"ESG Impact on Enterprise Value (Base: ${base_val:,}M)",
            xaxis_title="Value Impact ($M)",
            height=380,
        )

        with col2:
            c1, c2 = st.columns(2)
            c1.metric("Base EV", f"${base_val:,.0f}M")
            c2.metric("ESG-Adjusted EV", f"${adj_value:,.0f}M",
                      delta=f"${net_impact:+,.0f}M ({net_impact/base_val*100:+.1f}%)")
            st.plotly_chart(fig3, use_container_width=True)
