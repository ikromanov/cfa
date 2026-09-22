"""Equity Valuation — DCF, Dividend Discount Model, relative valuation."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ── helpers ───────────────────────────────────────────────────────────────────

def gordon_growth(d1: float, r: float, g: float):
    """Gordon Growth Model: P = D1 / (r - g)."""
    if r <= g:
        return None
    return d1 / (r - g)


def two_stage_ddm(d0, g1, g2, r, n_stage1):
    """Two-stage DDM: high-growth phase then stable growth terminal value."""
    pv = 0.0
    d = d0
    for t in range(1, n_stage1 + 1):
        d *= 1 + g1
        pv += d / (1 + r) ** t
    # Terminal value at end of stage 1
    d_terminal = d * (1 + g2)
    tv = d_terminal / (r - g2) if r > g2 else np.nan
    pv += tv / (1 + r) ** n_stage1
    return pv


def dcf_value(fcf0, g_explicit, g_terminal, wacc, n_explicit):
    """DCF with explicit forecast period then terminal value (Gordon Growth)."""
    pv = 0.0
    fcf = fcf0
    for t in range(1, n_explicit + 1):
        fcf *= 1 + g_explicit
        pv  += fcf / (1 + wacc) ** t
    tv  = fcf * (1 + g_terminal) / (wacc - g_terminal) if wacc > g_terminal else np.nan
    pv  += tv / (1 + wacc) ** n_explicit
    return pv, tv


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Equity Valuation")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dividend Discount Model", "DCF Valuation", "Relative Valuation",
        "Residual Income Model", "Private Company Valuation",
    ])

    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
## Dividend Discount Model (DDM)

The DDM values a stock as the **present value of all future dividends**.

### Gordon Growth Model (constant growth)

$$P_0 = \\frac{D_1}{r - g} = \\frac{D_0 (1+g)}{r - g}$$

- $D_1$ = next year's expected dividend
- $r$ = required rate of return (from CAPM: $r = r_f + \\beta (r_m - r_f)$)
- $g$ = constant dividend growth rate

### Two-stage DDM

For companies with a temporary high-growth phase before settling at a stable rate:

$$P_0 = \\sum_{t=1}^{n} \\frac{D_t}{(1+r)^t} + \\frac{P_n}{(1+r)^n}$$

where $P_n = D_{n+1} / (r - g_2)$ is the **terminal value**.

> **Exam tip:** DDM is most appropriate for firms that pay dividends and have
> predictable growth (e.g., mature, regulated utilities).
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            d0   = st.slider("D₀ — current annual dividend ($)", 0.10, 10.0, 2.0, 0.10)
            g1   = st.slider("Stage 1 growth rate (%)", 0.0, 30.0, 15.0, 0.5) / 100
            n1   = st.slider("Stage 1 duration (years)", 1, 15, 5)
            g2   = st.slider("Terminal growth rate (%)", 0.0, 10.0, 3.0, 0.5) / 100
            r    = st.slider("Required return (%)", 1.0, 20.0, 9.0, 0.5) / 100

        # Sensitivity grid: price vs r and g2
        r_vals  = np.linspace(0.05, 0.18, 50)
        g2_vals = np.linspace(0.00, 0.08, 50)
        price_grid = np.array([
            [two_stage_ddm(d0, g1, g2v, rv, n1) for rv in r_vals]
            for g2v in g2_vals
        ])
        price_grid = np.where(np.isfinite(price_grid), price_grid, np.nan)
        price_base = two_stage_ddm(d0, g1, g2, r, n1)

        # Dividend path
        years = np.arange(0, n1 + 11)
        divs  = np.zeros(len(years))
        d     = d0
        for i, yr in enumerate(years):
            if yr == 0:
                divs[i] = d0
            elif yr <= n1:
                d *= 1 + g1
                divs[i] = d
            else:
                d *= 1 + g2
                divs[i] = d

        fig = make_subplots(1, 2, subplot_titles=["Dividend Path", "Sensitivity: Price vs r & g_terminal"])

        stage1_mask = years <= n1
        fig.add_trace(go.Bar(
            x=years[stage1_mask], y=divs[stage1_mask],
            name=f"Stage 1 (g={g1*100:.1f}%)", marker_color="steelblue",
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=years[~stage1_mask], y=divs[~stage1_mask],
            name=f"Stage 2 (g={g2*100:.1f}%)", marker_color="mediumseagreen",
        ), row=1, col=1)
        fig.update_xaxes(title_text="Year", row=1, col=1)
        fig.update_yaxes(title_text="Dividend ($)", row=1, col=1)

        fig.add_trace(go.Heatmap(
            z=price_grid,
            x=r_vals * 100,
            y=g2_vals * 100,
            colorscale="RdYlGn",
            colorbar=dict(title="Price ($)", len=0.7),
            zmin=0, zmax=np.nanpercentile(price_grid, 95),
            hovertemplate="r=%{x:.1f}%  g_term=%{y:.1f}%  Price=$%{z:.1f}<extra></extra>",
        ), row=1, col=2)

        # Current point
        fig.add_trace(go.Scatter(
            x=[r * 100], y=[g2 * 100],
            mode="markers",
            marker=dict(symbol="x", size=14, color="black", line=dict(width=2)),
            name="Current", showlegend=True,
        ), row=1, col=2)
        fig.update_xaxes(title_text="Required return r (%)", row=1, col=2)
        fig.update_yaxes(title_text="Terminal growth g₂ (%)", row=1, col=2)
        fig.update_layout(height=480, barmode="overlay")

        with col2:
            if np.isfinite(price_base):
                st.metric("Two-stage DDM price", f"${price_base:.2f}")
            else:
                st.error("r must be greater than terminal growth rate g₂")
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
## Discounted Cash Flow (DCF)

DCF values the firm by discounting **free cash flows to the firm (FCFF)** at the
**weighted average cost of capital (WACC)**:

$$V_0 = \\sum_{t=1}^{n} \\frac{FCFF_t}{(1+WACC)^t} + \\frac{TV_n}{(1+WACC)^n}$$

**Terminal value** (Gordon Growth):

$$TV_n = \\frac{FCFF_n \\times (1 + g)}{WACC - g}$$

### WACC

$$WACC = \\frac{E}{V} r_e + \\frac{D}{V} r_d (1 - t)$$

where $r_e$ is the cost of equity (CAPM), $r_d$ is the pre-tax cost of debt, $t$ is the tax rate.

> **Exam tip:** A higher WACC lowers valuation.  Changes in capital structure affect WACC
> through the debt tax shield and financial distress costs.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            fcf0       = st.slider("FCFF₀ ($M)", 10, 500, 100, 10)
            g_explicit = st.slider("Explicit growth rate (%)", -5.0, 40.0, 12.0, 0.5) / 100
            n_explicit = st.slider("Explicit forecast years", 3, 15, 7)
            g_terminal = st.slider("Terminal growth rate (%)", 0.0, 6.0, 2.5, 0.25) / 100
            wacc       = st.slider("WACC (%)", 4.0, 20.0, 9.0, 0.25) / 100
            shares     = st.slider("Shares outstanding (M)", 1, 500, 50, 5)
            net_debt   = st.slider("Net debt ($M)", 0, 1000, 200, 10)

        value, tv = dcf_value(fcf0, g_explicit, g_terminal, wacc, n_explicit)
        equity_value = value - net_debt
        price_per_share = equity_value / shares if shares > 0 else np.nan

        # Build FCF schedule
        yr_list, fcf_list, pv_list = [], [], []
        fcf = fcf0
        for t in range(1, n_explicit + 1):
            fcf *= 1 + g_explicit
            pv   = fcf / (1 + wacc) ** t
            yr_list.append(t)
            fcf_list.append(fcf)
            pv_list.append(pv)

        pv_tv = tv / (1 + wacc) ** n_explicit if np.isfinite(tv) else np.nan

        fig2 = make_subplots(
            1, 2,
            subplot_titles=["FCF Schedule & Present Values", "Value Composition"],
        )

        fig2.add_trace(go.Bar(
            x=yr_list, y=fcf_list,
            name="FCFF ($M)", marker_color="steelblue", opacity=0.7,
        ), row=1, col=1)
        fig2.add_trace(go.Scatter(
            x=yr_list, y=pv_list,
            name="PV of FCFF", mode="lines+markers",
            line=dict(color="crimson", width=2),
        ), row=1, col=1)
        fig2.update_xaxes(title_text="Year", row=1, col=1)
        fig2.update_yaxes(title_text="$M", row=1, col=1)

        # Waterfall-style: explicit PV vs terminal PV
        pv_explicit_total = sum(pv_list)
        labels = [f"Y{t} PV" for t in yr_list] + ["Terminal Value PV"]
        values = pv_list + [pv_tv]
        colors = ["steelblue"] * len(pv_list) + ["mediumseagreen"]

        fig2.add_trace(go.Bar(
            x=labels, y=values,
            marker_color=colors, name="Value breakdown",
        ), row=1, col=2)
        fig2.update_xaxes(title_text="Component", row=1, col=2)
        fig2.update_yaxes(title_text="Present Value ($M)", row=1, col=2)
        fig2.update_layout(height=460, showlegend=True)

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Enterprise Value", f"${value:,.0f}M")
            c2.metric("Equity Value", f"${equity_value:,.0f}M")
            if np.isfinite(price_per_share):
                c3.metric("Implied Price/Share", f"${price_per_share:.2f}")
            st.plotly_chart(fig2, use_container_width=True)
            if np.isfinite(pv_tv) and value > 0:
                tv_pct = pv_tv / value * 100
                st.info(f"Terminal value = {tv_pct:.1f}% of total enterprise value — "
                        + ("typical for high-growth firms." if tv_pct > 70 else "relatively low; explicit FCFs dominate."))

    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
## Relative Valuation (Multiples)

Instead of discounting cash flows, relative valuation compares a company's price
to a benchmark using **valuation multiples**.

| Multiple | Formula | Best for |
|---|---|---|
| **P/E** | Price / EPS | Profitable companies, same-sector peers |
| **EV/EBITDA** | Enterprise Value / EBITDA | Capital-structure-neutral comparison |
| **P/B** | Price / Book Value per share | Banks, asset-heavy firms |
| **P/S** | Price / Revenue | Pre-profit growth companies |
| **EV/Sales** | EV / Revenue | Capital-structure-neutral; pre-EBITDA stages |

### Justified P/E from fundamentals

Using the Gordon Growth Model, the **justified trailing P/E** is:

$$\\frac{P_0}{E_0} = \\frac{(1-b)(1+g)}{r - g}$$

where $b$ is the **plowback (retention) ratio** and $1-b$ is the **payout ratio**.

> **Exam tip:** A higher growth rate *or* lower required return → higher justified P/E.
> But beware: earnings can be manipulated — EV/EBITDA is often more robust.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            payout = st.slider("Dividend payout ratio (1-b)", 0.10, 1.0, 0.40, 0.05)
            g_pe   = st.slider("Growth rate (%)", 0.0, 15.0, 5.0, 0.5) / 100
            r_pe   = st.slider("Required return (%)", 4.0, 20.0, 10.0, 0.5) / 100

        justified_pe = payout * (1 + g_pe) / (r_pe - g_pe) if r_pe > g_pe else np.nan

        # Sensitivity: P/E vs g and r
        g_range = np.linspace(0.0, 0.12, 60)
        r_range = np.linspace(0.05, 0.18, 60)
        pe_grid = np.array([
            [payout * (1 + gv) / (rv - gv) if rv > gv else np.nan for rv in r_range]
            for gv in g_range
        ])

        fig3 = go.Figure(go.Heatmap(
            z=pe_grid,
            x=r_range * 100,
            y=g_range * 100,
            colorscale="RdYlGn",
            zmin=0, zmax=40,
            colorbar=dict(title="Justified P/E"),
            hovertemplate="r=%{x:.1f}%  g=%{y:.1f}%  P/E=%{z:.1f}x<extra></extra>",
        ))
        fig3.add_trace(go.Scatter(
            x=[r_pe * 100], y=[g_pe * 100],
            mode="markers",
            marker=dict(symbol="x", size=14, color="black", line=dict(width=2)),
            name="Current",
        ))
        fig3.update_layout(
            title="Justified P/E Sensitivity (payout fixed)",
            xaxis_title="Required return r (%)",
            yaxis_title="Growth rate g (%)",
            height=440,
        )

        with col2:
            if np.isfinite(justified_pe):
                st.metric("Justified trailing P/E", f"{justified_pe:.1f}x")
            else:
                st.error("Required return must exceed growth rate.")
            st.plotly_chart(fig3, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("""
## Residual Income Model (RI / EVA)

The **Residual Income** model values a stock as book value plus the present value of
all future *economic profits* — earnings in excess of the required return on equity.

$$P_0 = B_0 + \\sum_{t=1}^{\\infty} \\frac{RI_t}{(1+r)^t}$$

where **Residual Income** is:

$$RI_t = E_t - r \\cdot B_{t-1} = (ROE_t - r) \\cdot B_{t-1}$$

### Clean Surplus Relationship

For the RI model to work, all changes in book value must flow through the income statement:

$$B_t = B_{t-1} + E_t - D_t$$

**Dirty surplus items** (OCI items like FX translation, pension actuarial gains/losses,
unrealised securities gains) violate clean surplus and must be adjusted.

### Economic Value Added (EVA)

$$EVA = NOPAT - (WACC \\times \\text{Invested Capital})$$

EVA is the firm-level equivalent of RI. Positive EVA = the firm is earning more than
its cost of capital → value creation.

> **Exam tip:** RI is especially useful for firms that do NOT pay dividends (DDM fails)
> and when FCF is negative (DCF difficult). It anchors valuation to observable book value.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            b0_ri   = st.slider("Book value per share B₀ ($)", 5.0, 100.0, 20.0, 1.0)
            roe1    = st.slider("ROE Stage 1 (%)", 5.0, 40.0, 18.0, 0.5) / 100
            roe2    = st.slider("ROE Stable (%)", 5.0, 20.0, 10.0, 0.5) / 100
            r_ri    = st.slider("Required return r (%)", 5.0, 20.0, 10.0, 0.5) / 100
            n_ri    = st.slider("High-ROE phase (years)", 1, 15, 6)
            payout_ri = st.slider("Payout ratio", 0.0, 1.0, 0.4, 0.05)

        # Build RI schedule
        ri_years, ri_vals, bv_vals, eps_vals = [], [], [], []
        bv = b0_ri
        pv_ri_total = 0.0
        n_total = n_ri + 12

        for t in range(1, n_total + 1):
            roe = roe1 if t <= n_ri else roe2
            eps = roe * bv
            ri  = eps - r_ri * bv
            div = payout_ri * eps
            pv_ri = ri / (1 + r_ri) ** t
            pv_ri_total += pv_ri
            ri_years.append(t)
            ri_vals.append(ri)
            bv_vals.append(bv)
            eps_vals.append(eps)
            bv = bv + eps - div

        price_ri = b0_ri + pv_ri_total
        spread_per_year = [(roe1 if t <= n_ri else roe2) - r_ri for t in ri_years]

        fig4 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Residual Income Schedule", "Price Decomposition"],
        )

        ri_colors = ["steelblue" if v > 0 else "tomato" for v in ri_vals]
        fig4.add_trace(go.Bar(
            x=ri_years[:20], y=ri_vals[:20],
            marker_color=ri_colors[:20], name="RI per year",
        ), row=1, col=1)
        fig4.add_vline(x=n_ri + 0.5, line_dash="dash", line_color="gray",
                       annotation_text="Stable ROE", row=1, col=1)
        fig4.add_hline(y=0, line_color="black", row=1, col=1)
        fig4.update_xaxes(title_text="Year", row=1, col=1)
        fig4.update_yaxes(title_text="RI per share ($)", row=1, col=1)

        # Price decomposition
        pv_high_growth = sum(ri_vals[t-1] / (1 + r_ri)**t for t in range(1, n_ri + 1))
        pv_stable      = sum(ri_vals[t-1] / (1 + r_ri)**t for t in range(n_ri + 1, n_total + 1))

        fig4.add_trace(go.Bar(
            x=["Book Value B₀", "PV of RI\n(Stage 1)", "PV of RI\n(Stable)", "Total Price"],
            y=[b0_ri, pv_high_growth, pv_stable, price_ri],
            marker_color=["steelblue", "mediumseagreen", "darkorange", "gold"],
            text=[f"${v:.2f}" for v in [b0_ri, pv_high_growth, pv_stable, price_ri]],
            textposition="auto",
            name="Value bridge",
        ), row=1, col=2)
        fig4.update_xaxes(title_text="Component", row=1, col=2)
        fig4.update_yaxes(title_text="$ per share", row=1, col=2)
        fig4.update_layout(height=430, showlegend=False)

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Intrinsic Price (RI)", f"${price_ri:.2f}")
            c2.metric("Book Value B₀", f"${b0_ri:.2f}")
            c3.metric("PV of Future RI", f"${pv_ri_total:.2f}")
            above_below = "above" if price_ri > b0_ri else "at/below"
            st.info(f"Price {above_below} book: ROE {'> ' if roe1 > r_ri else '< '}r means "
                    f"the firm {'creates' if roe1 > r_ri else 'destroys'} shareholder value.")
            st.plotly_chart(fig4, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("""
## Private Company Valuation

Private companies lack market prices, requiring explicit adjustments for differences
in liquidity, control, and information relative to public peers.

### Three main approaches

| Approach | Methods | When to use |
|---|---|---|
| **Income** | DCF with size premium | When future cash flows are predictable |
| **Market** | Guideline public company multiples, M&A transactions | When comparable companies exist |
| **Asset** | Adjusted book value | Asset-heavy firms, liquidation scenarios |

### Key valuation discounts (and premiums)

$$\\text{Private Company Value} = \\text{Control Value} \\times (1 - DLOM)$$

$$\\text{Control Value} = \\text{Minority Value} \\times (1 + \\text{Control Premium})$$

| Adjustment | Typical range | Rationale |
|---|---|---|
| **Control Premium** | 20–40% | Acquirer gains strategic control |
| **DLOC** (Discount for Lack of Control) | 15–30% | Minority shareholder has no influence |
| **DLOM** (Discount for Lack of Marketability) | 20–35% | Cannot sell quickly at full value |
| **Key Person Discount** | 5–20% | Business depends on individual(s) |

> **Exam tip:** DLOC and DLOM are *multiplicative*, not additive.
> Applying 25% DLOC then 30% DLOM: value = base × (1 − 0.25) × (1 − 0.30) = 0.525 × base.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            peer_ev_ebitda   = st.slider("Peer group EV/EBITDA multiple", 5.0, 20.0, 10.0, 0.5)
            private_ebitda   = st.slider("Private company EBITDA ($M)", 5, 200, 30, 5)
            size_discount    = st.slider("Size/illiquidity WACC premium (%)", 0.0, 5.0, 2.0, 0.25) / 100
            dloc             = st.slider("DLOC — Lack of Control (%)", 0, 40, 20, 1) / 100
            dlom             = st.slider("DLOM — Lack of Marketability (%)", 0, 40, 25, 1) / 100
            key_person       = st.slider("Key Person Discount (%)", 0, 25, 8, 1) / 100
            net_debt_priv    = st.slider("Net Debt ($M)", 0, 500, 50, 10)
            shares_priv      = st.slider("Shares outstanding (M)", 1, 50, 10)

        # Control value (peer multiple, no discounts)
        control_ev     = peer_ev_ebitda * private_ebitda
        control_equity = control_ev - net_debt_priv
        control_ps     = control_equity / shares_priv

        # Minority (non-control) value
        minority_ps = control_ps * (1 - dloc)

        # Marketable minority value already in minority_ps
        # Apply DLOM for illiquid minority interest
        illiquid_minority_ps = minority_ps * (1 - dlom)

        # Key person discount on top
        final_ps = illiquid_minority_ps * (1 - key_person)

        # Build waterfall
        wf_labels = [
            "Control Value/share",
            "DLOC",
            "DLOM",
            "Key Person",
            "Final Value/share",
        ]
        wf_vals = [
            control_ps,
            -control_ps * dloc,
            -minority_ps * dlom,
            -illiquid_minority_ps * key_person,
            0,
        ]
        wf_measures = ["absolute", "relative", "relative", "relative", "total"]

        fig5 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Valuation Discount Waterfall", "Sensitivity: DLOC vs DLOM"],
        )

        fig5.add_trace(go.Waterfall(
            orientation="v",
            measure=wf_measures,
            x=wf_labels,
            y=wf_vals,
            decreasing=dict(marker_color="tomato"),
            increasing=dict(marker_color="mediumseagreen"),
            totals=dict(marker_color="steelblue"),
            text=[f"${abs(v):.2f}" if m == "relative" else f"${v:.2f}"
                  for v, m in zip(wf_vals, wf_measures)],
            textposition="outside",
            connector=dict(line=dict(color="gray")),
        ), row=1, col=1)
        fig5.update_yaxes(title_text="$ per share", row=1, col=1)

        # Sensitivity heatmap
        dloc_r = np.linspace(0, 0.40, 40)
        dlom_r = np.linspace(0, 0.40, 40)
        grid   = np.outer(
            1 - dloc_r,
            1 - dlom_r,
        ) * control_ps * (1 - key_person)

        fig5.add_trace(go.Heatmap(
            z=grid,
            x=dlom_r * 100,
            y=dloc_r * 100,
            colorscale="RdYlGn",
            colorbar=dict(title="Value/share ($)", len=0.7),
            hovertemplate="DLOC=%{y:.0f}%  DLOM=%{x:.0f}%  Value=$%{z:.2f}<extra></extra>",
        ), row=1, col=2)
        fig5.add_trace(go.Scatter(
            x=[dlom * 100], y=[dloc * 100],
            mode="markers",
            marker=dict(symbol="x", size=14, color="white", line=dict(width=2)),
            name="Current", showlegend=True,
        ), row=1, col=2)
        fig5.update_xaxes(title_text="DLOM (%)", row=1, col=2)
        fig5.update_yaxes(title_text="DLOC (%)", row=1, col=2)
        fig5.update_layout(height=450, showlegend=False)

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Control Value/share", f"${control_ps:.2f}")
            c2.metric("After DLOC + DLOM", f"${illiquid_minority_ps:.2f}",
                      delta=f"−{(1 - illiquid_minority_ps/control_ps)*100:.0f}%")
            c3.metric("Final Value/share", f"${final_ps:.2f}",
                      delta=f"−{(1 - final_ps/control_ps)*100:.0f}% total discount")
            st.plotly_chart(fig5, use_container_width=True)
