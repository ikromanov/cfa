"""Derivatives — option payoffs, Black-Scholes, forwards/futures, swaps."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st


# ── Black-Scholes ─────────────────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes option price."""
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def bs_greeks(S, K, T, r, sigma, option_type="call"):
    if T <= 1e-6:
        return dict(delta=np.nan, gamma=np.nan, theta=np.nan, vega=np.nan, rho=np.nan)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    sign = 1 if option_type == "call" else -1
    delta = sign * stats.norm.cdf(sign * d1)
    gamma = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * stats.norm.pdf(d1) * np.sqrt(T) / 100          # per 1% vol move
    theta = (-(S * stats.norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
             - sign * r * K * np.exp(-r * T) * stats.norm.cdf(sign * d2)) / 365
    rho   = sign * K * T * np.exp(-r * T) * stats.norm.cdf(sign * d2) / 100
    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Derivatives")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Option Payoffs", "Black-Scholes Pricing", "Option Greeks",
        "Forwards & Futures", "Swap Valuation",
    ])

    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
## Option Payoffs

An **option** gives the buyer the *right* (but not the obligation) to buy or sell an
asset at the **strike price K** on or before expiry.

| | Call | Put |
|---|---|---|
| **Payoff at expiry** | $\\max(S_T - K, 0)$ | $\\max(K - S_T, 0)$ |
| **Profit** | Payoff $-$ Premium | Payoff $-$ Premium |
| **Max loss (buyer)** | Premium paid | Premium paid |
| **Max gain (buyer)** | Unlimited | $K -$ Premium |

### Put-Call Parity

$$C - P = S_0 - K e^{-rT}$$

Any violation creates a **risk-free arbitrage** opportunity.

### Common strategies

- **Bull spread**: Buy call at $K_1$, sell call at $K_2 > K_1$
- **Bear spread**: Buy put at $K_2$, sell put at $K_1 < K_2$
- **Straddle**: Buy call + put at same K — profits from large moves (long volatility)
- **Covered call**: Long stock + short call — income strategy
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            strategy = st.selectbox("Strategy", [
                "Long Call", "Short Call", "Long Put", "Short Put",
                "Bull Call Spread", "Bear Put Spread",
                "Long Straddle", "Covered Call",
            ])
            K1  = st.slider("Strike K₁ ($)", 50, 200, 100, 5)
            K2  = st.slider("Strike K₂ (for spreads/straddle) ($)", 50, 200, 110, 5)
            prem = st.slider("Premium per option ($)", 1, 30, 10, 1)
            prem2 = st.slider("Premium K₂ option ($)", 1, 30, 7, 1)

        S_range = np.linspace(50, 200, 400)

        def payoff_profit(s_range, strategy, K1, K2, prem, prem2):
            K1, K2 = float(K1), float(K2)
            if strategy == "Long Call":
                payoff = np.maximum(s_range - K1, 0)
                profit = payoff - prem
            elif strategy == "Short Call":
                payoff = -np.maximum(s_range - K1, 0)
                profit = payoff + prem
            elif strategy == "Long Put":
                payoff = np.maximum(K1 - s_range, 0)
                profit = payoff - prem
            elif strategy == "Short Put":
                payoff = -np.maximum(K1 - s_range, 0)
                profit = payoff + prem
            elif strategy == "Bull Call Spread":
                payoff = np.maximum(s_range - K1, 0) - np.maximum(s_range - K2, 0)
                profit = payoff - (prem - prem2)
            elif strategy == "Bear Put Spread":
                payoff = np.maximum(K2 - s_range, 0) - np.maximum(K1 - s_range, 0)
                profit = payoff - (prem2 - prem)
            elif strategy == "Long Straddle":
                payoff = np.maximum(s_range - K1, 0) + np.maximum(K1 - s_range, 0)
                profit = payoff - (prem + prem)
            elif strategy == "Covered Call":
                payoff = s_range - np.maximum(s_range - K1, 0)
                profit = payoff - s_range[len(s_range)//2] + prem   # normalised
            return payoff, profit

        payoff, profit = payoff_profit(S_range, strategy, K1, K2, prem, prem2)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=S_range, y=payoff,
            name="Payoff at expiry", line=dict(color="steelblue", width=2.5),
        ))
        fig.add_trace(go.Scatter(
            x=S_range, y=profit,
            name="Profit (net of premium)", line=dict(color="crimson", width=2, dash="dash"),
        ))
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.add_vline(x=K1, line_dash="dot", line_color="gray",
                      annotation_text=f"K₁={K1}", annotation_font_size=9)
        if strategy in ("Bull Call Spread", "Bear Put Spread", "Long Straddle"):
            fig.add_vline(x=K2, line_dash="dot", line_color="darkgray",
                          annotation_text=f"K₂={K2}", annotation_font_size=9)

        # Fill profitable region
        profit_region = np.where(profit >= 0, profit, 0)
        loss_region   = np.where(profit < 0, profit, 0)
        fig.add_trace(go.Scatter(
            x=S_range, y=profit_region, fill="tozeroy",
            fillcolor="rgba(0,180,0,0.15)", line=dict(width=0),
            name="Profit zone", showlegend=True,
        ))
        fig.add_trace(go.Scatter(
            x=S_range, y=loss_region, fill="tozeroy",
            fillcolor="rgba(220,50,50,0.15)", line=dict(width=0),
            name="Loss zone", showlegend=True,
        ))

        fig.update_layout(
            title=f"Payoff Diagram — {strategy}",
            xaxis_title="Stock Price at Expiry ($)",
            yaxis_title="Profit / Loss ($)",
            height=480, hovermode="x unified",
        )

        with col2:
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
## Black-Scholes Model

Black, Scholes & Merton (1973) derived a **closed-form** option pricing formula
under the following assumptions:
- Log-normal stock price process with constant volatility $\\sigma$
- Continuous trading, no dividends, no transaction costs
- Risk-free rate $r$ is constant

**Call price:**

$$C = S_0 N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \\frac{\\ln(S/K) + (r + \\sigma^2/2)T}{\\sigma\\sqrt{T}}, \\quad d_2 = d_1 - \\sigma\\sqrt{T}$$

**Put price** (via put-call parity):

$$P = K e^{-rT} N(-d_2) - S_0 N(-d_1)$$

> **Exam tip:** Higher volatility **always** increases option value (for both calls and puts)
> because it raises the chance of a large favourable move.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            S0    = st.slider("Stock price S₀ ($)", 10, 300, 100, 5)
            K_bs  = st.slider("Strike K ($)", 10, 300, 100, 5)
            T_bs  = st.slider("Time to expiry (years)", 0.05, 3.0, 1.0, 0.05)
            r_bs  = st.slider("Risk-free rate (%)", 0.0, 10.0, 4.0, 0.25) / 100
            sig   = st.slider("Volatility σ (%)", 1, 80, 25, 1) / 100

        call_p = bs_price(S0, K_bs, T_bs, r_bs, sig, "call")
        put_p  = bs_price(S0, K_bs, T_bs, r_bs, sig, "put")

        # Price surface: S vs sigma
        S_range_bs  = np.linspace(max(10, S0 - 60), S0 + 60, 60)
        sig_range   = np.linspace(0.05, 0.80, 60)
        SS, SS_sig  = np.meshgrid(S_range_bs, sig_range)
        call_surface = np.vectorize(lambda s, sv: bs_price(s, K_bs, T_bs, r_bs, sv, "call"))(SS, SS_sig)
        put_surface  = np.vectorize(lambda s, sv: bs_price(s, K_bs, T_bs, r_bs, sv, "put"))(SS, SS_sig)

        fig_bs = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "surface"}, {"type": "surface"}]],
            subplot_titles=["Call Price Surface", "Put Price Surface"],
        )

        fig_bs.add_trace(go.Surface(
            z=call_surface, x=S_range_bs, y=sig_range * 100,
            colorscale="Blues", showscale=False, name="Call",
            hovertemplate="S=%{x}  σ=%{y:.0f}%  C=$%{z:.2f}<extra></extra>",
        ), row=1, col=1)
        fig_bs.add_trace(go.Surface(
            z=put_surface, x=S_range_bs, y=sig_range * 100,
            colorscale="Reds", showscale=False, name="Put",
            hovertemplate="S=%{x}  σ=%{y:.0f}%  P=$%{z:.2f}<extra></extra>",
        ), row=1, col=2)

        fig_bs.update_layout(
            height=500,
            scene =dict(xaxis_title="S ($)", yaxis_title="σ (%)", zaxis_title="Call ($)"),
            scene2=dict(xaxis_title="S ($)", yaxis_title="σ (%)", zaxis_title="Put ($)"),
        )

        with col2:
            c1, c2 = st.columns(2)
            c1.metric("Call price", f"${call_p:.2f}")
            c2.metric("Put price", f"${put_p:.2f}")
            pcp_check = call_p - put_p - (S0 - K_bs * np.exp(-r_bs * T_bs))
            st.caption(f"Put-call parity check: C - P - (S - Ke^{{-rT}}) = {pcp_check:.6f}  (should be ≈ 0)")
            st.plotly_chart(fig_bs, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
## Option Greeks

The **Greeks** measure option price sensitivity to various inputs.

| Greek | Measures sensitivity to… | Typical sign (long call) |
|---|---|---|
| **Delta** Δ | Underlying price | +0 to +1 |
| **Gamma** Γ | Rate of change of Delta | + (always for long options) |
| **Theta** Θ | Time decay (per day) | − (long options lose time value) |
| **Vega** ν | Volatility (per 1% move) | + (long options benefit from higher vol) |
| **Rho** ρ | Interest rate (per 1% move) | + for calls, − for puts |

> **Exam tip:** At-the-money options have the highest Gamma and Vega.
> Deep in-the-money options have Delta approaching 1 (call) or -1 (put).
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            S_g   = st.slider("Stock price S ($)", 10, 300, 100, 5, key="g_S")
            K_g   = st.slider("Strike K ($)", 10, 300, 100, 5, key="g_K")
            T_g   = st.slider("Time to expiry (years)", 0.05, 3.0, 1.0, 0.05, key="g_T")
            r_g   = st.slider("Risk-free rate (%)", 0.0, 10.0, 4.0, 0.25, key="g_r") / 100
            sig_g = st.slider("Volatility σ (%)", 1, 80, 25, 1, key="g_sig") / 100
            opt_type = st.radio("Option type", ["call", "put"])

        greeks = bs_greeks(S_g, K_g, T_g, r_g, sig_g, opt_type)
        g_cols = st.columns(5)
        for col, (name, val) in zip(g_cols, greeks.items()):
            col.metric(name.capitalize(), f"{val:.4f}" if np.isfinite(val) else "n/a")

        # Greeks vs stock price
        S_arr = np.linspace(max(10, S_g - 80), S_g + 80, 300)
        delta_arr = np.array([bs_greeks(s, K_g, T_g, r_g, sig_g, opt_type)["delta"] for s in S_arr])
        gamma_arr = np.array([bs_greeks(s, K_g, T_g, r_g, sig_g, opt_type)["gamma"] for s in S_arr])
        theta_arr = np.array([bs_greeks(s, K_g, T_g, r_g, sig_g, opt_type)["theta"] for s in S_arr])
        vega_arr  = np.array([bs_greeks(s, K_g, T_g, r_g, sig_g, opt_type)["vega"]  for s in S_arr])

        fig_g = make_subplots(
            rows=2, cols=2,
            subplot_titles=["Delta", "Gamma", "Theta (per day)", "Vega (per 1% vol)"],
        )
        for (row, col_idx), (label, arr, color) in zip(
            [(1,1),(1,2),(2,1),(2,2)],
            [("Delta", delta_arr, "royalblue"), ("Gamma", gamma_arr, "seagreen"),
             ("Theta", theta_arr, "tomato"), ("Vega", vega_arr, "darkorange")],
        ):
            fig_g.add_trace(go.Scatter(
                x=S_arr, y=arr, name=label,
                line=dict(color=color, width=2),
            ), row=row, col=col_idx)
            fig_g.add_vline(x=S_g, line_dash="dot", line_color="gray",
                            annotation_text=f"S={S_g}", annotation_font_size=8,
                            row=row, col=col_idx)
            fig_g.add_vline(x=K_g, line_dash="dot", line_color="black",
                            annotation_text=f"K={K_g}", annotation_font_size=8,
                            row=row, col=col_idx)

        fig_g.update_layout(
            height=560, showlegend=False,
            title=f"Greeks vs Stock Price — {opt_type.capitalize()}",
        )
        for row, col_idx in [(1,1),(1,2),(2,1),(2,2)]:
            fig_g.update_xaxes(title_text="S ($)", row=row, col=col_idx)

        with col2:
            st.plotly_chart(fig_g, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown("""
## Forwards & Futures — Cost of Carry Model

The **no-arbitrage forward price** is the price that prevents riskless profit from
a cash-and-carry or reverse cash-and-carry strategy.

### General formula

$$F_0 = S_0 \\cdot e^{(r + u - q - y)T}$$

where:
- $r$ = risk-free rate (financing cost)
- $u$ = storage costs (% of spot, per year)
- $q$ = dividend / convenience yield (% received by holder)
- $y$ = convenience yield (value of holding physical commodity)
- $T$ = time to delivery

### Key applications

| Asset | Formula | Notes |
|---|---|---|
| **Non-dividend stock** | $F = S_0 e^{rT}$ | No income; only financing cost |
| **Dividend-paying stock** | $F = S_0 e^{(r-q)T}$ | Dividend yield $q$ reduces forward |
| **Bond (continuous)** | $F = (S_0 - PV(coupons)) e^{rT}$ | Similar to dividend stock |
| **Currency** | $F = S_0 e^{(r_d - r_f)T}$ | Covered IRP |
| **Commodity** | $F = S_0 e^{(r+u-y)T}$ | Storage + convenience yield |

### Backwardation vs Contango

- **Contango** ($F > S$): futures premium — common when storage costs dominate
- **Backwardation** ($F < S$): futures discount — occurs when convenience yield > carrying costs

> **Exam tip:** An asset with NO income and NO convenience yield trades in contango.
> Commodities with high convenience yield (e.g., energy during crises) can trade in
> strong backwardation.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            asset_type = st.selectbox("Asset type", [
                "Non-dividend stock", "Dividend-paying stock",
                "Currency (FX)", "Commodity",
            ])
            s_fwd  = st.slider("Spot price S₀", 10.0, 500.0, 100.0, 5.0)
            r_fwd  = st.slider("Risk-free rate r (%)", 0.0, 10.0, 4.0, 0.25) / 100
            T_fwd  = st.slider("Delivery horizon T (years)", 0.1, 3.0, 1.0, 0.1)

            q_fwd = u_fwd = y_fwd = rf_fwd = 0.0
            if asset_type == "Dividend-paying stock":
                q_fwd = st.slider("Dividend yield q (%/yr)", 0.0, 10.0, 3.0, 0.25) / 100
            elif asset_type == "Currency (FX)":
                rf_fwd = st.slider("Foreign risk-free rate r_f (%)", 0.0, 10.0, 2.0, 0.25) / 100
                q_fwd  = rf_fwd
            elif asset_type == "Commodity":
                u_fwd = st.slider("Storage cost u (%/yr)", 0.0, 8.0, 2.0, 0.25) / 100
                y_fwd = st.slider("Convenience yield y (%/yr)", 0.0, 15.0, 3.0, 0.25) / 100

        net_carry = r_fwd + u_fwd - q_fwd - y_fwd
        T_arr     = np.linspace(0.05, 3.0, 200)
        F_arr     = s_fwd * np.exp(net_carry * T_arr)
        F_current = s_fwd * np.exp(net_carry * T_fwd)

        basis_arr  = F_arr - s_fwd   # forward premium
        structure  = "Contango" if F_current > s_fwd else "Backwardation"

        fig_fwd = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Forward Price vs Delivery Horizon", "Cost-of-Carry Decomposition"],
        )

        fig_fwd.add_trace(go.Scatter(
            x=T_arr, y=F_arr,
            mode="lines", line=dict(color="steelblue", width=2.5),
            name="Forward price F(T)",
        ), row=1, col=1)
        fig_fwd.add_hline(y=s_fwd, line_color="gray", line_dash="dot",
                          annotation_text=f"Spot S₀={s_fwd:.0f}", row=1, col=1)
        fig_fwd.add_vline(x=T_fwd, line_color="crimson", line_dash="dash",
                          annotation_text=f"T={T_fwd:.1f}", row=1, col=1)
        fig_fwd.add_trace(go.Scatter(
            x=[T_fwd], y=[F_current],
            mode="markers",
            marker=dict(size=12, color="crimson", symbol="circle"),
            name=f"F={F_current:.2f}",
        ), row=1, col=1)
        fig_fwd.update_xaxes(title_text="T (years)", row=1, col=1)
        fig_fwd.update_yaxes(title_text="Forward Price", row=1, col=1)

        # Decomposition bars
        components   = ["Spot S₀", "Financing (r)", "Storage (u)", "− Income (q/y)", "Forward F"]
        comp_values  = [s_fwd, s_fwd*(np.exp(r_fwd*T_fwd)-1),
                        s_fwd*(np.exp(u_fwd*T_fwd)-1),
                        -s_fwd*(np.exp((q_fwd+y_fwd)*T_fwd)-1), F_current]
        comp_colors  = ["steelblue","darkorange","tomato","mediumseagreen","gold"]
        comp_measures = ["absolute","relative","relative","relative","total"]

        fig_fwd.add_trace(go.Waterfall(
            orientation="v",
            measure=comp_measures,
            x=components,
            y=comp_values,
            decreasing=dict(marker_color="mediumseagreen"),
            increasing=dict(marker_color="darkorange"),
            totals=dict(marker_color="steelblue"),
            text=[f"{v:+.2f}" if m == "relative" else f"{v:.2f}"
                  for v, m in zip(comp_values, comp_measures)],
            textposition="outside",
            connector=dict(line=dict(color="gray")),
        ), row=1, col=2)
        fig_fwd.update_yaxes(title_text="Price", row=1, col=2)
        fig_fwd.update_layout(height=450, showlegend=False)

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Forward Price F₀", f"{F_current:.4f}")
            c2.metric("Basis (F - S)", f"{F_current - s_fwd:+.4f}")
            c3.metric("Market structure", structure,
                      delta_color="normal" if structure == "Contango" else "off")
            st.plotly_chart(fig_fwd, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("""
## Interest Rate Swaps — Valuation

A **plain vanilla interest rate swap** exchanges fixed-rate payments for floating-rate
payments on a notional principal.

### At initiation

The **fixed rate** (swap rate) is set so the swap has **zero value**:

$$r_{\\text{swap}} = \\frac{1 - DF_n}{\\sum_{t=1}^{n} DF_t \\cdot \\Delta t}$$

where $DF_t = \\frac{1}{(1+z_t)^t}$ is the discount factor at time $t$.

### Mark-to-Market value

After initiation, if rates change, the swap develops positive or negative value.
For the **pay-fixed** party:

$$V_{\\text{pay-fixed}} = V_{\\text{floating bond}} - V_{\\text{fixed bond}}$$

$$= \\left[1 - DF_n + L \\cdot \\sum DF_t \\right] - r_{\\text{fixed}} \\sum DF_t \\cdot \\Delta t - DF_n$$

where $L$ is the current LIBOR/SOFR fixing.

### Duration of a swap

A **pay-fixed swap** has:
- Duration ≈ duration of floating bond − duration of fixed bond
- ≈ short duration (floating nearly zero) − long duration (fixed)
- ≈ **negative duration** → useful for hedging fixed-income portfolios against rising rates

> **Exam tip:** Know which party benefits when rates rise vs fall.
> Pay-fixed / receive-floating: benefits when rates **rise** (floating leg increases).
> Receive-fixed / pay-floating: benefits when rates **fall** (fixed leg is locked in high).
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            notional_sw  = st.slider("Notional ($M)", 1, 500, 100, 10)
            fixed_rate   = st.slider("Fixed rate agreed at inception (%)", 1.0, 10.0, 4.0, 0.25) / 100
            current_rate = st.slider("Current market rate (%)", 0.5, 12.0, 5.5, 0.25) / 100
            freq         = st.selectbox("Payment frequency", ["Semi-annual", "Quarterly", "Annual"])
            maturity_sw  = st.slider("Remaining maturity (years)", 1, 10, 5)
            party        = st.radio("You are the", ["Pay-fixed (receive floating)", "Receive-fixed (pay floating)"])

        freq_map = {"Annual": 1, "Semi-annual": 2, "Quarterly": 4}
        m        = freq_map[freq]
        n_pay    = maturity_sw * m
        dt       = 1 / m

        # Flat yield curve at current rate (simplified)
        t_arr    = np.array([(i + 1) * dt for i in range(n_pay)])
        df_arr   = 1 / (1 + current_rate * dt) ** np.arange(1, n_pay + 1)

        # Fixed leg PV
        fixed_cf = fixed_rate * dt * notional_sw
        pv_fixed = fixed_cf * df_arr.sum() + notional_sw * df_arr[-1]

        # Floating leg PV (at market rate, always ≈ par for a freshly reset floating bond)
        pv_float = notional_sw   # simplified: floating bond always worth par at reset

        v_pay_fixed  = pv_float - pv_fixed
        v_recv_fixed = pv_fixed - pv_float
        swap_value   = v_pay_fixed if "Pay" in party else v_recv_fixed

        # Swap value vs current rate
        rate_arr     = np.linspace(0.005, 0.12, 100)
        values_arr   = []
        for cr in rate_arr:
            dfs_  = 1 / (1 + cr * dt) ** np.arange(1, n_pay + 1)
            pv_f_ = fixed_cf * dfs_.sum() + notional_sw * dfs_[-1]
            v_    = notional_sw - pv_f_ if "Pay" in party else pv_f_ - notional_sw
            values_arr.append(v_)

        # Cash flow diagram
        cf_times  = t_arr.tolist()
        fixed_cfs = [-fixed_cf] * n_pay
        float_cfs_approx = [current_rate * dt * notional_sw] * n_pay

        fig_sw = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Swap MTM Value vs Current Rate", "Cash Flows (Pay-Fixed perspective)"],
        )

        fig_sw.add_trace(go.Scatter(
            x=rate_arr * 100, y=values_arr,
            mode="lines", line=dict(color="steelblue", width=2.5),
            name="Swap MTM ($M)",
        ), row=1, col=1)
        fig_sw.add_hline(y=0, line_color="black", line_dash="dash", row=1, col=1)
        fig_sw.add_vline(x=fixed_rate * 100, line_color="gray", line_dash="dot",
                         annotation_text=f"Fixed={fixed_rate*100:.2f}%\n(breakeven)", row=1, col=1)
        fig_sw.add_vline(x=current_rate * 100, line_color="crimson", line_dash="dash",
                         annotation_text=f"Current={current_rate*100:.2f}%", row=1, col=1)
        fig_sw.add_trace(go.Scatter(
            x=[current_rate * 100], y=[swap_value],
            mode="markers",
            marker=dict(size=12, color="crimson"),
            name=f"Current MTM=${swap_value:.2f}M",
        ), row=1, col=1)
        fig_sw.update_xaxes(title_text="Current Market Rate (%)", row=1, col=1)
        fig_sw.update_yaxes(title_text="Swap MTM Value ($M)", row=1, col=1)

        # Cash flows
        fig_sw.add_trace(go.Bar(
            x=cf_times, y=fixed_cfs,
            name="Fixed payments (out)", marker_color="tomato",
        ), row=1, col=2)
        fig_sw.add_trace(go.Bar(
            x=cf_times, y=float_cfs_approx,
            name="Floating receipts (in)", marker_color="steelblue",
        ), row=1, col=2)
        fig_sw.add_hline(y=0, line_color="black", row=1, col=2)
        fig_sw.update_xaxes(title_text="Year", row=1, col=2)
        fig_sw.update_yaxes(title_text="Cash Flow ($M)", row=1, col=2)
        fig_sw.update_layout(height=450, barmode="overlay", showlegend=True)

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("PV (Fixed leg)", f"${pv_fixed:.2f}M")
            c2.metric("PV (Floating leg)", f"${pv_float:.2f}M")
            direction = "positive" if swap_value > 0 else "negative"
            c3.metric("Swap MTM Value", f"${swap_value:.2f}M",
                      delta=f"In your favour" if swap_value > 0 else "Against you",
                      delta_color="normal" if swap_value > 0 else "inverse")
            st.plotly_chart(fig_sw, use_container_width=True)
            if current_rate > fixed_rate:
                st.success("Rates rose above fixed rate → pay-fixed party benefits (floating receipts > fixed payments).")
            else:
                st.info("Rates fell below fixed rate → receive-fixed party benefits.")
