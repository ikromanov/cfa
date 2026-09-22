"""Economics — FX (PPP/IRP), Balance of Payments, Solow growth model."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ── helpers ───────────────────────────────────────────────────────────────────

def ppp_expected_fx(s0, inf_d, inf_f, periods):
    """Purchasing Power Parity: expected exchange rate path."""
    t = np.arange(periods + 1)
    return s0 * ((1 + inf_d) / (1 + inf_f)) ** t


def irp_forward(s0, r_d, r_f, T):
    """Covered Interest Rate Parity: F = S × (1+r_d)/(1+r_f)."""
    return s0 * (1 + r_d) ** T / (1 + r_f) ** T


def solow_steady_state(s, delta, n, g, alpha):
    """Solow steady-state capital per effective worker k*."""
    return (s / (delta + n + g)) ** (1 / (1 - alpha))


def solow_path(k0, s, delta, n, g, alpha, T=80):
    """Simulate Solow capital accumulation path."""
    k = np.zeros(T + 1)
    k[0] = k0
    for t in range(T):
        f_k = k[t] ** alpha                  # output per effective worker
        inv = s * f_k                         # investment
        dep = (delta + n + g) * k[t]          # break-even investment
        k[t + 1] = k[t] + inv - dep
        k[t + 1] = max(k[t + 1], 1e-6)
    return k


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Economics")

    tab1, tab2, tab3 = st.tabs([
        "FX: PPP & Interest Rate Parity",
        "Balance of Payments",
        "Solow Growth Model",
    ])

    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
## Foreign Exchange — PPP and Interest Rate Parity

### Purchasing Power Parity (PPP)

The **Law of One Price** aggregated across all goods. PPP states that exchange rates
should adjust to equalize the purchasing power of currencies:

$$\\frac{E[S_{t+T}]}{S_0} = \\left(\\frac{1 + \\pi_d}{1 + \\pi_f}\\right)^T$$

A currency with *higher inflation* should *depreciate*.

### Interest Rate Parity

**Covered IRP** (no arbitrage, using forward contract):

$$F_T = S_0 \\cdot \\frac{(1 + r_d)^T}{(1 + r_f)^T}$$

**Uncovered IRP** (expectation, no hedge):

$$E[S_T] = S_0 \\cdot \\frac{(1 + r_d)^T}{(1 + r_f)^T}$$

The **forward premium/discount** on a currency equals the interest rate differential.

### The Carry Trade

Uncovered IRP *fails* empirically in the short run — high-rate currencies tend to
*appreciate* rather than depreciate (the "forward premium puzzle"). This is exploited
by the carry trade: borrow in low-rate currency, invest in high-rate currency.
Risk: sudden reversals (carry trade crashes) when risk appetite collapses.

> **Exam tip:** Covered IRP holds by arbitrage (riskless). Uncovered IRP is an
> *expectation* that often fails. The difference: forward rate ≠ expected future spot rate.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            s0      = st.number_input("Spot rate S₀ (d/f)", 0.5, 5.0, 1.25, 0.05)
            inf_d   = st.slider("Domestic inflation (%/yr)", 0.0, 15.0, 4.0, 0.25) / 100
            inf_f   = st.slider("Foreign inflation (%/yr)", 0.0, 15.0, 2.0, 0.25) / 100
            r_d     = st.slider("Domestic risk-free rate (%)", 0.0, 15.0, 5.0, 0.25) / 100
            r_f     = st.slider("Foreign risk-free rate (%)", 0.0, 15.0, 3.0, 0.25) / 100
            horizon = st.slider("Horizon (years)", 1, 10, 5)

        ppp_path  = ppp_expected_fx(s0, inf_d, inf_f, horizon)
        irp_fwds  = np.array([irp_forward(s0, r_d, r_f, t) for t in range(horizon + 1)])

        # Forward premium on domestic currency (in percent)
        fwd_prem = (irp_fwds[-1] / s0 - 1) * 100
        ppp_chg  = (ppp_path[-1] / s0 - 1) * 100
        basis    = fwd_prem - ppp_chg

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Exchange Rate Paths (PPP vs IRP)",
                "No-Arbitrage Decomposition",
            ],
        )
        t_range = list(range(horizon + 1))

        fig.add_trace(go.Scatter(
            x=t_range, y=ppp_path,
            mode="lines+markers", line=dict(color="steelblue", width=2.5),
            name="PPP expected rate",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t_range, y=irp_fwds,
            mode="lines+markers", line=dict(color="crimson", width=2.5, dash="dash"),
            name="Covered IRP forward",
        ), row=1, col=1)
        fig.add_hline(y=s0, line_color="gray", line_dash="dot",
                      annotation_text="S₀", row=1, col=1)
        fig.update_xaxes(title_text="Years", row=1, col=1)
        fig.update_yaxes(title_text="Exchange Rate (d/f)", row=1, col=1)

        # Decomposition bar: show IRP components
        irp_premium = (r_d - r_f) * horizon * 100   # approx
        ppp_change  = (inf_d - inf_f) * horizon * 100
        real_rate   = irp_premium - ppp_change

        fig.add_trace(go.Bar(
            x=["Interest rate differential", "Inflation differential", "Real rate differential"],
            y=[irp_premium, ppp_change, real_rate],
            marker_color=["steelblue", "darkorange", "mediumseagreen"],
            name="Decomposition",
        ), row=1, col=2)
        fig.add_hline(y=0, line_color="black", row=1, col=2)
        fig.update_xaxes(title_text="Component", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative effect (%) over horizon", row=1, col=2)
        fig.update_layout(height=440, showlegend=True)

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric(f"PPP rate (year {horizon})", f"{ppp_path[-1]:.4f}",
                      delta=f"{ppp_chg:+.1f}%")
            c2.metric(f"Forward rate (year {horizon})", f"{irp_fwds[-1]:.4f}",
                      delta=f"{fwd_prem:+.1f}%")
            c3.metric("IRP−PPP basis", f"{basis:+.2f}%")
            st.plotly_chart(fig, use_container_width=True)
            if abs(basis) > 1:
                st.info(f"The {basis:+.1f}% basis reflects a real interest rate differential — "
                        "IRP and PPP can diverge when real rates differ between countries.")

    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
## Balance of Payments (BOP)

The BOP must always sum to zero:

$$\\text{Current Account} + \\text{Capital Account} + \\text{Financial Account} = 0$$

| Account | Includes | Sign |
|---|---|---|
| **Current Account (CA)** | Trade in goods & services, income, transfers | Deficit = outflow |
| **Capital Account (KA)** | Non-produced, non-financial assets; debt forgiveness | Usually small |
| **Financial Account (FA)** | FDI, portfolio investment, reserves | Surplus offsets CA deficit |

### What drives exchange rates?

A **persistent CA deficit** financed by short-term capital inflows is fragile — if
confidence falls, capital reverses and the currency crashes. Long-term FDI financing
is more stable.

> **Exam tip:** The CA deficit = domestic investment − domestic savings.
> A country can run a CA deficit sustainably if it is investing productively
> (e.g., productive imports) and foreign investors trust the return.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            ca_bal    = st.slider("Current Account (% GDP/yr)", -10.0, 5.0, -3.0, 0.5)
            fdi_share = st.slider("% of deficit financed by FDI", 0, 100, 30, 5)
            crisis_yr = st.slider("Crisis trigger at quarter", 4, 20, 10)
            quarters  = 24

        # Simulate BOP dynamics
        t_q   = np.arange(quarters)
        ca    = np.full(quarters, ca_bal)
        fdi   = np.full(quarters, fdi_share / 100 * abs(ca_bal))
        port  = np.full(quarters, (1 - fdi_share / 100) * abs(ca_bal))

        # Crisis: portfolio reverses
        port_adj = port.copy()
        ca_adj   = ca.copy()
        port_adj[crisis_yr:] = -abs(ca_bal) * 0.5   # reversal
        ca_adj[crisis_yr:]   = ca_bal * 0.7          # improvement forced by currency fall

        # Implied FX pressure (reserves must absorb the gap)
        reserve_change = -(ca_adj + fdi + port_adj)

        fig2 = make_subplots(
            rows=2, cols=1,
            subplot_titles=[
                "BOP Components (% GDP)",
                "Implied Reserve Change / FX Pressure",
            ],
            row_heights=[0.6, 0.4],
        )

        fig2.add_trace(go.Bar(x=t_q, y=ca_adj, name="Current Account", marker_color="tomato"), row=1, col=1)
        fig2.add_trace(go.Bar(x=t_q, y=fdi, name="FDI inflows", marker_color="steelblue"), row=1, col=1)
        fig2.add_trace(go.Bar(x=t_q, y=port_adj, name="Portfolio flows", marker_color="mediumseagreen"), row=1, col=1)
        fig2.update_layout(barmode="stack")
        fig2.add_vline(x=crisis_yr - 0.5, line_dash="dash", line_color="red",
                       annotation_text="Capital reversal", row=1, col=1)

        fig2.add_trace(go.Scatter(
            x=t_q, y=reserve_change,
            mode="lines+markers",
            line=dict(color="darkorange", width=2),
            name="Reserve change (− = loss)",
            fill="tozeroy",
            fillcolor="rgba(255,165,0,0.15)",
        ), row=2, col=1)
        fig2.add_hline(y=0, line_color="black", row=2, col=1)

        fig2.update_xaxes(title_text="Quarter", row=2, col=1)
        fig2.update_yaxes(title_text="% of GDP", row=1, col=1)
        fig2.update_yaxes(title_text="% of GDP", row=2, col=1)
        fig2.update_layout(height=520)

        with col2:
            st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
## Solow Growth Model

The Solow model explains long-run economic growth through capital accumulation,
population growth, and technological progress.

**Production function** (Cobb-Douglas per effective worker):

$$y = k^\\alpha, \\quad y = Y/(AL), \\quad k = K/(AL)$$

**Capital accumulation:**

$$\\dot{k} = s \\cdot f(k) - (\\delta + n + g) \\cdot k$$

**Steady state** $k^*$ where investment = break-even investment:

$$k^* = \\left(\\frac{s}{\\delta + n + g}\\right)^{1/(1-\\alpha)}$$

At $k^*$: output per worker grows at rate $g$ (technological progress only).

### Convergence hypothesis

**Absolute convergence**: poor countries grow faster and eventually catch up.
**Conditional convergence**: countries converge to their *own* steady state.
The data support conditional convergence — institutions and policies determine $k^*$.

### Golden Rule

The savings rate that maximises consumption per effective worker at steady state:

$$s^{\\text{golden}} = \\alpha$$

> **Exam tip:** In the Solow model, a permanent rise in the savings rate *raises* the
> level of output per worker but does NOT change the long-run growth rate (which is $g$).
> Only improvements in technology ($g$) can permanently raise the growth rate.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            s_rate = st.slider("Savings rate s", 0.05, 0.60, 0.25, 0.01)
            delta  = st.slider("Depreciation δ", 0.02, 0.15, 0.08, 0.01)
            n_pop  = st.slider("Population growth n (%)", 0.0, 4.0, 1.5, 0.25) / 100
            g_tech = st.slider("Tech. progress g (%)", 0.0, 4.0, 2.0, 0.25) / 100
            alpha  = st.slider("Capital share α", 0.20, 0.60, 0.33, 0.01)

        k_star = solow_steady_state(s_rate, delta, n_pop, g_tech, alpha)
        s_golden = alpha
        k_golden = solow_steady_state(s_golden, delta, n_pop, g_tech, alpha)
        c_star   = (1 - s_rate) * k_star ** alpha

        k_range = np.linspace(0.01, k_star * 2.2, 300)
        sf_k    = s_rate * k_range ** alpha
        breakeven = (delta + n_pop + g_tech) * k_range

        # Two convergence paths (one rich, one poor)
        k0_rich = k_star * 1.8
        k0_poor = k_star * 0.2
        path_rich = solow_path(k0_rich, s_rate, delta, n_pop, g_tech, alpha)
        path_poor = solow_path(k0_poor, s_rate, delta, n_pop, g_tech, alpha)

        fig3 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Solow Diagram (Phase Diagram)", "Convergence Paths"],
        )

        # Phase diagram
        fig3.add_trace(go.Scatter(
            x=k_range, y=sf_k,
            mode="lines", line=dict(color="steelblue", width=2.5),
            name=f"Investment s·f(k), s={s_rate:.2f}",
        ), row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=k_range, y=breakeven,
            mode="lines", line=dict(color="crimson", width=2.5, dash="dash"),
            name=f"Break-even (δ+n+g)k = {(delta+n_pop+g_tech)*100:.1f}%·k",
        ), row=1, col=1)
        # Golden rule investment line
        sf_k_gold = s_golden * k_range ** alpha
        fig3.add_trace(go.Scatter(
            x=k_range, y=sf_k_gold,
            mode="lines", line=dict(color="gold", width=1.5, dash="dot"),
            name=f"Golden Rule s={s_golden:.2f}",
        ), row=1, col=1)

        # Steady state marker
        fig3.add_trace(go.Scatter(
            x=[k_star], y=[s_rate * k_star ** alpha],
            mode="markers+text",
            marker=dict(size=14, color="steelblue", symbol="circle"),
            text=["k*"], textposition="top right",
            name=f"Steady state k*={k_star:.2f}",
        ), row=1, col=1)
        fig3.add_vline(x=k_star, line_dash="dot", line_color="gray", row=1, col=1)

        fig3.update_xaxes(title_text="Capital per eff. worker k", row=1, col=1)
        fig3.update_yaxes(title_text="Per effective worker", row=1, col=1)

        # Convergence paths
        t_sim = list(range(81))
        fig3.add_trace(go.Scatter(
            x=t_sim, y=path_rich,
            mode="lines", line=dict(color="crimson", width=2),
            name=f"Rich economy (k₀={k0_rich:.1f})",
        ), row=1, col=2)
        fig3.add_trace(go.Scatter(
            x=t_sim, y=path_poor,
            mode="lines", line=dict(color="steelblue", width=2),
            name=f"Poor economy (k₀={k0_poor:.1f})",
        ), row=1, col=2)
        fig3.add_hline(y=k_star, line_dash="dash", line_color="gray",
                       annotation_text=f"k*={k_star:.2f}", row=1, col=2)

        fig3.update_xaxes(title_text="Time periods", row=1, col=2)
        fig3.update_yaxes(title_text="Capital per eff. worker", row=1, col=2)
        fig3.update_layout(height=480, showlegend=True,
                           legend=dict(orientation="h", yanchor="bottom", y=-0.3))

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Steady-state k*", f"{k_star:.2f}")
            c2.metric("Output y* = (k*)^α", f"{k_star**alpha:.2f}")
            c3.metric("Consumption c*", f"{c_star:.2f}")
            if abs(s_rate - s_golden) < 0.02:
                st.success(f"Near the Golden Rule! s ≈ α = {s_golden:.2f}")
            elif s_rate > s_golden:
                st.info(f"Oversaving: s={s_rate:.2f} > α={s_golden:.2f}. Reducing savings would raise consumption.")
            else:
                st.info(f"Undersaving: s={s_rate:.2f} < α={s_golden:.2f}. Raising savings would raise consumption.")
            st.plotly_chart(fig3, use_container_width=True)
