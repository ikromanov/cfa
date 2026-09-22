"""Risk Management — Value-at-Risk and Expected Shortfall (CVaR)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st


# ── helpers ───────────────────────────────────────────────────────────────────

def var_historical(returns, confidence):
    var_pct  = np.percentile(returns, (1 - confidence) * 100)
    tail     = returns[returns <= var_pct]
    cvar_pct = tail.mean() if len(tail) else var_pct
    return -var_pct, -cvar_pct


def var_parametric(returns, confidence):
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z         = stats.norm.ppf(1 - confidence)
    var_pct   = mu + z * sigma
    cvar_pct  = mu - sigma * stats.norm.pdf(-z) / (1 - confidence)
    return -var_pct, -cvar_pct


def var_monte_carlo(returns, confidence, n_sims, rng):
    mu, sigma   = returns.mean(), returns.std(ddof=1)
    sim         = rng.normal(mu, sigma, size=n_sims)
    var_pct     = np.percentile(sim, (1 - confidence) * 100)
    tail        = sim[sim <= var_pct]
    cvar_pct    = tail.mean() if len(tail) else var_pct
    return -var_pct, -cvar_pct


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Risk Management")

    st.markdown("""
## Value-at-Risk (VaR)

**VaR** answers: *"What is the maximum loss I can expect over a given time horizon
at a given confidence level?"*

$$\\text{VaR}_{\\alpha} = \\inf\\{l \\in \\mathbb{R} : P(L > l) \\leq 1-\\alpha\\}$$

A **95% 1-day VaR of \$20,000** means: *"There is only a 5% chance that losses will
exceed \$20,000 in a single day."*

### Three standard methods

| Method | Assumption | Pros | Cons |
|---|---|---|---|
| **Historical Simulation** | No distributional assumption — uses actual past returns | Simple, captures fat tails | Depends heavily on the lookback window |
| **Parametric (Variance-Covariance)** | Returns are normally distributed | Analytically tractable | Underestimates tail risk |
| **Monte Carlo** | Specified distribution (often normal) | Flexible, scalable | Computationally intensive |

## Conditional VaR (CVaR) / Expected Shortfall

VaR tells us nothing about *how bad* losses are when they exceed the threshold.
**CVaR** is the *expected loss given that loss exceeds VaR*:

$$\\text{CVaR}_{\\alpha} = E[L \\mid L > \\text{VaR}_{\\alpha}]$$

CVaR is **coherent** (satisfies sub-additivity) while VaR is not — which is why
Basel III/IV regulators shifted to Expected Shortfall.
""")

    st.markdown("---")
    st.markdown("## Interactive Simulation")

    col1, col2 = st.columns([1, 2])

    with col1:
        portfolio_value = st.number_input(
            "Portfolio value ($)", min_value=10_000, max_value=100_000_000,
            value=1_000_000, step=10_000, format="%d",
        )
        confidence = st.slider("Confidence level", 0.90, 0.99, 0.95, 0.01,
                               format="%.2f")
        n_days    = st.slider("Historical window (trading days)", 100, 1000, 500, 50)
        n_sims    = st.slider("Monte Carlo simulations", 10_000, 200_000, 100_000, 10_000)
        seed      = st.number_input("Random seed", 0, 9999, 42)

        # Fat tail toggle
        fat_tails = st.checkbox("Add fat-tail shocks to history", value=True)

    rng = np.random.default_rng(int(seed))
    daily_returns = rng.normal(loc=0.0005, scale=0.012, size=n_days)
    if fat_tails:
        crash_days = rng.choice(n_days, size=max(1, n_days // 100), replace=False)
        daily_returns[crash_days] -= rng.uniform(0.04, 0.08, size=len(crash_days))

    h_var,  h_cvar  = var_historical(daily_returns, confidence)
    p_var,  p_cvar  = var_parametric(daily_returns, confidence)
    mc_var, mc_cvar = var_monte_carlo(daily_returns, confidence, n_sims, rng)

    dollar = lambda x: x * portfolio_value

    # ── Left chart: return distribution ───────────────────────────────────
    dollar_returns = daily_returns * portfolio_value
    tail_mask      = daily_returns <= -h_var / portfolio_value * 0

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Return Distribution with VaR Cutoffs",
                        "Cumulative Portfolio Value"],
    )

    counts, bins = np.histogram(dollar_returns, bins=60)
    bin_centers  = (bins[:-1] + bins[1:]) / 2

    # Colour bars: tail red, rest blue
    var_cutoff = -h_var
    colors_bar = ["crimson" if b <= var_cutoff else "steelblue" for b in bin_centers]

    fig.add_trace(go.Bar(
        x=bin_centers, y=counts,
        marker_color=colors_bar, name="Daily P&L",
        hovertemplate="P&L=%{x:$,.0f}  Count=%{y}<extra></extra>",
    ), row=1, col=1)

    for label, var, color, dash in [
        (f"Historical VaR {confidence:.0%}", dollar(h_var), "tomato", "dash"),
        (f"Parametric VaR {confidence:.0%}", dollar(p_var), "darkorange", "dot"),
        (f"Monte Carlo VaR {confidence:.0%}", dollar(mc_var), "gold", "dashdot"),
    ]:
        fig.add_vline(x=-var, line_color=color, line_dash=dash,
                      annotation_text=label, annotation_font_size=9,
                      row=1, col=1)

    fig.add_vline(x=-dollar(h_cvar), line_color="tomato", line_dash="solid",
                  annotation_text=f"CVaR {confidence:.0%}", annotation_font_size=9,
                  row=1, col=1)

    # ── Right chart: cumulative path ───────────────────────────────────────
    cumulative = (1 + daily_returns).cumprod() * portfolio_value

    # Rolling 20-day parametric VaR band
    window = 20
    rolling_var = np.full(n_days, np.nan)
    for i in range(window, n_days):
        rv, _ = var_parametric(daily_returns[i - window:i], confidence)
        rolling_var[i] = rv * cumulative[i]

    days = np.arange(n_days)
    fig.add_trace(go.Scatter(
        x=days, y=cumulative,
        name="Portfolio value", line=dict(color="steelblue", width=1.5),
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.concatenate([days, days[::-1]]),
        y=np.concatenate([cumulative - rolling_var,
                          (cumulative - rolling_var)[::-1]]),
        fill="toself", fillcolor="rgba(220,50,50,0.15)",
        line=dict(width=0), name=f"1-day VaR band ({window}d rolling)",
        showlegend=True,
    ), row=1, col=2)

    fig.update_xaxes(title_text="Daily P&L ($)", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    fig.update_xaxes(title_text="Trading Day", row=1, col=2)
    fig.update_yaxes(title_text="Portfolio Value ($)",
                     tickformat="$,.0f", row=1, col=2)
    fig.update_layout(height=520, showlegend=True, hovermode="x unified")

    with col2:
        st.plotly_chart(fig, use_container_width=True)

        # Results table
        st.markdown("### Results")
        rows = {
            "Historical Simulation": (h_var, h_cvar),
            "Parametric (Normal)":   (p_var, p_cvar),
            "Monte Carlo":           (mc_var, mc_cvar),
        }
        cols = st.columns(3)
        for col, (method, (var, cvar)) in zip(cols, rows.items()):
            col.metric(f"{method} VaR", f"${dollar(var):,.0f}")
            col.caption(f"CVaR: ${dollar(cvar):,.0f}")
