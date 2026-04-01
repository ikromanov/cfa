"""Portfolio Management — Efficient Frontier, Active Management, Factor Models."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import minimize
from scipy import stats


# ── helpers ───────────────────────────────────────────────────────────────────

def _portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray):
    ret = weights @ mu
    vol = np.sqrt(weights @ cov @ weights)
    return ret, vol


def _efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 200):
    n = len(mu)
    target_returns = np.linspace(mu.min(), mu.max(), n_points)
    vols, rets = [], []
    for target in target_returns:
        res = minimize(
            lambda w: np.sqrt(w @ cov @ w),
            x0=np.ones(n) / n,
            method="SLSQP",
            constraints=[
                {"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "eq", "fun": lambda w: w @ mu - target},
            ],
            bounds=[(0, 1)] * n,
        )
        if res.success:
            vols.append(res.fun)
            rets.append(target)
    return np.array(vols), np.array(rets)


def _random_portfolios(mu, cov, n=3000, seed=42):
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    w = rng.dirichlet(np.ones(n_assets), size=n)
    ret = w @ mu
    vol = np.sqrt(np.einsum("ij,jk,ik->i", w, cov, w))
    sharpe = ret / vol
    return vol, ret, sharpe


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Portfolio Management")

    tab_mpt, tab_active, tab_factor = st.tabs([
        "Efficient Frontier & CML",
        "Active Management",
        "Factor Models (Fama-French)",
    ])

    with tab_mpt:
      st.markdown("## Modern Portfolio Theory (MPT)")

    st.markdown("""
**Harry Markowitz (1952)** showed that investors can reduce risk without sacrificing
return by combining assets whose returns are not perfectly correlated.

### Key ideas

| Concept | Formula | Intuition |
|---|---|---|
| **Portfolio return** | $E[R_p] = \\sum w_i E[R_i]$ | Weighted average of individual returns |
| **Portfolio variance** | $\\sigma_p^2 = \\mathbf{w}^T \\Sigma \\mathbf{w}$ | Depends on correlations, not just individual variances |
| **Efficient frontier** | Min $\\sigma_p$ s.t. $E[R_p] = \\bar{r}$ | The set of portfolios with the *lowest* risk for each return level |
| **Sharpe ratio** | $S = (E[R_p] - r_f) / \\sigma_p$ | Return earned per unit of total risk |

### The Capital Market Line (CML)

When a **risk-free asset** is introduced, investors can combine it with the **tangency
portfolio** (the market portfolio on MPT's efficient frontier).
The CML dominates the curved frontier for all risk-averse investors:

$$E[R_p] = r_f + \\frac{E[R_m] - r_f}{\\sigma_m} \\cdot \\sigma_p$$

> **Exam tip:** The *Security Market Line (SML)* uses **beta** (systematic risk).
> The *CML* uses **total standard deviation** and applies only to *efficient portfolios*.
""")

    st.markdown("---")
    st.markdown("## Interactive Chart")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Asset assumptions (annualised)**")
        mu_a = st.slider("Asset A — expected return (%)", 2, 20, 8) / 100
        mu_b = st.slider("Asset B — expected return (%)", 2, 20, 12) / 100
        mu_c = st.slider("Asset C — expected return (%)", 2, 20, 6) / 100

        sigma_a = st.slider("Asset A — volatility (%)", 5, 40, 15) / 100
        sigma_b = st.slider("Asset B — volatility (%)", 5, 40, 25) / 100
        sigma_c = st.slider("Asset C — volatility (%)", 5, 40, 10) / 100

        rho_ab = st.slider("Correlation A-B", -1.0, 1.0, 0.2, 0.05)
        rho_ac = st.slider("Correlation A-C", -1.0, 1.0, -0.1, 0.05)
        rho_bc = st.slider("Correlation B-C", -1.0, 1.0, 0.3, 0.05)

        rf = st.slider("Risk-free rate (%)", 0, 10, 3) / 100

    mu = np.array([mu_a, mu_b, mu_c])
    cov = np.array([
        [sigma_a**2,               rho_ab * sigma_a * sigma_b, rho_ac * sigma_a * sigma_c],
        [rho_ab * sigma_a * sigma_b, sigma_b**2,               rho_bc * sigma_b * sigma_c],
        [rho_ac * sigma_a * sigma_c, rho_bc * sigma_b * sigma_c, sigma_c**2],
    ])

    # Check positive semi-definite
    eigvals = np.linalg.eigvalsh(cov)
    if np.any(eigvals < -1e-8):
        with col2:
            st.error("The correlation inputs produce an invalid (non-positive-semidefinite) covariance matrix. Adjust the correlations.")
        return

    # Random portfolios
    r_vol, r_ret, r_sharpe = _random_portfolios(mu, cov)

    # Efficient frontier
    ef_vol, ef_ret = _efficient_frontier(mu, cov)

    # Tangency portfolio
    def neg_sharpe(w):
        ret, vol = _portfolio_stats(w, mu, cov)
        return -(ret - rf) / vol if vol > 1e-8 else 0

    res = minimize(
        neg_sharpe,
        x0=np.ones(3) / 3,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
        bounds=[(0, 1)] * 3,
    )
    tan_ret, tan_vol = _portfolio_stats(res.x, mu, cov)

    # CML
    cml_vols = np.linspace(0, tan_vol * 1.8, 100)
    cml_rets = rf + (tan_ret - rf) / tan_vol * cml_vols

    fig = go.Figure()

    # Random portfolio cloud coloured by Sharpe
    fig.add_trace(go.Scatter(
        x=r_vol * 100, y=r_ret * 100,
        mode="markers",
        marker=dict(
            color=r_sharpe, colorscale="Viridis", size=3, opacity=0.5,
            colorbar=dict(title="Sharpe", thickness=12, len=0.6),
        ),
        name="Random portfolios",
        hovertemplate="σ=%{x:.1f}%  E[R]=%{y:.1f}%<extra></extra>",
    ))

    # Efficient frontier
    fig.add_trace(go.Scatter(
        x=ef_vol * 100, y=ef_ret * 100,
        mode="lines", line=dict(color="crimson", width=3),
        name="Efficient frontier",
    ))

    # CML
    fig.add_trace(go.Scatter(
        x=cml_vols * 100, y=cml_rets * 100,
        mode="lines", line=dict(color="royalblue", width=2, dash="dash"),
        name="Capital Market Line",
    ))

    # Individual assets
    for label, r, v in zip(["A", "B", "C"], mu, [sigma_a, sigma_b, sigma_c]):
        fig.add_trace(go.Scatter(
            x=[v * 100], y=[r * 100],
            mode="markers+text",
            marker=dict(size=14, symbol="diamond", color="orange",
                        line=dict(width=1, color="black")),
            text=[f"Asset {label}"], textposition="top center",
            name=f"Asset {label}", showlegend=False,
        ))

    # Tangency portfolio
    fig.add_trace(go.Scatter(
        x=[tan_vol * 100], y=[tan_ret * 100],
        mode="markers+text",
        marker=dict(size=16, symbol="star", color="gold",
                    line=dict(width=1, color="black")),
        text=["Tangency"], textposition="top right",
        name="Tangency portfolio",
    ))

    # Risk-free rate
    fig.add_trace(go.Scatter(
        x=[0], y=[rf * 100],
        mode="markers+text",
        marker=dict(size=12, symbol="circle", color="green",
                    line=dict(width=1, color="black")),
        text=[f"r_f={rf*100:.1f}%"], textposition="middle right",
        name="Risk-free rate",
    ))

    fig.update_layout(
        title="Efficient Frontier & Capital Market Line",
        xaxis_title="Volatility σ (%)",
        yaxis_title="Expected Return E[R] (%)",
        height=560,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode="closest",
    )

    with col2:
        st.plotly_chart(fig, use_container_width=True)
        tan_sharpe = (tan_ret - rf) / tan_vol
        st.metric("Tangency Sharpe ratio", f"{tan_sharpe:.3f}")
        st.caption(
            f"Tangency portfolio:  E[R] = {tan_ret*100:.1f}%,  σ = {tan_vol*100:.1f}%  |  "
            f"weights A={res.x[0]:.2f}  B={res.x[1]:.2f}  C={res.x[2]:.2f}"
        )

    # ══════════════════════════════════════════════════════════════════════
    with tab_active:
        st.markdown("""
## Active Portfolio Management

### The Fundamental Law of Active Management (Grinold)

$$IR = IC \\times \\sqrt{BR}$$

- **IR** (Information Ratio) = active return / active risk = α / ω — measures manager skill
- **IC** (Information Coefficient) = correlation between manager's forecasts and outcomes
- **BR** (Breadth) = number of independent investment decisions per year

### Transfer Coefficient (TC)

In practice, constraints (no short selling, position limits) prevent full implementation
of the optimal portfolio. The **Transfer Coefficient** (0–1) captures this:

$$IR \\approx TC \\times IC \\times \\sqrt{BR}$$

### Appraisal vs Information Ratio

| Ratio | Formula | Benchmark |
|---|---|---|
| **Sharpe** | $(R_p - r_f) / \\sigma_p$ | Risk-free rate; uses total risk |
| **Information Ratio** | $\\alpha / \\omega$ | Benchmark; uses active risk (tracking error) |
| **Appraisal** | $\\alpha / \\sigma_{\\varepsilon}$ | Model residual; Jensen's alpha / residual risk |

> **Exam tip:** A manager with IC = 0.05 and BR = 100 bets has IR = 0.5. Doubling
> breadth (to 200 bets at same IC) raises IR to 0.71 — a bigger benefit than raising IC
> from 0.05 to 0.07 at same breadth.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            ic   = st.slider("IC (Information Coefficient)", 0.01, 0.20, 0.05, 0.01)
            br   = st.slider("Breadth (bets/year)", 5, 500, 100, 5)
            tc   = st.slider("Transfer Coefficient", 0.1, 1.0, 0.7, 0.05)
            vol_active = st.slider("Active risk ω (%)", 1.0, 10.0, 4.0, 0.25) / 100

        ir_unconstrained = ic * np.sqrt(br)
        ir_constrained   = tc * ic * np.sqrt(br)
        alpha            = ir_constrained * vol_active

        # Heatmap: IR vs IC and BR
        ic_range = np.linspace(0.01, 0.20, 50)
        br_range = np.linspace(5, 500, 50)
        ir_grid  = np.outer(ic_range, np.sqrt(br_range)) * tc

        fig_act = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "IR = TC × IC × √BR (heatmap)",
                "Skill vs Luck: Alpha Distribution",
            ],
        )

        fig_act.add_trace(go.Heatmap(
            z=ir_grid,
            x=br_range,
            y=ic_range,
            colorscale="RdYlGn",
            zmin=0, zmax=1.5,
            colorbar=dict(title="IR", len=0.7),
            hovertemplate="IC=%{y:.2f}  BR=%{x}  IR=%{z:.2f}<extra></extra>",
        ), row=1, col=1)
        fig_act.add_trace(go.Scatter(
            x=[br], y=[ic],
            mode="markers",
            marker=dict(symbol="x", size=14, color="white", line=dict(width=2)),
            name="Current",
        ), row=1, col=1)
        fig_act.update_xaxes(title_text="Breadth (BR)", row=1, col=1)
        fig_act.update_yaxes(title_text="IC", row=1, col=1)

        # Simulated alpha distribution: skill vs luck
        rng_am = np.random.default_rng(42)
        n_mgrs = 1000
        # True alpha (from skill)
        true_alpha = ic * vol_active * np.sqrt(br) * rng_am.normal(1, 0.3, n_mgrs)
        # Noise component
        noise      = vol_active / np.sqrt(br) * rng_am.normal(0, 1, n_mgrs)
        obs_alpha  = true_alpha + noise

        fig_act.add_trace(go.Histogram(
            x=obs_alpha * 100, nbinsx=40,
            marker_color="steelblue", opacity=0.7,
            name="Observed alpha (%)",
        ), row=1, col=2)
        fig_act.add_trace(go.Histogram(
            x=noise * 100, nbinsx=40,
            marker_color="tomato", opacity=0.5,
            name="Pure noise (luck)",
        ), row=1, col=2)
        fig_act.add_vline(x=alpha * 100, line_color="gold", line_dash="dash",
                          annotation_text=f"Expected α={alpha*100:.2f}%",
                          row=1, col=2)
        fig_act.add_vline(x=0, line_color="black", row=1, col=2)
        fig_act.update_xaxes(title_text="Observed Alpha (%)", row=1, col=2)
        fig_act.update_yaxes(title_text="Count", row=1, col=2)
        fig_act.update_layout(height=500, barmode="overlay", showlegend=True,
                              legend=dict(orientation="h", yanchor="bottom", y=-0.25))

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("IR (unconstrained)", f"{ir_unconstrained:.3f}")
            c2.metric("IR (with TC constraints)", f"{ir_constrained:.3f}")
            c3.metric("Expected Alpha (α)", f"{alpha*100:.2f}%")
            st.plotly_chart(fig_act, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab_factor:
        st.markdown("""
## Fama-French Three-Factor Model

The **CAPM** explains expected return using only market risk (beta). Fama and French
(1992) found two additional priced factors:

$$R_i - r_f = \\alpha + \\beta_{MKT}(R_m - r_f) + \\beta_{SMB} \\cdot SMB + \\beta_{HML} \\cdot HML + \\varepsilon$$

| Factor | Longname | Long | Short | Historical premium |
|---|---|---|---|---|
| **MKT** | Market risk premium | All stocks | Risk-free | ~5–7%/yr |
| **SMB** | Small Minus Big (size) | Small caps | Large caps | ~2–3%/yr |
| **HML** | High Minus Low (value) | High B/M (value) | Low B/M (growth) | ~2–5%/yr |

### Why do these factors exist?

- **Risk-based**: SMB and HML proxy for distress risk not captured by beta alone
- **Behavioural**: Investors systematically overprice growth stocks and underprice value stocks

### Practical applications at Level II

- **Performance attribution**: Decompose active returns into factor exposures vs genuine alpha
- **Portfolio construction**: Tilt toward value/size if you believe the premium persists
- **Risk analysis**: A portfolio with high HML loading is a "value bet" — it will
  underperform during growth rallies

> **Exam tip:** Alpha (α) in the FF model is smaller than CAPM alpha because SMB/HML
> capture systematic risk that CAPM misclassifies as alpha.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            beta_mkt = st.slider("β_MKT (market beta)", 0.0, 2.0, 1.1, 0.05)
            beta_smb = st.slider("β_SMB (size loading)", -1.0, 2.0, 0.4, 0.05)
            beta_hml = st.slider("β_HML (value loading)", -1.0, 2.0, 0.3, 0.05)
            alpha_ff = st.slider("Alpha α (%/yr)", -5.0, 5.0, 0.5, 0.1) / 100
            rf_ff    = st.slider("Risk-free rate (%)", 0.0, 6.0, 3.0, 0.25) / 100

        # Stylised factor premiums (annualised)
        mkt_premium = 0.06
        smb_premium = 0.02
        hml_premium = 0.03

        expected_return = rf_ff + beta_mkt * mkt_premium + beta_smb * smb_premium + beta_hml * hml_premium + alpha_ff
        capm_return     = rf_ff + beta_mkt * mkt_premium

        # Simulate monthly returns
        rng_ff = np.random.default_rng(99)
        n_months = 120   # 10 years
        mkt_sim  = rng_ff.normal(mkt_premium / 12, 0.04, n_months)
        smb_sim  = rng_ff.normal(smb_premium / 12, 0.025, n_months)
        hml_sim  = rng_ff.normal(hml_premium / 12, 0.025, n_months)
        eps_sim  = rng_ff.normal(0, 0.02, n_months)
        port_ret = (rf_ff / 12 + alpha_ff / 12 +
                    beta_mkt * mkt_sim + beta_smb * smb_sim + beta_hml * hml_sim + eps_sim)

        # Rolling 36-month factor attribution
        window_f = 36
        months   = list(range(n_months))
        roll_mkt, roll_smb, roll_hml, roll_alpha = [], [], [], []
        for i in range(window_f, n_months):
            pr = port_ret[i - window_f:i]
            mk = mkt_sim[i - window_f:i]
            sm = smb_sim[i - window_f:i]
            hl = hml_sim[i - window_f:i]
            X  = np.column_stack([np.ones(window_f), mk, sm, hl])
            try:
                b, _, _, _ = np.linalg.lstsq(X, pr, rcond=None)
            except Exception:
                b = [0, beta_mkt, beta_smb, beta_hml]
            roll_alpha.append(b[0] * 12 * 100)
            roll_mkt.append(b[1])
            roll_smb.append(b[2])
            roll_hml.append(b[3])

        t_roll = list(range(window_f, n_months))

        fig_ff = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Factor Loadings (static)",
                "Expected Return Decomposition",
                "Rolling 36m Beta Estimates",
                "Rolling Alpha (%/yr)",
            ],
        )

        # Factor loading bar chart
        factors    = ["MKT", "SMB", "HML"]
        loadings   = [beta_mkt, beta_smb, beta_hml]
        load_colors = ["steelblue", "darkorange", "mediumseagreen"]
        fig_ff.add_trace(go.Bar(
            x=factors, y=loadings, marker_color=load_colors,
            text=[f"{v:.2f}" for v in loadings], textposition="auto",
            name="Factor loadings",
        ), row=1, col=1)
        fig_ff.add_hline(y=0, line_color="black", row=1, col=1)
        fig_ff.update_yaxes(title_text="Loading (β)", row=1, col=1)

        # Return decomposition
        components_ff   = ["r_f", "β·MKT", "β·SMB", "β·HML", "Alpha", "E[R]"]
        return_comp_vals = [
            rf_ff * 100,
            beta_mkt * mkt_premium * 100,
            beta_smb * smb_premium * 100,
            beta_hml * hml_premium * 100,
            alpha_ff * 100,
            0,
        ]
        measures_ff = ["absolute","relative","relative","relative","relative","total"]
        ff_colors   = ["steelblue","crimson","darkorange","mediumseagreen","gold","steelblue"]

        fig_ff.add_trace(go.Waterfall(
            orientation="v",
            measure=measures_ff,
            x=components_ff,
            y=return_comp_vals,
            decreasing=dict(marker_color="tomato"),
            increasing=dict(marker_color="mediumseagreen"),
            totals=dict(marker_color="steelblue"),
            text=[f"{v:+.2f}%" if m == "relative" else f"{v:.2f}%"
                  for v, m in zip(return_comp_vals, measures_ff)],
            textposition="outside",
            connector=dict(line=dict(color="gray")),
        ), row=1, col=2)
        fig_ff.update_yaxes(title_text="Return (%/yr)", row=1, col=2)

        # Rolling betas
        for beta_roll, name, color in [
            (roll_mkt, "MKT", "steelblue"),
            (roll_smb, "SMB", "darkorange"),
            (roll_hml, "HML", "mediumseagreen"),
        ]:
            fig_ff.add_trace(go.Scatter(
                x=t_roll, y=beta_roll,
                mode="lines", line=dict(color=color, width=1.5),
                name=f"β_{name}",
            ), row=2, col=1)
        fig_ff.add_hline(y=0, line_color="black", line_dash="dash", row=2, col=1)
        fig_ff.update_xaxes(title_text="Month", row=2, col=1)
        fig_ff.update_yaxes(title_text="Beta", row=2, col=1)

        # Rolling alpha
        fig_ff.add_trace(go.Scatter(
            x=t_roll, y=roll_alpha,
            mode="lines", line=dict(color="gold", width=1.5),
            name="Alpha (%/yr)",
            fill="tozeroy",
            fillcolor="rgba(255,200,0,0.15)",
        ), row=2, col=2)
        fig_ff.add_hline(y=0, line_color="black", line_dash="dash", row=2, col=2)
        fig_ff.update_xaxes(title_text="Month", row=2, col=2)
        fig_ff.update_yaxes(title_text="Annualised Alpha (%)", row=2, col=2)
        fig_ff.update_layout(height=600, showlegend=True,
                             legend=dict(orientation="h", yanchor="bottom", y=-0.15))

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("FF Expected Return", f"{expected_return*100:.2f}%")
            c2.metric("CAPM Expected Return", f"{capm_return*100:.2f}%")
            c3.metric("α (FF model)", f"{alpha_ff*100:.2f}%",
                      delta="Outperforms FF benchmark" if alpha_ff > 0 else "Underperforms",
                      delta_color="normal" if alpha_ff > 0 else "inverse")
            st.plotly_chart(fig_ff, use_container_width=True)
