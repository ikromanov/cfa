"""Fixed Income — bond pricing, yield curves, duration & convexity, term structure, credit."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st


# ── helpers ───────────────────────────────────────────────────────────────────

def bond_price(face: float, coupon_rate: float, ytm: float, n_periods: int, freq: int = 2) -> float:
    """Price a fixed-rate bond with semi-annual (or other) compounding."""
    c = face * coupon_rate / freq
    y = ytm / freq
    periods = n_periods * freq
    if abs(y) < 1e-10:
        return c * periods + face
    t = np.arange(1, periods + 1)
    pv_coupons = np.sum(c / (1 + y) ** t)
    pv_face = face / (1 + y) ** periods
    return pv_coupons + pv_face


def modified_duration(face, coupon_rate, ytm, n_periods, freq=2):
    """Modified duration via first derivative of price w.r.t. ytm."""
    dp = 1e-4
    p_up   = bond_price(face, coupon_rate, ytm + dp, n_periods, freq)
    p_down = bond_price(face, coupon_rate, ytm - dp, n_periods, freq)
    p      = bond_price(face, coupon_rate, ytm, n_periods, freq)
    return -(p_up - p_down) / (2 * dp * p)


def convexity(face, coupon_rate, ytm, n_periods, freq=2):
    """Convexity via second derivative of price w.r.t. ytm."""
    dp = 1e-4
    p_up   = bond_price(face, coupon_rate, ytm + dp, n_periods, freq)
    p_down = bond_price(face, coupon_rate, ytm - dp, n_periods, freq)
    p      = bond_price(face, coupon_rate, ytm, n_periods, freq)
    return (p_up + p_down - 2 * p) / (dp**2 * p)


# ── main render ───────────────────────────────────────────────────────────────

def merton_default_prob(asset_value, debt_face, r, sigma_a, T):
    """Merton structural credit model: risk-neutral default probability."""
    if asset_value <= 0 or sigma_a <= 0 or T <= 0:
        return np.nan, np.nan, np.nan
    d2 = (np.log(asset_value / debt_face) + (r - 0.5 * sigma_a**2) * T) / (sigma_a * np.sqrt(T))
    d1 = d2 + sigma_a * np.sqrt(T)
    eq_value    = asset_value * stats.norm.cdf(d1) - debt_face * np.exp(-r * T) * stats.norm.cdf(d2)
    default_prob = stats.norm.cdf(-d2)   # risk-neutral prob of default
    credit_spread = -np.log(stats.norm.cdf(d2) + (asset_value / debt_face) * np.exp(r * T) * stats.norm.cdf(d1)) / T
    return eq_value, default_prob, max(credit_spread, 0)


def render():
    st.title("Fixed Income")

    tab_bond, tab_term, tab_credit, tab_cds = st.tabs([
        "Bond Pricing & Duration",
        "Term Structure Theories",
        "Credit Analysis (Merton Model)",
        "Credit Default Swaps",
    ])

    with tab_bond:
      st.markdown("""
## Bond Pricing

A **bond** is a debt instrument promising periodic coupon payments and return of
face value at maturity.  Its fair price is the **present value of all future cash flows**
discounted at the **yield-to-maturity (YTM)**:

$$P = \\sum_{t=1}^{N} \\frac{C}{(1+y/m)^t} + \\frac{F}{(1+y/m)^N}$$

where $C = F \\times c / m$ is the periodic coupon, $y$ is the YTM, $m$ is the
compounding frequency, $F$ is face value, and $N = T \\times m$ is total periods.

### Price–Yield relationship
- When YTM **rises** → price **falls** (inverse relationship)
- When coupon = YTM → bond trades **at par**
- Coupon > YTM → **premium** bond;  Coupon < YTM → **discount** bond

## Duration & Convexity

**Modified duration** (MD) measures the *linear* price sensitivity to yield changes:

$$\\Delta P \\approx -MD \\times P \\times \\Delta y$$

**Convexity** captures the *curvature* — for large yield moves, duration alone
underestimates price increases and overestimates price drops:

$$\\Delta P \\approx \\left(-MD \\times \\Delta y + \\tfrac{1}{2} \\times Conv \\times (\\Delta y)^2\\right) \\times P$$

> **Exam tip:** Higher coupon bonds and shorter maturity bonds have *lower* duration.
> Zero-coupon bonds have duration equal to their maturity.
""")

    st.markdown("---")
    st.markdown("## Interactive Charts")

    col1, col2 = st.columns([1, 2])

    with col1:
        face       = 1000.0
        coupon_pct = st.slider("Coupon rate (%)", 0.0, 15.0, 6.0, 0.25)
        maturity   = st.slider("Maturity (years)", 1, 30, 10)
        ytm_pct    = st.slider("Current YTM (%)", 0.5, 20.0, 6.0, 0.25)

        coupon = coupon_pct / 100
        ytm    = ytm_pct    / 100

        price = bond_price(face, coupon, ytm, maturity)
        md    = modified_duration(face, coupon, ytm, maturity)
        cvx   = convexity(face, coupon, ytm, maturity)

        st.metric("Bond Price", f"${price:,.2f}")
        st.metric("Modified Duration", f"{md:.3f} years")
        st.metric("Convexity", f"{cvx:.2f}")

        if price > face * 1.001:
            st.info("Premium bond (coupon > YTM)")
        elif price < face * 0.999:
            st.info("Discount bond (coupon < YTM)")
        else:
            st.info("Par bond (coupon ≈ YTM)")

    # ── Chart 1: Price vs YTM ──────────────────────────────────────────────
    ytm_range = np.linspace(0.005, 0.20, 300)
    prices    = [bond_price(face, coupon, y, maturity) for y in ytm_range]

    # Duration-only approximation
    price_dur_approx = [
        price * (1 - md * (y - ytm)) for y in ytm_range
    ]
    # Duration + convexity approximation
    price_full_approx = [
        price * (1 - md * (y - ytm) + 0.5 * cvx * (y - ytm)**2)
        for y in ytm_range
    ]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Price vs YTM", "Yield Curve Shapes"],
    )

    fig.add_trace(go.Scatter(
        x=ytm_range * 100, y=prices,
        name="Actual price", line=dict(color="royalblue", width=2.5),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ytm_range * 100, y=price_dur_approx,
        name="Duration approx.", line=dict(color="orange", width=1.5, dash="dash"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ytm_range * 100, y=price_full_approx,
        name="Dur + Conv approx.", line=dict(color="green", width=1.5, dash="dot"),
    ), row=1, col=1)

    # Current YTM marker
    fig.add_vline(x=ytm * 100, line_dash="dot", line_color="red",
                  annotation_text=f"YTM={ytm_pct:.2f}%", row=1, col=1)
    fig.add_hline(y=price, line_dash="dot", line_color="red", row=1, col=1)

    # ── Chart 2: Yield curve shapes ────────────────────────────────────────
    maturities = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

    # Normal (upward sloping)
    normal = 2.0 + 2.5 * (1 - np.exp(-maturities / 8))
    # Inverted
    inverted = 5.5 - 2.0 * (1 - np.exp(-maturities / 5))
    # Flat
    flat = np.full_like(maturities, 4.0)
    # Humped
    humped = 2.5 + 2.0 * maturities / 5 * np.exp(1 - maturities / 5)

    for label, curve, color in [
        ("Normal (upward)", normal, "royalblue"),
        ("Inverted", inverted, "crimson"),
        ("Flat", flat, "gray"),
        ("Humped", humped, "darkorange"),
    ]:
        fig.add_trace(go.Scatter(
            x=maturities, y=curve,
            name=label, mode="lines+markers",
            line=dict(color=color, width=2),
        ), row=1, col=2)

    fig.update_xaxes(title_text="YTM (%)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Maturity (years)", row=1, col=2)
    fig.update_yaxes(title_text="Yield (%)", row=1, col=2)

    fig.update_layout(height=520, hovermode="x unified")

    with col2:
        st.plotly_chart(fig, use_container_width=True)

    # ── Duration by maturity heat ──────────────────────────────────────────
    st.markdown("### Duration across coupon × maturity")
    mats  = np.arange(1, 31)
    cpns  = np.arange(1, 15) / 100
    grid  = np.array([
        [modified_duration(face, c, ytm, m) for m in mats]
        for c in cpns
    ])

    fig2 = go.Figure(go.Heatmap(
        z=grid,
        x=mats,
        y=[f"{c*100:.0f}%" for c in cpns],
        colorscale="RdYlGn_r",
        colorbar=dict(title="Mod. Duration (years)"),
        hovertemplate="Maturity=%{x}yr  Coupon=%{y}  Duration=%{z:.2f}<extra></extra>",
    ))
    fig2.update_layout(
        title=f"Modified Duration (YTM fixed at {ytm_pct:.2f}%)",
        xaxis_title="Maturity (years)",
        yaxis_title="Coupon rate",
        height=380,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab_term:
        st.markdown("""
## Term Structure Theories

The **yield curve** (term structure of interest rates) can take many shapes. Three
theories explain *why* different shapes arise:

### 1. Pure Expectations Theory
Forward rates are **unbiased predictors** of future short rates. An upward-sloping
curve implies the market expects rates to rise.

$$f(t, t+1) = \\frac{(1 + z_{t+1})^{t+1}}{(1 + z_t)^t} - 1$$

### 2. Liquidity Preference Theory
Investors require a **liquidity premium** for holding longer-term bonds (more price
risk). Forward rates = expected future rates + liquidity premium. This explains why
yield curves are normally upward-sloping even when no rate increases are expected.

### 3. Preferred Habitat / Market Segmentation
Supply and demand in **specific maturity segments** drives rates. Institutional
investors (pension funds, insurers) have preferred maturities and will only move
if compensated sufficiently. This explains humps and dips in the curve.

### Bootstrapping Spot Rates from Par Rates

The **spot rate** (zero-coupon yield) is derived from par yields iteratively:
$$z_n = \\left(\\frac{1 + c_n/2}{1 - c_n/2 \\sum_{t=1}^{n-1} d_t}\\right)^{1/n} - 1 \\quad \\text{(simplified)}$$

> **Exam tip:** The forward rate can be locked in today using a forward rate agreement
> or a bond position. If the actual future short rate < forward rate, the long-term
> bond outperforms the rolling short-term strategy.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            liq_prem_slope = st.slider("Liquidity premium slope (bps/yr)", 0, 30, 8, 1)
            short_rate     = st.slider("Expected short rate (flat) (%)", 0.5, 8.0, 3.0, 0.25)
            curve_slope    = st.slider("Rate expectations slope (bps/yr)", -20, 30, 5, 1)

        mats_ts  = np.array([0.5, 1, 2, 3, 5, 7, 10, 20, 30], dtype=float)

        # Pure expectations: rates rising at curve_slope bps per year
        pe_curve = short_rate + curve_slope / 100 * mats_ts

        # Liquidity preference: add liquidity premium
        lp_curve = pe_curve + liq_prem_slope / 100 * np.sqrt(mats_ts)

        # Preferred habitat: add a bump at 5-7yr (institutional demand)
        hab_bump = 0.4 * np.exp(-0.5 * ((mats_ts - 6) / 2)**2)
        hab_curve = lp_curve + hab_bump

        # Bootstrap spot rates from simplified par rates
        par_rates = lp_curve / 100
        spot_rates = np.zeros(len(mats_ts))
        spot_rates[0] = par_rates[0]
        for i in range(1, len(mats_ts)):
            n  = mats_ts[i]
            c  = par_rates[i] / 2
            # Sum of discount factors for all prior periods (simplified, annual)
            dfs = sum((1 + spot_rates[j])**(-mats_ts[j]) for j in range(i))
            num = 1 + c
            den = 1 + c * dfs
            spot_rates[i] = (num / den) ** (1 / n) - 1

        # Forward rates between adjacent maturities
        fwd_rates = np.zeros(len(mats_ts) - 1)
        for i in range(len(mats_ts) - 1):
            t1, t2 = mats_ts[i], mats_ts[i + 1]
            fwd_rates[i] = ((1 + spot_rates[i + 1])**t2 / (1 + spot_rates[i])**t1)**(1/(t2 - t1)) - 1

        fig_ts = go.Figure()
        for label, curve, color, dash in [
            ("Pure Expectations", pe_curve, "gray", "dot"),
            ("+ Liquidity Premium", lp_curve, "steelblue", "dash"),
            ("+ Preferred Habitat bump", hab_curve, "crimson", "solid"),
            ("Bootstrapped Spot Rates", spot_rates * 100, "mediumseagreen", "dashdot"),
        ]:
            fig_ts.add_trace(go.Scatter(
                x=mats_ts, y=curve if "Spot" not in label else spot_rates * 100,
                mode="lines+markers", line=dict(color=color, width=2, dash=dash),
                name=label,
            ))

        fig_ts.update_layout(
            title="Yield Curve Decomposition by Theory",
            xaxis_title="Maturity (years)",
            yaxis_title="Yield (%)",
            height=450,
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )

        with col2:
            st.plotly_chart(fig_ts, use_container_width=True)
            st.markdown("**Implied forward rates (between adjacent maturities)**")
            fwd_labels = [f"{mats_ts[i]:.0f}→{mats_ts[i+1]:.0f}yr" for i in range(len(fwd_rates))]
            ff = go.Figure(go.Bar(
                x=fwd_labels, y=fwd_rates * 100,
                marker_color="darkorange",
                text=[f"{v*100:.2f}%" for v in fwd_rates],
                textposition="auto",
            ))
            ff.update_layout(height=220, yaxis_title="Forward Rate (%)",
                             title="Implied Forward Rates (from spot curve)")
            st.plotly_chart(ff, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab_credit:
        st.markdown("""
## Credit Analysis — Merton Structural Model

The **Merton model** treats a firm's equity as a **call option** on the firm's assets:
shareholders own the assets but have sold a put to the debtholders (limited liability).

$$\\text{Equity} = V_A \\cdot N(d_1) - F e^{-rT} N(d_2)$$

$$d_1 = \\frac{\\ln(V_A/F) + (r + \\sigma_A^2/2)T}{\\sigma_A\\sqrt{T}}, \\quad d_2 = d_1 - \\sigma_A\\sqrt{T}$$

**Risk-neutral default probability** = $N(-d_2)$

**Key insight:** Default probability rises with:
- Higher **leverage** (debt / assets)
- Higher **asset volatility** σ_A
- Shorter **time horizon** (less time to recover)

### Credit Spread

$$\\text{Spread} \\approx -\\frac{1}{T}\\ln\\left[N(d_2) + \\frac{V_A}{F}e^{rT}N(d_1)\\right]$$

> **Exam tip:** Know the Merton model conceptually. The exam tests whether you understand
> the relationship between leverage, volatility, and default probability — not whether
> you can derive the formula.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            va      = st.slider("Asset value V_A ($M)", 50, 2000, 500, 25)
            f_debt  = st.slider("Debt face value F ($M)", 10, 1500, 300, 25)
            sigma_a = st.slider("Asset volatility σ_A (%)", 5, 60, 20, 1) / 100
            r_mert  = st.slider("Risk-free rate (%)", 0.5, 8.0, 3.0, 0.25) / 100
            T_mert  = st.slider("Debt maturity (years)", 0.5, 10.0, 3.0, 0.5)

        eq_val, dp, cs = merton_default_prob(va, f_debt, r_mert, sigma_a, T_mert)

        # Sensitivity: default prob vs leverage ratio
        lev_range   = np.linspace(0.1, 0.95, 80)
        dp_vs_lev   = [merton_default_prob(va, va * l, r_mert, sigma_a, T_mert)[1] for l in lev_range]
        cs_vs_lev   = [merton_default_prob(va, va * l, r_mert, sigma_a, T_mert)[2] * 10000 for l in lev_range]

        # Default prob vs volatility
        sig_range   = np.linspace(0.02, 0.60, 80)
        dp_vs_sig   = [merton_default_prob(va, f_debt, r_mert, s, T_mert)[1] for s in sig_range]

        fig_cr = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "Default Prob & Credit Spread vs Leverage",
                "Default Probability vs Asset Volatility",
            ],
        )

        current_lev = f_debt / va
        fig_cr.add_trace(go.Scatter(
            x=lev_range * 100, y=[d * 100 for d in dp_vs_lev],
            mode="lines", line=dict(color="crimson", width=2),
            name="Default prob (%)",
        ), row=1, col=1)
        fig_cr.add_trace(go.Scatter(
            x=lev_range * 100, y=cs_vs_lev,
            mode="lines", line=dict(color="darkorange", width=2, dash="dash"),
            name="Credit spread (bps)", yaxis="y3",
        ), row=1, col=1)
        fig_cr.add_vline(x=current_lev * 100, line_dash="dot", line_color="steelblue",
                         annotation_text=f"Current D/V={current_lev*100:.0f}%", row=1, col=1)

        fig_cr.add_trace(go.Scatter(
            x=sig_range * 100, y=[d * 100 for d in dp_vs_sig],
            mode="lines", line=dict(color="steelblue", width=2),
            name="Default prob (%)",
        ), row=1, col=2)
        fig_cr.add_vline(x=sigma_a * 100, line_dash="dot", line_color="crimson",
                         annotation_text=f"σ={sigma_a*100:.0f}%", row=1, col=2)

        fig_cr.update_xaxes(title_text="Leverage D/V (%)", row=1, col=1)
        fig_cr.update_yaxes(title_text="Default Probability (%)", row=1, col=1)
        fig_cr.update_xaxes(title_text="Asset Volatility σ_A (%)", row=1, col=2)
        fig_cr.update_yaxes(title_text="Default Probability (%)", row=1, col=2)
        fig_cr.update_layout(height=440, showlegend=True,
                             legend=dict(orientation="h", yanchor="bottom", y=-0.25))

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Equity Value (Merton)", f"${eq_val:.1f}M" if np.isfinite(eq_val) else "N/A")
            c2.metric("Default Probability", f"{dp*100:.2f}%" if np.isfinite(dp) else "N/A")
            c3.metric("Credit Spread", f"{cs*10000:.0f} bps" if np.isfinite(cs) else "N/A")
            st.plotly_chart(fig_cr, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab_cds:
        st.markdown("""
## Credit Default Swaps (CDS)

A **CDS** is an insurance contract on credit risk:
- **Protection buyer** pays a periodic premium (the CDS spread) and receives a payment if default occurs
- **Protection seller** collects the premium and pays the loss given default

### CDS spread ≈ Credit spread

In theory: **CDS spread ≈ Bond yield − Risk-free rate = Credit spread**

The **CDS basis** = CDS spread − bond credit spread. If basis ≠ 0, an arbitrage may exist:
- **Negative basis** (CDS cheap): buy bond + buy CDS protection → near risk-free with positive carry
- **Positive basis** (CDS expensive): sell protection + short bond

### Cash settlement at default

$$\\text{CDS Payment} = \\text{Notional} \\times (1 - \\text{Recovery Rate})$$

Where **recovery rate** is the fraction of face value recovered in bankruptcy (typically 30–40% for senior unsecured bonds).

### Uses of CDS

| Use | Mechanism |
|---|---|
| **Hedging** | Bond holder buys protection to remove credit risk |
| **Speculation** | Sell protection to earn premium; buy protection for credit short |
| **Synthetic exposure** | Sell protection without buying the bond |
| **Basis trade** | Exploit CDS spread vs bond spread divergence |

> **Exam tip:** CDS are *unfunded* (no upfront payment). A CDS spread widening
> means credit deterioration. Know that the protection seller is synthetically *long*
> credit risk (same as owning the bond).
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            notional     = st.slider("CDS Notional ($M)", 1, 500, 10, 1)
            cds_spread   = st.slider("CDS spread (bps/yr)", 10, 1000, 150, 10)
            bond_spread  = st.slider("Bond credit spread (bps/yr)", 10, 1000, 160, 10)
            recovery     = st.slider("Recovery rate (%)", 0, 80, 40, 5) / 100
            years_to_mat = st.slider("CDS maturity (years)", 1, 10, 5)

        basis        = cds_spread - bond_spread
        annual_prem  = notional * cds_spread / 10000
        default_pmt  = notional * (1 - recovery)
        total_prem   = annual_prem * years_to_mat

        # Payoff scenarios
        fig_cds = make_subplots(
            rows=1, cols=2,
            subplot_titles=["CDS Cash Flows", "CDS Spread vs Bond Spread (Basis)"],
        )

        # Cash flows timeline
        t_arr = list(range(years_to_mat + 1))
        prem_cfs  = [-annual_prem] * years_to_mat + [0]  # buyer pays
        default_scenario = [0] * (years_to_mat // 2) + [default_pmt] + [0] * (years_to_mat - years_to_mat // 2)

        fig_cds.add_trace(go.Bar(
            x=t_arr[:-1], y=prem_cfs[:-1],
            name="Premium payments (buyer)", marker_color="tomato",
        ), row=1, col=1)
        fig_cds.add_trace(go.Bar(
            x=[years_to_mat // 2], y=[default_pmt],
            name=f"Default payment (if default at yr {years_to_mat//2})",
            marker_color="mediumseagreen",
        ), row=1, col=1)
        fig_cds.add_hline(y=0, line_color="black", row=1, col=1)
        fig_cds.update_xaxes(title_text="Year", row=1, col=1)
        fig_cds.update_yaxes(title_text="Cash Flow ($M)", row=1, col=1)

        # Basis over simulated time
        rng_cds = np.random.default_rng(31)
        t_sim   = np.arange(60)
        basis_sim = basis + rng_cds.normal(0, 5, 60).cumsum() * 0.1
        colors_basis = ["crimson" if b < 0 else "steelblue" for b in basis_sim]

        fig_cds.add_trace(go.Bar(
            x=t_sim, y=basis_sim,
            marker_color=colors_basis, name="CDS basis (bps)",
        ), row=1, col=2)
        fig_cds.add_hline(y=0, line_color="black", row=1, col=2)
        fig_cds.add_annotation(x=15, y=max(basis_sim) * 0.8,
                               text="Positive basis:\nCDS expensive",
                               font=dict(color="steelblue", size=9), showarrow=False,
                               row=1, col=2)
        fig_cds.add_annotation(x=15, y=min(basis_sim) * 0.8,
                               text="Negative basis:\nCDS cheap → arb",
                               font=dict(color="crimson", size=9), showarrow=False,
                               row=1, col=2)
        fig_cds.update_xaxes(title_text="Trading Day", row=1, col=2)
        fig_cds.update_yaxes(title_text="Basis (bps)", row=1, col=2)
        fig_cds.update_layout(height=420, barmode="overlay", showlegend=True)

        with col2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Annual Premium", f"${annual_prem:.2f}M")
            c2.metric("Default Payment", f"${default_pmt:.1f}M")
            c3.metric("CDS Basis", f"{basis:+.0f} bps",
                      delta="Negative basis (arb opportunity)" if basis < 0 else "Positive basis",
                      delta_color="normal" if basis < 0 else "off")
            st.plotly_chart(fig_cds, use_container_width=True)
