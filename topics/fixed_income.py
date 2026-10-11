"""Fixed Income — bond pricing, yield curves, duration & convexity, term structure, credit."""

import numpy as np
import pandas as pd
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


def option_smoothing_k(face, coupon_rate, n_periods, vol_scale, freq=2):
    """Convert a yield-volatility slider into a soft-min/soft-max sharpness constant
    `k`, scaled by the bond's price sensitivity at the coupon (the option's trigger).
    Higher rate volatility → smaller k → a wider, smoother transition band, i.e. more
    time value in the embedded option near the money."""
    trigger      = coupon_rate
    price_t      = bond_price(face, coupon_rate, trigger, n_periods, freq)
    dur_t        = modified_duration(face, coupon_rate, trigger, n_periods, freq)
    price_scale  = max(dur_t * price_t * vol_scale, 1.0)
    return 1.0 / price_scale


def callable_bond_price(face, coupon_rate, ytm, n_periods, call_price, k, freq=2):
    """Callable bond price = smooth-min(straight price, call price) — approaches the
    straight bond when out of the money, and is capped near the call price when the
    issuer's call option is in the money (YTM well below the coupon). Guaranteed
    non-increasing in yield, unlike a naive weighted blend of the two price levels."""
    straight = bond_price(face, coupon_rate, ytm, n_periods, freq)
    return -np.logaddexp(-k * straight, -k * call_price) / k


def putable_bond_price(face, coupon_rate, ytm, n_periods, put_price, k, freq=2):
    """Putable bond price = smooth-max(straight price, put price) — approaches the
    straight bond when out of the money, and is floored near the put price when the
    investor's put option is in the money (YTM well above the coupon)."""
    straight = bond_price(face, coupon_rate, ytm, n_periods, freq)
    return np.logaddexp(k * straight, k * put_price) / k


def effective_duration_convexity(price_fn, ytm, dy=0.0025):
    """Effective duration/convexity via finite differences on a full reprice —
    required for embedded-option bonds since cash flows change with the yield level,
    so closed-form modified duration/convexity (fixed cash flows) no longer applies."""
    p0     = price_fn(ytm)
    p_up   = price_fn(ytm + dy)
    p_down = price_fn(ytm - dy)
    eff_dur  = (p_down - p_up) / (2 * dy * p0)
    eff_conv = (p_up + p_down - 2 * p0) / (dy**2 * p0)
    return eff_dur, eff_conv


# ── binomial interest-rate tree ─────────────────────────────────────────────

def price_bond_on_tree(tree, coupon, face, maturity, call_price=None, put_price=None,
                        spread=0.0, return_all=False, first_exercise=0):
    """Backward induction through a calibrated binomial interest-rate tree.

    `tree` is a list of length `maturity`; `tree[t]` holds the one-period rates
    (length t+1, node index i = number of up-moves) that discount cash flows from
    time t to t+1. `spread` (the OAS) is added to every node rate before discounting.
    At each node the risk-neutral value is [0.5*(V_up + V_down) + coupon] / (1+r);
    a callable bond's value is capped at `call_price`, a putable bond's floored at
    `put_price` — the issuer/investor exercises whenever it is advantageous, at any
    node from time `first_exercise` onward (earlier nodes are a lockout period).
    """
    N = maturity
    values = np.full(N + 1, face, dtype=float)
    snapshots  = {N: values.copy()}
    exercised  = {}
    for t in range(N - 1, -1, -1):
        rates_t = tree[t] + spread
        new_values = np.empty(t + 1)
        ex_flags   = np.zeros(t + 1, dtype=bool)
        for i in range(t + 1):
            held = (0.5 * (values[i + 1] + values[i]) + coupon) / (1 + rates_t[i])
            pv = held
            if t < first_exercise:
                pass
            elif call_price is not None and held > call_price:
                pv = call_price
                ex_flags[i] = True
            elif put_price is not None and held < put_price:
                pv = put_price
                ex_flags[i] = True
            new_values[i] = pv
        values = new_values
        snapshots[t] = values.copy()
        exercised[t] = ex_flags
    if return_all:
        return values[0], snapshots, exercised
    return values[0]


def calibrate_binomial_tree(par_rates, vol, iters=80):
    """Calibrate a lognormal binomial interest-rate tree to a benchmark par curve.

    For each maturity t+1, adjacent-node rates at step t are related by
    r_{t,i} = r_{t,0} * exp(2*vol*i). `r_{t,0}` is found by bisection so that an
    on-the-run par bond of that maturity, priced on the tree built so far, is worth
    exactly par — the standard calibration used before valuing embedded-option bonds.
    """
    tree = []
    for t, par_rate in enumerate(par_rates):
        coupon = par_rate * 100.0

        def price_at(r0, t=t, coupon=coupon):
            rates_t = r0 * np.exp(2 * vol * np.arange(t + 1))
            return price_bond_on_tree(tree + [rates_t], coupon, 100.0, t + 1)

        lo, hi = 1e-6, 2.0
        for _ in range(iters):
            mid = (lo + hi) / 2
            if price_at(mid) > 100.0:
                lo = mid
            else:
                hi = mid
        r0 = (lo + hi) / 2
        tree.append(r0 * np.exp(2 * vol * np.arange(t + 1)))
    return tree


def solve_oas(tree, coupon, face, maturity, market_price, call_price=None, put_price=None,
              iters=80):
    """Bisection for the Option-Adjusted Spread: the constant spread added to every
    node rate that makes the tree's model price equal the observed market price."""
    lo, hi = -0.05, 0.15
    for _ in range(iters):
        mid = (lo + hi) / 2
        p = price_bond_on_tree(tree, coupon, face, maturity, call_price, put_price, spread=mid)
        if p > market_price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


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

    tab_bond, tab_term, tab_credit, tab_cds, tab_embedded, tab_binom = st.tabs([
        "Bond Pricing & Duration",
        "Term Structure Theories",
        "Credit Analysis (Merton Model)",
        "Credit Default Swaps",
        "Embedded Options & Spreads",
        "Binomial Tree Pricing",
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

    # ══════════════════════════════════════════════════════════════════════
    with tab_embedded:
        st.markdown("""
## Bonds with Embedded Options

Many corporate bonds give one side the right to alter the bond's cash flows before
maturity. The two classic structures:

| | Option owner | Exercised when... | Effect on investor |
|---|---|---|---|
| **Callable bond** | Issuer (investor is short a call) | Rates **fall** — issuer refinances cheaper | Price upside is **capped** near the call price |
| **Putable bond** | Investor (investor is long a put) | Rates **rise** — investor "sells back" near par | Price downside is **floored** near the put price |

### Valuation

$$V_{\\text{callable}} = V_{\\text{straight}} - V_{\\text{call option}}$$
$$V_{\\text{putable}} = V_{\\text{straight}} + V_{\\text{put option}}$$

### Moneyness

- **Out of the money (OTM):** current YTM is far from the coupon/trigger — the option
  is nearly worthless and the bond behaves like an option-free (straight) bond.
- **At/near the money:** the option's value is most sensitive to yield changes — this
  is where **negative convexity** (callable) or the sharpest **positive convexity**
  (putable) appears.
- **In the money (ITM):** the option dominates pricing. A callable bond compresses
  toward the **call price** (upside capped — "**price compression**" / negative
  convexity). A putable bond is supported at the **put price** (a price floor).

### Why *effective*, not modified, duration/convexity

A callable/putable bond's cash flows depend on whether the option gets exercised, so
the closed-form modified-duration formulas (which assume fixed cash flows) break
down. Instead, reprice the bond after shocking yields up and down:

$$ED = \\frac{P_{-\\Delta y} - P_{+\\Delta y}}{2 \\times \\Delta y \\times P_0}, \\qquad
EC = \\frac{P_{+\\Delta y} + P_{-\\Delta y} - 2P_0}{(\\Delta y)^2 \\times P_0}$$

Deep ITM, a callable bond's price barely moves as yields fall further (it's pinned
near the call price) — **effective duration shrinks toward zero** and **effective
convexity turns negative**. A putable bond, by contrast, keeps **positive convexity**
across the board because the put floor only ever helps the investor.

### Option-Adjusted Spread (OAS)

$$OAS = Z\\text{-spread} - \\text{Option Cost}$$

- Callable bond: option cost > 0 → **OAS < Z-spread**
- Putable bond: option cost < 0 → **OAS > Z-spread**

OAS strips the option's value out of the spread so callable, putable, and
option-free bonds can be compared on a like-for-like credit basis.

> **Exam tip:** Higher assumed interest-rate **volatility** raises the value of *any*
> embedded option (call or put), since it raises the odds the option ends up in the
> money. Higher volatility → **lower** callable bond price, **higher** putable bond
> price — even if the *level* of rates doesn't change.
""")

        st.markdown("---")
        st.markdown("## Interactive: Callable vs. Putable vs. Straight Bond")

        col1, col2 = st.columns([1, 2])

        with col1:
            face_e      = 1000.0
            coupon_e_pct = st.slider("Coupon rate (%)", 1.0, 12.0, 6.0, 0.25, key="emb_coupon")
            maturity_e   = st.slider("Maturity (years)", 2, 30, 10, key="emb_mat")
            ytm_e_pct    = st.slider("Current YTM (%)", 0.5, 15.0, 6.0, 0.25, key="emb_ytm")
            call_px_pct  = st.slider("Call price (% of par)", 100.0, 110.0, 102.0, 0.5, key="emb_call")
            put_px_pct   = st.slider("Put price (% of par)", 90.0, 105.0, 100.0, 0.5, key="emb_put")
            vol_pct      = st.slider("Rate volatility / option time-value (%)", 0.2, 3.0, 1.0, 0.1,
                                      key="emb_vol",
                                      help="Wider = more time value in the option, e.g. higher assumed rate volatility")

            coupon_e = coupon_e_pct / 100
            ytm_e    = ytm_e_pct / 100
            vol_e    = vol_pct / 100
            call_px  = face_e * call_px_pct / 100
            put_px   = face_e * put_px_pct / 100
            k_e      = option_smoothing_k(face_e, coupon_e, maturity_e, vol_e)

            straight_price = bond_price(face_e, coupon_e, ytm_e, maturity_e)
            call_price_now = callable_bond_price(face_e, coupon_e, ytm_e, maturity_e, call_px, k_e)
            put_price_now  = putable_bond_price(face_e, coupon_e, ytm_e, maturity_e, put_px, k_e)

            st.metric("Straight (option-free) price", f"${straight_price:,.2f}")
            st.metric("Callable bond price", f"${call_price_now:,.2f}",
                      delta=f"{call_price_now - straight_price:,.2f} (call option value)")
            st.metric("Putable bond price", f"${put_price_now:,.2f}",
                      delta=f"{put_price_now - straight_price:,.2f} (put option value)")

            gap_pct = (ytm_e - coupon_e) * 100
            if gap_pct < -0.5:
                st.warning("Callable bond: **IN THE MONEY** — YTM well below coupon, issuer likely to refinance/call.")
            elif gap_pct > 0.5:
                st.success("Callable bond: **OUT OF THE MONEY** — behaves like a straight bond.")
            else:
                st.info("Callable bond: **NEAR THE MONEY** — sharpest negative convexity here.")

            if gap_pct > 0.5:
                st.warning("Putable bond: **IN THE MONEY** — YTM well above coupon, investor likely to put.")
            elif gap_pct < -0.5:
                st.success("Putable bond: **OUT OF THE MONEY** — behaves like a straight bond.")
            else:
                st.info("Putable bond: **NEAR THE MONEY** — sharpest positive convexity here.")

        # ── Chart: Price vs YTM for straight / callable / putable ──────────
        ytm_range_e = np.linspace(0.005, 0.20, 300)
        straight_curve = [bond_price(face_e, coupon_e, y, maturity_e) for y in ytm_range_e]
        callable_curve = [callable_bond_price(face_e, coupon_e, y, maturity_e, call_px, k_e) for y in ytm_range_e]
        putable_curve  = [putable_bond_price(face_e, coupon_e, y, maturity_e, put_px, k_e) for y in ytm_range_e]

        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(
            x=ytm_range_e * 100, y=straight_curve,
            name="Straight bond", line=dict(color="gray", width=2, dash="dot"),
        ))
        fig_e.add_trace(go.Scatter(
            x=ytm_range_e * 100, y=callable_curve,
            name="Callable bond", line=dict(color="crimson", width=2.5),
        ))
        fig_e.add_trace(go.Scatter(
            x=ytm_range_e * 100, y=putable_curve,
            name="Putable bond", line=dict(color="mediumseagreen", width=2.5),
        ))
        fig_e.add_hline(y=call_px, line_dash="dash", line_color="crimson",
                        annotation_text=f"Call price ${call_px:,.0f}", annotation_position="bottom right")
        fig_e.add_hline(y=put_px, line_dash="dash", line_color="mediumseagreen",
                        annotation_text=f"Put price ${put_px:,.0f}", annotation_position="top right")
        fig_e.add_vline(x=coupon_e_pct, line_dash="dot", line_color="black",
                        annotation_text=f"Coupon={coupon_e_pct:.2f}% (trigger)")
        fig_e.add_vline(x=ytm_e_pct, line_dash="dot", line_color="steelblue",
                        annotation_text=f"Current YTM={ytm_e_pct:.2f}%")

        fig_e.update_layout(
            title="Price–Yield Profiles: Negative vs. Positive Convexity",
            xaxis_title="YTM (%)", yaxis_title="Price ($)",
            height=520, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        )

        with col2:
            st.plotly_chart(fig_e, use_container_width=True)

        # ── Effective duration/convexity comparison ─────────────────────────
        st.markdown("### Effective Duration & Convexity — Straight vs. Callable vs. Putable")

        ed_straight, ec_straight = effective_duration_convexity(
            lambda y: bond_price(face_e, coupon_e, y, maturity_e), ytm_e)
        ed_call, ec_call = effective_duration_convexity(
            lambda y: callable_bond_price(face_e, coupon_e, y, maturity_e, call_px, k_e), ytm_e)
        ed_put, ec_put = effective_duration_convexity(
            lambda y: putable_bond_price(face_e, coupon_e, y, maturity_e, put_px, k_e), ytm_e)

        d1, d2, d3 = st.columns(3)
        d1.metric("Straight — Eff. Duration", f"{ed_straight:.2f}")
        d1.metric("Straight — Eff. Convexity", f"{ec_straight:.1f}")
        d2.metric("Callable — Eff. Duration", f"{ed_call:.2f}",
                  delta=f"{ed_call - ed_straight:+.2f} vs straight")
        d2.metric("Callable — Eff. Convexity", f"{ec_call:.1f}",
                  delta="Negative convexity" if ec_call < 0 else "Positive",
                  delta_color="inverse" if ec_call < 0 else "normal")
        d3.metric("Putable — Eff. Duration", f"{ed_put:.2f}",
                  delta=f"{ed_put - ed_straight:+.2f} vs straight")
        d3.metric("Putable — Eff. Convexity", f"{ec_put:.1f}",
                  delta="Positive convexity", delta_color="normal")

        # ── Effective duration across the yield range ────────────────────────
        ed_range_call = [effective_duration_convexity(
            lambda yy: callable_bond_price(face_e, coupon_e, yy, maturity_e, call_px, k_e), y)[0]
            for y in ytm_range_e]
        ed_range_put = [effective_duration_convexity(
            lambda yy: putable_bond_price(face_e, coupon_e, yy, maturity_e, put_px, k_e), y)[0]
            for y in ytm_range_e]
        ed_range_straight = [effective_duration_convexity(
            lambda yy: bond_price(face_e, coupon_e, yy, maturity_e), y)[0]
            for y in ytm_range_e]

        fig_ed = go.Figure()
        fig_ed.add_trace(go.Scatter(x=ytm_range_e * 100, y=ed_range_straight,
                                     name="Straight", line=dict(color="gray", width=2, dash="dot")))
        fig_ed.add_trace(go.Scatter(x=ytm_range_e * 100, y=ed_range_call,
                                     name="Callable", line=dict(color="crimson", width=2.5)))
        fig_ed.add_trace(go.Scatter(x=ytm_range_e * 100, y=ed_range_put,
                                     name="Putable", line=dict(color="mediumseagreen", width=2.5)))
        fig_ed.add_hline(y=0, line_color="black", line_width=1)
        fig_ed.add_vline(x=coupon_e_pct, line_dash="dot", line_color="black")
        fig_ed.update_layout(
            title="Effective Duration collapses (callable) or stays elevated (putable) near the money",
            xaxis_title="YTM (%)", yaxis_title="Effective Duration (years)",
            height=380,
        )
        st.plotly_chart(fig_ed, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab_binom:
        st.markdown(r"""
## Binomial Interest Rate Tree Pricing

The price-yield curves in the previous tab used a smooth shortcut. In practice
(and on the exam), callable/putable bonds are valued on a **binomial interest
rate tree**: a lattice of possible future short rates, discounted back to today
one period at a time, applying the call/put decision at every node along the way.

### 1. Build the tree

At each time step $t$, there are $t+1$ possible one-period rates, indexed by the
number of "up" moves $i = 0, \dots, t$. Adjacent nodes are related by a constant
multiplicative spread driven by assumed interest-rate **volatility** $\sigma$:

$$r_{t,i} = r_{t,0} \times e^{2\sigma i}$$

so $r_{t,0}$ (the bottom node) pins down the whole time-step. Up and down moves
are **equally likely (50/50) under the risk-neutral measure** — all the real-world
probability information is already embedded in the rate levels themselves.

### 2. Calibrate to the benchmark yield curve

$r_{t,0}$ is *not* free — it is found by **bisection** so that an on-the-run par
bond of maturity $t+1$, priced on the tree built so far, comes out to exactly par
(100). This repeats one maturity at a time, so the tree exactly reproduces the
benchmark par curve when it is used to price option-free bonds.

### 3. Value the bond by backward induction

Starting at maturity (where every node is worth the redemption value) and working
backward to today, each node's value is the probability-weighted average of its
two successors, discounted one period, plus the coupon received at that node:

$$V_{t,i} = \frac{0.5\,(V_{t+1,i+1} + V_{t+1,i}) + CF_{t+1}}{1 + r_{t,i}}$$

**Embedded option logic** — compare the "held" value above to the exercise price
at every node *before* discounting further back:

$$V_{t,i}^{\text{callable}} = \min\!\big(V_{t,i}, \text{Call Price}\big), \qquad
V_{t,i}^{\text{putable}} = \max\!\big(V_{t,i}, \text{Put Price}\big)$$

The root node's value, $V_{0,0}$, is today's price.

### 4. Option-Adjusted Spread (OAS)

If the model price doesn't match the bond's observed market price, add a constant
spread to **every node rate** and re-run backward induction until it does — that
spread is the OAS, the bond's compensation for credit/liquidity risk *after*
stripping out the value of the embedded option.

> **Exam tip:** Higher volatility widens the spread between up/down nodes, which
> raises the value of *any* embedded option — lowering a callable bond's price and
> raising a putable bond's price, exactly as in the smooth-model chart above.
""")

        st.markdown("---")
        st.markdown("## Worked Example — Step by Step")

        col1, col2 = st.columns([1, 2])

        with col1:
            n_periods = st.select_slider("Number of periods (years)", options=[2, 3, 4], value=3,
                                          key="binom_n")
            st.caption("Benchmark par curve (annual-pay, par = 100)")
            par_rates = []
            for yr in range(1, n_periods + 1):
                default = 3.0 + 0.5 * yr
                pr = st.slider(f"{yr}-yr par rate (%)", 1.0, 10.0, default, 0.1, key=f"binom_par_{yr}")
                par_rates.append(pr / 100)

            vol_binom = st.slider("Interest rate volatility σ (%)", 5.0, 30.0, 15.0, 1.0, key="binom_vol") / 100
            coupon_binom_pct = st.slider("Bond coupon rate (%)", 1.0, 10.0, 5.0, 0.25, key="binom_coupon")
            coupon_binom = coupon_binom_pct  # dollars per $100 face, annual

            is_callable = st.checkbox("Callable", value=True, key="binom_callable")
            call_px_binom = st.slider("Call price", 95.0, 110.0, 100.0, 0.5, key="binom_callpx",
                                       disabled=not is_callable)
            is_putable = st.checkbox("Putable", value=False, key="binom_putable")
            put_px_binom = st.slider("Put price", 90.0, 105.0, 100.0, 0.5, key="binom_putpx",
                                      disabled=not is_putable)

        tree = calibrate_binomial_tree(par_rates, vol_binom)

        call_arg = call_px_binom if is_callable else None
        put_arg  = put_px_binom if is_putable else None

        straight_price, straight_snap, _ = price_bond_on_tree(
            tree, coupon_binom, 100.0, n_periods, return_all=True)
        callable_price, callable_snap, callable_ex = price_bond_on_tree(
            tree, coupon_binom, 100.0, n_periods, call_price=call_arg, return_all=True)
        putable_price, putable_snap, putable_ex = price_bond_on_tree(
            tree, coupon_binom, 100.0, n_periods, put_price=put_arg, return_all=True)

        # ── Step 1: calibrated tree ────────────────────────────────────────
        st.markdown(f"### Step 1 — Calibrated {n_periods}-period rate tree")
        st.caption("Each column t holds the one-period rates available from time t to t+1, "
                   "calibrated so a par bond of that maturity prices to exactly 100.")

        tree_rows = []
        for i in range(n_periods - 1, -1, -1):
            row = {"node": f"i={i}"}
            for t in range(n_periods):
                row[f"t={t}"] = f"{tree[t][i]*100:.3f}%" if i <= t else ""
            tree_rows.append(row)
        tree_df = pd.DataFrame(tree_rows).set_index("node")
        st.dataframe(tree_df, use_container_width=True)

        # Tree diagram
        edge_x, edge_y = [], []
        node_x, node_y, node_text = [], [], []
        for t in range(n_periods):
            for i in range(t + 1):
                y = i - t / 2
                node_x.append(t); node_y.append(y)
                node_text.append(f"{tree[t][i]*100:.2f}%")
                edge_x += [t, t + 1, None]; edge_y += [y, i - (t + 1) / 2, None]
                edge_x += [t, t + 1, None]; edge_y += [y, i + 1 - (t + 1) / 2, None]
        for i in range(n_periods + 1):
            y = i - n_periods / 2
            node_x.append(n_periods); node_y.append(y)
            node_text.append("Maturity")

        fig_tree = go.Figure()
        fig_tree.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                       line=dict(color="lightgray", width=1), hoverinfo="skip",
                                       showlegend=False))
        fig_tree.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                       marker=dict(size=26, color="steelblue"),
                                       text=node_text, textposition="middle center",
                                       textfont=dict(size=9, color="white"), showlegend=False))
        fig_tree.update_layout(
            title="Binomial Rate Lattice (up move → higher node)",
            xaxis_title="Time step t", yaxis=dict(visible=False),
            height=380, margin=dict(l=20, r=20, t=40, b=20),
        )
        with col2:
            st.plotly_chart(fig_tree, use_container_width=True)

        # ── Step 2: worked backward-induction arithmetic ─────────────────────
        st.markdown("### Step 2 — Backward induction, one step at a time")

        r_top = tree[n_periods - 1][n_periods - 1]
        v_top = (100.0 + coupon_binom) / (1 + r_top)
        r_bot = tree[n_periods - 1][0]
        v_bot = (100.0 + coupon_binom) / (1 + r_bot)

        st.markdown(f"""
**From maturity back to $t={n_periods-1}$:** every node at maturity is worth the
redemption value (\\${100:.2f}), so at each $t={n_periods-1}$ node both successors
equal \\$100.00.

- Top node ($i={n_periods-1}$, rate $r={r_top*100:.3f}\\%$):
$$V = \\frac{{0.5\\times(100.00+100.00) + {coupon_binom:.2f}}}{{1+{r_top:.5f}}} = \\${v_top:.3f}$$

- Bottom node ($i=0$, rate $r={r_bot*100:.3f}\\%$):
$$V = \\frac{{0.5\\times(100.00+100.00) + {coupon_binom:.2f}}}{{1+{r_bot:.5f}}} = \\${v_bot:.3f}$$

This repeats at every node, one time step at a time, using the **two already-computed
successor values** — never re-touching maturity — until it reaches $t=0$.
""")

        if is_callable:
            found = None
            for t in range(n_periods - 1, -1, -1):
                idx = np.where(callable_ex[t])[0]
                if len(idx):
                    i = idx[0]
                    held = (0.5 * (callable_snap[t + 1][i + 1] + callable_snap[t + 1][i])
                            + coupon_binom) / (1 + tree[t][i])
                    found = (t, i, held)
                    break
            if found:
                t, i, held = found
                st.markdown(f"""
**Where the call kicks in:** at node $t={t}, i={i}$ the held (unconstrained) value is
\\${held:.3f}, above the call price of \\${call_px_binom:.2f} — the issuer calls the
bond, so $V_{{{t},{i}}}$ is capped at \\${call_px_binom:.2f} (marked **\\*** below)
instead of the higher uncapped value.
""")
            else:
                st.caption("With these inputs, the call price is never reached — the callable "
                           "bond prices identically to the straight bond.")

        if is_putable:
            found = None
            for t in range(n_periods - 1, -1, -1):
                idx = np.where(putable_ex[t])[0]
                if len(idx):
                    i = idx[0]
                    held = (0.5 * (putable_snap[t + 1][i + 1] + putable_snap[t + 1][i])
                            + coupon_binom) / (1 + tree[t][i])
                    found = (t, i, held)
                    break
            if found:
                t, i, held = found
                st.markdown(f"""
**Where the put kicks in:** at node $t={t}, i={i}$ the held (unconstrained) value is
\\${held:.3f}, below the put price of \\${put_px_binom:.2f} — the investor puts the
bond back, so $V_{{{t},{i}}}$ is floored at \\${put_px_binom:.2f} (marked **\\*** below).
""")
            else:
                st.caption("With these inputs, the put price is never reached — the putable "
                           "bond prices identically to the straight bond.")

        # ── Step 3: full backward-induction tables ───────────────────────────
        st.markdown("### Step 3 — Full backward induction")

        def snapshot_df(snapshots, exercised=None):
            rows = []
            for i in range(n_periods, -1, -1):
                row = {"node": f"i={i}"}
                for t in range(n_periods, -1, -1):
                    if i <= t:
                        cell = f"{snapshots[t][i]:.3f}"
                        if exercised is not None and t in exercised and i < len(exercised[t]) and exercised[t][i]:
                            cell += " *"
                        row[f"t={t}"] = cell
                    else:
                        row[f"t={t}"] = ""
                rows.append(row)
            cols = ["node"] + [f"t={t}" for t in range(n_periods + 1)]
            return pd.DataFrame(rows)[cols].set_index("node")

        st.markdown("**Straight (option-free) bond**")
        st.dataframe(snapshot_df(straight_snap), use_container_width=True)

        if is_callable:
            st.markdown("**Callable bond** (\\* = called at this node)")
            st.dataframe(snapshot_df(callable_snap, callable_ex), use_container_width=True)

        if is_putable:
            st.markdown("**Putable bond** (\\* = put back at this node)")
            st.dataframe(snapshot_df(putable_snap, putable_ex), use_container_width=True)

        # ── Step 4: results ───────────────────────────────────────────────
        st.markdown("### Step 4 — Results")
        m1, m2, m3 = st.columns(3)
        m1.metric("Straight bond price", f"${straight_price:.3f}")
        if is_callable:
            m2.metric("Callable bond price", f"${callable_price:.3f}",
                      delta=f"{callable_price - straight_price:+.3f} (call option value)")
        if is_putable:
            m3.metric("Putable bond price", f"${putable_price:.3f}",
                      delta=f"{putable_price - straight_price:+.3f} (put option value)")

        # ── Bonus: OAS solver ───────────────────────────────────────────────
        with st.expander("Bonus — solve for the Option-Adjusted Spread (OAS)"):
            st.markdown("Enter this bond's observed market price to back out its OAS: the "
                        "constant spread added to every tree node that reprices it exactly.")
            model_price = callable_price if is_callable else (putable_price if is_putable else straight_price)
            market_price = st.number_input("Observed market price ($)", value=round(model_price, 3),
                                           step=0.1, key="binom_market_price")
            oas = solve_oas(tree, coupon_binom, 100.0, n_periods, market_price,
                            call_price=call_arg, put_price=put_arg)
            st.metric("Option-Adjusted Spread (OAS)", f"{oas*10000:.1f} bps")
