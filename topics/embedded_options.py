"""Bonds with Embedded Options — pricing on a binomial tree, effective duration/convexity,
one-sided and key rate durations, capped/floored floaters (Fixed Income Module 3)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from topics.fixed_income import calibrate_binomial_tree, price_bond_on_tree


# ── helpers ───────────────────────────────────────────────────────────────────

def par_curve(short_rate, long_rate, maturity):
    """Linear par curve from the 1-year to the `maturity`-year rate (decimals)."""
    if maturity == 1:
        return (short_rate,)
    return tuple(np.linspace(short_rate, long_rate, maturity))


@st.cache_data(show_spinner=False, max_entries=2000)
def build_tree(par_rates, vol):
    """Calibrated tree, cached — effective duration and the charts below rebuild the
    tree for many shifted curves, and calibration is the expensive step."""
    par_rates = tuple(max(r, 1e-4) for r in par_rates)
    return calibrate_binomial_tree(list(par_rates), vol)


def tree_price(par_rates, vol, coupon, call_price=None, put_price=None, oas=0.0,
               first_exercise=1):
    """Price a bond maturing at the end of the curve on a tree calibrated to `par_rates`,
    exercisable at every node from year `first_exercise` onward."""
    tree = build_tree(tuple(par_rates), vol)
    return price_bond_on_tree(tree, coupon, 100.0, len(par_rates),
                              call_price=call_price, put_price=put_price, spread=oas,
                              first_exercise=first_exercise)


def effective_measures(par_rates, vol, coupon, call_price=None, put_price=None,
                       oas=0.0, d_curve=0.003, first_exercise=1):
    """Curriculum procedure: hold the OAS constant, shift the benchmark par curve
    down/up by ΔCurve, re-calibrate the tree, and reprice."""
    par = np.asarray(par_rates)
    pv0   = tree_price(par, vol, coupon, call_price, put_price, oas, first_exercise)
    pv_dn = tree_price(par - d_curve, vol, coupon, call_price, put_price, oas, first_exercise)
    pv_up = tree_price(par + d_curve, vol, coupon, call_price, put_price, oas, first_exercise)
    return {
        "PV0": pv0, "PV−": pv_dn, "PV+": pv_up,
        "Effective duration": (pv_dn - pv_up) / (2 * d_curve * pv0),
        "Effective convexity": (pv_dn + pv_up - 2 * pv0) / (d_curve**2 * pv0),
        "One-sided up duration": (pv0 - pv_up) / (d_curve * pv0),
        "One-sided down duration": (pv_dn - pv0) / (d_curve * pv0),
    }


def key_rate_durations(par_rates, vol, coupon, call_price=None, put_price=None,
                       oas=0.0, d_curve=0.003, first_exercise=1):
    """Shift one par rate at a time; the key rate durations sum to ≈ effective duration."""
    par = np.asarray(par_rates)
    pv0 = tree_price(par, vol, coupon, call_price, put_price, oas, first_exercise)
    krds = []
    for k in range(len(par)):
        bump = np.zeros(len(par))
        bump[k] = d_curve
        pv_dn = tree_price(par - bump, vol, coupon, call_price, put_price, oas, first_exercise)
        pv_up = tree_price(par + bump, vol, coupon, call_price, put_price, oas, first_exercise)
        krds.append((pv_dn - pv_up) / (2 * d_curve * pv0))
    return krds


def price_floater_on_tree(tree, maturity, cap=None, floor=None, face=100.0):
    """Floater paying the one-period rate set at t (in arrears at t+1), optionally capped
    or floored. Uncapped, every node is worth par; the cap/floor value is the gap."""
    values = np.full(maturity + 1, face)
    for t in range(maturity - 1, -1, -1):
        r = tree[t]
        c = r.copy()
        if cap is not None:
            c = np.minimum(c, cap)
        if floor is not None:
            c = np.maximum(c, floor)
        values = (0.5 * (values[1:] + values[:-1]) + c * face) / (1 + r)
    return values[0]


# ── page ──────────────────────────────────────────────────────────────────────

def render():
    st.title("Bonds with Embedded Options")
    st.caption("Fixed Income · Module 3 — Valuation and Analysis of Bonds with Embedded Options")

    with st.sidebar:
        st.markdown("---")
        st.markdown("**Bond & curve inputs**")
        maturity = st.select_slider("Maturity (years)", options=list(range(2, 11)), value=5,
                                    key="eo_n")
        short = st.slider("1-yr par rate (%)", 0.5, 10.0, 3.0, 0.1, key="eo_short") / 100
        long_ = st.slider(f"{maturity}-yr par rate (%)", 0.5, 10.0, 4.0, 0.1, key="eo_long") / 100
        vol = st.slider("Interest rate volatility σ (%)", 1.0, 40.0, 15.0, 1.0, key="eo_vol") / 100
        coupon = st.slider("Coupon rate (%)", 1.0, 10.0, 4.0, 0.25, key="eo_cpn")
        call_px = st.slider("Call price", 95.0, 110.0, 100.0, 0.5, key="eo_call")
        put_px = st.slider("Put price", 90.0, 105.0, 100.0, 0.5, key="eo_put")
        first_ex = 1
        if maturity > 2:
            first_ex = st.slider("First call/put date (year)", 1, maturity - 1, 1,
                                 key="eo_first")
        oas = st.slider("OAS (bps)", -100, 300, 0, 5, key="eo_oas") / 10000

    par = par_curve(short, long_, maturity)
    bonds = {
        "Straight": dict(),
        "Callable": dict(call_price=call_px, first_exercise=first_ex),
        "Putable":  dict(put_price=put_px, first_exercise=first_ex),
    }
    colors = {"Straight": "steelblue", "Callable": "crimson", "Putable": "seagreen"}

    tab_price, tab_dur, tab_float, tab_sheet = st.tabs([
        "Pricing in General",
        "Effective Duration & Convexity",
        "Capped & Floored Floaters",
        "Cheat Sheet",
    ])

    # ── Pricing ──────────────────────────────────────────────────────────
    with tab_price:
        st.markdown(r"""
## Pricing bonds with embedded options

Every bond with an embedded option is a **straight (option-free) bond plus or minus
an option**. Who owns the option sets the sign:

$$V_{\text{callable}} = V_{\text{straight}} - V_{\text{call}} \qquad\qquad
V_{\text{putable}} = V_{\text{straight}} + V_{\text{put}}$$

- **Callable** — the *issuer* owns the call; it refinances when rates fall, so the
  investor is short the call and accepts a lower price (higher yield).
- **Putable** — the *investor* owns the put; they sell back at the put price when
  rates rise, so they pay a higher price (lower yield).
- **Extendible** bond ≈ putable bond of the longer maturity (same cash-flow pattern).

### Valuation procedure (arbitrage-free framework)

1. **Calibrate** a binomial tree to the benchmark par curve for an assumed volatility σ.
2. **Backward induction** — at each node, value = $\dfrac{0.5(V_u + V_d) + C}{1 + r}$.
3. **Apply the exercise rule at every node**: callable → $\min(V, \text{call price})$;
   putable → $\max(V, \text{put price})$.
4. If the bond is risky, add the **OAS** to every node rate (constant spread that makes
   the model price equal the market price).

Value the straight bond on the same tree, and the option value falls out by subtraction.

### What drives the option values

| Driver | Call option value | Callable bond | Put option value | Putable bond |
|---|---|---|---|---|
| ↑ Interest rate volatility | ↑ | ↓ | ↑ | ↑ |
| ↑ Rate level | ↓ | — | ↑ | — |
| Curve steeper (upward sloping) | ↓ | ↑ | ↑ | ↑ |
| Curve flattens / inverts | ↑ | ↓ | ↓ | ↓ |

> **Exam tip:** an upward-sloping curve means higher *forward* rates → less likely the
> issuer calls (call worth less) and more likely the investor puts (put worth more).
> Volatility is the one driver that raises **both** option values.
""")
        st.markdown("---")
        prices = {name: tree_price(par, vol, coupon, oas=oas, **kw) for name, kw in bonds.items()}
        call_val = prices["Straight"] - prices["Callable"]
        put_val = prices["Putable"] - prices["Straight"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Straight", f"{prices['Straight']:.3f}")
        c2.metric("Callable", f"{prices['Callable']:.3f}")
        c3.metric("Call option", f"{call_val:.3f}")
        c4.metric("Putable", f"{prices['Putable']:.3f}")
        c5.metric("Put option", f"{put_val:.3f}")
        st.caption("Inputs are in the sidebar. Prices per 100 par, annual coupons; the option "
                   "is exercisable at every node from the first call/put date onward.")

        col_a, col_b = st.columns(2)

        vols = np.arange(0.01, 0.41, 0.03)
        fig_vol = go.Figure()
        for name, kw in bonds.items():
            fig_vol.add_trace(go.Scatter(
                x=vols * 100, y=[tree_price(par, v, coupon, oas=oas, **kw) for v in vols],
                name=name, line=dict(color=colors[name], width=2.5)))
        fig_vol.add_vline(x=vol * 100, line_dash="dot", line_color="gray")
        fig_vol.update_layout(title="Price vs volatility",
                              xaxis_title="Interest rate volatility σ (%)",
                              yaxis_title="Price", height=380, legend=dict(x=0.01, y=0.99))
        col_a.plotly_chart(fig_vol, use_container_width=True)

        mid = (short + long_) / 2
        slopes = np.linspace(-0.02, 0.03, 11)
        call_vals, put_vals = [], []
        for s in slopes:
            p = par_curve(mid - s / 2, mid + s / 2, maturity)
            straight = tree_price(p, vol, coupon, oas=oas)
            call_vals.append(straight - tree_price(p, vol, coupon, call_price=call_px, oas=oas,
                                                           first_exercise=first_ex))
            put_vals.append(tree_price(p, vol, coupon, put_price=put_px, oas=oas,
                                     first_exercise=first_ex) - straight)
        fig_slope = go.Figure()
        fig_slope.add_trace(go.Scatter(x=slopes * 10000, y=call_vals, name="Call option",
                                       line=dict(color="crimson", width=2.5)))
        fig_slope.add_trace(go.Scatter(x=slopes * 10000, y=put_vals, name="Put option",
                                       line=dict(color="seagreen", width=2.5)))
        fig_slope.add_vline(x=(long_ - short) * 10000, line_dash="dot", line_color="gray")
        fig_slope.update_layout(
            title=f"Option value vs curve slope (average level {mid*100:.2f}% held fixed)",
            xaxis_title=f"{maturity}-yr minus 1-yr par rate (bps)",
            yaxis_title="Option value", height=380, legend=dict(x=0.01, y=0.99))
        col_b.plotly_chart(fig_slope, use_container_width=True)

    # ── Effective duration ───────────────────────────────────────────────
    with tab_dur:
        st.markdown(r"""
## Effective duration and convexity

Cash flows of a callable/putable bond change when rates change, so **modified
duration (fixed cash flows) does not apply**. Instead, reprice the bond on the tree
after shifting the **benchmark curve** — effective duration is a **curve duration**,
not a yield duration.

$$\text{EffDur} = \frac{PV_{-} - PV_{+}}{2\,(\Delta\text{Curve})\,PV_0}
\qquad\qquad
\text{EffCon} = \frac{PV_{-} + PV_{+} - 2\,PV_0}{(\Delta\text{Curve})^2\,PV_0}$$

**Procedure**
1. Given the bond's price, solve for its **OAS** on the calibrated tree.
2. Shift the benchmark par curve **down** by ΔCurve, re-calibrate the tree, value the
   bond **holding the OAS constant** → $PV_-$.
3. Shift **up** by ΔCurve, same steps → $PV_+$.
4. Plug into the formulas.

**Key definitions**
- **Effective duration** — % price change per 100 bps parallel shift in the
  benchmark curve, allowing cash flows (exercise) to respond.
- **Effective convexity** — second-order effect. Straight and putable: **positive**.
  Callable: **negative** when rates are low and the call is near/in the money
  (price compressed toward the call price → *price compression*).
- **One-sided durations** — $\dfrac{PV_0 - PV_+}{\Delta\text{Curve}\,PV_0}$ (up) and
  $\dfrac{PV_- - PV_0}{\Delta\text{Curve}\,PV_0}$ (down). Callable near the money:
  **down-duration < up-duration** (price capped on the way up). Putable near the
  money: **up-duration < down-duration** (price floored on the way down).
- **Key rate (partial) durations** — sensitivity to shifting *one* par rate; they
  sum to ≈ effective duration and reveal **shape** risk. An OTM callable's risk sits
  at maturity; an ITM callable's shifts toward the earlier (call-date) key rates.

**Ranking:** callable ≤ straight, putable ≤ straight (an option can only shorten the
expected life). Zero-coupon ≈ maturity; fixed-coupon bond < maturity; floater ≈ time
to next reset (near zero).

> **Exam tip:** in the vignette you will be given $PV_0$, $PV_-$, $PV_+$ and ΔCurve
> (often 30 bps). Watch the signs: $PV_-$ is the *higher* price.
""")
        st.markdown("---")
        d_curve = st.select_slider("ΔCurve (bps)", options=[10, 20, 25, 30, 50, 100], value=30,
                                   key="eo_dcurve") / 10000

        rows = []
        for name, kw in bonds.items():
            m = effective_measures(par, vol, coupon, oas=oas, d_curve=d_curve, **kw)
            rows.append({"Bond": name, **m})
        eff_df = pd.DataFrame(rows).set_index("Bond")
        st.dataframe(eff_df.style.format("{:.3f}"), width="stretch")

        col_a, col_b = st.columns(2)

        shifts = np.arange(-0.03, 0.0301, 0.0025)
        fig_ed = go.Figure()
        for name, kw in bonds.items():
            fig_ed.add_trace(go.Scatter(
                x=shifts * 10000,
                y=[effective_measures(np.asarray(par) + s, vol, coupon, oas=oas,
                                      d_curve=d_curve, **kw)["Effective duration"]
                   for s in shifts],
                name=name, line=dict(color=colors[name], width=2.5)))
        fig_ed.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_ed.update_layout(title="Effective duration vs parallel curve shift",
                             xaxis_title="Parallel shift of the par curve (bps)",
                             yaxis_title="Effective duration", height=400,
                             legend=dict(x=0.01, y=0.01))
        col_a.plotly_chart(fig_ed, use_container_width=True)

        fig_krd = go.Figure()
        tenors = [f"{k}y" for k in range(1, maturity + 1)]
        for name, kw in bonds.items():
            fig_krd.add_trace(go.Bar(
                x=tenors,
                y=key_rate_durations(par, vol, coupon, oas=oas, d_curve=d_curve, **kw),
                name=name, marker_color=colors[name]))
        fig_krd.update_layout(title="Key rate durations (one par rate shifted at a time)",
                              xaxis_title="Key rate", yaxis_title="Key rate duration",
                              barmode="group", height=400, legend=dict(x=0.01, y=0.99))
        col_b.plotly_chart(fig_krd, use_container_width=True)
        st.caption("Try a high coupon (call deep in the money): the callable's duration "
                   "collapses and its key rate risk moves to the short end. Lower the coupon "
                   "and it converges to the straight bond.")

    # ── Floaters ─────────────────────────────────────────────────────────
    with tab_float:
        st.markdown(r"""
## Capped and floored floaters

A floater whose coupon is set at the start of each period and paid at the end, at
the same rate used to discount it, is worth **par** at every node. Adding a cap or
floor turns it into an option-embedded bond:

$$V_{\text{capped floater}} = V_{\text{straight floater}} - V_{\text{cap}} \qquad
V_{\text{floored floater}} = V_{\text{straight floater}} + V_{\text{floor}}$$

- **Cap** — benefits the *issuer* (limits coupons when rates rise) → investor is short it.
- **Floor** — benefits the *investor* (guaranteed minimum coupon) → investor is long it.
- On the tree, at each node use coupon $= \min(r, \text{cap})$ or $\max(r, \text{floor})$
  but discount at the actual node rate $r$.
- Higher volatility raises both cap and floor values.
""")
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        with col1:
            cap = st.slider("Cap rate (%)", 1.0, 10.0, 4.5, 0.25, key="eo_cap") / 100
            floor = st.slider("Floor rate (%)", 0.0, 8.0, 3.0, 0.25, key="eo_floor") / 100
            tree = build_tree(par, vol)
            capped = price_floater_on_tree(tree, maturity, cap=cap)
            floored = price_floater_on_tree(tree, maturity, floor=floor)
            st.metric("Capped floater", f"{capped:.3f}", delta=f"cap value {100 - capped:.3f}",
                      delta_color="off")
            st.metric("Floored floater", f"{floored:.3f}",
                      delta=f"floor value {floored - 100:.3f}", delta_color="off")
            st.caption("No OAS / quoted margin — the uncapped floater is exactly 100.")
        with col2:
            fig_fl = go.Figure()
            fig_fl.add_trace(go.Scatter(
                x=vols * 100,
                y=[100 - price_floater_on_tree(build_tree(par, v), maturity, cap=cap) for v in vols],
                name="Cap value", line=dict(color="crimson", width=2.5)))
            fig_fl.add_trace(go.Scatter(
                x=vols * 100,
                y=[price_floater_on_tree(build_tree(par, v), maturity, floor=floor) - 100
                   for v in vols],
                name="Floor value", line=dict(color="seagreen", width=2.5)))
            fig_fl.add_vline(x=vol * 100, line_dash="dot", line_color="gray")
            fig_fl.update_layout(title="Cap and floor value vs volatility",
                                 xaxis_title="Interest rate volatility σ (%)",
                                 yaxis_title="Option value (per 100)", height=380,
                                 legend=dict(x=0.01, y=0.99))
            st.plotly_chart(fig_fl, use_container_width=True)

    # ── Cheat sheet ──────────────────────────────────────────────────────
    with tab_sheet:
        st.markdown(r"""
## Cheat sheet

| Concept | Callable | Putable |
|---|---|---|
| Option owner | Issuer | Investor |
| Bond value | Straight − call | Straight + put |
| OAS vs Z-spread | OAS < Z-spread | OAS > Z-spread |
| ↑ Volatility | Price ↓ | Price ↑ |
| Upward-sloping curve | Call worth less | Put worth more |
| Effective duration | ≤ straight | ≤ straight |
| Effective convexity | Negative when call is ITM | Positive |
| One-sided durations | Down < up (near the money) | Up < down (near the money) |

**Formulas**

$$\text{EffDur} = \frac{PV_{-} - PV_{+}}{2\,\Delta\text{Curve}\,PV_0}
\qquad \text{EffCon} = \frac{PV_{-} + PV_{+} - 2PV_0}{(\Delta\text{Curve})^2\,PV_0}$$

$$\%\Delta P \approx -\text{EffDur}\cdot\Delta\text{Curve}
+ \tfrac{1}{2}\,\text{EffCon}\cdot(\Delta\text{Curve})^2$$

$$\text{Option cost} = Z\text{-spread} - OAS$$

**Common traps**
- Effective duration uses a shift in the **benchmark curve** and holds the **OAS**
  constant — not the bond's own YTM.
- **Higher OAS ≠ automatically cheap** if the volatility assumption is off: assuming
  *too low* σ makes a callable look cheap (OAS overstated).
- Callable and putable durations both shorten as their option moves in the money,
  but for opposite rate moves (callable when rates **fall**, putable when they **rise**).
""")
