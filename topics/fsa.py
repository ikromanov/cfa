"""Financial Statement Analysis — intercorporate investments, pensions, multinational FX."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st


# ── helpers ───────────────────────────────────────────────────────────────────

def intercorporate_ratios(
    parent_assets, parent_equity, parent_net_income,
    investee_assets, investee_liabilities, investee_net_income,
    ownership,
):
    """Return key ratios under fair value, equity method, proportionate consolidation and acquisition method."""

    # ── Fair value (minority: <20%, held as investment at fair value) ──────
    fv_assets      = parent_assets              # investee not on B/S
    fv_equity      = parent_equity
    fv_net_income  = parent_net_income          # dividends only, ignored here for simplicity

    # ── Equity method (associate: 20–50%) ─────────────────────────────────
    em_investee_bv = investee_assets - investee_liabilities
    em_assets      = parent_assets + ownership * em_investee_bv   # one line on B/S
    em_equity      = parent_equity + ownership * em_investee_bv   # mirrored
    em_net_income  = parent_net_income + ownership * investee_net_income

    # ── Proportionate consolidation (share of each line, no NCI) ───────────
    pc_assets      = parent_assets + ownership * investee_assets
    pc_equity      = em_equity                   # same equity as equity method
    pc_net_income  = em_net_income               # same NI as equity method

    # ── Acquisition method / full consolidation (subsidiary: >50%) ─────────
    cons_assets    = parent_assets + investee_assets
    nci            = (1 - ownership) * em_investee_bv
    cons_equity    = parent_equity + ownership * em_investee_bv + nci   # includes NCI
    cons_net_inc   = parent_net_income + ownership * investee_net_income   # attributable to parent (excl. NCI)

    methods = {
        "Fair Value / FVTPL": dict(
            assets=fv_assets, equity=fv_equity, net_income=fv_net_income,
        ),
        "Equity Method": dict(
            assets=em_assets, equity=em_equity, net_income=em_net_income,
        ),
        "Proportionate": dict(
            assets=pc_assets, equity=pc_equity, net_income=pc_net_income,
        ),
        "Acquisition Method": dict(
            assets=cons_assets, equity=cons_equity, net_income=cons_net_inc,
        ),
    }

    rows = []
    for method, vals in methods.items():
        a, e, ni = vals["assets"], vals["equity"], vals["net_income"]
        rows.append({
            "Method": method,
            "Total Assets ($M)": round(a, 1),
            "Equity ($M)": round(e, 1),
            "Net Income ($M)": round(ni, 1),
            "ROE (%)": round(ni / e * 100, 1) if e > 0 else None,
            "ROA (%)": round(ni / a * 100, 1) if a > 0 else None,
            "D/E": round((a - e) / e, 2) if e > 0 else None,
        })
    return pd.DataFrame(rows).set_index("Method")


def bond_price(face, coupon_rate, ytm, years):
    """Price of an annual-pay bond."""
    if years <= 0:
        return float(face)
    t = np.arange(1, years + 1)
    return float(np.sum(face * coupon_rate / (1 + ytm) ** t) + face / (1 + ytm) ** years)


def amortized_cost_schedule(face, coupon_rate, eir, years, price0):
    """Carrying amount at each year-end under the effective interest method."""
    values = [price0]
    for _ in range(years):
        values.append(values[-1] * (1 + eir) - face * coupon_rate)
    return values


def financial_asset_year1(face, coupon_rate, yield_purchase, years, yield_end, sold):
    """Year-1 reporting of a bond held at amortized cost, FVOCI and FVTPL.

    Bought at the start of the year at `yield_purchase`; at year-end the market
    yield is `yield_end`. If `sold`, the bond is sold at year-end fair value.
    Ignores expected credit losses and transaction costs.
    """
    price0          = bond_price(face, coupon_rate, yield_purchase, years)
    interest_income = price0 * yield_purchase                 # effective interest method
    ac_end          = price0 + interest_income - face * coupon_rate
    fv_end          = bond_price(face, coupon_rate, yield_end, years - 1)
    fv_change       = fv_end - ac_end                         # vs amortized cost

    base = {"interest": interest_income, "fv_pl": 0.0, "fv_oci": 0.0, "realized": 0.0}
    methods = {
        "Amortized Cost": {**base, "carrying": ac_end},
        "FVOCI":          {**base, "fv_oci": fv_change, "carrying": fv_end},
        "FVTPL":          {**base, "fv_pl": fv_change, "carrying": fv_end},
    }
    if sold:
        methods["Amortized Cost"].update(realized=fv_change)
        methods["FVOCI"].update(fv_oci=0.0, realized=fv_change)   # OCI recycled to P&L
        for m in methods.values():
            m["carrying"] = 0.0

    rows = []
    for method, v in methods.items():
        net_income = v["interest"] + v["fv_pl"] + v["realized"]
        rows.append({
            "Method": method,
            "Interest income (P&L)": v["interest"],
            "FV change (P&L)": v["fv_pl"],
            "Realized gain (P&L)": v["realized"],
            "Net income": net_income,
            "OCI": v["fv_oci"],
            "Total comprehensive income": net_income + v["fv_oci"],
            "Carrying amount": v["carrying"],
        })
    return price0, ac_end, fv_end, pd.DataFrame(rows).set_index("Method")


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Financial Statement Analysis")

    tab1, tab_fa, tab2, tab3, tab4 = st.tabs([
        "Intercorporate Investments",
        "Financial Assets (IFRS 9)",
        "Pension Accounting",
        "Multinational FX Translation",
        "Financial Reporting Quality",
    ])

    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
## Intercorporate Investments — Accounting Methods

The accounting treatment depends on the ownership percentage and degree of influence:

| Ownership | Influence | Method | Balance Sheet Impact |
|---|---|---|---|
| **< 20%** | Little/none | Financial asset: FVTPL / FVOCI (see *Financial Assets* tab) | Investment at fair value only |
| **20–50%** | Significant | **Equity method** | One-line "Investment in associate" |
| Shared control | Joint control | **Equity method** for joint ventures (proportionate consolidation only in narrow cases) | One line (or share of each line) |
| **> 50%** | Control | **Acquisition method** (full consolidation) | 100% of assets + liabilities on B/S, plus NCI |

Ownership % is only a guide. **Influence and control** decide. For example, board
representation or technology dependence can mean significant influence below 20%.

### Why this matters for analysis

The **same underlying economics** produce **radically different** financial ratios
depending on the method. A financial analyst must *undo* consolidation or the equity
method to compare companies on a like-for-like basis.

Key difference: Under consolidation, **100% of the subsidiary's debt** appears on the
parent's balance sheet (even if the parent only owns 60%). This inflates leverage ratios.

> **Exam tip:** A Level II vignette will often show a company switching from equity method
> to consolidation (or vice versa due to a change in ownership). Know how each ratio
> (ROE, ROA, D/E, net margin) changes under each method.
""")

        with st.expander("Equity method — associates (20–50%) and joint ventures", expanded=False):
            st.markdown("""
**Mechanics**

$$\\text{Investment}_{end} = \\text{Investment}_{begin} + \\text{Share of investee NI} - \\text{Dividends received}$$

- Initially recorded **at cost**, in one line on the balance sheet ("Investment in associate").
- The investor's **share of investee net income** goes to the income statement in one line.
- **Dividends received reduce the investment.** They are a return of capital, **not income**.
- Losses reduce the investment, normally not below zero.

**Excess of purchase price over book value**

| Component | Treatment |
|---|---|
| Excess of fair value over book value of **identifiable assets** (e.g. PP&E) | Depreciated/amortized. **Reduces** equity income each year |
| Remainder = **goodwill** | Not amortized. Part of the investment carrying amount; not tested separately |

**Unrealized intercompany profit (upstream and downstream sales)**

Profit on goods not yet sold to a third party is deferred **in proportion to the
ownership share**:

$$\\text{Deferred profit} = \\text{Ownership \\%} \\times \\text{Profit on goods still in inventory}$$

It is recognized when the goods are sold to an outside party.

**Impairment**

- **IFRS:** needs objective evidence of a loss event; impairment reversals allowed.
- **US GAAP:** a decline in fair value below carrying amount that is *other than temporary*; **no reversal**.

**Fair value option:** US GAAP allows it for equity-method investments. IFRS limits it
to venture capital firms, mutual funds and similar entities.

> **Exam tip:** Equity-method income is often the "hidden" part of earnings. The
> investor records NI but receives cash only as dividends. Compare the equity
> income line to dividends received to check earnings quality.
""")

        with st.expander("Acquisition method — business combinations (control, usually > 50%)", expanded=False):
            st.markdown("""
**Mechanics (IFRS 3 / ASC 805)**

- The acquirer consolidates **100% of the acquiree's identifiable assets and liabilities
  at fair value** on the acquisition date. This includes intangibles the acquiree
  never recognized (brands, customer lists, IPR&D).
- **Acquisition-related costs are expensed** (not capitalized).
- **Contingent consideration** is measured at fair value at acquisition.
- **All intercompany transactions and balances are eliminated in full** (100%, not the
  ownership share).
- **Non-controlling interest (NCI)** is shown in equity, separate from parent equity.
  Consolidated NI is split into the parent's and the NCI's share.

**Goodwill**

$$\\text{Goodwill} = \\text{Consideration} + \\text{FV of NCI} + \\text{FV of previously held interest} - \\text{FV of identifiable net assets}$$

| | **Full goodwill** | **Partial goodwill** |
|---|---|---|
| NCI measured at | Fair value of NCI | NCI share of identifiable net assets |
| Allowed under | IFRS **and US GAAP (required)** | IFRS only |
| Effect | Higher goodwill, assets and equity (NCI) | Lower assets and equity → higher ROA and ROE |

- **Bargain purchase** (negative goodwill): recognized immediately as a **gain in P&L**.
- **Goodwill is not amortized.** It is tested for impairment at least annually:
  - **IFRS:** one step. Compare the recoverable amount of the cash-generating unit to its carrying amount.
  - **US GAAP:** compare the reporting unit's fair value to its carrying amount. The impairment is the excess, capped at goodwill.

> **Exam tip:** Full vs partial goodwill changes **assets, equity and NCI** but **not
> the parent's net income**. Partial goodwill gives higher ROA and ROE. An impairment
> under full goodwill is larger, because there is more goodwill to write off.
""")

        with st.expander("Proportionate consolidation", expanded=False):
            st.markdown("""
**Mechanics**

- The investor reports its **share of each asset, liability, revenue and expense** of
  the investee, line by line.
- **No NCI**, because only the investor's own share is brought in.

**Where it is used**

- **IFRS 11:** *not* allowed for **joint ventures**, which must use the equity method.
  For **joint operations**, the operator records its own assets, liabilities, revenue
  and expenses, which has a similar result.
- **US GAAP:** generally not allowed. Limited exceptions for unincorporated entities
  in industries such as oil & gas and construction.

**Compared with the equity method**

| | Equity method | Proportionate consolidation |
|---|---|---|
| Net income | **Same** | **Same** |
| Shareholders' equity | **Same** | **Same** |
| Assets, liabilities | Lower (one line) | Higher (share of each line) |
| Revenue | Lower (not included) | Higher |
| Net margin, ROA | **Higher** | Lower |
| Leverage (D/E) | Lower | Higher |

Analysts sometimes **restate equity-method joint ventures on a proportionate basis**
to show the investor's real share of debt and operating assets.
""")

        st.markdown("""
### Side-by-side: same economics, different statements

| | Equity method | Proportionate consolidation | Acquisition method |
|---|---|---|---|
| Assets & liabilities | Lowest | Middle | **Highest** (100% of investee) |
| Revenue | Lowest | Middle | **Highest** |
| Net income to parent | Same | Same | Same |
| Equity | Same | Same | **Higher** (includes NCI) |
| Net margin, ROA | **Highest** | Middle | Lowest |
| Leverage (D/E) | Lowest | Middle | **Highest** |
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Parent company**")
            p_assets  = st.slider("Parent total assets ($M)", 100, 2000, 500, 50)
            p_equity  = st.slider("Parent equity ($M)", 50, 1000, 200, 25)
            p_ni      = st.slider("Parent net income ($M)", 5, 200, 30, 5)

            st.markdown("**Investee / subsidiary**")
            i_assets  = st.slider("Investee total assets ($M)", 50, 1000, 300, 25)
            i_liab    = st.slider("Investee total liabilities ($M)", 0, 800, 180, 25)
            i_ni      = st.slider("Investee net income ($M)", 1, 100, 20, 5)
            ownership = st.slider("Ownership stake (%)", 5, 95, 35, 5) / 100

        df = intercorporate_ratios(p_assets, p_equity, p_ni, i_assets, i_liab, i_ni, ownership)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["ROE (%)", "ROA (%)", "Debt/Equity"],
        )
        methods = df.index.tolist()
        colors  = ["steelblue", "darkorange", "mediumpurple", "crimson"]

        for col_idx, metric in enumerate(["ROE (%)", "ROA (%)", "D/E"], start=1):
            vals = df[metric].tolist()
            fig.add_trace(go.Bar(
                x=methods, y=vals,
                marker_color=colors, name=metric,
                text=[f"{v:.1f}" for v in vals],
                textposition="outside",
            ), row=1, col=col_idx)

        fig.update_layout(height=400, showlegend=False, title_text="Ratio Comparison by Accounting Method")

        with col2:
            st.dataframe(
                df.style.format({
                    "Total Assets ($M)": "{:.1f}",
                    "Equity ($M)": "{:.1f}",
                    "Net Income ($M)": "{:.1f}",
                    "ROE (%)": "{:.1f}",
                    "ROA (%)": "{:.1f}",
                    "D/E": "{:.2f}",
                }).background_gradient(cmap="RdYlGn", subset=["ROE (%)", "ROA (%)"]),
                use_container_width=True,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab_fa:
        st.markdown("""
## Investments in Financial Assets — IFRS 9 Classification

Passive investments (no significant influence) are classified by **two tests**:

1. **Business model:** what does the company do with the asset — *hold to collect*
   contractual cash flows, *hold to collect and sell*, or trade?
2. **Cash-flow characteristics (SPPI):** are the cash flows *solely payments of
   principal and interest*? Plain bonds pass; equity and derivatives fail.

| Category | Instruments | Criteria | Balance sheet | Unrealized FV changes | On sale |
|---|---|---|---|---|---|
| **Amortized cost** | Debt only | Hold to collect **+** SPPI | Amortized cost (EIR method) | Not recognized | Realized G/L → P&L |
| **FVOCI — debt** | Debt | Hold to collect **and sell** + SPPI | Fair value | **OCI** | OCI balance **recycled** to P&L |
| **FVOCI — equity** | Equity | Irrevocable election at purchase; not held for trading | Fair value | **OCI** | **No recycling** — stays in equity |
| **FVTPL** | Default: equity, derivatives, debt failing the tests | Anything else (or FV option to fix an accounting mismatch) | Fair value | **P&L** | Already in P&L |

### What goes where

| | Amortized cost | FVOCI — debt | FVOCI — equity | FVTPL |
|---|---|---|---|---|
| Interest / dividends | P&L (EIR) | P&L (EIR) | Dividends → P&L | P&L |
| Expected credit losses | P&L | P&L | n/a | n/a |
| Transaction costs | Capitalized | Capitalized | Capitalized | **Expensed** |

**Effective interest method:** interest income = carrying amount × yield at purchase,
not the coupon. A bond bought at a discount accretes up to par; one bought at a
premium amortizes down.

$$\\text{AC}_{t} = \\text{AC}_{t-1} \\times (1 + \\text{EIR}) - \\text{Coupon}$$

**US GAAP mapping (debt):** held-to-maturity ≈ amortized cost, available-for-sale ≈ FVOCI,
trading ≈ FVTPL. Equity investments go through net income. US GAAP has **no** FVOCI
election for equity.

**Reclassification:** debt only, and only when the business model changes (rare,
prospective). Equity FVOCI election is irrevocable.

> **Exam tip:** Given the same bond and the same price moves:
> - **Net income:** amortized cost = FVOCI. FVTPL differs by the unrealized gain/loss.
> - **Balance sheet:** FVOCI = FVTPL (fair value). Amortized cost differs.
> - **Total comprehensive income:** FVOCI = FVTPL.
> - **Once sold:** net income is the same under all three, because FVOCI-debt recycles.
>   FVOCI-equity does **not** recycle, so its gains never hit net income.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Bond bought at start of year**")
            fa_face   = st.slider("Par value ($M)", 10, 500, 100, 10)
            fa_coupon = st.slider("Coupon rate (%)", 0.0, 10.0, 4.0, 0.25) / 100
            fa_y0     = st.slider("Market yield at purchase = EIR (%)", 0.5, 10.0, 5.0, 0.25) / 100
            fa_years  = st.slider("Years to maturity at purchase", 2, 10, 5, 1)
            st.markdown("**At year-end**")
            fa_y1     = st.slider("Market yield at year-end (%)", 0.5, 10.0, 6.0, 0.25) / 100
            fa_sold   = st.checkbox("Sold at year-end fair value", value=False)

        price0, ac_end, fv_end, df_fa = financial_asset_year1(
            fa_face, fa_coupon, fa_y0, fa_years, fa_y1, fa_sold,
        )

        # Stacked income components per method
        components = [
            ("Interest income (P&L)", "steelblue"),
            ("FV change (P&L)", "darkorange"),
            ("Realized gain (P&L)", "crimson"),
            ("OCI", "mediumseagreen"),
        ]
        fig_fa = go.Figure()
        for comp, color in components:
            vals = df_fa[comp].tolist()
            fig_fa.add_trace(go.Bar(
                x=df_fa.index.tolist(), y=vals, name=comp, marker_color=color,
                text=[f"{v:+.2f}" if abs(v) > 1e-9 else "" for v in vals],
                textposition="inside",
            ))
        for method, tci in df_fa["Total comprehensive income"].items():
            fig_fa.add_annotation(
                x=method, y=max(tci, df_fa.loc[method, "Net income"], 0),
                text=f"NI {df_fa.loc[method, 'Net income']:.2f} · TCI {tci:.2f}",
                showarrow=False, yshift=14,
            )
        fig_fa.update_layout(
            barmode="relative", height=400, yaxis_title="$M",
            title="Year-1 income by classification",
            legend=dict(orientation="h", y=-0.15),
        )

        with col2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Purchase price", f"${price0:,.2f}M")
            m2.metric("Amortized cost (year-end)", f"${ac_end:,.2f}M")
            m3.metric("Fair value (year-end)", f"${fv_end:,.2f}M",
                      delta=f"{fv_end - ac_end:+.2f}M vs AC")
            st.dataframe(df_fa.style.format("{:,.2f}"), use_container_width=True)
            st.plotly_chart(fig_fa, use_container_width=True)

        # Carrying amount over the bond's life
        st.markdown("### Carrying amount over the bond's life")
        st.caption(
            "Amortized cost pulls to par along the purchase yield. Fair value assumes the "
            "market yield stays at the year-end level. The gap is the accumulated OCI "
            "under FVOCI, or the cumulative unrealized P&L under FVTPL."
        )
        years_axis = list(range(fa_years + 1))
        ac_path = amortized_cost_schedule(fa_face, fa_coupon, fa_y0, fa_years, price0)
        fv_path = [price0] + [bond_price(fa_face, fa_coupon, fa_y1, fa_years - t)
                              for t in years_axis[1:]]

        fig_path = go.Figure()
        fig_path.add_trace(go.Scatter(
            x=years_axis, y=ac_path, mode="lines+markers", name="Amortized cost",
            line=dict(color="steelblue", width=2.5),
        ))
        fig_path.add_trace(go.Scatter(
            x=years_axis, y=fv_path, mode="lines+markers", name="Fair value (FVOCI / FVTPL)",
            line=dict(color="darkorange", width=2.5, dash="dash"),
        ))
        fig_path.add_hline(y=fa_face, line_dash="dot", line_color="gray",
                           annotation_text="Par")
        fig_path.update_layout(
            xaxis_title="Years since purchase", yaxis_title="$M", height=320,
        )
        st.plotly_chart(fig_path, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
## Pension Accounting — Defined Benefit Plans

Under IAS 19 / US GAAP, companies must recognise the **net pension liability**
(or asset) = Projected Benefit Obligation (PBO) − Fair Value of Plan Assets.

### PBO Roll-Forward

$$\\text{PBO}_{end} = \\text{PBO}_{begin} + \\text{Service Cost} + \\text{Interest Cost} + \\text{Actuarial Gains/Losses} − \\text{Benefits Paid}$$

| Component | Driven by | Income Statement? |
|---|---|---|
| **Service cost** | Current year benefit earned | Yes — operating |
| **Interest cost** | PBO × discount rate | Yes — financing (or operating) |
| **Actuarial G/L** | Changes in discount rate, mortality | OCI (IFRS) or corridor (old GAAP) |
| **Benefits paid** | Actual payments | No (balance sheet only) |

### Key sensitivity

The PBO is a **discounted liability** — a fall in the discount rate (e.g., falling
bond yields) *dramatically increases* the PBO. This is the primary source of
pension risk for plan sponsors.

> **Exam tip:** Under IFRS, all actuarial gains/losses go to OCI immediately.
> This means the P&L is "cleaner" but equity is more volatile.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            pbo_begin    = st.slider("PBO beginning ($M)", 100, 5000, 1000, 50)
            service_cost = st.slider("Service cost ($M)", 5, 200, 40, 5)
            discount_r   = st.slider("Discount rate (%)", 1.0, 10.0, 4.5, 0.25) / 100
            salary_g     = st.slider("Salary growth assumption (%)", 1.0, 8.0, 3.0, 0.25) / 100
            benefits_pd  = st.slider("Benefits paid ($M)", 0, 300, 60, 10)
            plan_assets  = st.slider("Plan assets fair value ($M)", 100, 5000, 850, 50)
            asset_return = st.slider("Actual plan asset return (%)", -10.0, 20.0, 6.0, 0.5) / 100

        interest_cost  = pbo_begin * discount_r
        # Actuarial loss: 1% change in discount rate has large impact
        actuarial_loss = -pbo_begin * (salary_g - 0.03) * 3   # simplified sensitivity
        pbo_end        = pbo_begin + service_cost + interest_cost + actuarial_loss - benefits_pd

        funded_status_begin = plan_assets - pbo_begin
        asset_gain    = plan_assets * asset_return
        plan_assets_end = plan_assets + asset_gain - benefits_pd
        funded_status_end = plan_assets_end - pbo_end

        # Waterfall
        labels   = ["PBO begin", "Service cost", "Interest cost", "Actuarial G/L",
                    "Benefits paid", "PBO end"]
        values   = [pbo_begin, service_cost, interest_cost, actuarial_loss,
                    -benefits_pd, 0]
        measures = ["absolute", "relative", "relative", "relative", "relative", "total"]
        wf_colors = ["steelblue", "crimson", "darkorange",
                     "green" if actuarial_loss < 0 else "crimson",
                     "steelblue", "mediumseagreen"]

        fig2 = go.Figure(go.Waterfall(
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector=dict(line=dict(color="gray", width=1)),
            decreasing=dict(marker_color="tomato"),
            increasing=dict(marker_color="crimson"),
            totals=dict(marker_color="steelblue"),
            text=[f"${v:+.0f}M" if m == "relative" else f"${v:.0f}M"
                  for v, m in zip(values, measures)],
            textposition="outside",
        ))
        fig2.update_layout(
            title="PBO Roll-Forward",
            yaxis_title="$M",
            height=420,
        )

        with col2:
            m1, m2, m3 = st.columns(3)
            m1.metric("PBO (end)", f"${pbo_end:,.0f}M",
                      delta=f"{pbo_end - pbo_begin:+.0f}M")
            m2.metric("Plan Assets (end)", f"${plan_assets_end:,.0f}M")
            status_sign = "Overfunded" if funded_status_end > 0 else "Underfunded"
            m3.metric("Funded Status", f"${funded_status_end:,.0f}M",
                      delta=status_sign,
                      delta_color="normal" if funded_status_end > 0 else "inverse")
            st.plotly_chart(fig2, use_container_width=True)

        # Sensitivity: discount rate vs PBO
        st.markdown("### PBO sensitivity to discount rate")
        dr_range = np.linspace(0.01, 0.10, 80)
        pbo_sens = [pbo_begin / (discount_r / dr) if dr > 0 else np.nan for dr in dr_range]
        # Rough approximation: PBO scales inversely with discount rate
        pbo_sens_approx = [pbo_begin * (discount_r / dr) ** 10 for dr in dr_range]

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=dr_range * 100, y=pbo_sens_approx,
            mode="lines", line=dict(color="steelblue", width=2.5),
            name="PBO (approx. duration=10)",
        ))
        fig3.add_vline(x=discount_r * 100, line_dash="dot", line_color="crimson",
                       annotation_text=f"Current rate {discount_r*100:.1f}%")
        fig3.update_layout(
            xaxis_title="Discount rate (%)", yaxis_title="PBO ($M)",
            height=280, title="PBO vs Discount Rate (inverse relationship)",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
## Multinational Operations — FX Translation

When a parent consolidates a **foreign subsidiary**, it must translate the
subsidiary's financials into the parent's **functional currency**.

### Two methods

| | **Current Rate Method** | **Temporal Method** |
|---|---|---|
| **Use when** | Foreign subsidiary operates independently (functional currency = local) | Foreign subsidiary is extension of parent (functional currency = parent's) |
| **Assets & Liabilities** | All at **current** (closing) rate | Monetary at current rate; non-monetary at **historical** rate |
| **Income Statement** | Average rate | Mixed (matched to underlying) |
| **Translation G/L** | Goes to **OCI** (CTA reserve) | Goes to **income statement** |
| **IFRS / US GAAP** | Default for independent subs | Required for hyperinflationary economies |

### Key exam trap

Under the **temporal method**, if a subsidiary holds mostly non-monetary assets
(e.g., property) and the local currency depreciates, the translation gain/loss
hits the income statement — creating earnings volatility not present under current rate.

> **Exam tip:** Identify which method applies, then trace the effect on ROE, D/E,
> and earnings. A weaker foreign currency reduces reported assets and equity under
> the current rate method (negative CTA), but the effect on earnings depends on
> the method.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            hist_rate    = st.number_input("Historical rate (FX/USD)", 0.5, 5.0, 2.0, 0.1)
            avg_rate     = st.number_input("Average rate this year (FX/USD)", 0.5, 5.0, 1.8, 0.1)
            curr_rate    = st.number_input("Current (closing) rate (FX/USD)", 0.5, 5.0, 1.6, 0.1)
            fx_appn = (curr_rate - hist_rate) / hist_rate * 100
            st.caption(f"FX change: {fx_appn:+.1f}% since asset purchase")

        # Simplified subsidiary B/S in local currency
        local = {
            "Cash":                   200,
            "Accounts Receivable":    150,
            "Inventory":              300,
            "Property, Plant & Eq.":  800,
            "Total Assets":          1450,
            "Accounts Payable":       100,
            "Long-term Debt":         400,
            "Common Equity":          950,
            "Total L+E":             1450,
        }

        # Classify: monetary vs non-monetary
        monetary    = {"Cash", "Accounts Receivable", "Accounts Payable", "Long-term Debt"}
        non_monetary = {"Inventory", "Property, Plant & Eq.", "Common Equity"}

        rows = []
        for item, lc_val in local.items():
            is_mon = item in monetary
            is_total = item.startswith("Total")

            # Current rate method
            if is_total:
                cr_rate_used = "—"
                cr_usd = None
            elif is_mon or item not in non_monetary:
                cr_rate_used = f"{curr_rate:.2f} (current)"
                cr_usd = lc_val / curr_rate
            else:
                cr_rate_used = f"{curr_rate:.2f} (current)"
                cr_usd = lc_val / curr_rate

            # Temporal method
            if is_total:
                tm_rate_used = "—"
                tm_usd = None
            elif is_mon:
                tm_rate_used = f"{curr_rate:.2f} (current)"
                tm_usd = lc_val / curr_rate
            elif item == "Common Equity":
                tm_rate_used = f"{hist_rate:.2f} (historical)"
                tm_usd = lc_val / hist_rate
            else:
                tm_rate_used = f"{hist_rate:.2f} (historical)"
                tm_usd = lc_val / hist_rate

            rows.append({
                "Item": item,
                "LC": lc_val,
                "Current Rate (USD)": round(cr_usd, 0) if cr_usd else None,
                "Temporal (USD)": round(tm_usd, 0) if tm_usd else None,
                "Rate used (Current)": cr_rate_used,
                "Rate used (Temporal)": tm_rate_used,
            })

        df_fx = pd.DataFrame(rows)
        # Compute totals
        cr_assets   = df_fx.loc[df_fx["Item"].isin(["Cash","Accounts Receivable","Inventory","Property, Plant & Eq."]), "Current Rate (USD)"].sum()
        tm_assets   = df_fx.loc[df_fx["Item"].isin(["Cash","Accounts Receivable","Inventory","Property, Plant & Eq."]), "Temporal (USD)"].sum()
        cr_liab     = df_fx.loc[df_fx["Item"].isin(["Accounts Payable","Long-term Debt"]), "Current Rate (USD)"].sum()
        tm_liab     = df_fx.loc[df_fx["Item"].isin(["Accounts Payable","Long-term Debt"]), "Temporal (USD)"].sum()
        cr_equity   = local["Common Equity"] / curr_rate
        tm_equity   = local["Common Equity"] / hist_rate
        cr_cta      = cr_assets - cr_liab - cr_equity   # goes to OCI
        tm_gl       = tm_assets - tm_liab - tm_equity   # goes to I/S

        with col2:
            st.dataframe(
                df_fx[["Item", "LC", "Current Rate (USD)", "Temporal (USD)"]].iloc[:8],
                use_container_width=True,
                hide_index=True,
            )

            fig_fx = go.Figure()
            categories = ["Total Assets", "Total Liabilities", "Equity (ex-CTA/GL)"]
            cr_vals = [cr_assets, cr_liab, cr_equity]
            tm_vals = [tm_assets, tm_liab, tm_equity]
            x = np.arange(len(categories))
            width = 0.35

            fig_fx.add_trace(go.Bar(
                x=categories, y=cr_vals, name="Current Rate",
                marker_color="steelblue", width=width,
                text=[f"${v:,.0f}" for v in cr_vals], textposition="auto",
                offset=-width/2,
            ))
            fig_fx.add_trace(go.Bar(
                x=categories, y=tm_vals, name="Temporal",
                marker_color="darkorange", width=width,
                text=[f"${v:,.0f}" for v in tm_vals], textposition="auto",
                offset=width/2,
            ))

            fig_fx.update_layout(
                title="USD Balance Sheet: Current Rate vs Temporal",
                yaxis_title="USD ($)",
                barmode="group",
                height=360,
            )
            st.plotly_chart(fig_fx, use_container_width=True)

            c1, c2 = st.columns(2)
            cta_str = "gain" if cr_cta > 0 else "loss"
            gl_str  = "gain" if tm_gl > 0 else "loss"
            c1.metric("CTA (Current Rate → OCI)", f"${cr_cta:+,.0f}", delta="OCI")
            c2.metric("Translation G/L (Temporal → I/S)", f"${tm_gl:+,.0f}",
                      delta="Income Statement", delta_color="inverse" if tm_gl < 0 else "normal")
