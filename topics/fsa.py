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
    """Return key ratios under fair value, equity method, and consolidation."""

    # ── Fair value (minority: <20%, held as investment at fair value) ──────
    fv_assets      = parent_assets              # investee not on B/S
    fv_equity      = parent_equity
    fv_net_income  = parent_net_income          # dividends only, ignored here for simplicity

    # ── Equity method (associate: 20–50%) ─────────────────────────────────
    em_investee_bv = investee_assets - investee_liabilities
    em_assets      = parent_assets + ownership * em_investee_bv   # one line on B/S
    em_equity      = parent_equity + ownership * em_investee_bv   # mirrored
    em_net_income  = parent_net_income + ownership * investee_net_income

    # ── Full consolidation (subsidiary: >50%) ──────────────────────────────
    cons_assets    = parent_assets + investee_assets
    cons_liab      = (parent_assets - parent_equity) + investee_liabilities
    nci            = (1 - ownership) * (investee_assets - investee_liabilities)
    cons_equity    = parent_equity + nci         # includes NCI
    cons_net_inc   = parent_net_income + investee_net_income   # then deduct NCI share

    methods = {
        "Fair Value / FVTPL": dict(
            assets=fv_assets, equity=fv_equity, net_income=fv_net_income,
        ),
        "Equity Method": dict(
            assets=em_assets, equity=em_equity, net_income=em_net_income,
        ),
        "Consolidation": dict(
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


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Financial Statement Analysis")

    tab1, tab2, tab3 = st.tabs([
        "Intercorporate Investments",
        "Pension Accounting",
        "Multinational FX Translation",
    ])

    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
## Intercorporate Investments — Accounting Methods

The accounting treatment depends on the ownership percentage and degree of influence:

| Ownership | Influence | Method | Balance Sheet Impact |
|---|---|---|---|
| **< 20%** | Little/none | Fair value through P&L (FVTPL) | Investment at fair value only |
| **20–50%** | Significant | **Equity method** | One-line "Investment in associate" |
| **> 50%** | Control | **Full consolidation** | 100% of assets + liabilities on B/S |

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
        colors  = ["steelblue", "darkorange", "crimson"]

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
