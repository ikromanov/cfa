"""Ethics & Professional Standards — Standards I-VII, GIPS, research objectivity."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Ethics & Professional Standards")
    st.markdown("""
Ethics is **10–15% of the exam** and is disproportionately important because a failing
ethics score can prevent candidates from passing even with high scores elsewhere.
Level II tests *application* in complex multi-party vignettes, not just recall.
""")

    tab1, tab2, tab3 = st.tabs([
        "Standards I–VII Overview",
        "GIPS & Performance Reporting",
        "Research Objectivity",
    ])

    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
## CFA Institute Standards of Professional Conduct

### The Seven Standards

| Standard | Title | Key Level II traps |
|---|---|---|
| **I** | Professionalism | Misrepresentation of credentials or model assumptions |
| **II** | Integrity of Capital Markets | Mosaic theory vs. material non-public information (MNPI) |
| **III** | Duties to Clients | Suitability, fair dealing, performance presentation |
| **IV** | Duties to Employers | Whistleblowing, independent practice |
| **V** | Investment Analysis | Diligence, reasonable basis, record retention |
| **VI** | Conflicts of Interest | Disclosure of referral fees, priority of transactions |
| **VII** | Responsibilities as CFA Member | Conduct during exam, reference to designation |

---

### Standard II(A) — Material Non-Public Information (MNPI)

This is the most frequently tested area at Level II. The key framework:

**Mosaic Theory:** An analyst may use public information combined with non-material
non-public information to reach a conclusion — this is *not* a violation.

$$\\text{MNPI} = \\text{Material} \\cap \\text{Non-Public}$$

Information is **material** if its disclosure would likely affect the security's price.
It is **non-public** if it has not been disseminated to the marketplace.

> **Scenario:** An analyst meets a company CFO who mentions *"we had a good quarter"*.
> Is this MNPI? It depends — if the market already expects a good quarter, it may not
> be material. If it contradicts consensus, it likely is.
""")

        st.markdown("### Interactive: Is this MNPI?")
        col1, col2 = st.columns([1, 1])
        with col1:
            info_public = st.radio("Is the information public?",
                                    ["Yes — widely disseminated", "No — not yet public"])
            info_material = st.radio("Is the information material?",
                                      ["Yes — would move the price", "No — already priced in or trivial"])
            action = st.radio("Analyst's proposed action",
                               ["Trade on this information", "Use as part of mosaic (with other public data)",
                                "Report to supervisor / compliance", "Do nothing"])

        is_public   = "Yes" in info_public
        is_material = "Yes" in info_material
        is_mnpi     = not is_public and is_material

        with col2:
            if is_mnpi and "Trade" in action:
                st.error("**VIOLATION** — Trading on MNPI violates Standard II(A). "
                          "The analyst must immediately stop and not trade until the "
                          "information is made public.")
            elif is_mnpi and "mosaic" in action:
                st.error("**VIOLATION** — Even using MNPI as *part* of the mosaic is "
                          "prohibited. Mosaic theory only protects use of non-material "
                          "non-public information combined with public information.")
            elif is_mnpi and "supervisor" in action:
                st.success("**CORRECT** — Reporting to compliance is the appropriate "
                            "action. The firm must implement information barriers and "
                            "the analyst should not trade.")
            elif not is_material:
                st.info("Not material → not MNPI. Analyst may use it as part of mosaic "
                         "analysis (Mosaic Theory applies).")
            elif is_public:
                st.info("Already public → not MNPI. May trade freely on this analysis.")

            # Classification matrix
            fig = go.Figure(go.Heatmap(
                z=[[0, 1], [0, 0]],
                x=["Not Material", "Material"],
                y=["Public", "Non-Public"],
                colorscale=[[0, "mediumseagreen"], [1, "crimson"]],
                showscale=False,
                text=[["OK to use", "OK (already priced in)"],
                      ["Mosaic Theory OK", "MNPI — DO NOT TRADE"]],
                texttemplate="%{text}",
                textfont=dict(size=13),
            ))
            fig.update_layout(
                title="MNPI Classification Matrix",
                height=300,
                xaxis_title="Materiality",
                yaxis_title="Public Status",
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("""
### Standard III(A) — Loyalty, Prudence, Care

The **fiduciary duty hierarchy**: client interests > employer interests > personal interests.
In a discretionary account, the manager must act in the client's *best interest*, not
just avoid obvious conflicts.

### Standard VI(B) — Priority of Transactions

Order of priority:
1. **Client accounts** — always first
2. **Employer accounts** — second
3. **Personal accounts** — last

Front-running (trading personal accounts ahead of client orders) is a clear violation.
""")

    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
## GIPS — Global Investment Performance Standards

GIPS provides a common framework for presenting investment performance so that
investors can make **like-for-like comparisons** across managers.

### Core requirements

- Performance must be calculated using **time-weighted returns (TWR)**, not money-weighted
- All **actual, fee-paying discretionary accounts** must be included in at least one composite
- **10 years** of compliant history (or since inception) must be presented
- **Composites** group accounts with similar investment objectives/strategies

### TWR vs MWR

$$TWR = \\prod_{t=1}^{n}(1 + r_t) - 1, \\quad r_t = \\frac{V_{end} - V_{begin} - CF}{V_{begin} + w_t \\cdot CF}$$

$$MWR = IRR: \\quad V_0 = \\sum_{t} \\frac{CF_t}{(1+IRR)^t}$$

**GIPS requires TWR** because it eliminates the effect of external cash flow timing
— which is controlled by the *client*, not the manager.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            v0         = st.slider("Beginning value ($)", 100_000, 2_000_000, 1_000_000, 50_000, format="%d")
            cf_midpoint = st.slider("Cash inflow at midpoint ($)", -500_000, 1_000_000, 200_000, 25_000, format="%d")
            r1         = st.slider("Return sub-period 1 (%)", -30.0, 30.0, 10.0, 0.5) / 100
            r2         = st.slider("Return sub-period 2 (%)", -30.0, 30.0, -5.0, 0.5) / 100

        v_mid_before_cf = v0 * (1 + r1)
        v_mid           = v_mid_before_cf + cf_midpoint
        v_end           = v_mid * (1 + r2)

        twr = (1 + r1) * (1 + r2) - 1

        # MWR: solve CF equation
        # v0 at t=0, cf at t=0.5, -v_end at t=1
        # Approximate MWR iteratively
        def mwr_obj(irr):
            return v0 + cf_midpoint / (1 + irr)**0.5 - v_end / (1 + irr)
        from scipy.optimize import brentq
        try:
            mwr = brentq(mwr_obj, -0.99, 5.0)
        except Exception:
            mwr = np.nan

        fig2 = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Portfolio Value Timeline", "TWR vs MWR Comparison"],
        )

        # Timeline
        times  = [0, 0.5, 0.5, 1.0]
        values = [v0, v_mid_before_cf, v_mid, v_end]
        fig2.add_trace(go.Scatter(
            x=times, y=values, mode="lines+markers",
            line=dict(color="steelblue", width=2.5),
            marker=dict(size=10),
            name="Portfolio value",
        ), row=1, col=1)
        if cf_midpoint != 0:
            fig2.add_annotation(
                x=0.5, y=max(v_mid_before_cf, v_mid),
                text=f"Cash flow: ${cf_midpoint:+,.0f}",
                arrowhead=2, ax=40, ay=-30,
                font=dict(color="crimson"),
                row=1, col=1,
            )
        fig2.update_xaxes(title_text="Year", row=1, col=1,
                          tickvals=[0, 0.5, 1], ticktext=["t=0", "t=0.5", "t=1"])
        fig2.update_yaxes(title_text="Portfolio Value ($)", tickformat="$,.0f", row=1, col=1)

        # Comparison
        methods_g = ["Time-Weighted Return\n(GIPS required)", "Money-Weighted Return\n(IRR)"]
        return_vals = [twr * 100, mwr * 100 if np.isfinite(mwr) else 0]
        bar_colors = ["steelblue", "darkorange"]
        fig2.add_trace(go.Bar(
            x=methods_g, y=return_vals, marker_color=bar_colors,
            text=[f"{v:.2f}%" for v in return_vals], textposition="auto",
            name="Return",
        ), row=1, col=2)
        fig2.add_hline(y=0, line_color="black", row=1, col=2)
        fig2.update_yaxes(title_text="Return (%)", row=1, col=2)
        fig2.update_layout(height=400, showlegend=False)

        with col2:
            c1, c2 = st.columns(2)
            c1.metric("TWR (GIPS)", f"{twr*100:.2f}%")
            c2.metric("MWR (IRR)", f"{mwr*100:.2f}%" if np.isfinite(mwr) else "N/A")
            if abs(twr - mwr) > 0.01:
                diff = twr - mwr
                st.info(f"TWR and MWR differ by {diff*100:.1f} ppts because the cash flow "
                         "timing benefited " + ("the late sub-period (large inflow before bad return)." if cf_midpoint > 0 and r2 < 0
                         else "the early sub-period."))
            st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
## Research Objectivity & Conflicts of Interest

### Standard V(A) — Diligence and Reasonable Basis

Analysts must have a **reasonable and adequate basis** for recommendations.
At Level II, this is tested in the context of:
- Relying on third-party research (must evaluate the process, not just the conclusion)
- Quantitative models (must understand the model's assumptions and limitations)
- Using outdated information

### Standard VI(A) — Disclosure of Conflicts

All material conflicts must be disclosed. Common conflicts:

| Conflict | Example | Required action |
|---|---|---|
| **Ownership** | Analyst owns shares in covered company | Disclose in report |
| **Underwriting relationship** | Bank is underwriting the company's IPO | Disclose; may need to restrict research |
| **Referral fees** | Advisor receives fee for directing clients to fund | Disclose to client |
| **Soft dollars** | Broker provides research in exchange for trading commissions | Disclose; must benefit clients |

### IPO Allocation — Fair Dealing (Standard III(B))

When shares are oversubscribed, allocation must be:
- **Pro-rata** based on order size (not selective favoring)
- All clients in the same composite/risk profile treated equally
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            total_shares  = st.slider("IPO shares available (M)", 1, 100, 10)
            total_demand  = st.slider("Total client demand (M)", 1, 200, 40)
            n_clients     = st.slider("Number of client accounts", 2, 10, 5)

        # Simulate client orders
        rng = np.random.default_rng(42)
        order_sizes = rng.integers(1, 20, size=n_clients).astype(float)
        order_sizes = order_sizes / order_sizes.sum() * total_demand

        # Pro-rata allocation
        fill_rate    = min(total_shares / total_demand, 1.0)
        pro_rata_all = order_sizes * fill_rate

        # "Unfair" allocation: first n/2 clients get more
        unfair_all   = order_sizes.copy()
        half         = n_clients // 2
        unfair_all[:half] *= min(1.5 * fill_rate, 1.0) / fill_rate
        unfair_all[:half] = np.minimum(unfair_all[:half], order_sizes[:half])
        # Scale to match total available
        unfair_all = unfair_all / unfair_all.sum() * min(total_shares, total_demand)
        unfair_all = np.minimum(unfair_all, order_sizes)

        fig3 = go.Figure()
        client_labels = [f"Client {i+1}" for i in range(n_clients)]

        fig3.add_trace(go.Bar(
            x=client_labels, y=order_sizes,
            name="Order size", marker_color="lightgray",
            text=[f"{v:.1f}M" for v in order_sizes], textposition="auto",
        ))
        fig3.add_trace(go.Bar(
            x=client_labels, y=pro_rata_all,
            name=f"Pro-rata allocation (fill={fill_rate:.0%})",
            marker_color="steelblue",
        ))
        fig3.add_trace(go.Bar(
            x=client_labels, y=unfair_all,
            name="Cherry-picked allocation (violation!)",
            marker_color="crimson", opacity=0.7,
        ))

        fig3.update_layout(
            title=f"IPO Allocation: {total_shares}M shares available vs {total_demand:.0f}M demand",
            yaxis_title="Shares Allocated (M)",
            barmode="overlay",
            height=420,
        )
        fig3.add_annotation(
            x=n_clients // 4, y=max(pro_rata_all) * 1.2,
            text="Pro-rata = always compliant",
            font=dict(color="steelblue"), showarrow=False,
        )

        with col2:
            st.plotly_chart(fig3, use_container_width=True)
            st.metric("Fill rate (pro-rata)", f"{fill_rate:.1%}")
            if fill_rate < 1:
                st.warning(f"Oversubscribed by {total_demand/total_shares:.1f}x. "
                           "Pro-rata allocation is the only compliant method.")
