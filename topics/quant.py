"""Quantitative Methods — regression diagnostics, time series, ML basics."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import streamlit as st


# ── helpers ───────────────────────────────────────────────────────────────────

def simulate_regression(n=100, heterosked=0.0, autocorr=0.0, seed=42):
    """Generate OLS data with controllable violations."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # Heteroskedastic errors: variance proportional to |x1|^heterosked
    scale = 1 + heterosked * np.abs(x1)
    eps = rng.normal(0, scale)
    # Add autocorrelation via AR(1) on residuals
    for t in range(1, n):
        eps[t] += autocorr * eps[t - 1]
    y = 2 + 0.8 * x1 - 0.4 * x2 + eps
    return x1, x2, y


def ols_fit(x1, x2, y):
    X = np.column_stack([np.ones(len(y)), x1, x2])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    resid = y - y_hat
    return beta, y_hat, resid


def durbin_watson(resid):
    diff = np.diff(resid)
    return np.sum(diff**2) / np.sum(resid**2)


def vif(x1, x2, corr_12):
    """Approximate VIF for two predictors given their correlation."""
    r2 = corr_12**2
    return 1 / (1 - r2) if r2 < 1 else np.inf


def simulate_ar1(n=200, b0=0.5, b1=0.8, sigma=1.0, seed=7):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = b0 / (1 - b1) if abs(b1) < 1 else 0.0
    for t in range(1, n):
        x[t] = b0 + b1 * x[t - 1] + rng.normal(0, sigma)
    return x


def acf(series, max_lag=20):
    n = len(series)
    mean = series.mean()
    var = np.var(series)
    lags, corrs = [], []
    for k in range(1, max_lag + 1):
        cov_k = np.mean((series[:n - k] - mean) * (series[k:] - mean))
        lags.append(k)
        corrs.append(cov_k / var if var > 0 else 0)
    return lags, corrs


# ── main render ───────────────────────────────────────────────────────────────

def render():
    st.title("Quantitative Methods")

    tab1, tab2, tab3 = st.tabs([
        "Multiple Regression Diagnostics",
        "Time Series: AR & Unit Roots",
        "ML Concepts: Bias-Variance & LASSO",
    ])

    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
## Multiple Regression — Level II Diagnostics

At Level II you must **interpret** regression output and identify violations of the
classical linear regression assumptions (CLRM).

### The key violations

| Violation | Symptom | Consequence | Fix |
|---|---|---|---|
| **Heteroskedasticity** | Residuals fan out with fitted values | Standard errors biased → invalid t-tests | Robust (White) SEs or WLS |
| **Serial correlation** | Residuals show patterns over time | SEs underestimated → Type I errors | Newey-West SEs or GLS |
| **Multicollinearity** | High $R^2$ but insignificant t-stats | Inflated SEs, unstable coefficients | Drop variable, PCA, or regularize |

### Key statistics to read from output

$$DW = \\frac{\\sum_{t=2}^{T}(\\hat{e}_t - \\hat{e}_{t-1})^2}{\\sum_{t=1}^{T}\\hat{e}_t^2} \\approx 2(1-\\hat{\\rho})$$

DW ≈ 2 → no autocorrelation. DW < 1 → positive serial correlation (common in financial time series).

$$VIF_j = \\frac{1}{1 - R_j^2}$$

VIF > 5 is problematic; VIF > 10 is severe multicollinearity.

> **Exam tip:** Heteroskedasticity and serial correlation do NOT bias the coefficients —
> they only bias the *standard errors*. Multicollinearity also leaves coefficients unbiased
> but makes them imprecise and highly sensitive to the sample.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            heterosked = st.slider("Heteroskedasticity severity", 0.0, 3.0, 0.0, 0.1)
            autocorr   = st.slider("Serial correlation (ρ)", -0.9, 0.9, 0.0, 0.05)
            corr_12    = st.slider("Correlation between X₁ and X₂", -0.95, 0.95, 0.1, 0.05)
            n_obs      = st.slider("Number of observations", 30, 300, 100, 10)

        x1, x2, y = simulate_regression(n_obs, heterosked, autocorr)
        # Inject collinearity
        x2_col = corr_12 * x1 + np.sqrt(1 - corr_12**2) * x2
        beta, y_hat, resid = ols_fit(x1, x2_col, y)
        dw    = durbin_watson(resid)
        vif_x = vif(x1, x2_col, corr_12)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                "Residuals vs Fitted (Heteroskedasticity)",
                "Residual ACF (Serial Correlation)",
                "VIF (Multicollinearity)",
            ],
        )

        # Panel 1: residuals vs fitted
        color_resid = ["crimson" if abs(r) > 2 * resid.std() else "steelblue" for r in resid]
        fig.add_trace(go.Scatter(
            x=y_hat, y=resid, mode="markers",
            marker=dict(color=color_resid, size=5, opacity=0.7),
            name="Residuals",
            hovertemplate="Ŷ=%{x:.2f}  ê=%{y:.2f}<extra></extra>",
        ), row=1, col=1)
        fig.add_hline(y=0, line_color="black", line_dash="dash", row=1, col=1)

        # Ideal funnel overlay if heterosked > 0
        if heterosked > 0.5:
            sorted_yhat = np.sort(y_hat)
            fig.add_trace(go.Scatter(
                x=sorted_yhat,
                y=2 * (1 + heterosked * np.abs(sorted_yhat - sorted_yhat.mean()) / sorted_yhat.std()),
                mode="lines", line=dict(color="tomato", dash="dot", width=1),
                name="±2σ envelope", showlegend=True,
            ), row=1, col=1)

        # Panel 2: ACF of residuals
        lags, acf_vals = acf(resid, max_lag=15)
        ci = 1.96 / np.sqrt(n_obs)
        bar_colors = ["crimson" if abs(v) > ci else "steelblue" for v in acf_vals]
        fig.add_trace(go.Bar(
            x=lags, y=acf_vals, marker_color=bar_colors, name="ACF",
        ), row=1, col=2)
        fig.add_hline(y=ci,  line_color="gray", line_dash="dash", row=1, col=2)
        fig.add_hline(y=-ci, line_color="gray", line_dash="dash", row=1, col=2)

        # Panel 3: VIF bars
        vif2 = vif(x2_col, x1, corr_12)
        vif_vals  = [vif_x, vif2]
        vif_names = ["X₁", "X₂"]
        vif_colors = ["crimson" if v > 5 else ("orange" if v > 2 else "steelblue") for v in vif_vals]
        fig.add_trace(go.Bar(
            x=vif_names, y=vif_vals, marker_color=vif_colors, name="VIF",
        ), row=1, col=3)
        fig.add_hline(y=5, line_color="orange", line_dash="dash",
                      annotation_text="VIF=5 (caution)", row=1, col=3)
        fig.add_hline(y=10, line_color="crimson", line_dash="dash",
                      annotation_text="VIF=10 (severe)", row=1, col=3)

        fig.update_xaxes(title_text="Fitted values Ŷ", row=1, col=1)
        fig.update_yaxes(title_text="Residuals ê", row=1, col=1)
        fig.update_xaxes(title_text="Lag", row=1, col=2)
        fig.update_yaxes(title_text="Autocorrelation", row=1, col=2)
        fig.update_yaxes(title_text="VIF", row=1, col=3)
        fig.update_layout(height=420, showlegend=False)

        with col2:
            m1, m2, m3 = st.columns(3)
            m1.metric("Durbin-Watson", f"{dw:.3f}",
                      delta="No serial corr." if 1.5 < dw < 2.5 else "Serial corr. detected",
                      delta_color="normal" if 1.5 < dw < 2.5 else "inverse")
            m2.metric("VIF (X₁)", f"{vif_x:.1f}" if np.isfinite(vif_x) else "∞",
                      delta="OK" if vif_x < 5 else "High",
                      delta_color="normal" if vif_x < 5 else "inverse")
            m3.metric("R²", f"{1 - np.var(resid)/np.var(y):.3f}")
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("""
## Time Series — AR Models and Unit Roots

### AR(1) process

$$x_t = b_0 + b_1 x_{t-1} + \\varepsilon_t, \\quad \\varepsilon_t \\sim N(0, \\sigma^2)$$

**Mean reversion** requires $|b_1| < 1$. The long-run mean is:

$$\\bar{x} = \\frac{b_0}{1 - b_1}$$

### Unit root (random walk)

When $b_1 = 1$: $x_t = b_0 + x_{t-1} + \\varepsilon_t$ — the series is **non-stationary**.
Spurious regression: two independent random walks can show high $R^2$ and significant
t-statistics purely by chance. Always test with **Dickey-Fuller** before regressing time series.

### Cointegration

Two non-stationary series $X_t$ and $Y_t$ are **cointegrated** if a linear combination
$Y_t - \\beta X_t$ is stationary. Cointegrated series have a long-run equilibrium — the
deviation from it (the "spread") is mean-reverting. This is the basis for **pairs trading**.

> **Exam tip:** First-differencing removes a unit root. But if two series are cointegrated,
> you should model them in *levels* using an error correction model — differencing would
> destroy the long-run information.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            b1      = st.slider("AR(1) coefficient b₁", -0.5, 1.15, 0.7, 0.01)
            b0_val  = st.slider("Intercept b₀", -1.0, 1.0, 0.3, 0.05)
            sigma   = st.slider("Error σ", 0.1, 3.0, 1.0, 0.1)
            n_ts    = st.slider("Series length", 50, 500, 200, 25)

        x_series = simulate_ar1(n_ts, b0_val, b1, sigma)

        # Classify regime
        if abs(b1) < 0.5:
            regime, regime_color = "Rapidly mean-reverting", "green"
        elif abs(b1) < 0.95:
            regime, regime_color = "Slowly mean-reverting (stationary)", "steelblue"
        elif abs(b1) < 1.05:
            regime, regime_color = "Unit root (non-stationary)", "orange"
        else:
            regime, regime_color = "Explosive (non-stationary)", "crimson"

        mean_lr = b0_val / (1 - b1) if abs(b1) < 1 else np.nan

        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "AR(1) Simulated Path",
                "Sample ACF",
                "Return to Mean (if stationary)",
                "Cointegrated Pair Example",
            ],
        )

        # Path — colour shifts when crossing unit root boundary
        path_color = regime_color
        fig2.add_trace(go.Scatter(
            x=list(range(n_ts)), y=x_series,
            mode="lines", line=dict(color=path_color, width=1.2),
            name="x_t",
        ), row=1, col=1)
        if np.isfinite(mean_lr):
            fig2.add_hline(y=mean_lr, line_color="black", line_dash="dash",
                           annotation_text=f"μ={mean_lr:.2f}", row=1, col=1)

        # ACF
        lags2, acf2 = acf(x_series, 20)
        ci2 = 1.96 / np.sqrt(n_ts)
        acf_colors = ["crimson" if abs(v) > ci2 else "steelblue" for v in acf2]
        fig2.add_trace(go.Bar(x=lags2, y=acf2, marker_color=acf_colors, name="ACF"), row=1, col=2)
        fig2.add_hline(y=ci2,  line_color="gray", line_dash="dash", row=1, col=2)
        fig2.add_hline(y=-ci2, line_color="gray", line_dash="dash", row=1, col=2)

        # Mean-reversion speed: impulse response
        horizon = 30
        impulse = np.array([b1**t for t in range(horizon)])
        fig2.add_trace(go.Scatter(
            x=list(range(horizon)), y=impulse,
            mode="lines+markers", line=dict(color="darkorange", width=2),
            name="Impulse response",
        ), row=2, col=1)
        fig2.add_hline(y=0, line_color="black", line_dash="dash", row=2, col=1)

        # Cointegrated pair (spread is stationary by construction)
        rng_coint = np.random.default_rng(99)
        common_trend = np.cumsum(rng_coint.normal(0, 1, n_ts))
        x_coint = common_trend + rng_coint.normal(0, 0.5, n_ts)
        y_coint = 1.5 * common_trend + rng_coint.normal(0, 0.5, n_ts)
        spread  = y_coint - 1.5 * x_coint   # stationary by construction

        fig2.add_trace(go.Scatter(
            x=list(range(n_ts)), y=spread,
            mode="lines", line=dict(color="purple", width=1.2),
            name="Cointegrating spread",
        ), row=2, col=2)
        fig2.add_hline(y=spread.mean(), line_color="black", line_dash="dash",
                       annotation_text="Long-run mean", row=2, col=2)

        fig2.update_layout(height=550, showlegend=False)
        for r, c, xt, yt in [
            (1,1,"Time","x_t"), (1,2,"Lag","ACF"),
            (2,1,"Horizon","Response to unit shock"), (2,2,"Time","Spread (Y − 1.5X)"),
        ]:
            fig2.update_xaxes(title_text=xt, row=r, col=c)
            fig2.update_yaxes(title_text=yt, row=r, col=c)

        with col2:
            st.markdown(f"**Regime:** :{regime_color}[{regime}]")
            if np.isfinite(mean_lr):
                st.metric("Long-run mean", f"{mean_lr:.2f}")
            st.plotly_chart(fig2, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("""
## Machine Learning — Bias-Variance Tradeoff & Regularization

### Bias-Variance decomposition

$$\\text{MSE} = \\text{Bias}^2 + \\text{Variance} + \\text{Irreducible noise}$$

- **High bias** (underfitting): model too simple, misses the true pattern
- **High variance** (overfitting): model memorises the training data, fails on new data
- **Optimal complexity**: minimises *test* error (not training error)

### Regularization

**LASSO** (L1): adds $\\lambda \\sum |\\beta_j|$ penalty → produces *sparse* models (some $\\beta_j = 0$). \\
**Ridge** (L2): adds $\\lambda \\sum \\beta_j^2$ penalty → shrinks coefficients, rarely zeroes them.

As $\\lambda$ increases, coefficients are forced toward zero — reducing variance at the cost of bias.

### Supervised vs unsupervised

| | Supervised | Unsupervised |
|---|---|---|
| **Labels** | Yes (known $y$) | No |
| **Goal** | Predict $y$ | Discover structure |
| **CFA examples** | Credit scoring, return prediction | Clustering investors by risk profile |

> **Exam tip:** The CFA curriculum tests *conceptual* ML — you will not be asked to
> code. Focus on: train/validation/test split, cross-validation, overfitting,
> precision vs recall, and why regularization helps in high-dimensional financial data.
""")

        col1, col2 = st.columns([1, 2])
        with col1:
            max_degree = st.slider("Max polynomial degree", 1, 15, 8)
            n_ml       = st.slider("Training samples", 20, 200, 40, 10)
            noise_sd   = st.slider("Noise level σ", 0.1, 2.0, 0.5, 0.1)
            lam_log    = st.slider("LASSO λ (log scale)", -4.0, 2.0, -1.0, 0.25)
            lam        = 10 ** lam_log

        rng_ml = np.random.default_rng(77)
        x_ml = np.sort(rng_ml.uniform(-1, 1, n_ml))
        y_true_fn = lambda x: np.sin(2 * np.pi * x)
        y_train = y_true_fn(x_ml) + rng_ml.normal(0, noise_sd, n_ml)

        x_test = np.linspace(-1, 1, 200)

        train_errors, test_errors = [], []
        degrees = list(range(1, max_degree + 1))
        for deg in degrees:
            V_train = np.vander(x_ml, deg + 1, increasing=True)
            coef = np.linalg.lstsq(V_train, y_train, rcond=None)[0]
            V_test = np.vander(x_test, deg + 1, increasing=True)
            train_pred = V_train @ coef
            test_pred  = V_test @ coef
            train_errors.append(np.mean((y_train - train_pred)**2))
            test_errors.append(np.mean((y_true_fn(x_test) - test_pred)**2))

        # Best fit for current max_degree
        V_train_full = np.vander(x_ml, max_degree + 1, increasing=True)
        coef_full = np.linalg.lstsq(V_train_full, y_train, rcond=None)[0]
        V_test_full = np.vander(x_test, max_degree + 1, increasing=True)
        y_pred_full = V_test_full @ coef_full

        # LASSO path (manual coordinate descent, simplified)
        def lasso_path(X, y, lambdas, n_iter=200):
            n_feat = X.shape[1]
            coefs  = []
            beta   = np.zeros(n_feat)
            for lam_val in lambdas:
                for _ in range(n_iter):
                    for j in range(n_feat):
                        r_j = y - X @ beta + X[:, j] * beta[j]
                        rho = X[:, j] @ r_j
                        xjj = X[:, j] @ X[:, j]
                        if xjj < 1e-10:
                            beta[j] = 0
                        else:
                            beta[j] = np.sign(rho) * max(abs(rho) - lam_val, 0) / xjj
                coefs.append(beta.copy())
            return np.array(coefs)

        # Simple 5-feature dataset for LASSO
        rng_l = np.random.default_rng(55)
        n_lasso = 60
        X_lasso = rng_l.normal(0, 1, (n_lasso, 5))
        true_beta = np.array([2.0, -1.5, 0.0, 0.8, 0.0])
        y_lasso   = X_lasso @ true_beta + rng_l.normal(0, 0.5, n_lasso)
        lambdas   = np.logspace(-2, 1, 40)
        lasso_coefs = lasso_path(X_lasso, y_lasso, lambdas)

        # ── Plots ─────────────────────────────────────────────────────────
        fig3 = make_subplots(
            rows=1, cols=3,
            subplot_titles=[
                "Fitted Curve (current complexity)",
                "Bias-Variance Tradeoff",
                "LASSO Coefficient Path",
            ],
        )

        # Panel 1: fitted vs truth
        fig3.add_trace(go.Scatter(
            x=x_ml, y=y_train, mode="markers",
            marker=dict(color="steelblue", size=6, opacity=0.7),
            name="Training data",
        ), row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=x_test, y=y_true_fn(x_test),
            mode="lines", line=dict(color="black", width=1.5, dash="dash"),
            name="True function",
        ), row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=x_test, y=y_pred_full,
            mode="lines", line=dict(color="crimson", width=2),
            name=f"Poly degree {max_degree}",
        ), row=1, col=1)

        # Panel 2: train vs test error
        fig3.add_trace(go.Scatter(
            x=degrees, y=train_errors,
            mode="lines+markers", line=dict(color="steelblue", width=2),
            name="Train MSE",
        ), row=1, col=2)
        fig3.add_trace(go.Scatter(
            x=degrees, y=test_errors,
            mode="lines+markers", line=dict(color="crimson", width=2),
            name="Test MSE",
        ), row=1, col=2)
        optimal_deg = int(np.argmin(test_errors)) + 1
        fig3.add_vline(x=optimal_deg, line_dash="dot", line_color="green",
                       annotation_text=f"Optimal={optimal_deg}", row=1, col=2)

        # Panel 3: LASSO path
        colors_lasso = ["steelblue", "crimson", "gray", "darkorange", "purple"]
        for j in range(5):
            fig3.add_trace(go.Scatter(
                x=np.log10(lambdas), y=lasso_coefs[:, j],
                mode="lines", line=dict(color=colors_lasso[j], width=1.5),
                name=f"β{j+1} (true={true_beta[j]})",
            ), row=1, col=3)
        fig3.add_vline(x=np.log10(lam), line_dash="dot", line_color="black",
                       annotation_text=f"Current λ", row=1, col=3)
        fig3.add_hline(y=0, line_color="black", line_width=0.5, row=1, col=3)

        fig3.update_layout(height=430, showlegend=True,
                           legend=dict(orientation="h", yanchor="bottom", y=-0.35))
        for r, c, xt, yt in [
            (1,1,"x","y"), (1,2,"Polynomial degree","MSE"), (1,3,"log₁₀(λ)","Coefficient"),
        ]:
            fig3.update_xaxes(title_text=xt, row=r, col=c)
            fig3.update_yaxes(title_text=yt, row=r, col=c)

        with col2:
            oc1, oc2 = st.columns(2)
            oc1.metric("Optimal degree (test MSE)", optimal_deg)
            oc2.metric("Current degree test MSE", f"{test_errors[max_degree-1]:.3f}")
            st.plotly_chart(fig3, use_container_width=True)
