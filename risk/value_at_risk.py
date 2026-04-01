"""
Value-at-Risk (VaR) Demonstration
==================================
VaR answers: "What is the maximum loss I can expect over a given time horizon
at a given confidence level?"

Three standard methods:
  1. Historical Simulation  — uses actual past returns, no distributional assumption
  2. Parametric (Variance-Covariance) — assumes normally distributed returns
  3. Monte Carlo Simulation — simulates many future paths

Also demonstrates Conditional VaR (CVaR / Expected Shortfall), which answers:
"Given that losses exceed VaR, what is the expected loss?"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ── Parameters ────────────────────────────────────────────────────────────────
CONFIDENCE_LEVEL = 0.95   # 95% confidence → 5% tail
PORTFOLIO_VALUE  = 1_000_000  # $1M portfolio
HOLDING_PERIOD   = 1          # days
N_SIMULATIONS    = 100_000
SEED             = 42

rng = np.random.default_rng(SEED)


# ── Generate synthetic daily return history (500 trading days ≈ 2 years) ──────
# Simulate a realistic return series: mostly normal, with occasional fat tails
n_days = 500
daily_returns = rng.normal(loc=0.0005, scale=0.012, size=n_days)
# Inject a few extreme down days to mimic real markets
crash_days = rng.choice(n_days, size=5, replace=False)
daily_returns[crash_days] -= rng.uniform(0.04, 0.08, size=5)


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — Historical Simulation
# ══════════════════════════════════════════════════════════════════════════════
def var_historical(returns: np.ndarray, confidence: float, portfolio_value: float) -> tuple[float, float]:
    """Sort past returns; the VaR is the (1-confidence) percentile."""
    var_pct  = np.percentile(returns, (1 - confidence) * 100)
    cvar_pct = returns[returns <= var_pct].mean()  # average of tail losses

    var_dollar  = -var_pct  * portfolio_value
    cvar_dollar = -cvar_pct * portfolio_value
    return var_dollar, cvar_dollar


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — Parametric (Variance-Covariance)
# ══════════════════════════════════════════════════════════════════════════════
def var_parametric(returns: np.ndarray, confidence: float, portfolio_value: float) -> tuple[float, float]:
    """
    Assumes returns ~ N(μ, σ²).
    VaR  = -(μ + z_α · σ)  where z_α = stats.norm.ppf(1 - confidence)
    CVaR = -(μ - σ · φ(z_α) / (1 - confidence))  where φ is the normal PDF
    """
    mu, sigma = returns.mean(), returns.std(ddof=1)
    z = stats.norm.ppf(1 - confidence)         # e.g. -1.645 at 95%

    var_pct  = mu + z * sigma
    # CVaR formula for normal distribution
    cvar_pct = mu - sigma * stats.norm.pdf(-z) / (1 - confidence)

    var_dollar  = -var_pct  * portfolio_value
    cvar_dollar = -cvar_pct * portfolio_value
    return var_dollar, cvar_dollar


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 3 — Monte Carlo Simulation
# ══════════════════════════════════════════════════════════════════════════════
def var_monte_carlo(
    returns: np.ndarray,
    confidence: float,
    portfolio_value: float,
    n_sims: int,
) -> tuple[float, float]:
    """
    Fit μ and σ from historical data, then simulate N future returns.
    VaR and CVaR are computed from the simulated distribution.
    """
    mu, sigma    = returns.mean(), returns.std(ddof=1)
    sim_returns  = rng.normal(mu, sigma, size=n_sims)

    var_pct  = np.percentile(sim_returns, (1 - confidence) * 100)
    cvar_pct = sim_returns[sim_returns <= var_pct].mean()

    var_dollar  = -var_pct  * portfolio_value
    cvar_dollar = -cvar_pct * portfolio_value
    return var_dollar, cvar_dollar


# ── Run all three methods ─────────────────────────────────────────────────────
hist_var,  hist_cvar  = var_historical(daily_returns, CONFIDENCE_LEVEL, PORTFOLIO_VALUE)
param_var, param_cvar = var_parametric(daily_returns, CONFIDENCE_LEVEL, PORTFOLIO_VALUE)
mc_var,    mc_cvar    = var_monte_carlo(daily_returns, CONFIDENCE_LEVEL, PORTFOLIO_VALUE, N_SIMULATIONS)


# ── Print results ─────────────────────────────────────────────────────────────
print(f"Portfolio Value : ${PORTFOLIO_VALUE:>12,.0f}")
print(f"Confidence Level: {CONFIDENCE_LEVEL:.0%}")
print(f"Holding Period  : {HOLDING_PERIOD} day(s)")
print()
print(f"{'Method':<28} {'VaR':>12} {'CVaR (ES)':>12}")
print("-" * 54)
print(f"{'Historical Simulation':<28} ${hist_var:>11,.0f} ${hist_cvar:>11,.0f}")
print(f"{'Parametric (Normal)':<28} ${param_var:>11,.0f} ${param_cvar:>11,.0f}")
print(f"{'Monte Carlo':<28} ${mc_var:>11,.0f} ${mc_cvar:>11,.0f}")
print()
print("Interpretation (Historical):")
print(f"  With {CONFIDENCE_LEVEL:.0%} confidence, the 1-day loss will NOT exceed ${hist_var:,.0f}.")
print(f"  But IF losses exceed VaR, they average ${hist_cvar:,.0f} (CVaR/Expected Shortfall).")


# ── Visualise ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Value-at-Risk (VaR) Demonstration", fontsize=14, fontweight="bold")

# — Left: return distribution with VaR cutoffs
ax = axes[0]
dollar_returns = daily_returns * PORTFOLIO_VALUE
ax.hist(dollar_returns, bins=40, color="steelblue", edgecolor="white", alpha=0.7, label="Daily P&L")

colors = {"Historical": ("tomato",    hist_var),
          "Parametric": ("darkorange", param_var),
          "Monte Carlo":("gold",       mc_var)}

for label, (color, var) in colors.items():
    ax.axvline(-var, color=color, linewidth=2, linestyle="--", label=f"{label} VaR: ${var:,.0f}")

ax.axvline(-hist_cvar, color="tomato", linewidth=1.5, linestyle=":", label=f"CVaR: ${hist_cvar:,.0f}")
ax.set_xlabel("Daily P&L ($)")
ax.set_ylabel("Frequency")
ax.set_title(f"Return Distribution with {CONFIDENCE_LEVEL:.0%} VaR")
ax.legend(fontsize=8)

# — Right: cumulative return path + rolling 1-day VaR band
ax2 = axes[1]
cumulative = (1 + daily_returns).cumprod() * PORTFOLIO_VALUE
ax2.plot(cumulative, color="steelblue", linewidth=1.5, label="Portfolio Value")

# Rolling 20-day parametric VaR band
window = 20
rolling_var = np.full(n_days, np.nan)
for i in range(window, n_days):
    window_returns = daily_returns[i - window:i]
    rv, _ = var_parametric(window_returns, CONFIDENCE_LEVEL, cumulative[i])
    rolling_var[i] = rv

ax2.fill_between(
    range(n_days),
    cumulative - rolling_var,
    cumulative,
    alpha=0.25, color="tomato", label=f"1-day VaR band ({window}-day rolling)"
)
ax2.set_xlabel("Trading Day")
ax2.set_ylabel("Portfolio Value ($)")
ax2.set_title("Portfolio Path with Rolling VaR Band")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig("risk/var_demonstration.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nChart saved to risk/var_demonstration.png")
