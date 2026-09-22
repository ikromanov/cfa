---
name: cfa-explain
description: Explain a CFA Level II concept, formula, or term in plain English with exam context. Use when the user asks to explain or clarify a CFA topic, formula, model, or term (e.g. "explain the Merton model", "what is DLOC", "how does IRP work").
argument-hint: "<concept or formula>"
---

# CFA Level II Concept Explainer

Explain the concept given in `$ARGUMENTS` clearly and concisely for a CFA Level II candidate.

## Output structure

Always produce exactly these five sections:

### 1. Plain English
One short paragraph. No jargon. Explain what the concept *is* and what problem it solves, as if speaking to a smart person who has never seen it before.

### 2. The Formula (if applicable)
Display the canonical formula using inline notation. Define every variable on its own line. If there is no formula, skip this section silently.

Example format:
```
F₀ = S₀ × e^((r + u − q)T)

F₀  = forward price
S₀  = current spot price
r   = risk-free rate
u   = storage cost (as % of spot)
q   = convenience yield or dividend yield
T   = time to expiration (years)
```

### 3. Intuition: What moves it and why
2–4 bullet points. Each bullet names one input or driver, states the direction of its effect, and gives a one-sentence reason. Focus on relationships that are non-obvious or counterintuitive — things a candidate would get wrong.

### 4. Level II Vignette Context
Describe the typical exam scenario in which this concept appears:
- What does the vignette usually set up? (company, analyst, data table)
- What are candidates asked to calculate or identify?
- What trap or distractor does the exam commonly use?

Keep this to 3–5 sentences.

### 5. One-Line Memory Hook
A single sentence — a mnemonic, analogy, or rule of thumb — that makes the concept stick. Should be memorable enough to recall under exam pressure.

---

## Tone and length

- Concise. Each section should be as short as it can be while being complete.
- Use plain language in sections 1 and 5. Use precise terminology in sections 2–4.
- Never pad. If a section has nothing useful to add, omit it.
- Do not add a preamble before the first section heading.

## Topic coverage

Cover any concept from the CFA Level II curriculum, including but not limited to:

**Ethics:** Mosaic theory, MNPI threshold, GIPS composite construction, TWR vs MWR, soft dollar standards, IPO allocation fairness

**Quantitative Methods:** Heteroskedasticity, serial correlation, multicollinearity, unit root, cointegration, Dickey-Fuller test, ARMA, ARCH, LASSO, ridge regression, bias-variance tradeoff

**Economics:** Covered/uncovered IRP, relative PPP, J-curve effect, Mundell-Fleming, BOP components, Solow steady state, convergence hypothesis

**FSA:** Equity method vs consolidation (ratio impacts), proportionate consolidation, PBO vs ABO, pension corridor, current rate vs temporal method, hyperinflationary subsidiary

**Corporate Issuers:** MM propositions I and II, trade-off theory, pecking order theory, signalling theory, dividend irrelevance, share repurchase mechanics, ESG integration

**Equity:** Gordon Growth Model, H-model, FCFE/FCFF discount, residual income model, EVA, clean surplus relation, price-to-book decomposition, DLOC, DLOM, control premium, EV/EBITDA

**Fixed Income:** Duration, convexity, DV01, key rate duration, bootstrapping, forward rates, par/spot/forward relationships, CIR/Vasicek term structure, OAS, Z-spread, Merton model (d1/d2/default probability), CDS spread, negative basis trade

**Derivatives:** Put-call parity, Black-Scholes assumptions, delta/gamma/vega/theta/rho, cost-of-carry model, contango vs backwardation, convenience yield, pay-fixed swap, FRA valuation, collar strategy

**Alternatives:** LBO mechanics, IRR vs MOIC, J-curve, carried interest, management fee drag, merger arbitrage, relative value, long/short equity beta, NAV vs GAV, REIT FFO/AFFO/NAV

**Portfolio Management:** Efficient frontier, CML vs SML, information ratio, information coefficient, breadth, transfer coefficient, fundamental law of active management, tracking error, Fama-French factors (MKT, SMB, HML), Carhart momentum, alpha vs factor exposure
