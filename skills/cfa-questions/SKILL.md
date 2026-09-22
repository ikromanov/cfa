---
name: cfa-questions
description: Generate CFA Level II vignette-style exam questions. Use when the user asks for practice questions, exam questions, or wants to drill a CFA topic. Accepts an optional topic argument (e.g. /cfa-questions equity).
argument-hint: "[topic|random] [n=1]"
---

# CFA Level II Question Generator

Generate realistic CFA Level II **vignette-style item sets** matching the actual exam format.

## Arguments

`$ARGUMENTS` may contain:
- A topic name (e.g. `equity`, `fixed income`, `ethics`, `fsa`, `quant`, `economics`, `derivatives`, `alternatives`, `portfolio`, `corporate`)
- A count hint like `n=2` for two vignettes
- `random` or blank → pick a topic at random, weighted by exam weight (higher-weight topics appear more often)

## Exam Weight Reference (use for random selection)

| Weight | Topics |
|--------|--------|
| 10–15% | Ethics, FSA, Equity, Fixed Income, Portfolio Management |
| 5–10%  | Quant, Economics, Corporate Issuers, Derivatives, Alternatives |

## Output Format

For each vignette, produce **exactly** this structure:

---

### Vignette: [Descriptive Title]
**Topic:** [Topic Name] · **Difficulty:** [Easy / Medium / Hard]

[3–5 sentences setting up a realistic scenario. Include a named analyst, company, or country. Embed the numerical data needed to answer the questions — a mini case study. For Ethics, include a specific Standards dilemma. For quantitative topics, include a small data table or regression output inline.]

**Exhibit [n]** *(if needed)*

| Column A | Column B | Column C |
|----------|----------|----------|
| data     | data     | data     |

---

**Question 1 of [N]**

[Question stem — specific and unambiguous. Never use "all of the following EXCEPT" — write positively framed questions.]

- A) [Option A]
- B) [Option B]
- C) [Option C]

<details>
<summary>Answer & Explanation</summary>

**Correct answer: [A / B / C]**

**Why [correct option] is right:** [1–3 sentences with the calculation or reasoning, citing the specific CFA rule or formula.]

**Why [wrong option 1] is wrong:** [1 sentence.]

**Why [wrong option 2] is wrong:** [1 sentence.]

**Exam tip:** [One high-yield trap or common mistake for this sub-topic, matching the style of the High-Yield Areas list below.]

</details>

---

*(Repeat for questions 2–N in the same vignette)*

---

## Content Standards by Topic

### Ethics
- Anchor to a specific Standard (e.g. II(A) Material Non-Public Information, III(B) Fair Dealing)
- Mosaic theory scenarios: mix public + non-public data, ask whether action is permitted
- GIPS: TWR vs MWR, composite construction, error correction
- Research Objectivity: IPO allocation fairness, front-running risk

### FSA
- Intercorporate investments: equity method vs consolidation → show ratio shifts (ROE, D/E)
- Pensions: PBO increases when discount rate falls (inverse); service cost vs interest cost
- FX translation: current rate vs temporal method → which asset types differ?

### Equity
- Residual Income: RI_t = E_t − r × B_{t−1}; price > book iff ROE > r
- DLOC and DLOM are **multiplicative** (apply sequentially), never additive
- DDM: multi-stage with terminal value; H-model approximation
- Private company: FCFF/FCFE with control premium, minority discount

### Fixed Income
- Merton model: default prob rises with leverage and asset volatility; equity = call on assets
- CDS basis: negative basis = bond spread > CDS spread → buy bond + buy protection = arb
- Term structure: bootstrap spot rates from par rates; forward rate from spot rates
- Duration/convexity: price impact = −D·Δy + ½·C·(Δy)²

### Derivatives
- Pay-fixed swap benefits when rates **rise** (floating receipts increase)
- Backwardation: F < S → convenience yield > r + storage cost
- Forward price: F₀ = S₀ × e^((r+u−q)T); dividend yield reduces forward price
- Option strategies: protective put vs covered call payoff profiles

### Quantitative Methods
- Heteroskedasticity / serial correlation → biased SEs, not biased coefficients
- Unit root → spurious regression; need cointegration test before regressing two I(1) series
- AR(1): mean-reverting if |b₁| < 1; explosive if |b₁| > 1
- LASSO: shrinks coefficients to zero (variable selection); ridge does not

### Economics
- PPP: currencies with higher inflation depreciate; use relative PPP for exchange rate forecasts
- IRP: higher domestic rate → forward discount on domestic currency (covered IRP)
- BOP: current account deficit financed by capital/financial account surplus
- Solow: steady-state capital k* = (s/(δ+n+g))^(1/(1−α)); convergence implies poor countries grow faster

### Corporate Issuers
- MM with taxes: V_L = V_U + T_c·D; trade-off adds distress costs → optimal D/V
- Dividends vs buybacks: mechanically identical in MM world; differ in signalling and taxes
- ESG integration: materiality depends on sector; governance premium in valuations

### Derivatives & Alternatives
- LBO: equity return driven by leverage, EBITDA growth, and multiple expansion
- J-curve: negative early returns due to fees + undeployed capital
- Hedge funds: L/S equity has lower market beta; merger arb is exposed to deal-break risk
- REIT: NAV = NOI / cap_rate − net_debt; FFO adds back D&A; AFFO subtracts maintenance capex

### Portfolio Management
- Information Ratio: IR = IC × √BR; doubling breadth is often better than improving IC
- Transfer coefficient (TC) scales IR when constraints prevent full expression of views
- Fama-French: SMB = small-cap premium; HML = value premium; alpha is residual after factor exposure
- Active risk = tracking error; active return = portfolio return − benchmark return

## Quality Rules

1. Every numerical value in the vignette must be **used** by at least one question.
2. Distractors must be **plausible** — based on common calculation errors or formula misapplication, not random.
3. Difficulty levels: Easy = direct formula application; Medium = multi-step or ratio interpretation; Hard = requires integrating two concepts or spotting a trap.
4. Never reveal the answer in the question stem. Never use double-negatives.
5. Match the 2026 CFA Level II curriculum — do not include Level I or Level III material.
6. When `$ARGUMENTS` specifies a topic, generate only that topic. When blank or `random`, pick one topic, weighted by exam weight.
