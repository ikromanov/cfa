# CFA Level II — Study Plan (Revised): September 2026 → November 2026

**Exam date:** Wednesday, November 18, 2026  
**Revised:** September 27, 2026  
**Remaining:** 7 weeks + 2 days  
**Target hours:** ~140 (≈20h/wk)

---

## Where Things Stand

The original plan (March 31) assumed ~280h and five mocks done by now. Actual
progress: one scan of all materials plus some practice questions — roughly the
end of the old Phase 1, with no mocks taken. This revision compresses the rest
into two phases:

1. **Questions-first learning.** You've already skimmed everything, so skip a
   second reading pass. For each topic spend ~20% on summary/formulas, ~60% on
   vignettes, ~20% reviewing mistakes (in the app, using the sliders).
2. **Allocate by exam weight, not full coverage.** The five high-weight topics
   are ~60% of the exam and get most of the hours.

At ~140h this is below the 300h CFA Institute recommends — the target is the
minimum passing score, not mastery. Cut accordingly.

---

## Topic Weights (CFA Institute 2026)

| Tier | Topic | Weight |
|---|---|---|
| High | Ethics & Professional Standards | 10–15% |
| High | Financial Statement Analysis | 10–15% |
| High | Equity Valuation | 10–15% |
| High | Fixed Income | 10–15% |
| High | Portfolio Management | 10–15% |
| Mid | Quantitative Methods | 5–10% |
| Mid | Economics | 5–10% |
| Mid | Corporate Issuers | 5–10% |
| Mid | Derivatives | 5–10% |
| Mid | Alternative Investments | 5–10% |

---

---

## Phase A — Fast Learning Pass by Weight (Sep 28 – Oct 25 · 4 weeks · ~80h)

**Day 1 (Mon Sep 28): half-mock baseline.** One 44-question session, timed
(2h12m). Its only job is to set the topic order within each week below.

| Week | Dates | Topics (hours) | App Modules |
|---|---|---|---|
| 1 | Sep 28 – Oct 4 | Ethics — Standards I–VII + cases (8h) · FSA — intercorporate investments, pensions, multinational FX (12h) | `ethics.py`, `fsa.py` |
| 2 | Oct 5–11 | Equity — FCFF/FCFE, residual income, multiples, private company (12h) · Fixed Income — term structure, arbitrage-free valuation (8h) | `equity.py`, `fixed_income.py` |
| 3 | Oct 12–18 | Fixed Income — embedded options, credit, CDS (6h) · Portfolio Mgmt (8h) · Derivatives — forwards/futures, swaps, options (6h) | `fixed_income.py`, `portfolio.py`, `derivatives.py` |
| 4 | Oct 19–24 | Quant — regression, time series (6h) · Economics — FX, IRP/PPP, growth (4h) · Corporate Issuers (4h) · Alternatives — real estate/REITs, PE, hedge funds (5h) | `quant.py`, `economics.py`, `corporate.py`, `alternatives.py` |
| — | **Sun Oct 25** | **Mock A** — full, timed (AM + PM) | |

**Daily habit (15 min, every day including weekends):** a few `/ask` questions
from the Telegram bot, or `/cfa-questions n=2`, on topics *already covered*.
This adds the spaced repetition the compressed pass would otherwise lack.

**If you fall behind:** don't push the whole schedule back. Cut the
lowest-weight material first (parts of Economics, the thinner Corporate
Issuers readings, ML in Quant). Never cut Ethics.

---

## Phase B — Mocks + Weak-Area Repair (Oct 26 – Nov 15 · 3 weeks · ~55h)

| Week | Dates | Focus | Hours |
|---|---|---|---|
| 5 | Oct 26 – Nov 1 | Review Mock A in depth (4–5h; write down the rule behind each miss, not just the answer) · drill the 2 weakest topics · Ethics second pass · **Mock B Sat Oct 31** | 20 |
| 6 | Nov 2–8 | Review Mock B · weakest topic + any high-weight topic below 60% · build a one-page formula sheet per topic · **Mock C Sat Nov 7** | 20 |
| 7 | Nov 9–15 | Review Mock C · final Ethics pass (re-read Standards, then 40–60 Qs) · **Mock D Wed Nov 11 or Sat Nov 14** (official CFA Institute mock if saved) · review mistake log | 15 |
| — | Nov 16–17 | Light only: formula sheets, mistake log, logistics. **No new material.** | 2–3 |
| — | **Wed Nov 18** | **Exam** | |

If Mock C comes in under 60%, drop Mock D and spend those hours on your single
weakest *high-weight* topic plus Ethics.

---

## Mock Exam Schedule Summary

| Mock | Date | Target | Action |
|---|---|---|---|
| Half baseline | Sep 28 | Ignore score | Set topic order within each week |
| Mock A | Oct 25 | ≥ 55% | Pick 2 weak topics for Week 5 |
| Mock B | Oct 31 | ≥ 62% | Check trend; adjust Week 6 |
| Mock C | Nov 7 | ≥ 65% | Lock in the final weak topic |
| Mock D | Nov 11–14 | ≥ 68–70% | Final calibration; exam-day pacing and reading order |

---

## Hours Summary

| Phase | Weeks | h/wk | Total |
|---|---|---|---|
| A — Fast learning pass | 4 | 20 | 80h |
| B — Mocks + repair | 3 | 15–20 | 55h |
| Final 2 days | — | — | ~3h |
| **Total** | **7+** | — | **~138h** |

---

## High-Yield Areas (Level II-Specific Traps)

These appear frequently in vignettes and are often mishandled:

- **Ethics:** Mosaic theory vs MNPI — know exactly when it crosses the line
- **FSA:** Same economics, different ratios under equity method vs consolidation
- **FSA:** PBO sensitivity to discount rate (inverse, convex relationship)
- **Equity:** Residual Income — when does price > book? When ROE > r
- **Equity:** DLOC and DLOM are multiplicative, not additive
- **Fixed Income:** Merton model — know direction of default prob vs leverage/vol
- **Fixed Income:** CDS basis arbitrage — negative basis = buy bond + buy protection
- **Derivatives:** Pay-fixed swap benefits when rates **rise** (floating receipts increase)
- **Derivatives:** Forward price in backwardation: convenience yield > r + storage
- **Portfolio:** IR = TC × IC × √BR — doubling breadth is often better than improving IC
- **Quant:** Heteroskedasticity/serial correlation bias SEs, not coefficients
- **Quant:** Unit root → never regress two random walks without testing cointegration

---

## App Usage Tips

Run with: `uv run streamlit run app.py`

- Use **sliders aggressively** — building intuition beats re-reading formulas
- After each vignette you get wrong, open the relevant app page and move the
  parameters to match the scenario — see the chart update in real time
- The **Exam tip** callouts in each section are curated from historically tested traps

---

*Generated: 2026-03-31 · Revised: 2026-09-27*
