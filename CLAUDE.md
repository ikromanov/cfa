# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Personal Python repository for studying CFA Level II (November 2026 exam) and experimenting with AI tools.

## Commands

```bash
# Run the Streamlit study app
uv run streamlit run app.py

# Run the Telegram practice-question bot (needs .env with TELEGRAM_BOT_TOKEN
# and ANTHROPIC_API_KEY, see .env.example)
uv run python -m telegram_bot.bot

# Install dependencies
uv sync

# Lint / format
uv run ruff check .
uv run ruff format .
```

Python version is pinned to 3.9 (`.python-version`). Use `uv` as the package manager (`pyproject.toml` + `uv.lock`).

## Architecture

### Streamlit app (`app.py` + `topics/`)

`app.py` is the single entry point. It imports one module per CFA topic from `topics/` and routes sidebar navigation to the selected module's `render()` function. Every topic module follows the same pattern:

- Pure Python helper functions implementing CFA formulas (e.g. `gordon_growth`, `two_stage_ddm`)
- A `render()` function that builds the Streamlit page: theory text, `st.slider` controls, and Plotly charts

The `risk/` directory contains standalone scripts (e.g. `value_at_risk.py`) not wired into the app.

### Telegram bot (`telegram_bot/`)

A Telegram bot for on-demand exam practice, separate from the Streamlit app:

- `config.py` — env vars (`TELEGRAM_BOT_TOKEN`, `ANTHROPIC_API_KEY`) and the topic/weight list
- `questions.py` — calls the Claude API (`claude-opus-5`, structured output via `messages.parse`)
  to generate a `VignetteQuestion` (case + 3-choice MCQ + explanation) for a given topic
- `storage.py` — SQLite (`data/bot_history.db`, gitignored) recording every attempt for later review
- `bot.py` — `python-telegram-bot` handlers: `/ask [topic]`, `/topics`, `/stats`, `/history`

Grading is exact-match on the MCQ letter (no second LLM call); the explanation generated
alongside the question is reused as feedback.

### Claude Code skills (`skills/`)

Two custom skills are registered via `.claude-plugin/plugin.json`:

- `/cfa-explain <concept>` — explains any CFA Level II concept in a structured 5-section format (plain English, formula, intuition, vignette context, memory hook)
- `/cfa-questions [topic] [n=N]` — generates vignette-style item sets in the actual exam format, weighted by topic exam weight

Skill definitions live in `skills/cfa-explain/SKILL.md` and `skills/cfa-questions/SKILL.md`.

### Study plan

`STUDY_PLAN.md` has the full 31-week schedule (Apr–Nov 2026) with mock exam dates and score targets. Each week maps to a specific `topics/*.py` module to use alongside reading.
