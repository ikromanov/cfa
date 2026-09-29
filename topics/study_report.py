"""Study report — hours studied and practice-question results from study_log.csv."""

from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

LOG_PATH = Path(__file__).resolve().parent.parent / "study_log.csv"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_log(path=LOG_PATH):
    """Read the study log and split 'FSA - Module 4: ...' into topic and module."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    parts = df["topic"].str.split(" - ", n=1, expand=True)
    df["module"] = df["topic"]
    df["topic"] = parts[0].str.strip()
    return df


def summarize(df, by):
    """Total hours, questions and correct answers per group, plus accuracy."""
    out = (
        df.groupby(by, sort=True)
        .agg(sessions=("hours", "size"), hours=("hours", "sum"),
             questions=("questions", "sum"), correct=("correct", "sum"))
        .reset_index()
    )
    out["accuracy"] = out["correct"] / out["questions"].where(out["questions"] > 0)
    return out


def _table(df):
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "hours": st.column_config.NumberColumn("Hours", format="%.2f"),
            "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
        },
    )


# ── page ──────────────────────────────────────────────────────────────────────

def render():
    st.title("Study Report")

    if not LOG_PATH.exists():
        st.warning(f"No study log found at `{LOG_PATH.name}`.")
        return

    df = load_log()
    if df.empty:
        st.info("The study log is empty.")
        return

    # Slider spans whole days so every entry is reachable in 30-minute steps.
    lo = datetime.combine(df["timestamp"].min().date(), time.min)
    hi = datetime.combine(df["timestamp"].max().date(), time.min) + timedelta(days=1)
    start, end = st.slider(
        "Period",
        min_value=lo, max_value=hi, value=(lo, hi),
        step=timedelta(minutes=30), format="YYYY-MM-DD HH:mm",
    )

    period = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    if period.empty:
        st.info("No study sessions in the selected period.")
        return

    total_q = period["questions"].sum()
    total_c = period["correct"].sum()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hours studied", f"{period['hours'].sum():.1f}")
    c2.metric("Sessions", len(period))
    c3.metric("Correct answers", f"{total_c} / {total_q}")
    c4.metric("Accuracy", f"{total_c / total_q:.0%}" if total_q else "—")

    st.subheader("By topic")
    _table(summarize(period, "topic"))

    st.subheader("By module")
    _table(summarize(period, ["topic", "module"]).drop(columns="topic"))

    with st.expander("Sessions in period"):
        st.dataframe(
            period.sort_values("timestamp", ascending=False),
            hide_index=True, width="stretch",
        )
