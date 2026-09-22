import sqlite3
from contextlib import closing
from datetime import datetime, timezone

from telegram_bot.config import DB_PATH
from telegram_bot.questions import VignetteQuestion

SCHEMA = """
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    username TEXT,
    topic TEXT NOT NULL,
    vignette TEXT NOT NULL,
    question TEXT NOT NULL,
    choice_a TEXT NOT NULL,
    choice_b TEXT NOT NULL,
    choice_c TEXT NOT NULL,
    correct_choice TEXT NOT NULL,
    user_choice TEXT NOT NULL,
    is_correct INTEGER NOT NULL,
    explanation TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(SCHEMA)
        conn.commit()


def save_attempt(
    user_id: int,
    username: str,
    topic: str,
    q: VignetteQuestion,
    user_choice: str,
) -> bool:
    is_correct = user_choice == q.correct_choice
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.execute(
            """INSERT INTO attempts
            (user_id, username, topic, vignette, question, choice_a, choice_b, choice_c,
             correct_choice, user_choice, is_correct, explanation, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id, username, topic, q.vignette, q.question,
                q.choice_a, q.choice_b, q.choice_c,
                q.correct_choice, user_choice, int(is_correct), q.explanation,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    return is_correct


def get_stats(user_id: int) -> list[tuple[str, int, int]]:
    """Returns (topic, correct_count, total_count) per topic, ordered by topic."""
    with closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            """SELECT topic, SUM(is_correct), COUNT(*)
            FROM attempts WHERE user_id = ?
            GROUP BY topic ORDER BY topic""",
            (user_id,),
        ).fetchall()
    return [(topic, correct or 0, total) for topic, correct, total in rows]


def get_recent(user_id: int, limit: int = 10) -> list[sqlite3.Row]:
    with closing(sqlite3.connect(DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM attempts WHERE user_id = ?
            ORDER BY created_at DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return rows
