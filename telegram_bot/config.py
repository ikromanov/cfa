import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Restrict the bot to a single Telegram user (get your ID from @userinfobot).
# If unset, the bot answers anyone who finds it - not recommended.
_allowed_user_id = os.environ.get("TELEGRAM_ALLOWED_USER_ID")
ALLOWED_USER_ID = int(_allowed_user_id) if _allowed_user_id else None

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "bot_history.db"

# CFA Level II topics with approximate 2026 exam weights (STUDY_PLAN.md)
TOPICS = {
    "Ethics & Professional Standards": "10-15%",
    "Financial Statement Analysis": "10-15%",
    "Equity Valuation": "10-15%",
    "Fixed Income": "10-15%",
    "Portfolio Management": "10-15%",
    "Quantitative Methods": "5-10%",
    "Economics": "5-10%",
    "Corporate Issuers": "5-10%",
    "Derivatives": "5-10%",
    "Alternative Investments": "5-10%",
}
