"""
CFA Level II exam-question Telegram bot.
Run with:  uv run python -m telegram_bot.bot
"""

import asyncio
import functools
import logging
from typing import Optional

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

from telegram_bot.config import ALLOWED_USER_ID, TELEGRAM_BOT_TOKEN, TOPICS
from telegram_bot.questions import VignetteQuestion, generate_question
from telegram_bot.storage import get_recent, get_stats, init_db, save_attempt

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

TOPIC_LIST = list(TOPICS.keys())


def restricted(handler):
    """Rejects everyone except ALLOWED_USER_ID (if set)."""

    @functools.wraps(handler)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if ALLOWED_USER_ID is not None and user.id != ALLOWED_USER_ID:
            logger.warning("Rejected unauthorized user %s (%s)", user.id, user.username)
            if update.callback_query:
                await update.callback_query.answer("Not authorized.", show_alert=True)
            elif update.message:
                await update.message.reply_text("Not authorized.")
            return
        return await handler(update, context)

    return wrapper


def _topic_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        [InlineKeyboardButton(f"{name} ({weight})", callback_data=f"topic|{i}")]
        for i, (name, weight) in enumerate(TOPICS.items())
    ]
    return InlineKeyboardMarkup(buttons)


def _answer_keyboard() -> InlineKeyboardMarkup:
    buttons = [[
        InlineKeyboardButton("A", callback_data="answer|A"),
        InlineKeyboardButton("B", callback_data="answer|B"),
        InlineKeyboardButton("C", callback_data="answer|C"),
    ]]
    return InlineKeyboardMarkup(buttons)


def _format_question(topic: str, q: VignetteQuestion) -> str:
    return (
        f"*{topic}*\n\n"
        f"{q.vignette}\n\n"
        f"*{q.question}*\n\n"
        f"A) {q.choice_a}\n"
        f"B) {q.choice_b}\n"
        f"C) {q.choice_c}"
    )


@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "CFA Level II practice bot.\n\n"
        "/ask - get a vignette question (pick a topic)\n"
        "/ask <topic> - jump straight to a topic\n"
        "/topics - list topics and exam weights\n"
        "/stats - your accuracy by topic\n"
        "/history - your last 10 attempts"
    )


@restricted
async def topics_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    lines = [f"{name} — {weight}" for name, weight in TOPICS.items()]
    await update.message.reply_text("\n".join(lines))


def _match_topic(text: str) -> Optional[str]:
    text = text.strip().lower()
    for name in TOPIC_LIST:
        if text == name.lower() or text in name.lower():
            return name
    return None


@restricted
async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if context.args:
        requested = " ".join(context.args)
        topic = _match_topic(requested)
        if topic is None:
            await update.message.reply_text(
                f"No topic matches '{requested}'. Use /topics to see valid names."
            )
            return
        await _send_question(update.message.chat_id, topic, context)
    else:
        await update.message.reply_text(
            "Pick a topic:", reply_markup=_topic_keyboard()
        )


async def _send_question(chat_id: int, topic: str, context: ContextTypes.DEFAULT_TYPE) -> None:
    await context.bot.send_message(chat_id, f"Generating a {topic} question…")
    q = await asyncio.to_thread(generate_question, topic)
    context.chat_data["pending"] = {"topic": topic, "question": q}
    await context.bot.send_message(
        chat_id,
        _format_question(topic, q),
        parse_mode="Markdown",
        reply_markup=_answer_keyboard(),
    )


@restricted
async def topic_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    index = int(query.data.split("|", 1)[1])
    topic = TOPIC_LIST[index]
    await query.edit_message_text(f"Generating a {topic} question…")
    q = await asyncio.to_thread(generate_question, topic)
    context.chat_data["pending"] = {"topic": topic, "question": q}
    await query.edit_message_text(
        _format_question(topic, q), parse_mode="Markdown", reply_markup=_answer_keyboard()
    )


@restricted
async def answer_selected(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    pending = context.chat_data.get("pending")
    if pending is None:
        await query.edit_message_text("No active question. Use /ask to get one.")
        return

    user_choice = query.data.split("|", 1)[1]
    topic, q = pending["topic"], pending["question"]
    user = update.effective_user
    is_correct = save_attempt(
        user_id=user.id,
        username=user.username or user.full_name,
        topic=topic,
        q=q,
        user_choice=user_choice,
    )
    context.chat_data["pending"] = None

    verdict = "✅ Correct!" if is_correct else f"❌ Incorrect — correct answer: {q.correct_choice}"
    await query.edit_message_text(
        f"{_format_question(topic, q)}\n\n"
        f"Your answer: {user_choice}\n{verdict}\n\n_{q.explanation}_",
        parse_mode="Markdown",
    )


@restricted
async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = get_stats(update.effective_user.id)
    if not rows:
        await update.message.reply_text("No attempts recorded yet. Try /ask.")
        return
    lines = []
    total_correct = total_count = 0
    for topic, correct, count in rows:
        total_correct += correct
        total_count += count
        lines.append(f"{topic}: {correct}/{count} ({correct / count:.0%})")
    lines.append(f"\nOverall: {total_correct}/{total_count} ({total_correct / total_count:.0%})")
    await update.message.reply_text("\n".join(lines))


@restricted
async def history_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    rows = get_recent(update.effective_user.id, limit=10)
    if not rows:
        await update.message.reply_text("No attempts recorded yet. Try /ask.")
        return
    lines = []
    for row in rows:
        mark = "✅" if row["is_correct"] else "❌"
        lines.append(f"{mark} [{row['created_at'][:16]}] {row['topic']}: {row['question'][:60]}")
    await update.message.reply_text("\n".join(lines))


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (see .env.example)")

    init_db()

    if ALLOWED_USER_ID is None:
        logger.warning(
            "TELEGRAM_ALLOWED_USER_ID is not set - anyone who finds this bot can use it "
            "and spend your Anthropic API credits."
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("topics", topics_cmd))
    app.add_handler(CommandHandler("ask", ask_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("history", history_cmd))
    app.add_handler(CallbackQueryHandler(topic_selected, pattern=r"^topic\|"))
    app.add_handler(CallbackQueryHandler(answer_selected, pattern=r"^answer\|"))

    logger.info("Bot starting…")
    app.run_polling()


if __name__ == "__main__":
    main()
