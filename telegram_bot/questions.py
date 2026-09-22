from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from telegram_bot.config import ANTHROPIC_API_KEY

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You are an expert CFA Level II exam item-set writer. You write \
original vignette-style questions in the exact style of the official CFA Institute \
Level II exam: a short case (vignette) with realistic figures/names, followed by one \
multiple-choice question with exactly three choices (A, B, C). Exactly one choice is \
correct. The question must require applying a Level II concept to the vignette \
(calculation or analysis), not simple recall. Write a concise explanation covering why \
the correct choice is right and why each distractor is wrong."""


class VignetteQuestion(BaseModel):
    vignette: str = Field(description="2-5 sentence case with concrete figures/names")
    question: str = Field(description="The item stem, a single question")
    choice_a: str
    choice_b: str
    choice_c: str
    correct_choice: Literal["A", "B", "C"]
    explanation: str = Field(description="Why the correct choice is right and the others wrong")


def generate_question(topic: str) -> VignetteQuestion:
    response = _client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Write one CFA Level II vignette item set for the topic: {topic}.",
        }],
        output_format=VignetteQuestion,
    )
    return response.parsed_output
