"""Conservative final-answer extraction for GSM8K-style numeric tasks.

Avoid parsing intermediate reasoning numbers as final answers.
"""
import re
from decimal import Decimal, InvalidOperation

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_GOLD = re.compile(r"####\s*(" + _NUMBER + r")(?=\s|$|[.,;!])")
_BOXED = re.compile(r"\\boxed\s*\{\s*(" + _NUMBER + r")\s*\}")
_FINAL = re.compile(
    r"(?:final answer(?: is|:)?|answer\s*:|the answer is)\s*\$?\s*("
    + _NUMBER + r")(?=\s|$|[.,;!])",
    re.IGNORECASE,
)

_QWEN_SPECIAL_TOKEN = re.compile(r"<\|[^>]*\|>")


def normalize_number(value):
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def gold_answer(text):
    matches = _GOLD.findall(text)
    return normalize_number(matches[-1]) if matches else None


def predicted_answer(text):
    """Read a numerical final answer, never an unfinished Qwen thinking trace."""
    if not isinstance(text, str):
        return None

    # Ignore reasoning that may contain conflicting intermediate guesses.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        # A truncated thinking trace has no usable final answer.
        return None

    # Qwen end-of-message tokens and Markdown bold markers are not answers.
    # Examples: "Final answer: 108<|im_end|>", "**Final answer:** 18".
    text = _QWEN_SPECIAL_TOKEN.sub("", text)
    text = text.replace("**", "")

    # Prefer the explicit final answer to other numeric expressions.
    for pattern in (_FINAL, _BOXED, _GOLD):
        matches = pattern.findall(text)
        if matches:
            return normalize_number(matches[-1])
    return None
