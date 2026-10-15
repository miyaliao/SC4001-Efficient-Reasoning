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


def normalize_number(value):
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError, TypeError):
        return None


def gold_answer(text):
    matches = _GOLD.findall(text)
    return normalize_number(matches[-1]) if matches else None


def predicted_answer(text):
    if not isinstance(text, str):
        return None
    for pattern in (_GOLD, _BOXED, _FINAL):
        matches = pattern.findall(text)
        if matches:
            return normalize_number(matches[-1])
    return None
