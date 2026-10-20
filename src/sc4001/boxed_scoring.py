"""Versioned canonical Boxed final-answer extraction (no Math-Verify imports).

The experiment prompt is unchanged. This parser accepts superficial Markdown/
LaTeX display wrappers, but requires ONE explicit final answer with ONE
balanced boxed expression and no substantive trailing text.
"""
from __future__ import annotations
import re

# Both "### Final Answer" and "Final answer:" are semantic final-answer labels.
# Require a whole label at line start; the following content is independently
# validated to be exactly one balanced boxed expression.
_FINAL = re.compile(r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?final[ \t]+answer\b[ \t]*:?")
_TOKEN = re.compile(r"<\|[^>]+\|>")
_BOX_START = re.compile(r"^\\boxed\s*\{")
_WRAPPERS = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))


def final_body(response: str) -> str:
    if not isinstance(response, str):
        return ""
    if "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    elif "<think>" in response:
        return ""
    return _TOKEN.sub("", response).strip()


def canonical_box(response: str) -> tuple[str | None, str, str | None]:
    """Return (boxed_latex, status, wrapper). No rescue of unboxed prose."""
    body = final_body(response)
    if not body:
        return None, "no_final_response", None
    # Presentation-only Markdown emphasis is permitted around the answer label.
    normalized = body.replace("**", "").replace("__", "")
    labels = list(_FINAL.finditer(normalized))
    if not labels:
        return None, "no_final_label", None
    if len(labels) != 1:
        return None, "multiple_final_labels", None
    tail = normalized[labels[0].end():].strip()
    wrapper = "none"
    for opening, closing in _WRAPPERS:
        if tail.startswith(opening):
            if not tail.endswith(closing) or len(tail) <= len(opening) + len(closing):
                return None, "unclosed_math_wrapper", None
            tail = tail[len(opening):-len(closing)].strip()
            wrapper = {"$$":"display_dollar","$":"inline_dollar",
                       r"\(":"inline_paren",r"\[":"display_bracket"}[opening]
            break

    start = _BOX_START.match(tail)
    if not start:
        return None, "missing_box", wrapper
    depth, end = 1, None
    for i in range(start.end(), len(tail)):
        if tail[i] == "{" and (i == 0 or tail[i-1] != "\\"):
            depth += 1
        elif tail[i] == "}" and (i == 0 or tail[i-1] != "\\"):
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None, "unclosed_box", wrapper
    if tail[end:].strip():
        return None, "trailing_text_or_extra_math", wrapper
    return tail[:end], "ok", wrapper


def numeric_percent_points(gold: str, boxed: str, question: str) -> tuple[str, bool]:
    """GSM-style percentage questions report 36 for answer '36 percent'.

    Change ONLY the percent unit from a bare numeric percentage answer in
    explicitly percentage-seeking questions. Do not strip percent from arbitrary
    expressions or symbolic math. Always record whether this was applied.
    """
    if not re.fullmatch(r"[-+]?\d[\d,]*(?:\.\d+)?", str(gold).strip()):
        return boxed, False
    if not re.search(r"\bpercent(?:age)?\b", question, re.I):
        return boxed, False
    m = re.fullmatch(r"\\boxed\s*\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:\\%|%)\s*\}", boxed)
    if not m:
        return boxed, False
    return r"\boxed{" + m.group(1) + "}", True
