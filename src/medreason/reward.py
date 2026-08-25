"""Strict answer parsing, binary task reward, and reasoning diagnostics."""

from __future__ import annotations

import re
from collections.abc import Collection
from typing import Any


ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)


def parse_final_answer(text: str, allowed_labels: Collection[str]) -> str | None:
    """Return the label only when text satisfies the frozen format gate."""
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    match = ANSWER_TAG_RE.search(text)
    if match is None or text[match.end() :].strip():
        return None
    label = match.group(1)
    if len(label) != 1 or label not in allowed_labels:
        return None
    return label


def task_success(text: str, gold_label: str, allowed_labels: Collection[str]) -> int:
    """Binary verifiable task-success reward."""
    return int(parse_final_answer(text, allowed_labels) == gold_label)


def reasoning_diagnostics(text: str, tokenizer: Any) -> dict[str, int | bool]:
    """Count tokens before the final-answer tag; this never changes reward."""
    match = ANSWER_TAG_RE.search(text)
    prefix = text[: match.start()] if match is not None else text
    pre_answer_tokens = len(tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"])
    return {
        "pre_answer_tokens": pre_answer_tokens,
        "is_direct_answer": pre_answer_tokens < 8,
    }

