"""The shared plain-text prompt and completion contract."""

from __future__ import annotations

from collections.abc import Mapping


PROMPT_VERSION = "plain_text_v1"
PROMPT_HEADER = """You are solving a medical multiple-choice exam question.
Reason briefly and choose the single best option.

Question:
{question}

Options:
{options}

End the response with exactly one final answer tag:
<answer>X</answer>"""


def render_options(options: Mapping[str, str]) -> str:
    """Render options in their existing order without relabeling or sorting."""
    return "\n".join(f"{label}. {text}" for label, text in options.items())


def render_prompt(sample: Mapping[str, object]) -> str:
    """Render every training and evaluation sample with plain_text_v1."""
    question = sample["question"]
    options = sample["options"]
    if not isinstance(question, str) or not isinstance(options, Mapping):
        raise TypeError("sample must contain a string question and an options mapping")
    return PROMPT_HEADER.format(question=question, options=render_options(options))


def render_sft_completion(sample: Mapping[str, object], eos_token: str) -> str:
    explanation = sample.get("explanation")
    answer = sample["answer"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("SFT completion requires a non-empty explanation")
    if not isinstance(answer, str):
        raise TypeError("sample answer must be a string")
    if not eos_token:
        raise ValueError("tokenizer eos_token must be non-empty")
    return f"\n\n{explanation.strip()}\n\n<answer>{answer}</answer>{eos_token}"


def encoded_length(tokenizer: object, text: str) -> int:
    """Measure length with the actual tokenizer and no truncation."""
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def validate_sft_length(sample: Mapping[str, object], tokenizer: object, max_length: int = 1024) -> int:
    eos_token = getattr(tokenizer, "eos_token", None)
    text = render_prompt(sample) + render_sft_completion(sample, eos_token)
    length = encoded_length(tokenizer, text)
    if length > max_length:
        raise ValueError(f"SFT sequence has {length} tokens, exceeding max_length={max_length}")
    return length


def validate_generation_context(
    sample: Mapping[str, object],
    tokenizer: object,
    max_completion_length: int,
    context_limit: int,
) -> int:
    prompt_tokens = encoded_length(tokenizer, render_prompt(sample))
    if prompt_tokens + max_completion_length > context_limit:
        raise ValueError(
            f"prompt_tokens ({prompt_tokens}) + max_completion_length ({max_completion_length}) "
            f"exceeds context_limit ({context_limit})"
        )
    return prompt_tokens
