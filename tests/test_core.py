from __future__ import annotations

import os

import pytest

from medreason.data import (
    clean_gold_leakage,
    configure_huggingface_downloads,
    dataset_distribution_id,
    exact_deduplicate,
    normalize_medmcqa,
    normalize_medqa,
    normalize_mmlu_pro,
    split_medmcqa_validation,
    stratified_sample,
    validate_sample,
)
from medreason.prompt import render_prompt, render_sft_completion, validate_generation_context, validate_sft_length
from medreason.reward import parse_final_answer, reasoning_diagnostics, task_success


class CharacterTokenizer:
    eos_token = "<eos>"
    model_max_length = 4096

    def __call__(self, text, **kwargs):
        assert kwargs.get("truncation") is False
        return {"input_ids": list(text)}


def sample(**updates):
    value = {
        "id": "x",
        "source": "fixture",
        "split": "train",
        "question": "Which option is correct?",
        "options": {"A": "First", "B": "Second", "C": "Third", "D": "Fourth"},
        "answer": "B",
        "explanation": "Because the second option is correct.",
        "subject": "medicine",
    }
    value.update(updates)
    return value


def test_schema_label_mapping_and_source_normalizers() -> None:
    medmcqa = normalize_medmcqa(
        {
            "id": "m1", "question": "Q", "opa": "a", "opb": "b", "opc": "c", "opd": "d",
            "cop": 1, "choice_type": "single", "exp": "Answer: B Explanation", "subject_name": "Pathology",
        },
        "train",
    )
    assert medmcqa["answer"] == "B"
    assert medmcqa["explanation"] == "Explanation"
    assert normalize_medqa(
        {"question": "Q", "options": {"A": "a", "B": "b", "C": "c", "D": "d"}, "answer_idx": "C"},
        "dev", 0,
    )["answer"] == "C"
    assert normalize_mmlu_pro(
        {"question_id": 7, "question": "Q", "options": list("abcdefghij"), "answer_index": 9,
         "answer": "J", "category": "health"},
        "test", 0,
    )["answer"] == "J"


def test_dataset_distribution_can_differ_from_original_source() -> None:
    assert dataset_distribution_id({"id": "jind11/MedQA", "distribution_id": "mirror/MedQA"}) == "mirror/MedQA"
    assert dataset_distribution_id({"id": "source/dataset"}) == "source/dataset"


def test_public_dataset_downloads_disable_xet(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    configure_huggingface_downloads()
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_schema_rejects_non_contiguous_labels_and_bad_answer() -> None:
    with pytest.raises(ValueError):
        validate_sample(sample(options={"A": "a", "C": "c"}))
    with pytest.raises(ValueError):
        validate_sample(sample(answer="E"))


def test_exact_dedup_uses_question_and_ordered_options() -> None:
    duplicate = sample(id="duplicate")
    reordered = sample(id="reordered", options={"B": "Second", "A": "First", "C": "Third", "D": "Fourth"})
    assert [row["id"] for row in exact_deduplicate([sample(), duplicate, reordered])] == ["x", "reordered"]
    forbidden = {(sample()["question"], tuple(sample()["options"].items()))}
    assert exact_deduplicate([duplicate], forbidden_keys=forbidden) == []


def test_renderer_and_completion_are_frozen() -> None:
    prompt = render_prompt(sample())
    assert "Question:\nWhich option is correct?" in prompt
    assert "Options:\nA. First\nB. Second\nC. Third\nD. Fourth" in prompt
    assert prompt.endswith("<answer>X</answer>")
    completion = render_sft_completion(sample(), "<eos>")
    assert completion.startswith("\n\nBecause")
    assert completion.endswith("<answer>B</answer><eos>")


def test_length_checks_never_silently_truncate() -> None:
    tokenizer = CharacterTokenizer()
    length = validate_sft_length(sample(), tokenizer, max_length=1000)
    assert length > 0
    with pytest.raises(ValueError, match="exceeding"):
        validate_sft_length(sample(), tokenizer, max_length=10)
    with pytest.raises(ValueError, match="exceeds context_limit"):
        validate_generation_context(sample(), tokenizer, max_completion_length=4000, context_limit=4096)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning... <answer>B</answer>", "B"),
        ("<answer>B</answer>", "B"),
        ("B", None),
        ("Answer: B", None),
        ("<answer>B</answer> extra text", None),
        ("<answer>B</answer><answer>B</answer>", None),
        ("<answer> B </answer>", None),
        ("<answer>E</answer>", None),
        ("<answer>B</answer>\n\t", "B"),
    ],
)
def test_strict_parser(text, expected) -> None:
    assert parse_final_answer(text, set("ABCD")) == expected


def test_binary_reward_and_direct_answer_are_independent() -> None:
    tokenizer = CharacterTokenizer()
    direct = "<answer>B</answer>"
    assert task_success(direct, "B", set("ABCD")) == 1
    assert reasoning_diagnostics(direct, tokenizer) == {"pre_answer_tokens": 0, "is_direct_answer": True}
    assert task_success("Long reasoning <answer>A</answer>", "B", set("ABCD")) == 0


def test_gold_leakage_only_removes_matching_opening_phrase() -> None:
    assert clean_gold_leakage("The correct option is D. Medical reason.", "D") == "Medical reason."
    assert clean_gold_leakage("Answer: A Medical reason.", "D") == "Answer: A Medical reason."
    assert clean_gold_leakage("Medical reason. Answer: D", "D") == "Medical reason. Answer: D"


def test_stratified_sample_is_deterministic_and_proportional() -> None:
    rows = [sample(id=f"a{i}", subject="a") for i in range(8)] + [sample(id=f"b{i}", subject="b") for i in range(2)]
    first = stratified_sample(rows, 5, seed=42)
    second = stratified_sample(rows, 5, seed=42)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert sum(row["subject"] == "a" for row in first) == 4
    assert sum(row["subject"] == "b" for row in first) == 1


def test_medmcqa_final_eval_does_not_require_sft_explanation() -> None:
    sft_candidates = [sample(id=f"dev{i}", question=f"dev {i}") for i in range(2)]
    no_explanation = sample(id="eval", question="eval only", explanation=None)
    sft_dev, eval_candidates = split_medmcqa_validation(
        sft_candidates,
        [*sft_candidates, no_explanation],
        dev_size=2,
        seed=42,
        forbidden_eval_keys=set(),
    )
    assert len(sft_dev) == 2
    assert eval_candidates == [no_explanation]
