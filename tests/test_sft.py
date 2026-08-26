from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from medreason.evaluate_sft import select_generation_candidate
from medreason.train_sft import (
    audit_tokenized_example,
    build_sft_record,
    completion_was_truncated,
    resolve_training_steps,
)


class FakeTokenizer:
    eos_token_id = 0

    def decode(self, token_ids, skip_special_tokens=False):
        mapping = {1: "Reason", 2: "<answer>", 3: "B", 4: "</answer>", 0: "<eos>"}
        return "".join(mapping[token_id] for token_id in token_ids)


def test_sft_config_matches_spec() -> None:
    config = yaml.safe_load(Path("configs/sft.yaml").read_text(encoding="utf-8"))
    assert config["training"] == {
        "max_length": 1024,
        "completion_only_loss": True,
        "packing": False,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1.0e-4,
        "num_train_epochs": 1,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "weight_decay": 0.0,
        "bf16": True,
        "gradient_checkpointing": True,
        "max_grad_norm": 1.0,
        "seed": 42,
    }
    assert config["smoke"]["samples"] == 32
    assert config["smoke"]["max_steps"] == 5
    assert config["evaluation"]["format_rate_gate"] == 0.95
    assert config["evaluation"]["repetition_penalty"] == 1.05
    assert config["evaluation"]["repetition_penalty_diagnostic"] == [1.0, 1.05, 1.1]


def test_build_sft_record_uses_prompt_completion_contract() -> None:
    sample = {
        "question": "Q",
        "options": {"A": "one", "B": "two"},
        "answer": "B",
        "explanation": "Reason",
    }
    record = build_sft_record(sample, "<eos>")
    assert record["prompt"].endswith("<answer>X</answer>")
    assert record["completion"] == "\n\nReason\n\n<answer>B</answer><eos>"


def test_warmup_ratio_is_resolved_for_locked_sft_api() -> None:
    assert resolve_training_steps(32, 1, 16, 1, 5, 0.03) == (5, 1)
    assert resolve_training_steps(12000, 1, 16, 1, -1, 0.03) == (750, 23)


def test_truncation_requires_cap_length_without_eos() -> None:
    assert not completion_was_truncated([1, 2, 0], max_new_tokens=3, eos_token_id=0)
    assert completion_was_truncated([1, 2, 3], max_new_tokens=3, eos_token_id=0)
    assert not completion_was_truncated([1, 2], max_new_tokens=3, eos_token_id=0)


def test_generation_candidate_selection_follows_predefined_order() -> None:
    results = [
        {"repetition_penalty": 1.0, "metrics": {"format_rate": 0.88}},
        {"repetition_penalty": 1.05, "metrics": {"format_rate": 0.96}},
        {"repetition_penalty": 1.1, "metrics": {"format_rate": 0.99}},
    ]
    assert select_generation_candidate(results, 0.95) == 1.05
    assert select_generation_candidate(results[:1], 0.95) is None


def test_label_audit_accepts_prompt_mask_and_trainable_answer_eos() -> None:
    report = audit_tokenized_example(
        {"input_ids": [9, 9, 1, 2, 3, 4, 0], "labels": [-100, -100, 1, 2, 3, 4, 0]},
        FakeTokenizer(),
    )
    assert report == {
        "sequence_tokens": 7,
        "masked_prompt_tokens": 2,
        "trainable_completion_tokens": 5,
    }


@pytest.mark.parametrize(
    "labels",
    [
        [9, -100, 1, 2, 3, 4, 0],
        [-100, -100, 1, 2, 3, 4, -100],
        [-100, -100, 1, 2, 3, 4, 9],
    ],
)
def test_label_audit_rejects_silent_masking_bugs(labels) -> None:
    with pytest.raises(RuntimeError):
        audit_tokenized_example({"input_ids": [9] * 7, "labels": labels}, FakeTokenizer())
