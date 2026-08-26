from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from medreason.freeze_protocol import build_frozen_protocol
from medreason.train_grpo import binary_grpo_rewards, build_grpo_record, infer_event_schedule


def test_candidate_grpo_config_matches_spec() -> None:
    config = yaml.safe_load(Path("configs/grpo.yaml").read_text(encoding="utf-8"))
    training = config["training"]
    assert {key: training[key] for key in (
        "learning_rate", "max_steps", "per_device_train_batch_size", "gradient_accumulation_steps",
        "steps_per_generation", "num_generations", "num_iterations", "max_completion_length",
        "temperature", "top_p", "loss_type", "epsilon", "epsilon_high", "beta", "scale_rewards",
        "mask_truncated_completions", "use_vllm", "bf16", "gradient_checkpointing", "max_grad_norm", "seed",
    )} == {
        "learning_rate": 1.0e-6, "max_steps": 300, "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4, "steps_per_generation": 4, "num_generations": 4,
        "num_iterations": 2, "max_completion_length": 256, "temperature": 0.8, "top_p": 0.95,
        "loss_type": "dapo", "epsilon": 0.2, "epsilon_high": 0.28, "beta": 0.0,
        "scale_rewards": "group", "mask_truncated_completions": True, "use_vllm": False,
        "bf16": True, "gradient_checkpointing": True, "max_grad_norm": 1.0, "seed": 42,
    }
    assert training["repetition_penalty"] == 1.0
    assert config["protocol"]["deterministic_evaluation_repetition_penalty"] == 1.05


def test_grpo_record_and_binary_direct_answer_reward() -> None:
    sample = {"question": "Q", "options": {"A": "one", "B": "two"}, "answer": "B"}
    record = build_grpo_record(sample)
    assert record["prompt"].endswith("<answer>X</answer>")
    assert record["allowed_labels"] == ["A", "B"]
    assert binary_grpo_rewards(["<answer>B</answer>", "B"], ["B", "B"], [["A", "B"], ["A", "B"]]) == [1.0, 0.0]


def test_sampler_budget_comes_from_observed_event_steps() -> None:
    assert infer_event_schedule([0, 2, 4], 300) == {
        "optimizer_steps_per_generation_event": 2,
        "planned_generation_events": 150,
    }
    with pytest.raises(ValueError):
        infer_event_schedule([0], 300)
    with pytest.raises(ValueError):
        infer_event_schedule([0, 2, 5], 300)


def test_protocol_freeze_resolves_numeric_fresh_budgets() -> None:
    config = yaml.safe_load(Path("configs/grpo.yaml").read_text(encoding="utf-8"))
    smoke = {
        "status": "passed", "trl": "1.10.0",
        "fresh_counters": {
            "optimizer_steps": 5, "fresh_generation_batches": 3, "fresh_prompt_groups": 3,
            "fresh_rollouts": 12, "fresh_generated_completion_tokens": 900,
            "generation_event_optimizer_steps": [0, 2, 4],
        },
    }
    protocol = build_frozen_protocol(config, smoke, {
        "r": 16, "lora_alpha": 32, "lora_dropout": 0.0, "bias": "none",
        "target_modules": ["q_proj"], "task_type": "CAUSAL_LM",
    })
    assert protocol["budget"]["planned_fresh_prompt_groups"] == 150
    assert protocol["budget"]["planned_fresh_rollouts"] == 600
    assert protocol["generation"]["repetition_penalty"] == 1.0
    assert protocol["deterministic_evaluation"]["repetition_penalty"] == 1.05
