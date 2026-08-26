from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from medreason.train_full_grpo import verify_branch_inputs, verify_budget_symmetry


def test_full_grpo_config_has_symmetric_branch_only_inputs() -> None:
    config = yaml.safe_load(Path("configs/full_grpo.yaml").read_text(encoding="utf-8"))
    assert config["branches"] == {
        "random": {
            "pool_path": "outputs/difficulty/random_pool.jsonl",
            "output_dir": "outputs/random_grpo",
        },
        "difficulty": {
            "pool_path": "outputs/difficulty/informative_pool.jsonl",
            "output_dir": "outputs/difficulty_grpo",
        },
    }
    assert config["failure_monitor"] == {
        "consecutive_logging_steps": 10,
        "minimum_format_rate": 0.90,
        "maximum_truncation_rate": 0.25,
    }


def identity() -> dict:
    return {
        "trl_version": "1.10.0", "model": {}, "quantization": {}, "sft_adapter": {},
        "prompt": {}, "generation": {}, "reward": {},
    }


def test_branch_pool_is_unique_protocol_bound_and_probe_free() -> None:
    protocol = {
        "trl_version": "1.10.0", "model": {}, "quantization": {}, "sft_adapter": {},
        "prompt": {}, "generation": {}, "reward": {},
    }
    summary = {"status": "passed", "frozen_protocol_identity": identity(), "pool_size": 2}
    verify_branch_inputs([{"id": "a"}, {"id": "b"}], {"probe"}, summary, protocol, "random")
    with pytest.raises(ValueError, match="probe"):
        verify_branch_inputs([{"id": "a"}, {"id": "probe"}], {"probe"}, summary, protocol, "random")
    with pytest.raises(ValueError, match="unique"):
        verify_branch_inputs([{"id": "a"}, {"id": "a"}], set(), summary, protocol, "difficulty")


def test_budget_symmetry_requires_two_passing_complete_branches() -> None:
    counters = {"optimizer_steps": 300, "fresh_prompt_groups": 150, "fresh_rollouts": 600}
    assert verify_budget_symmetry(
        {"status": "passed", "fresh_counters": counters},
        {"status": "passed", "fresh_counters": dict(counters)},
    ) == counters
    with pytest.raises(ValueError, match="budget-comparable"):
        verify_budget_symmetry(
            {"status": "passed", "fresh_counters": counters},
            {"status": "passed", "fresh_counters": {**counters, "fresh_rollouts": 596}},
        )
    with pytest.raises(ValueError, match="budget-comparable"):
        verify_budget_symmetry(
            {"status": "failed", "fresh_counters": counters},
            {"status": "passed", "fresh_counters": counters},
        )
