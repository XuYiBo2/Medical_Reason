from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml

from medreason.phase6 import paired_bootstrap, persistence_metrics, select_checkpoint


def test_phase6_config_freezes_eval_execution_and_statistics() -> None:
    config = yaml.safe_load(Path("configs/eval.yaml").read_text(encoding="utf-8"))
    assert config["checkpoint_steps"] == [100, 200, 300]
    assert config["execution"] == {"prompt_batch_size": 8, "persistence_seed": 42}
    assert config["bootstrap"] == {"resamples": 5000, "confidence_level": 0.95, "seed": 42}
    assert set(config["datasets"]) == {"medqa_dev", "medqa_test", "medmcqa_eval", "mmlu_pro_health"}


def test_checkpoint_selection_uses_frozen_lexicographic_priority() -> None:
    candidates = [
        {"step": 100, "metrics": {"strict_task_success_accuracy": 0.6, "format_rate": 0.99,
                                    "truncation_rate": 0.0, "mean_completion_tokens": 80}},
        {"step": 200, "metrics": {"strict_task_success_accuracy": 0.7, "format_rate": 0.95,
                                    "truncation_rate": 0.03, "mean_completion_tokens": 90}},
        {"step": 300, "metrics": {"strict_task_success_accuracy": 0.7, "format_rate": 0.96,
                                    "truncation_rate": 0.05, "mean_completion_tokens": 100}},
    ]
    assert select_checkpoint(candidates)["step"] == 300
    candidates[2]["metrics"]["format_rate"] = 0.95
    assert select_checkpoint(candidates)["step"] == 200


def test_persistence_metrics_use_frozen_i0_strata() -> None:
    step0 = [
        {"sample_id": "i1", "pass_rate": 0.5, "is_informative": True},
        {"sample_id": "i2", "pass_rate": 0.25, "is_informative": True},
        {"sample_id": "n1", "pass_rate": 0.0, "is_informative": False},
        {"sample_id": "n2", "pass_rate": 1.0, "is_informative": False},
    ]
    current = [
        {"sample_id": "i1", "pass_rate": 0.75, "is_informative": True},
        {"sample_id": "i2", "pass_rate": 1.0, "is_informative": False},
        {"sample_id": "n1", "pass_rate": 0.25, "is_informative": True},
        {"sample_id": "n2", "pass_rate": 1.0, "is_informative": False},
    ]
    metrics = persistence_metrics(step0, current)
    assert metrics["p_informative_given_i0_1"] == 0.5
    assert metrics["p_informative_given_i0_0"] == 0.5
    assert metrics["persistence_gap"] == 0.0
    assert metrics["informative_retention"] == 0.5
    assert metrics["informative_jaccard"] == 1 / 3
    assert math.isfinite(metrics["spearman_pass_rate"])


def test_step_zero_persistence_reuses_scan_exactly() -> None:
    rows = [
        {"sample_id": "i", "pass_rate": 0.5, "is_informative": True},
        {"sample_id": "n", "pass_rate": 0.0, "is_informative": False},
    ]
    metrics = persistence_metrics(rows, rows)
    assert metrics["p_informative_given_i0_1"] == 1.0
    assert metrics["p_informative_given_i0_0"] == 0.0
    assert metrics["persistence_gap"] == 1.0
    assert metrics["spearman_pass_rate"] == pytest.approx(1.0)
    assert metrics["informative_retention"] == 1.0
    assert metrics["informative_jaccard"] == 1.0


def test_paired_bootstrap_is_paired_deterministic_and_labeled() -> None:
    result = paired_bootstrap([0, 0, 1, 1], [1, 0, 1, 1], 5000, 0.95, 42)
    assert result["delta"] == 0.25
    assert result["confidence_interval"][0] <= result["delta"] <= result["confidence_interval"][1]
    assert result["label"] == "single-training-seed conditional CI"
