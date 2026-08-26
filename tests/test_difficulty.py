from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from medreason.difficulty import (
    required_pool_size,
    scan_gate_counts,
    next_scan_target,
    select_probe_and_pools,
    validate_frozen_protocol,
)


def frozen_protocol() -> dict:
    return {
        "phase": "Phase 3 GRPO protocol freeze",
        "trl_version": "1.10.0",
        "model": {"name_or_path": "base", "revision": "rev"},
        "quantization": {"load_in_4bit": True},
        "sft_adapter": {"path": "outputs/sft/adapter"},
        "prompt": {"serialization": "plain_text", "version": "plain_text_v1"},
        "generation": {
            "num_generations": 4, "temperature": 0.8, "top_p": 0.95,
            "repetition_penalty": 1.0, "max_completion_length": 256,
            "decode_skip_special_tokens": True,
        },
        "reward": {
            "parser_version": "strict_answer_v1", "reward_version": "binary_task_success_v1", "values": [0, 1]
        },
        "budget": {"planned_fresh_prompt_groups": 150},
    }


def test_difficulty_config_matches_spec() -> None:
    config = yaml.safe_load(Path("configs/difficulty.yaml").read_text(encoding="utf-8"))
    assert config["scan"] == {"initial_samples": 1000, "expand_step": 500, "prompt_batch_size": 8, "seed": 42}
    assert config["probe"] == {"target": 256, "per_stratum_target": 128, "seed": 42}
    assert required_pool_size(frozen_protocol(), config["pools"]["minimum_size"]) == 256
    assert scan_gate_counts(256, 128) == {"informative_required": 384, "noninformative_required": 128}
    assert next_scan_target(0, 6000, 1000, 500) == 1000
    assert next_scan_target(1200, 6000, 1000, 500) == 1500
    assert next_scan_target(1500, 1600, 1000, 500) == 1600


def test_frozen_protocol_contract_is_checked() -> None:
    protocol = frozen_protocol()
    validate_frozen_protocol(protocol)
    protocol["generation"]["num_generations"] = 1
    with pytest.raises(ValueError):
        validate_frozen_protocol(protocol)


def test_probe_is_excluded_and_pools_have_equal_frozen_size() -> None:
    rows = []
    samples = {}
    for index in range(12):
        sample_id = f"i{index}"
        rows.append({"sample_id": sample_id, "is_informative": True})
        samples[sample_id] = {"id": sample_id}
    for index in range(8):
        sample_id = f"n{index}"
        rows.append({"sample_id": sample_id, "is_informative": False})
        samples[sample_id] = {"id": sample_id}
    selection = select_probe_and_pools(
        rows, samples, pool_size=6, per_stratum_target=2, probe_seed=42, pool_seed=42,
        full_universe_scanned=False,
    )
    probe_ids = set(selection["probe"]["all_ids"])
    info_ids = set(selection["informative_pool_ids"])
    random_ids = set(selection["random_pool_ids"])
    assert len(info_ids) == len(random_ids) == 6
    assert not probe_ids & info_ids
    assert not probe_ids & random_ids
    assert all(sample_id.startswith("i") for sample_id in info_ids)
    assert selection["pool_overlap_count"] == len(info_ids & random_ids)
    assert selection["pool_overlap_ratio"] == len(info_ids & random_ids) / 6


def test_scan_must_expand_until_probe_and_info_pool_fit() -> None:
    rows = ([{"sample_id": f"i{x}", "is_informative": True} for x in range(7)]
            + [{"sample_id": f"n{x}", "is_informative": False} for x in range(2)])
    samples = {row["sample_id"]: {"id": row["sample_id"]} for row in rows}
    with pytest.raises(ValueError, match="expand"):
        select_probe_and_pools(rows, samples, 6, 2, 42, 42, full_universe_scanned=False)


def test_full_universe_uses_symmetric_available_probe_and_marks_low_support() -> None:
    rows = ([{"sample_id": f"i{x}", "is_informative": True} for x in range(8)]
            + [{"sample_id": f"n{x}", "is_informative": False} for x in range(3)])
    samples = {row["sample_id"]: {"id": row["sample_id"]} for row in rows}
    selection = select_probe_and_pools(rows, samples, 4, 5, 42, 42, full_universe_scanned=True)
    assert selection["probe"]["per_stratum"] == 3
    assert selection["probe"]["actual"] == 6
    assert selection["probe"]["support"] == "exploratory_low_support"
