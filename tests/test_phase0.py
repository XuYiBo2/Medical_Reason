from dataclasses import make_dataclass
from pathlib import Path

import yaml

from medreason.phase0 import GRPO_REQUIRED_FIELDS, SFT_REQUIRED_FIELDS


def test_frozen_api_field_sets_are_complete() -> None:
    expected_sft = {"max_length", "completion_only_loss"}
    expected_grpo = {
        "loss_type",
        "epsilon",
        "epsilon_high",
        "beta",
        "scale_rewards",
        "num_iterations",
        "num_generations",
        "max_completion_length",
        "steps_per_generation",
        "generation_batch_size",
        "mask_truncated_completions",
    }
    assert SFT_REQUIRED_FIELDS == expected_sft
    assert GRPO_REQUIRED_FIELDS == expected_grpo


def test_required_fields_can_detect_a_missing_field() -> None:
    FakeConfig = make_dataclass("FakeConfig", [("max_length", int)])
    available = set(FakeConfig.__dataclass_fields__)
    assert SFT_REQUIRED_FIELDS - available == {"completion_only_loss"}


def test_phase0_config_matches_frozen_qlora_contract() -> None:
    config = yaml.safe_load(Path("configs/phase0.yaml").read_text(encoding="utf-8"))
    assert config["model"]["name_or_path"] == "Qwen/Qwen3-8B-Base"
    assert config["quantization"] == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bfloat16",
    }
    assert config["lora"] == {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
        ],
    }
