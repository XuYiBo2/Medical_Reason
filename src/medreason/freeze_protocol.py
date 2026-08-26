"""Freeze the Phase 3 protocol after a passing GRPO smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from medreason.prompt import PROMPT_VERSION
from medreason.train_grpo import PARSER_VERSION, REWARD_VERSION, infer_event_schedule


def build_frozen_protocol(config: dict[str, Any], smoke: dict[str, Any], adapter_config: dict[str, Any]) -> dict[str, Any]:
    if smoke.get("status") != "passed":
        raise ValueError("protocol freeze requires a passing smoke report")
    cfg = config["training"]
    counters = smoke["fresh_counters"]
    events = counters["fresh_generation_batches"]
    if events <= 0:
        raise ValueError("smoke report contains no real generation event")
    if counters["fresh_prompt_groups"] % events or counters["fresh_rollouts"] % events:
        raise ValueError("smoke event counters do not resolve to integral per-event budgets")
    groups_per_event = counters["fresh_prompt_groups"] // events
    rollouts_per_event = counters["fresh_rollouts"] // events
    if groups_per_event != 1 or rollouts_per_event != cfg["num_generations"]:
        raise ValueError(
            f"unexpected locked sampler shape: groups/event={groups_per_event}, rollouts/event={rollouts_per_event}"
        )
    schedule = infer_event_schedule(counters["generation_event_optimizer_steps"], cfg["max_steps"])
    planned_events = schedule["planned_generation_events"]
    lora_fields = {
        key: adapter_config.get(key)
        for key in ("r", "lora_alpha", "lora_dropout", "bias", "target_modules", "task_type")
    }
    return {
        "phase": "Phase 3 GRPO protocol freeze",
        "trl_version": smoke["trl"],
        "model": config["model"],
        "quantization": config["quantization"],
        "sft_adapter": {
            "path": config["initialization"]["sft_adapter_dir"],
            "identity": "immutable Phase 2 adapter; reload independently for every downstream branch",
            "lora_config": lora_fields,
        },
        "prompt": {"serialization": "plain_text", "version": PROMPT_VERSION},
        "generation": {
            "eos_token_required": True,
            "decode_skip_special_tokens": True,
            "num_generations": cfg["num_generations"],
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "repetition_penalty": cfg["repetition_penalty"],
            "max_completion_length": cfg["max_completion_length"],
            "truncation_definition": "completion ends without EOS/PAD at the generation cap",
        },
        "reward": {"parser_version": PARSER_VERSION, "reward_version": REWARD_VERSION, "values": [0, 1]},
        "training": {
            key: cfg[key] for key in (
                "learning_rate", "optimizer", "lr_scheduler_type", "per_device_train_batch_size",
                "gradient_accumulation_steps", "steps_per_generation", "num_iterations", "loss_type",
                "epsilon", "epsilon_high", "beta", "scale_rewards", "mask_truncated_completions",
                "max_grad_norm", "bf16", "gradient_checkpointing", "max_steps", "seed",
            )
        },
        "budget": {
            **schedule,
            "fresh_prompt_groups_per_generation_event": groups_per_event,
            "fresh_rollouts_per_generation_event": rollouts_per_event,
            "planned_fresh_prompt_groups": planned_events * groups_per_event,
            "planned_fresh_rollouts": planned_events * rollouts_per_event,
        },
        "checkpoint_and_dev_selection": config["protocol"]["checkpoint_selection"],
        "checkpointing": {
            "save_strategy": cfg["save_strategy"],
            "save_steps": cfg["save_steps"],
            "save_total_limit": cfg["save_total_limit"],
        },
        "deterministic_evaluation": {
            "do_sample": False,
            "repetition_penalty": config["protocol"]["deterministic_evaluation_repetition_penalty"],
        },
        "source_smoke": {
            "report": str(Path(config["smoke"]["output_dir"]) / "train_metrics.json"),
            "observed_counters": counters,
        },
        "invalidation_rules": {
            "rollout_affecting": "invalidate scan, p0, probe, pools, E2 and E3; smoke, freeze and scan again",
            "training_only": "rerun E2 and E3 from theta0; recompute budgets and pools when budget changes",
            "evaluation_only": "do not retrain; re-evaluate every affected branch",
        },
    }


def freeze(config_path: Path, smoke_path: Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    smoke_path = smoke_path or Path(config["smoke"]["output_dir"]) / "train_metrics.json"
    output_path = Path(config["protocol"]["output_path"])
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol: {output_path}")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    adapter_path = Path(config["initialization"]["sft_adapter_dir"]) / "adapter_config.json"
    adapter_config = json.loads(adapter_path.read_text(encoding="utf-8"))
    protocol = build_frozen_protocol(config, smoke, adapter_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(protocol, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/grpo.yaml"))
    parser.add_argument("--smoke-report", type=Path)
    args = parser.parse_args()
    print(yaml.safe_dump(freeze(args.config, args.smoke_report), allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()
