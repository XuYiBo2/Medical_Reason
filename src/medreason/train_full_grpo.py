"""Phase 5 controlled E2/E3 GRPO training from the same immutable SFT adapter."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

from medreason.difficulty import protocol_identity, validate_frozen_protocol
from medreason.prompt import PROMPT_VERSION
from medreason.reward import parse_final_answer, reasoning_diagnostics
from medreason.train_grpo import binary_grpo_rewards, build_grpo_record, _load_model_and_tokenizer
from medreason.train_sft import _adapter_state, _updated_adapter_parameters, _write_json


def verify_branch_inputs(
    pool_rows: list[dict[str, Any]], probe_ids: set[str], difficulty_summary: dict[str, Any],
    protocol: dict[str, Any], branch: str,
) -> None:
    if difficulty_summary.get("status") != "passed":
        raise ValueError("Phase 5 requires a passing Phase 4 summary")
    if difficulty_summary["frozen_protocol_identity"] != protocol_identity(protocol):
        raise ValueError("training pools were built under a different frozen protocol")
    expected_size = difficulty_summary["pool_size"]
    if len(pool_rows) != expected_size or len({row["id"] for row in pool_rows}) != expected_size:
        raise ValueError(f"{branch} pool must contain exactly {expected_size} unique samples")
    leaked = probe_ids & {row["id"] for row in pool_rows}
    if leaked:
        raise ValueError(f"{branch} pool contains {len(leaked)} persistence probe IDs")


def verify_budget_symmetry(random_metrics: dict[str, Any], difficulty_metrics: dict[str, Any]) -> dict[str, int]:
    fields = ("optimizer_steps", "fresh_prompt_groups", "fresh_rollouts")
    random_counters = random_metrics["fresh_counters"]
    difficulty_counters = difficulty_metrics["fresh_counters"]
    mismatches = {
        field: (random_counters[field], difficulty_counters[field])
        for field in fields if random_counters[field] != difficulty_counters[field]
    }
    if random_metrics.get("status") != "passed" or difficulty_metrics.get("status") != "passed" or mismatches:
        raise ValueError(f"E2/E3 are not budget-comparable: {mismatches}")
    return {field: random_counters[field] for field in fields}


def _normalized_log_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping = {
        "reward": "reward_mean",
        "reward_std": "reward_std",
        "frac_reward_zero_std": "frac_reward_zero_std",
        "format_rate": "format_rate",
        "pre_answer_tokens_mean": "pre_answer_tokens_mean",
        "direct_answer_rate": "direct_answer_rate",
        "completions/mean_length": "completion_length_mean",
        "completions/clipped_ratio": "truncation_rate",
        "clip_ratio/low_mean": "clip_ratio_low",
        "clip_ratio/high_mean": "clip_ratio_high",
        "loss": "loss",
        "grad_norm": "gradient_norm",
        "learning_rate": "learning_rate",
    }
    rows = []
    for source in history:
        row = {target: source[key] for key, target in mapping.items() if key in source}
        if row:
            row["optimizer_step"] = int(source.get("step", 0))
            rows.append(row)
    return rows


def run(config_path: Path, branch: str) -> dict[str, Any]:
    import torch
    import trl
    from datasets import load_dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    if trl.__version__ != "1.10.0":
        raise RuntimeError(f"Expected trl==1.10.0, found {trl.__version__}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 5 requires a CUDA GPU with bfloat16 support")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(Path(config["protocol_path"]).read_text(encoding="utf-8"))
    validate_frozen_protocol(protocol)
    branch_cfg = config["branches"][branch]
    output_dir = Path(branch_cfg["output_dir"])
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite an existing formal branch: {output_dir}")
    pool = list(load_dataset("json", data_files=branch_cfg["pool_path"], split="train"))
    probe = json.loads(Path(config["probe_ids_path"]).read_text(encoding="utf-8"))
    difficulty_summary = json.loads(Path(config["difficulty_summary_path"]).read_text(encoding="utf-8"))
    verify_branch_inputs(pool, set(probe["all_ids"]), difficulty_summary, protocol, branch)
    dataset = load_dataset("json", data_files=branch_cfg["pool_path"], split="train")
    dataset = dataset.map(build_grpo_record, remove_columns=dataset.column_names, desc=f"Rendering {branch} GRPO pool")
    model, tokenizer = _load_model_and_tokenizer({
        "model": protocol["model"], "quantization": protocol["quantization"],
        "initialization": {"sft_adapter_dir": protocol["sft_adapter"]["path"]},
    }, is_trainable=True)

    reward_audit: dict[str, list[Any]] = {
        "values": [], "format": [], "pre_answer_tokens": [], "direct": [], "group_zero_std": [],
    }
    group_size = protocol["generation"]["num_generations"]

    def binary_task_success_reward(completions, answer, allowed_labels, log_metric=None, **kwargs):
        rewards = binary_grpo_rewards(completions, answer, allowed_labels)
        diagnostics = [reasoning_diagnostics(text, tokenizer) for text in completions]
        formats = [float(parse_final_answer(text, set(labels)) is not None)
                   for text, labels in zip(completions, allowed_labels)]
        reward_audit["values"].extend(rewards)
        reward_audit["format"].extend(formats)
        reward_audit["pre_answer_tokens"].extend(item["pre_answer_tokens"] for item in diagnostics)
        reward_audit["direct"].extend(float(item["is_direct_answer"]) for item in diagnostics)
        for first in range(0, len(rewards), group_size):
            group = rewards[first : first + group_size]
            if len(group) != group_size:
                raise RuntimeError("reward batch does not contain complete G-sized prompt groups")
            reward_audit["group_zero_std"].append(float(statistics.pstdev(group) == 0.0))
        if log_metric is not None:
            log_metric("format_rate", statistics.fmean(formats))
            log_metric("pre_answer_tokens_mean", statistics.fmean(item["pre_answer_tokens"] for item in diagnostics))
            log_metric("direct_answer_rate", statistics.fmean(item["is_direct_answer"] for item in diagnostics))
        return rewards

    class AuditedGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs):
            self.generation_event_steps: list[int] = []
            self.fresh_rollouts = 0
            self.fresh_generated_completion_tokens = 0
            self.fresh_truncated_rollouts = 0
            super().__init__(*args, **kwargs)

        def _generate(self, prompts):
            result = super()._generate(prompts)
            if self.model.training:
                completion_ids = result[1]
                eos_and_pad = {self._tokenizer.eos_token_id, self._tokenizer.pad_token_id}
                self.generation_event_steps.append(int(self.state.global_step))
                self.fresh_rollouts += len(completion_ids)
                self.fresh_generated_completion_tokens += sum(len(ids) for ids in completion_ids)
                self.fresh_truncated_rollouts += sum(ids[-1] not in eos_and_pad for ids in completion_ids)
            return result

    monitor_cfg = config["failure_monitor"]

    class FailureMonitor(TrainerCallback):
        def __init__(self):
            self.bad_format = 0
            self.bad_truncation = 0
            self.reason: str | None = None

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = logs or {}
            if "format_rate" in logs:
                self.bad_format = self.bad_format + 1 if logs["format_rate"] < monitor_cfg["minimum_format_rate"] else 0
            if "completions/clipped_ratio" in logs:
                self.bad_truncation = (
                    self.bad_truncation + 1
                    if logs["completions/clipped_ratio"] > monitor_cfg["maximum_truncation_rate"] else 0
                )
            window = monitor_cfg["consecutive_logging_steps"]
            if self.bad_format >= window:
                self.reason = f"format_rate remained below {monitor_cfg['minimum_format_rate']} for {window} logs"
            if self.bad_truncation >= window:
                self.reason = f"truncation_rate remained above {monitor_cfg['maximum_truncation_rate']} for {window} logs"
            if self.reason:
                control.should_training_stop = True
            return control

    training = protocol["training"]
    generation = protocol["generation"]
    checkpointing = protocol["checkpointing"]
    args = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"), learning_rate=training["learning_rate"],
        optim=training["optimizer"], lr_scheduler_type=training["lr_scheduler_type"], max_steps=training["max_steps"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        steps_per_generation=training["steps_per_generation"], num_generations=generation["num_generations"],
        num_iterations=training["num_iterations"], max_completion_length=generation["max_completion_length"],
        temperature=generation["temperature"], top_p=generation["top_p"],
        repetition_penalty=generation["repetition_penalty"], loss_type=training["loss_type"],
        epsilon=training["epsilon"], epsilon_high=training["epsilon_high"], beta=training["beta"],
        scale_rewards=training["scale_rewards"], mask_truncated_completions=training["mask_truncated_completions"],
        use_vllm=False, bf16=training["bf16"], gradient_checkpointing=training["gradient_checkpointing"],
        max_grad_norm=training["max_grad_norm"], save_strategy=checkpointing["save_strategy"],
        save_steps=checkpointing["save_steps"], save_total_limit=checkpointing["save_total_limit"],
        seed=training["seed"], logging_steps=1, report_to="none",
    )
    monitor = FailureMonitor()
    before = _adapter_state(model)
    trainer = AuditedGRPOTrainer(
        model=model, reward_funcs=binary_task_success_reward, args=args,
        train_dataset=dataset, processing_class=tokenizer, callbacks=[monitor],
    )
    if trainer.ref_model is not None or set(model.peft_config) != {"default"}:
        raise RuntimeError("formal beta=0 run unexpectedly created a reference model/adapter")
    result = trainer.train()
    losses = [row["loss"] for row in trainer.state.log_history if "loss" in row]
    finite_loss = bool(losses) and all(math.isfinite(float(loss)) for loss in losses)
    gradient_norms = [row["grad_norm"] for row in trainer.state.log_history if "grad_norm" in row]
    finite_gradients = bool(gradient_norms) and all(math.isfinite(float(value)) for value in gradient_norms)
    updated_count = _updated_adapter_parameters(before, model)
    after = _adapter_state(model)
    adapter_finite = all(torch.isfinite(tensor).all().item() for tensor in after.values())
    events = len(trainer.generation_event_steps)
    counters = {
        "optimizer_steps": int(trainer.state.global_step),
        "fresh_generation_batches": events,
        "fresh_prompt_groups": trainer.fresh_rollouts // group_size,
        "fresh_rollouts": trainer.fresh_rollouts,
        "fresh_generated_completion_tokens": trainer.fresh_generated_completion_tokens,
        "generation_event_optimizer_steps": trainer.generation_event_steps,
    }
    expected = protocol["budget"]
    completed_budget = (
        counters["optimizer_steps"] == training["max_steps"]
        and counters["fresh_prompt_groups"] == expected["planned_fresh_prompt_groups"]
        and counters["fresh_rollouts"] == expected["planned_fresh_rollouts"]
    )
    checkpoint_steps = [100, 200, 300]
    checkpoints_present = [step for step in checkpoint_steps if (output_dir / "checkpoints" / f"checkpoint-{step}").is_dir()]
    status = "passed" if (
        monitor.reason is None and finite_loss and finite_gradients and updated_count > 0 and adapter_finite
        and completed_budget and checkpoints_present == checkpoint_steps
    ) else "failed"
    metrics = {
        **result.metrics,
        "status": status,
        "branch": branch,
        "trl": trl.__version__,
        "frozen_protocol_identity": protocol_identity(protocol),
        "source_pool": branch_cfg["pool_path"],
        "pool_size": len(pool),
        "probe_overlap_count": 0,
        "reference_model_loaded": trainer.ref_model is not None,
        "adapter_names": sorted(model.peft_config),
        "updated_lora_parameter_count": updated_count,
        "adapter_parameters_finite": adapter_finite,
        "finite_loss": finite_loss,
        "finite_gradient_norms": finite_gradients,
        "failure_monitor_reason": monitor.reason,
        "fresh_counters": counters,
        "reward_values_observed": sorted(set(reward_audit["values"])),
        "frac_reward_zero_std": statistics.fmean(reward_audit["group_zero_std"]),
        "format_rate": statistics.fmean(reward_audit["format"]),
        "pre_answer_tokens_mean": statistics.fmean(reward_audit["pre_answer_tokens"]),
        "direct_answer_rate": statistics.fmean(reward_audit["direct"]),
        "truncation_rate": trainer.fresh_truncated_rollouts / trainer.fresh_rollouts,
        "checkpoints_present": checkpoints_present,
        "training_log": _normalized_log_history(trainer.state.log_history),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0),
    }
    _write_json(output_dir / "train_metrics.json", metrics)
    if status != "passed":
        raise RuntimeError(f"{branch} formal GRPO failed its gate; see {output_dir / 'train_metrics.json'}")
    return metrics


def compare(config_path: Path) -> dict[str, int]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    random_metrics = json.loads((Path(config["branches"]["random"]["output_dir"]) / "train_metrics.json").read_text())
    difficulty_metrics = json.loads(
        (Path(config["branches"]["difficulty"]["output_dir"]) / "train_metrics.json").read_text()
    )
    result = verify_budget_symmetry(random_metrics, difficulty_metrics)
    _write_json(Path("outputs/grpo_budget_comparison.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/full_grpo.yaml"))
    parser.add_argument("--branch", choices=("random", "difficulty"))
    parser.add_argument("--compare-budgets", action="store_true")
    args = parser.parse_args()
    if args.compare_budgets:
        payload = compare(args.config)
    elif args.branch:
        payload = run(args.config, args.branch)
    else:
        parser.error("provide --branch or --compare-budgets")
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
