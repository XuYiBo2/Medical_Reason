"""Phase 3 GRPO smoke from the immutable SFT adapter."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import yaml

from medreason.prompt import PROMPT_VERSION, render_prompt
from medreason.reward import parse_final_answer, reasoning_diagnostics, task_success
from medreason.train_sft import _adapter_state, _updated_adapter_parameters, _write_json


PARSER_VERSION = "strict_answer_v1"
REWARD_VERSION = "binary_task_success_v1"


def build_grpo_record(sample: dict[str, Any]) -> dict[str, Any]:
    """Preserve reward columns while applying the one shared renderer."""
    return {
        "prompt": render_prompt(sample),
        "answer": sample["answer"],
        "allowed_labels": list(sample["options"]),
    }


def binary_grpo_rewards(completions: list[str], answers: list[str], allowed_labels: list[list[str]]) -> list[float]:
    if not (len(completions) == len(answers) == len(allowed_labels)):
        raise ValueError("completion, answer, and label batches must have equal length")
    rewards = [float(task_success(text, gold, set(labels))) for text, gold, labels in zip(
        completions, answers, allowed_labels
    )]
    if any(value not in (0.0, 1.0) for value in rewards):
        raise RuntimeError("GRPO reward escaped the binary {0,1} contract")
    return rewards


def infer_event_schedule(event_steps: list[int], planned_optimizer_steps: int) -> dict[str, int]:
    """Project the full budget only after smoke exposes a stable real event cadence."""
    if len(event_steps) < 2 or event_steps[0] != 0:
        raise ValueError("smoke must observe at least two fresh generation events starting at optimizer step 0")
    gaps = [right - left for left, right in zip(event_steps, event_steps[1:])]
    if min(gaps) <= 0 or len(set(gaps)) != 1:
        raise ValueError(f"generation event cadence is not stable: {event_steps}")
    cadence = gaps[0]
    events = len(range(event_steps[0], planned_optimizer_steps, cadence))
    return {"optimizer_steps_per_generation_event": cadence, "planned_generation_events": events}


def _load_model_and_tokenizer(config: dict[str, Any]):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg, quant_cfg = config["model"], config["quantization"]
    adapter_dir = Path(config["initialization"]["sft_adapter_dir"])
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"immutable SFT adapter not found: {adapter_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"], revision=model_cfg["revision"],
        trust_remote_code=model_cfg["trust_remote_code"], padding_side="left",
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer must define EOS")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"], revision=model_cfg["revision"],
        trust_remote_code=model_cfg["trust_remote_code"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        ), dtype=torch.bfloat16, device_map={"": 0},
    )
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    if set(model.peft_config) != {"default"}:
        raise RuntimeError(f"expected exactly the existing SFT adapter, found {list(model.peft_config)}")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("the existing SFT LoRA must be the only trainable parameter set")
    return model, tokenizer


def run(config_path: Path) -> dict[str, Any]:
    import torch
    import trl
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    if trl.__version__ != "1.10.0":
        raise RuntimeError(f"Expected trl==1.10.0, found {trl.__version__}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 3 smoke requires a CUDA GPU with bfloat16 support")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config["smoke"]["output_dir"])
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        raise FileExistsError(f"refusing to overwrite disposable smoke adapter: {adapter_dir}")
    model, tokenizer = _load_model_and_tokenizer(config)
    raw = load_dataset("json", data_files=config["data"]["train_path"], split="train")
    sample_count = config["smoke"]["samples"]
    if len(raw) < sample_count:
        raise RuntimeError(f"GRPO smoke requires {sample_count} rows, found {len(raw)}")
    dataset = raw.select(range(sample_count)).map(
        build_grpo_record, remove_columns=raw.column_names, desc="Rendering GRPO smoke prompts"
    )
    direct_case = binary_grpo_rewards(["<answer>A</answer>"], ["A"], [["A", "B"]]) == [1.0]
    if not direct_case:
        raise RuntimeError("direct-answer valid case failed")

    reward_audit: dict[str, list[Any]] = {"values": [], "format": [], "pre_answer_tokens": [], "direct": []}

    def binary_task_success_reward(completions, answer, allowed_labels, log_metric=None, **kwargs):
        rewards = binary_grpo_rewards(completions, answer, allowed_labels)
        diagnostics = [reasoning_diagnostics(text, tokenizer) for text in completions]
        formats = [float(parse_final_answer(text, set(labels)) is not None) for text, labels in zip(
            completions, allowed_labels
        )]
        reward_audit["values"].extend(rewards)
        reward_audit["format"].extend(formats)
        reward_audit["pre_answer_tokens"].extend(item["pre_answer_tokens"] for item in diagnostics)
        reward_audit["direct"].extend(float(item["is_direct_answer"]) for item in diagnostics)
        if log_metric is not None:
            log_metric("format_rate", sum(formats) / len(formats))
            log_metric("pre_answer_tokens_mean", sum(item["pre_answer_tokens"] for item in diagnostics) / len(diagnostics))
            log_metric("direct_answer_rate", sum(item["is_direct_answer"] for item in diagnostics) / len(diagnostics))
        return rewards

    class AuditedGRPOTrainer(GRPOTrainer):
        def __init__(self, *args, **kwargs):
            self.generation_event_steps: list[int] = []
            self.fresh_rollouts = 0
            self.fresh_generated_completion_tokens = 0
            super().__init__(*args, **kwargs)

        def _generate(self, prompts):
            result = super()._generate(prompts)
            if self.model.training:
                completion_ids = result[1]
                self.generation_event_steps.append(int(self.state.global_step))
                self.fresh_rollouts += len(completion_ids)
                self.fresh_generated_completion_tokens += sum(len(ids) for ids in completion_ids)
            return result

    cfg = config["training"]
    args = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"), learning_rate=cfg["learning_rate"], optim=cfg["optimizer"],
        lr_scheduler_type=cfg["lr_scheduler_type"], max_steps=config["smoke"]["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        steps_per_generation=cfg["steps_per_generation"], num_generations=cfg["num_generations"],
        num_iterations=cfg["num_iterations"], max_completion_length=cfg["max_completion_length"],
        temperature=cfg["temperature"], top_p=cfg["top_p"], repetition_penalty=cfg["repetition_penalty"],
        loss_type=cfg["loss_type"], epsilon=cfg["epsilon"], epsilon_high=cfg["epsilon_high"], beta=cfg["beta"],
        scale_rewards=cfg["scale_rewards"], mask_truncated_completions=cfg["mask_truncated_completions"],
        use_vllm=cfg["use_vllm"], bf16=cfg["bf16"], gradient_checkpointing=cfg["gradient_checkpointing"],
        max_grad_norm=cfg["max_grad_norm"], save_strategy="no", seed=cfg["seed"], logging_steps=1, report_to="none",
    )
    before = _adapter_state(model)
    trainer = AuditedGRPOTrainer(
        model=model, reward_funcs=binary_task_success_reward, args=args,
        train_dataset=dataset, processing_class=tokenizer,
    )
    if trainer.ref_model is not None or set(model.peft_config) != {"default"}:
        raise RuntimeError("beta=0 smoke unexpectedly created a reference model/adapter")
    result = trainer.train()
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    if not losses or any(not math.isfinite(float(loss)) for loss in losses):
        raise RuntimeError(f"missing or non-finite GRPO loss: {losses}")
    updated_count = _updated_adapter_parameters(before, model)
    if updated_count == 0:
        raise RuntimeError("GRPO smoke did not update the existing SFT LoRA")
    after = _adapter_state(model)
    if any(not torch.isfinite(tensor).all().item() for tensor in after.values()):
        raise RuntimeError("GRPO smoke produced a non-finite LoRA parameter")
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    events = len(trainer.generation_event_steps)
    if trainer.fresh_rollouts % cfg["num_generations"]:
        raise RuntimeError("fresh rollout count is not divisible by G")
    if trainer.fresh_rollouts != events * cfg["num_generations"]:
        raise RuntimeError(
            "single-process smoke did not produce exactly one prompt group and G rollouts per generation event"
        )
    counters = {
        "optimizer_steps": int(trainer.state.global_step),
        "fresh_generation_batches": events,
        "fresh_prompt_groups": trainer.fresh_rollouts // cfg["num_generations"],
        "fresh_rollouts": trainer.fresh_rollouts,
        "fresh_generated_completion_tokens": trainer.fresh_generated_completion_tokens,
        "generation_event_optimizer_steps": trainer.generation_event_steps,
    }
    schedule = infer_event_schedule(trainer.generation_event_steps, cfg["max_steps"])
    history_keys = {key for row in trainer.state.log_history for key in row}
    required_metrics = {"clip_ratio/low_mean", "clip_ratio/high_mean", "completions/clipped_ratio"}
    if not required_metrics <= history_keys:
        raise RuntimeError(f"objective/truncation metrics missing: {sorted(required_metrics - history_keys)}")
    metrics = {
        **result.metrics, "status": "passed", "trl": trl.__version__, "direct_answer_valid": direct_case,
        "updated_lora_parameter_count": updated_count, "adapter_parameters_finite": True,
        "adapter_names": sorted(model.peft_config),
        "reference_model_loaded": trainer.ref_model is not None, "reward_values_observed": sorted(set(reward_audit["values"])),
        "format_rate": sum(reward_audit["format"]) / len(reward_audit["format"]),
        "truncation_rate": next((row["completions/clipped_ratio"] for row in reversed(trainer.state.log_history)
                                 if "completions/clipped_ratio" in row), None),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0), "fresh_counters": counters,
        "resolved_sampler_semantics": schedule, "objective_metrics_present": sorted(required_metrics),
        "prompt_version": PROMPT_VERSION, "parser_version": PARSER_VERSION, "reward_version": REWARD_VERSION,
        "smoke_adapter_disposable": True,
    }
    _write_json(output_dir / "train_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/grpo.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
