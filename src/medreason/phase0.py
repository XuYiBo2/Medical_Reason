"""Phase 0 gates for the locked TRL API and the 4-bit QLoRA runtime."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import platform
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml


SFT_REQUIRED_FIELDS = {
    "max_length",
    "completion_only_loss",
}

GRPO_REQUIRED_FIELDS = {
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def check_trl_api(output_path: Path) -> dict[str, Any]:
    """Fail if the installed TRL is not the API frozen by the SPEC."""
    import trl
    from trl import GRPOConfig, SFTConfig

    if trl.__version__ != "1.10.0":
        raise RuntimeError(f"Expected trl==1.10.0, found {trl.__version__}")

    sft_fields = {field.name for field in fields(SFTConfig)}
    grpo_fields = {field.name for field in fields(GRPOConfig)}
    missing_sft = sorted(SFT_REQUIRED_FIELDS - sft_fields)
    missing_grpo = sorted(GRPO_REQUIRED_FIELDS - grpo_fields)
    if missing_sft or missing_grpo:
        raise RuntimeError(
            f"TRL API mismatch; missing SFT fields={missing_sft}, GRPO fields={missing_grpo}"
        )

    # The formal run uses steps_per_generation, never generation_batch_size.
    grpo = GRPOConfig(
        output_dir=str(output_path.parent / "grpo_api_check"),
        bf16=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        steps_per_generation=4,
        generation_batch_size=None,
        num_generations=4,
        num_iterations=2,
        loss_type="dapo",
        epsilon=0.20,
        epsilon_high=0.28,
        beta=0.0,
        scale_rewards="group",
        max_completion_length=256,
        mask_truncated_completions=True,
        use_vllm=False,
        report_to="none",
    )
    # TRL derives this after validating that only steps_per_generation was supplied.
    if grpo.steps_per_generation != 4 or grpo.generation_batch_size != 4:
        raise RuntimeError(
            "Unexpected generation batch semantics: "
            f"steps={grpo.steps_per_generation}, batch={grpo.generation_batch_size}"
        )

    report = {
        "status": "passed",
        "python": platform.python_version(),
        "trl": trl.__version__,
        "sft_required_fields": sorted(SFT_REQUIRED_FIELDS),
        "grpo_required_fields": sorted(GRPO_REQUIRED_FIELDS),
        "derived_generation_batch_size": grpo.generation_batch_size,
        "grpo_init_signature": str(inspect.signature(GRPOConfig)),
    }
    _write_json(output_path, report)
    return report


def run_qlora_smoke(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Exercise 4-bit load, LoRA update, adapter save/reload, and generation."""
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 0 QLoRA smoke requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not support bfloat16")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    lora_cfg = config["lora"]
    smoke_cfg = config["smoke"]
    torch.manual_seed(smoke_cfg["seed"])
    torch.cuda.manual_seed_all(smoke_cfg["seed"])

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {output_dir}")
    adapter_dir = output_dir / "adapter"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    load_kwargs = {
        "revision": model_cfg["revision"],
        "trust_remote_code": model_cfg["trust_remote_code"],
        "quantization_config": bnb_config,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
    }
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name_or_path"], **{
        "revision": model_cfg["revision"],
        "trust_remote_code": model_cfg["trust_remote_code"],
    })
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **load_kwargs)
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("Transformers did not load the base model in 4-bit")
    resolved_revision = getattr(model.config, "_commit_hash", None)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM",
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        target_modules=lora_cfg["target_modules"],
    ))

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise RuntimeError("Expected LoRA parameters to be the only trainable parameters")
    before = {name: parameter.detach().float().cpu().clone() for name, parameter in trainable}

    batch = tokenizer(
        smoke_cfg["prompt"],
        return_tensors="pt",
        truncation=True,
        max_length=smoke_cfg["max_input_tokens"],
    ).to("cuda")
    optimizer = torch.optim.AdamW((parameter for _, parameter in trainable), lr=smoke_cfg["learning_rate"])
    model.train()
    loss = model(**batch, labels=batch["input_ids"]).loss
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke loss: {loss.item()}")
    loss.backward()
    if not any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for _, parameter in trainable):
        raise RuntimeError("No non-zero LoRA gradient was produced")
    optimizer.step()
    updated_names = [
        name for name, parameter in trainable
        if not torch.equal(before[name], parameter.detach().float().cpu())
    ]
    if not updated_names:
        raise RuntimeError("Optimizer step did not update any LoRA parameter")
    loss_value = loss.detach().float().item()

    model.save_pretrained(adapter_dir, safe_serialization=True)
    del optimizer, model, trainable, before, loss
    gc.collect()
    torch.cuda.empty_cache()

    base = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **load_kwargs)
    reloaded = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    if not any(parameter.requires_grad for parameter in reloaded.parameters()):
        raise RuntimeError("Reloaded adapter is not trainable")
    reloaded.eval()
    with torch.no_grad():
        generated = reloaded.generate(**batch, max_new_tokens=2, do_sample=False)
    if generated.shape[1] <= batch["input_ids"].shape[1]:
        raise RuntimeError("Reloaded model did not generate a continuation")

    report = {
        "status": "passed",
        "model": model_cfg["name_or_path"],
        "requested_revision": model_cfg["revision"],
        "resolved_revision": resolved_revision,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "cuda_memory_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0),
        "loss": loss_value,
        "updated_lora_parameter_count": len(updated_names),
        "adapter_reloaded_trainable": True,
        "generation_new_tokens": generated.shape[1] - batch["input_ids"].shape[1],
    }
    _write_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    api_parser = subparsers.add_parser("api-check")
    api_parser.add_argument("--output", type=Path, default=Path("outputs/phase0/api_check.json"))
    smoke_parser = subparsers.add_parser("qlora-smoke")
    smoke_parser.add_argument("--config", type=Path, default=Path("configs/phase0.yaml"))
    smoke_parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase0/qlora_smoke"))
    args = parser.parse_args()

    if args.command == "api-check":
        report = check_trl_api(args.output)
    else:
        report = run_qlora_smoke(args.config, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
