"""Phase 2 QLoRA SFT smoke and full training entry point."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

from medreason.prompt import render_prompt, render_sft_completion
from medreason.reward import parse_final_answer, reasoning_diagnostics, task_success


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_sft_record(sample: dict[str, Any], eos_token: str) -> dict[str, str]:
    """Convert an internal sample to TRL's prompt-completion dataset contract."""
    return {
        "prompt": render_prompt(sample),
        "completion": render_sft_completion(sample, eos_token),
    }


def audit_tokenized_example(example: dict[str, list[int]], tokenizer: Any) -> dict[str, int]:
    """Fail on the silent completion-only masking bugs called out by the SPEC."""
    input_ids = example["input_ids"]
    labels = example["labels"]
    if len(input_ids) != len(labels) or not labels:
        raise RuntimeError("tokenized input_ids/labels are empty or have different lengths")
    try:
        first_trainable = next(index for index, label in enumerate(labels) if label != -100)
    except StopIteration as error:
        raise RuntimeError("completion-only example has no trainable labels") from error
    if first_trainable == 0 or any(label != -100 for label in labels[:first_trainable]):
        raise RuntimeError("prompt positions are not fully masked")
    if any(label == -100 for label in labels[first_trainable:]):
        raise RuntimeError("completion contains unexpectedly masked positions")
    if labels[-1] != tokenizer.eos_token_id:
        raise RuntimeError("EOS is missing from the trainable labels")
    completion_text = tokenizer.decode(labels[first_trainable:], skip_special_tokens=False)
    if "<answer>" not in completion_text or "</answer>" not in completion_text:
        raise RuntimeError("answer tag is missing from the trainable labels")
    return {
        "sequence_tokens": len(input_ids),
        "masked_prompt_tokens": first_trainable,
        "trainable_completion_tokens": len(labels) - first_trainable,
    }


def _load_json_dataset(path: Path, tokenizer: Any):
    from datasets import load_dataset

    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {path}")
    dataset = load_dataset("json", data_files=str(path), split="train")
    return dataset.map(
        lambda sample: build_sft_record(sample, tokenizer.eos_token),
        remove_columns=dataset.column_names,
        desc=f"Rendering {path.name}",
    )


def _model_and_tokenizer(config: dict[str, Any]):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    lora_cfg = config["lora"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg["revision"],
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg["revision"],
        trust_remote_code=model_cfg["trust_remote_code"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("base model was not loaded in 4-bit")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            target_modules=lora_cfg["target_modules"],
        ),
    )
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names or any("lora_" not in name for name in trainable_names):
        raise RuntimeError("LoRA parameters must be the only trainable parameters")
    return model, tokenizer


def _adapter_state(model: Any) -> dict[str, Any]:
    from peft import get_peft_model_state_dict

    return {name: tensor.detach().float().cpu().clone() for name, tensor in get_peft_model_state_dict(model).items()}


def _updated_adapter_parameters(before: dict[str, Any], model: Any) -> int:
    from peft import get_peft_model_state_dict

    after = get_peft_model_state_dict(model)
    return sum(not before[name].equal(tensor.detach().float().cpu()) for name, tensor in after.items())


def _reload_adapter(config: dict[str, Any], adapter_dir: Path, is_trainable: bool):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    model_cfg = config["model"]
    quant_cfg = config["quantization"]
    base = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"],
        revision=model_cfg["revision"],
        trust_remote_code=model_cfg["trust_remote_code"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    return PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)


def evaluate_dev(model: Any, tokenizer: Any, samples: list[dict[str, Any]], config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    import torch

    generation_cfg = config["evaluation"]
    predictions_path = output_dir / "dev_predictions.jsonl"
    successes = []
    formats = []
    lengths = []
    truncations = []
    pre_answer_tokens = []
    direct_answers = []
    model.eval()
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            prompt = render_prompt(sample)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    do_sample=generation_cfg["do_sample"],
                    num_beams=generation_cfg["num_beams"],
                    max_new_tokens=generation_cfg["max_new_tokens"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            completion_ids = output_ids[0, inputs["input_ids"].shape[1] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
            allowed = set(sample["options"])
            parsed = parse_final_answer(completion, allowed)
            success = task_success(completion, sample["answer"], allowed)
            diagnostics = reasoning_diagnostics(completion, tokenizer)
            truncated = len(completion_ids) == generation_cfg["max_new_tokens"]
            formats.append(int(parsed is not None))
            successes.append(success)
            lengths.append(len(completion_ids))
            truncations.append(int(truncated))
            pre_answer_tokens.append(diagnostics["pre_answer_tokens"])
            direct_answers.append(int(diagnostics["is_direct_answer"]))
            handle.write(json.dumps({
                "id": sample["id"],
                "gold_answer": sample["answer"],
                "parsed_answer": parsed,
                "completion": completion,
                "completion_tokens": len(completion_ids),
                "truncated": truncated,
                "task_success": success,
                **diagnostics,
            }, ensure_ascii=False) + "\n")
    metrics = {
        "samples": len(samples),
        "strict_task_success_accuracy": statistics.fmean(successes),
        "format_rate": statistics.fmean(formats),
        "pre_answer_tokens_mean": statistics.fmean(pre_answer_tokens),
        "direct_answer_rate": statistics.fmean(direct_answers),
        "completion_tokens_mean": statistics.fmean(lengths),
        "completion_tokens_median": statistics.median(lengths),
        "truncation_rate": statistics.fmean(truncations),
    }
    _write_json(output_dir / "dev_metrics.json", metrics)
    return metrics


def run(config_path: Path, mode: str) -> dict[str, Any]:
    import torch
    import trl
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    if trl.__version__ != "1.10.0":
        raise RuntimeError(f"Expected trl==1.10.0, found {trl.__version__}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 2 requires a CUDA GPU with bfloat16 support")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = Path(config[mode]["output_dir"])
    adapter_dir = output_dir / "adapter"
    if adapter_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing adapter: {adapter_dir}")

    model, tokenizer = _model_and_tokenizer(config)
    train_dataset = _load_json_dataset(Path(config["data"]["train_path"]), tokenizer)
    if mode == "smoke":
        train_dataset = train_dataset.select(range(config["smoke"]["samples"]))
    training_cfg = config["training"]
    args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_length=training_cfg["max_length"],
        completion_only_loss=training_cfg["completion_only_loss"],
        packing=training_cfg["packing"],
        per_device_train_batch_size=training_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation_steps"],
        learning_rate=training_cfg["learning_rate"],
        num_train_epochs=training_cfg["num_train_epochs"],
        max_steps=config["smoke"]["max_steps"] if mode == "smoke" else -1,
        warmup_ratio=training_cfg["warmup_ratio"],
        lr_scheduler_type=training_cfg["lr_scheduler_type"],
        weight_decay=training_cfg["weight_decay"],
        bf16=training_cfg["bf16"],
        gradient_checkpointing=training_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=training_cfg["max_grad_norm"],
        seed=training_cfg["seed"],
        save_strategy="no",
        logging_steps=1,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_dataset, processing_class=tokenizer)
    label_audit = audit_tokenized_example(trainer.train_dataset[0], tokenizer)
    before = _adapter_state(model)
    result = trainer.train()
    train_loss = float(result.metrics["train_loss"])
    if not math.isfinite(train_loss):
        raise RuntimeError(f"non-finite train loss: {train_loss}")
    updated_count = _updated_adapter_parameters(before, model)
    if updated_count == 0:
        raise RuntimeError("SFT did not update any LoRA parameter")
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    metrics = {
        **result.metrics,
        "mode": mode,
        "trl": trl.__version__,
        "base_model": config["model"]["name_or_path"],
        "base_revision": config["model"]["revision"],
        "updated_lora_parameter_count": updated_count,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0),
        "label_audit": label_audit,
    }
    _write_json(output_dir / "train_metrics.json", metrics)

    del before, trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = _reload_adapter(config, adapter_dir, is_trainable=True)
    metrics["adapter_reloaded_trainable"] = any(parameter.requires_grad for parameter in reloaded.parameters())
    if not metrics["adapter_reloaded_trainable"]:
        raise RuntimeError("saved adapter did not reload as trainable")

    raw_dev = load_dataset("json", data_files=config["data"]["dev_path"], split="train")
    if mode == "smoke":
        sample = raw_dev[0]
        dev_metrics = evaluate_dev(reloaded, tokenizer, [sample], config, output_dir)
    else:
        dev_metrics = evaluate_dev(reloaded, tokenizer, list(raw_dev), config, output_dir)
    metrics["dev_metrics"] = dev_metrics
    _write_json(output_dir / "train_metrics.json", metrics)
    if mode == "full" and dev_metrics["format_rate"] < config["evaluation"]["format_rate_gate"]:
        raise RuntimeError(
            f"SFT format_rate {dev_metrics['format_rate']:.4f} is below "
            f"the required gate {config['evaluation']['format_rate_gate']:.4f}"
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.mode), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

