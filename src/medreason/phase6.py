"""Phase 6 checkpoint selection, persistence probe, and unified final evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
from pathlib import Path
from typing import Any

import yaml

from medreason.difficulty import _scan_batch, protocol_identity, validate_frozen_protocol
from medreason.prompt import render_prompt, validate_generation_context
from medreason.reward import parse_final_answer, reasoning_diagnostics, task_success
from medreason.train_grpo import _load_model_and_tokenizer
from medreason.train_sft import completion_was_truncated, _write_json


def select_checkpoint(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the frozen lexicographic MedQA-dev selection priority."""
    if not candidates:
        raise ValueError("checkpoint selection requires candidates")
    return max(candidates, key=lambda item: (
        item["metrics"]["strict_task_success_accuracy"],
        item["metrics"]["format_rate"],
        -item["metrics"]["truncation_rate"],
        -item["metrics"]["mean_completion_tokens"],
    ))


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def _spearman_with_ties(left: list[float], right: list[float]) -> float:
    left_ranks, right_ranks = _average_ranks(left), _average_ranks(right)
    left_mean, right_mean = statistics.fmean(left_ranks), statistics.fmean(right_ranks)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left_ranks, right_ranks))
    left_scale = sum((x - left_mean) ** 2 for x in left_ranks) ** 0.5
    right_scale = sum((y - right_mean) ** 2 for y in right_ranks) ** 0.5
    return numerator / (left_scale * right_scale) if left_scale and right_scale else float("nan")


def persistence_metrics(step0: list[dict[str, Any]], current: list[dict[str, Any]]) -> dict[str, float]:
    initial = {row["sample_id"]: row for row in step0}
    now = {row["sample_id"]: row for row in current}
    if set(initial) != set(now):
        raise ValueError("persistence steps must contain exactly the frozen probe IDs")
    informative0 = {sample_id for sample_id, row in initial.items() if row["is_informative"]}
    noninformative0 = set(initial) - informative0
    informativet = {sample_id for sample_id, row in now.items() if row["is_informative"]}
    if not informative0 or not noninformative0:
        raise ValueError("persistence probe requires both I0 strata")
    p_info = len(informative0 & informativet) / len(informative0)
    p_noninfo = len(noninformative0 & informativet) / len(noninformative0)
    union = informative0 | informativet
    ordered_ids = list(initial)
    rho = _spearman_with_ties(
        [initial[sample_id]["pass_rate"] for sample_id in ordered_ids],
        [now[sample_id]["pass_rate"] for sample_id in ordered_ids],
    )
    return {
        "p_informative_given_i0_1": p_info,
        "p_informative_given_i0_0": p_noninfo,
        "persistence_gap": p_info - p_noninfo,
        "spearman_pass_rate": rho if math.isfinite(rho) else None,
        "informative_retention": len(informative0 & informativet) / len(informative0),
        "informative_jaccard": len(informative0 & informativet) / len(union),
    }


def paired_bootstrap(
    reference: list[int], candidate: list[int], resamples: int, confidence_level: float, seed: int,
) -> dict[str, Any]:
    import numpy as np

    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired bootstrap requires equal non-empty prediction arrays")
    differences = np.asarray(candidate, dtype=float) - np.asarray(reference, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(differences), size=(resamples, len(differences)))
    deltas = differences[draws].mean(axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "delta": float(differences.mean()),
        "confidence_interval": [float(np.quantile(deltas, alpha)), float(np.quantile(deltas, 1.0 - alpha))],
        "confidence_level": confidence_level,
        "resamples": resamples,
        "label": "single-training-seed conditional CI",
    }


def error_analysis_candidates(predictions: dict[str, list[dict[str, Any]]], limit: int = 30) -> list[dict[str, Any]]:
    by_model = {name: {row["sample_id"]: row for row in rows} for name, rows in predictions.items()}
    sample_ids = [row["sample_id"] for row in predictions["E3_difficulty"]]
    buckets: dict[str, list[str]] = {
        "E1_wrong_E3_correct": [], "E1_correct_E3_wrong": [],
        "E2_E3_disagreement": [], "format_or_truncation_abnormal": [],
    }
    for sample_id in sample_ids:
        e1, e2, e3 = (by_model[name][sample_id] for name in ("E1_sft", "E2_random", "E3_difficulty"))
        if not e1["task_success"] and e3["task_success"]:
            buckets["E1_wrong_E3_correct"].append(sample_id)
        if e1["task_success"] and not e3["task_success"]:
            buckets["E1_correct_E3_wrong"].append(sample_id)
        if e2["task_success"] != e3["task_success"]:
            buckets["E2_E3_disagreement"].append(sample_id)
        if any(row["parsed_answer"] is None or row["truncated"] for row in (e1, e2, e3)):
            buckets["format_or_truncation_abnormal"].append(sample_id)
    selected: list[tuple[str, str]] = []
    seen = set()
    while len(selected) < limit:
        added = False
        for category, ids in buckets.items():
            while ids and ids[0] in seen:
                ids.pop(0)
            if ids and len(selected) < limit:
                sample_id = ids.pop(0)
                selected.append((category, sample_id))
                seen.add(sample_id)
                added = True
        if not added:
            break
    return [{
        "category": category,
        "sample_id": sample_id,
        "gold_answer": by_model["E3_difficulty"][sample_id]["gold_answer"],
        "models": {
            name: {key: by_model[name][sample_id][key] for key in (
                "task_success", "parsed_answer", "completion", "completion_tokens", "truncated"
            )}
            for name in ("E1_sft", "E2_random", "E3_difficulty")
        },
    } for category, sample_id in selected]


def _load_eval_model(protocol: dict[str, Any], adapter_path: Path | None):
    if adapter_path is not None:
        return _load_model_and_tokenizer({
            "model": protocol["model"], "quantization": protocol["quantization"],
            "initialization": {"sft_adapter_dir": str(adapter_path)},
        }, is_trainable=False)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg, quant_cfg = protocol["model"], protocol["quantization"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["name_or_path"], revision=model_cfg["revision"],
        trust_remote_code=model_cfg["trust_remote_code"], padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name_or_path"], revision=model_cfg["revision"],
        trust_remote_code=model_cfg["trust_remote_code"],
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
            bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
            bnb_4bit_compute_dtype=torch.bfloat16,
        ), dtype=torch.bfloat16, device_map={"": 0},
    )
    model.config.use_cache = True
    model.eval()
    return model, tokenizer


def _release() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _read_cached_eval(path: Path, protocol: dict[str, Any], sample_count: int) -> dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if metrics["frozen_protocol_identity"] != protocol_identity(protocol) or metrics["samples"] != sample_count:
        raise RuntimeError(f"cached evaluation belongs to a different protocol/dataset: {path.parent}")
    if not (path.parent / "predictions.jsonl").exists():
        raise RuntimeError(f"cached metrics has no predictions: {path.parent}")
    return metrics


def evaluate_dataset(
    model: Any, tokenizer: Any, samples: list[dict[str, Any]], protocol: dict[str, Any],
    batch_size: int, output_dir: Path,
) -> dict[str, Any]:
    import torch

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.jsonl"
    if metrics_path.exists():
        return _read_cached_eval(metrics_path, protocol, len(samples))
    generation = protocol["generation"]
    deterministic = protocol["deterministic_evaluation"]
    context_limit = int(model.config.max_position_embeddings)
    predictions: list[dict[str, Any]] = []
    model.eval()
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        for sample in batch:
            validate_generation_context(sample, tokenizer, generation["max_completion_length"], context_limit)
        inputs = tokenizer(
            [render_prompt(sample) for sample in batch], return_tensors="pt", padding=True, add_special_tokens=False
        ).to("cuda")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, do_sample=False, num_beams=1,
                repetition_penalty=deterministic["repetition_penalty"],
                max_new_tokens=generation["max_completion_length"],
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        for sample, ids, text in zip(batch, completion_ids, completions):
            eos_positions = (ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            length = int(eos_positions[0].item()) + 1 if len(eos_positions) else len(ids)
            effective_ids = ids[:length]
            allowed = set(sample["options"])
            diagnostics = reasoning_diagnostics(text, tokenizer)
            predictions.append({
                "sample_id": sample["id"], "gold_answer": sample["answer"],
                "parsed_answer": parse_final_answer(text, allowed), "completion": text,
                "completion_tokens": length,
                "truncated": completion_was_truncated(
                    effective_ids, generation["max_completion_length"], tokenizer.eos_token_id
                ),
                "task_success": task_success(text, sample["answer"], allowed), **diagnostics,
            })
        if len(predictions) % (batch_size * 10) == 0 or len(predictions) == len(samples):
            print(f"Evaluation progress: {len(predictions)}/{len(samples)} -> {output_dir}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    lengths = [row["completion_tokens"] for row in predictions]
    metrics = {
        "samples": len(predictions),
        "strict_task_success_accuracy": statistics.fmean(row["task_success"] for row in predictions),
        "format_rate": statistics.fmean(row["parsed_answer"] is not None for row in predictions),
        "pre_answer_tokens_mean": statistics.fmean(row["pre_answer_tokens"] for row in predictions),
        "direct_answer_rate": statistics.fmean(row["is_direct_answer"] for row in predictions),
        "mean_completion_tokens": statistics.fmean(lengths),
        "median_completion_tokens": statistics.median(lengths),
        "truncation_rate": statistics.fmean(row["truncated"] for row in predictions),
        "generation": {"do_sample": False, "num_beams": 1,
                       "max_new_tokens": generation["max_completion_length"],
                       "repetition_penalty": deterministic["repetition_penalty"]},
        "frozen_protocol_identity": protocol_identity(protocol),
    }
    _write_json(metrics_path, metrics)
    return metrics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_probe(
    model: Any, tokenizer: Any, probe_samples: list[dict[str, Any]], protocol: dict[str, Any],
    batch_size: int, seed: int, output_path: Path,
) -> list[dict[str, Any]]:
    if output_path.exists():
        rows = _read_jsonl(output_path)
        if {row["sample_id"] for row in rows} != {sample["id"] for sample in probe_samples}:
            raise RuntimeError(f"cached persistence output has the wrong IDs: {output_path}")
        return rows
    rows = []
    for start in range(0, len(probe_samples), batch_size):
        rows.extend(_scan_batch(model, tokenizer, probe_samples[start : start + batch_size], protocol, start, seed))
        if len(rows) % (batch_size * 4) == 0 or len(rows) == len(probe_samples):
            print(f"Persistence progress: {len(rows)}/{len(probe_samples)} -> {output_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def checkpoint_and_persistence(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(Path(config["protocol_path"]).read_text(encoding="utf-8"))
    validate_frozen_protocol(protocol)
    batch_size = config["execution"]["prompt_batch_size"]
    dev_samples = _read_jsonl(Path(config["datasets"]["medqa_dev"]))
    output_dir = Path(config["output_dir"]) / "checkpoint_selection"
    persistence_dir = Path(config["persistence_output_dir"])
    steps = config["checkpoint_steps"]

    # Step 0 is evaluated once from immutable theta0 and shared by both curves.
    step0_dir = output_dir / "sft_step_0"
    if (step0_dir / "metrics.json").exists():
        sft_metrics = _read_cached_eval(step0_dir / "metrics.json", protocol, len(dev_samples))
    else:
        model, tokenizer = _load_eval_model(protocol, Path(config["sft_adapter_path"]))
        sft_metrics = evaluate_dataset(model, tokenizer, dev_samples, protocol, batch_size, step0_dir)
        del model, tokenizer
        _release()

    branch_candidates: dict[str, list[dict[str, Any]]] = {"random": [], "difficulty": []}
    persistence_rows: dict[int, list[dict[str, Any]]] = {}
    scan_by_id = {row["sample_id"]: row for row in _read_jsonl(Path(config["scan_path"]))}
    probe = json.loads(Path(config["probe_ids_path"]).read_text(encoding="utf-8"))
    probe_ids = probe["all_ids"]
    if any(sample_id not in scan_by_id for sample_id in probe_ids):
        raise RuntimeError("Formal Scan does not contain every frozen probe ID")
    step0_probe = [scan_by_id[sample_id] for sample_id in probe_ids]
    train_by_id = {row["id"]: row for row in _read_jsonl(Path(config["medqa_train_path"]))}
    probe_samples = [train_by_id[sample_id] for sample_id in probe_ids]
    persistence_state = {
        "frozen_protocol_identity": protocol_identity(protocol),
        "probe_ids": probe_ids,
        "seed": config["execution"]["persistence_seed"],
        "prompt_batch_size": batch_size,
    }
    persistence_state_path = persistence_dir / "state.json"
    if persistence_state_path.exists():
        if json.loads(persistence_state_path.read_text(encoding="utf-8")) != persistence_state:
            raise RuntimeError("cached persistence probe belongs to a different protocol or probe set")
    else:
        _write_json(persistence_state_path, persistence_state)

    for branch, root_key in (("random", "random_checkpoint_root"), ("difficulty", "difficulty_checkpoint_root")):
        for step in steps:
            checkpoint = Path(config[root_key]) / f"checkpoint-{step}"
            if not checkpoint.is_dir():
                raise FileNotFoundError(f"missing formal checkpoint: {checkpoint}")
            eval_dir = output_dir / branch / f"step_{step}"
            eval_cached = (eval_dir / "metrics.json").exists()
            probe_path = persistence_dir / f"probe_step_{step}.jsonl"
            probe_needed = branch == "difficulty" and not probe_path.exists()
            if eval_cached and not probe_needed:
                metrics = _read_cached_eval(eval_dir / "metrics.json", protocol, len(dev_samples))
                if branch == "difficulty":
                    persistence_rows[step] = _read_jsonl(probe_path)
            else:
                model, tokenizer = _load_eval_model(protocol, checkpoint)
                metrics = evaluate_dataset(model, tokenizer, dev_samples, protocol, batch_size, eval_dir)
                if branch == "difficulty":
                    persistence_rows[step] = _run_probe(
                        model, tokenizer, probe_samples, protocol, batch_size,
                        config["execution"]["persistence_seed"], probe_path,
                    )
                del model, tokenizer
                _release()
            branch_candidates[branch].append({"step": step, "checkpoint": str(checkpoint), "metrics": metrics})

    selections = {
        branch: {
            "step_0_sft_metrics": sft_metrics,
            "candidates": candidates,
            "selected": select_checkpoint(candidates),
            "selection_dataset": "MedQA dev only",
            "selection_priority": [
                "higher strict_task_success_accuracy", "higher format_rate",
                "lower truncation_rate", "lower mean_completion_tokens",
            ],
        }
        for branch, candidates in branch_candidates.items()
    }
    selections["status"] = "passed"
    _write_json(output_dir / "summary.json", selections)

    persistence_summary: dict[str, Any] = {
        "status": "passed",
        "probe_samples": len(step0_probe),
        "probe_rollouts_per_step": len(step0_probe) * protocol["generation"]["num_generations"],
        "step_0_source": "Formal Scan; no re-rollout",
        "steps": {"0": {**persistence_metrics(step0_probe, step0_probe), "generated_tokens": 0}},
        "frozen_protocol_identity": protocol_identity(protocol),
    }
    for step in steps:
        rows = persistence_rows.get(step) or _read_jsonl(persistence_dir / f"probe_step_{step}.jsonl")
        persistence_summary["steps"][str(step)] = {
            **persistence_metrics(step0_probe, rows),
            "generated_tokens": sum(row["completion_tokens"] for row in rows),
        }
    persistence_summary["total_new_probe_rollouts"] = (
        len(step0_probe) * protocol["generation"]["num_generations"] * len(steps)
    )
    persistence_summary["total_new_probe_generated_tokens"] = sum(
        item["generated_tokens"] for key, item in persistence_summary["steps"].items() if key != "0"
    )
    _write_json(persistence_dir / "summary.json", persistence_summary)
    return {"checkpoint_selection": selections, "persistence": persistence_summary}


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise RuntimeError(f"duplicate prediction IDs: {path}")
    return rows


def final_evaluation(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol = yaml.safe_load(Path(config["protocol_path"]).read_text(encoding="utf-8"))
    validate_frozen_protocol(protocol)
    selection_path = Path(config["output_dir"]) / "checkpoint_selection" / "summary.json"
    selections = json.loads(selection_path.read_text(encoding="utf-8"))
    models: dict[str, Path | None] = {
        "E0_base": None,
        "E1_sft": Path(config["sft_adapter_path"]),
        "E2_random": Path(selections["random"]["selected"]["checkpoint"]),
        "E3_difficulty": Path(selections["difficulty"]["selected"]["checkpoint"]),
    }
    final_root = Path(config["output_dir"]) / "final"
    all_metrics: dict[str, dict[str, Any]] = {}
    for model_name, adapter_path in models.items():
        pending = [name for name in config["datasets"] if name != "medqa_dev"
                   and not (final_root / model_name / name / "metrics.json").exists()]
        model = tokenizer = None
        if pending:
            model, tokenizer = _load_eval_model(protocol, adapter_path)
        all_metrics[model_name] = {}
        for dataset_name, dataset_path in config["datasets"].items():
            if dataset_name == "medqa_dev":
                continue
            samples = _read_jsonl(Path(dataset_path))
            metrics = evaluate_dataset(
                model, tokenizer, samples, protocol, config["execution"]["prompt_batch_size"],
                final_root / model_name / dataset_name,
            ) if model is not None else _read_cached_eval(
                final_root / model_name / dataset_name / "metrics.json", protocol, len(samples)
            )
            all_metrics[model_name][dataset_name] = metrics
        if model is not None:
            del model, tokenizer
            _release()

    bootstrap_cfg = config["bootstrap"]
    bootstrap_results: dict[str, Any] = {}
    for dataset_name in (name for name in config["datasets"] if name != "medqa_dev"):
        predictions = {
            model_name: _load_predictions(final_root / model_name / dataset_name / "predictions.jsonl")
            for model_name in models
        }
        reference_ids = [row["sample_id"] for row in predictions["E3_difficulty"]]
        aligned = {}
        for model_name, rows in predictions.items():
            by_id = {row["sample_id"]: row for row in rows}
            if set(by_id) != set(reference_ids):
                raise RuntimeError(f"models evaluated different examples for {dataset_name}")
            aligned[model_name] = [by_id[sample_id]["task_success"] for sample_id in reference_ids]
        bootstrap_results[dataset_name] = {
            "E3_minus_E1": paired_bootstrap(
                aligned["E1_sft"], aligned["E3_difficulty"], bootstrap_cfg["resamples"],
                bootstrap_cfg["confidence_level"], bootstrap_cfg["seed"],
            ),
            "E3_minus_E2": paired_bootstrap(
                aligned["E2_random"], aligned["E3_difficulty"], bootstrap_cfg["resamples"],
                bootstrap_cfg["confidence_level"], bootstrap_cfg["seed"],
            ),
        }
    medqa_predictions = {
        model_name: _load_predictions(final_root / model_name / "medqa_test" / "predictions.jsonl")
        for model_name in models
    }
    error_candidates = error_analysis_candidates(medqa_predictions, limit=30)
    error_path = final_root / "medqa_test_error_analysis_candidates.jsonl"
    with error_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in error_candidates:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    difficulty_summary = json.loads(Path(config["difficulty_summary_path"]).read_text(encoding="utf-8"))
    random_train = json.loads(Path(config["random_train_metrics_path"]).read_text(encoding="utf-8"))
    difficulty_train = json.loads(Path(config["difficulty_train_metrics_path"]).read_text(encoding="utf-8"))
    persistence_summary = json.loads((Path(config["persistence_output_dir"]) / "summary.json").read_text(encoding="utf-8"))
    summary = {
        "status": "passed",
        "selected_checkpoints": {key: None if value is None else str(value) for key, value in models.items()},
        "metrics": all_metrics,
        "paired_bootstrap": bootstrap_results,
        "medqa_test_error_analysis_candidates": len(error_candidates),
        "compute_accounting": {
            "scan": {
                "prompts": difficulty_summary["scan_samples"],
                "rollouts": difficulty_summary["scan_rollouts"],
                "generated_tokens": difficulty_summary["scan_completion_tokens"],
            },
            "E2_random_grpo": random_train["fresh_counters"],
            "E3_difficulty_grpo": difficulty_train["fresh_counters"],
            "persistence_probe": {
                "rollouts": persistence_summary["total_new_probe_rollouts"],
                "generated_tokens": persistence_summary["total_new_probe_generated_tokens"],
            },
        },
        "frozen_protocol_identity": protocol_identity(protocol),
    }
    _write_json(final_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    parser.add_argument("--checkpoint-and-persistence", action="store_true")
    parser.add_argument("--final-evaluation", action="store_true")
    args = parser.parse_args()
    if args.checkpoint_and_persistence == args.final_evaluation:
        parser.error("choose exactly one Phase 6 action")
    payload = checkpoint_and_persistence(args.config) if args.checkpoint_and_persistence else final_evaluation(args.config)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
