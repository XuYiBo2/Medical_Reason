"""Phase 4 formal MedQA scan and frozen probe/training-pool construction."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

import yaml

from medreason.prompt import PROMPT_VERSION, render_prompt, validate_generation_context
from medreason.reward import parse_final_answer, task_success
from medreason.train_grpo import PARSER_VERSION, REWARD_VERSION, _load_model_and_tokenizer
from medreason.train_sft import completion_was_truncated, _write_json


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def protocol_identity(protocol: dict[str, Any]) -> dict[str, Any]:
    """Fields that uniquely define scan rollouts and rewards."""
    return {
        "trl_version": protocol["trl_version"],
        "model": protocol["model"],
        "quantization": protocol["quantization"],
        "sft_adapter": protocol["sft_adapter"],
        "prompt": protocol["prompt"],
        "generation": protocol["generation"],
        "reward": protocol["reward"],
    }


def validate_frozen_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("phase") != "Phase 3 GRPO protocol freeze":
        raise ValueError("Formal Scan requires the Phase 3 frozen protocol")
    if protocol["prompt"]["version"] != PROMPT_VERSION:
        raise ValueError("frozen prompt version differs from the shared renderer")
    if protocol["reward"] != {
        "parser_version": PARSER_VERSION,
        "reward_version": REWARD_VERSION,
        "values": [0, 1],
    }:
        raise ValueError("frozen parser/reward contract differs from the implementation")
    generation = protocol["generation"]
    if generation["num_generations"] <= 1 or not generation["decode_skip_special_tokens"]:
        raise ValueError("invalid frozen generation contract")


def required_pool_size(protocol: dict[str, Any], minimum: int) -> int:
    return max(minimum, int(protocol["budget"]["planned_fresh_prompt_groups"]))


def scan_gate_counts(pool_size: int, per_stratum_target: int) -> dict[str, int]:
    return {
        "informative_required": pool_size + per_stratum_target,
        "noninformative_required": per_stratum_target,
    }


def next_scan_target(scanned: int, universe_size: int, initial: int, expand_step: int) -> int:
    """Finish the current fixed-size scan tranche after an interrupted run."""
    if scanned < initial:
        target = initial
    else:
        target = initial + ((scanned - initial) // expand_step + 1) * expand_step
    return min(universe_size, target)


def select_probe_and_pools(
    scan_rows: list[dict[str, Any]],
    samples_by_id: dict[str, dict[str, Any]],
    pool_size: int,
    per_stratum_target: int,
    probe_seed: int,
    pool_seed: int,
    full_universe_scanned: bool,
) -> dict[str, Any]:
    informative_ids = [row["sample_id"] for row in scan_rows if row["is_informative"]]
    noninformative_ids = [row["sample_id"] for row in scan_rows if not row["is_informative"]]
    if full_universe_scanned:
        per_stratum = min(per_stratum_target, len(informative_ids), len(noninformative_ids))
    else:
        per_stratum = per_stratum_target
        gates = scan_gate_counts(pool_size, per_stratum_target)
        if len(informative_ids) < gates["informative_required"] or len(noninformative_ids) < gates["noninformative_required"]:
            raise ValueError("scan must expand before probe and pools can be frozen")
    if per_stratum == 0:
        raise ValueError("both I0 strata are required for the persistence probe")
    probe_rng = random.Random(probe_seed)
    probe_informative = probe_rng.sample(informative_ids, per_stratum)
    probe_noninformative = probe_rng.sample(noninformative_ids, per_stratum)
    probe_ids = set(probe_informative + probe_noninformative)
    eligible_ids = [row["sample_id"] for row in scan_rows if row["sample_id"] not in probe_ids]
    informative_candidates = [
        row["sample_id"] for row in scan_rows if row["is_informative"] and row["sample_id"] not in probe_ids
    ]
    if len(informative_candidates) < pool_size or len(eligible_ids) < pool_size:
        raise ValueError("complete MedQA train does not contain enough eligible prompts for the frozen pool budget")
    info_rng = random.Random(pool_seed)
    random_rng = random.Random(pool_seed)
    informative_pool_ids = info_rng.sample(informative_candidates, pool_size)
    random_pool_ids = random_rng.sample(eligible_ids, pool_size)
    overlap = set(informative_pool_ids) & set(random_pool_ids)
    return {
        "probe": {
            "informative_ids": probe_informative,
            "noninformative_ids": probe_noninformative,
            "all_ids": probe_informative + probe_noninformative,
            "per_stratum": per_stratum,
            "target": per_stratum_target * 2,
            "actual": per_stratum * 2,
            "support": "exploratory_low_support" if per_stratum < 64 else "supported",
        },
        "informative_pool_ids": informative_pool_ids,
        "random_pool_ids": random_pool_ids,
        "informative_candidates_after_probe": len(informative_candidates),
        "eligible_count": len(eligible_ids),
        "pool_overlap_count": len(overlap),
        "pool_overlap_ratio": len(overlap) / pool_size,
        "informative_pool": [samples_by_id[sample_id] for sample_id in informative_pool_ids],
        "random_pool": [samples_by_id[sample_id] for sample_id in random_pool_ids],
    }


def _read_existing_scan(path: Path, expected_ids: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)) or ids != expected_ids[: len(ids)]:
        raise RuntimeError("existing scan.jsonl is not a valid prefix of the frozen scan order")
    return rows


def _scan_batch(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    protocol: dict[str, Any],
    start_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    import torch

    generation = protocol["generation"]
    context_limit = int(getattr(model.config, "max_position_embeddings"))
    for sample in samples:
        validate_generation_context(sample, tokenizer, generation["max_completion_length"], context_limit)
    prompts = [render_prompt(sample) for sample in samples]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    torch.manual_seed(seed + start_index)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=generation["num_generations"],
            temperature=generation["temperature"],
            top_p=generation["top_p"],
            repetition_penalty=generation["repetition_penalty"],
            max_new_tokens=generation["max_completion_length"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    completion_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
    group_size = generation["num_generations"]
    rows = []
    for offset, sample in enumerate(samples):
        first = offset * group_size
        group_ids = completion_ids[first : first + group_size]
        group_text = completions[first : first + group_size]
        allowed = set(sample["options"])
        rewards = [task_success(text, sample["answer"], allowed) for text in group_text]
        formats = [int(parse_final_answer(text, allowed) is not None) for text in group_text]
        lengths: list[int] = []
        truncations: list[int] = []
        for ids in group_ids:
            # Generated tensors are rectangular; remove right padding but retain a terminating EOS.
            eos_positions = (ids == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
            length = int(eos_positions[0].item()) + 1 if len(eos_positions) else len(ids)
            effective_ids = ids[:length]
            lengths.append(length)
            truncations.append(int(completion_was_truncated(
                effective_ids, generation["max_completion_length"], tokenizer.eos_token_id
            )))
        correct_count = sum(rewards)
        pass_rate = correct_count / group_size
        rows.append({
            "sample_id": sample["id"],
            "scan_index": start_index + offset,
            "num_rollouts": group_size,
            "correct_count": correct_count,
            "pass_rate": pass_rate,
            "reward_std_population": statistics.pstdev(rewards),
            "is_informative": 0.0 < pass_rate < 1.0,
            "format_rate": sum(formats) / len(formats),
            "truncation_rate": sum(truncations) / len(truncations),
            "mean_completion_tokens": statistics.fmean(lengths),
            "completion_tokens": sum(lengths),
        })
    return rows


def run(config_path: Path) -> dict[str, Any]:
    from datasets import load_dataset

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["probe"]["target"] != 2 * config["probe"]["per_stratum_target"]:
        raise ValueError("probe target must equal two symmetric stratum targets")
    protocol_path = Path(config["protocol_path"])
    if not protocol_path.is_file():
        raise FileNotFoundError("Formal Scan cannot run before outputs/protocol/frozen_protocol.yaml exists")
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    validate_frozen_protocol(protocol)
    raw = load_dataset("json", data_files=config["medqa_train_path"], split="train")
    samples = list(raw)
    if len({sample["id"] for sample in samples}) != len(samples):
        raise RuntimeError("MedQA train sample IDs must be unique")
    order = list(range(len(samples)))
    random.Random(config["scan"]["seed"]).shuffle(order)
    ordered_samples = [samples[index] for index in order]
    ordered_ids = [sample["id"] for sample in ordered_samples]
    samples_by_id = {sample["id"]: sample for sample in samples}
    output_dir = Path(config["output_dir"])
    scan_path = output_dir / "scan.jsonl"
    state_path = output_dir / "scan_state.json"
    frozen_outputs = (
        output_dir / "probe_ids.json",
        output_dir / "informative_pool.jsonl",
        output_dir / "random_pool.jsonl",
    )
    existing_frozen_outputs = [str(path) for path in frozen_outputs if path.exists()]
    if existing_frozen_outputs:
        raise FileExistsError(f"refusing to overwrite frozen Phase 4 outputs: {existing_frozen_outputs}")
    scan_state = {
        "frozen_protocol_identity": protocol_identity(protocol),
        "scan_seed": config["scan"]["seed"],
        "prompt_batch_size": config["scan"]["prompt_batch_size"],
        "medqa_train_size": len(samples),
    }
    if scan_path.exists() and not state_path.exists():
        raise RuntimeError("existing scan.jsonl has no protocol-bound scan_state.json")
    if state_path.exists():
        existing_state = json.loads(state_path.read_text(encoding="utf-8"))
        legacy_state = {key: value for key, value in scan_state.items() if key != "prompt_batch_size"}
        if existing_state == legacy_state:
            _write_json(state_path, scan_state)
        elif existing_state != scan_state:
            raise RuntimeError("existing Formal Scan belongs to a different frozen protocol or universe")
    else:
        _write_json(state_path, scan_state)
    scan_rows = _read_existing_scan(scan_path, ordered_ids)
    pool_size = required_pool_size(protocol, config["pools"]["minimum_size"])
    gates = scan_gate_counts(pool_size, config["probe"]["per_stratum_target"])
    model, tokenizer = _load_model_and_tokenizer({
        "model": protocol["model"], "quantization": protocol["quantization"],
        "initialization": {"sft_adapter_dir": protocol["sft_adapter"]["path"]},
    }, is_trainable=False)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    with scan_path.open("a", encoding="utf-8", newline="\n") as handle:
        while True:
            informative = sum(row["is_informative"] for row in scan_rows)
            noninformative = len(scan_rows) - informative
            if informative >= gates["informative_required"] and noninformative >= gates["noninformative_required"]:
                break
            if len(scan_rows) == len(ordered_samples):
                break
            next_target = next_scan_target(
                len(scan_rows), len(ordered_samples),
                config["scan"]["initial_samples"], config["scan"]["expand_step"],
            )
            print(f"Formal Scan tranche: {len(scan_rows)} -> {next_target}", flush=True)
            while len(scan_rows) < next_target:
                scan_index = len(scan_rows)
                batch_end = min(next_target, scan_index + config["scan"]["prompt_batch_size"])
                new_rows = _scan_batch(
                    model, tokenizer, ordered_samples[scan_index:batch_end], protocol,
                    scan_index, config["scan"]["seed"],
                )
                scan_rows.extend(new_rows)
                for row in new_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if len(scan_rows) % 40 == 0 or len(scan_rows) == next_target:
                    informative_so_far = sum(item["is_informative"] for item in scan_rows)
                    print(
                        f"Formal Scan progress: {len(scan_rows)}/{len(ordered_samples)}; "
                        f"informative={informative_so_far}",
                        flush=True,
                    )
    try:
        selection = select_probe_and_pools(
            scan_rows, samples_by_id, pool_size, config["probe"]["per_stratum_target"],
            config["probe"]["seed"], config["pools"]["seed"], len(scan_rows) == len(ordered_samples),
        )
    except ValueError as error:
        if len(scan_rows) != len(ordered_samples):
            raise
        negative_summary = {
            "status": "failed_insufficient_pool_support",
            "reason": str(error),
            "frozen_protocol_identity": protocol_identity(protocol),
            "medqa_train_size": len(samples),
            "scan_samples": len(scan_rows),
            "scan_rollouts": sum(row["num_rollouts"] for row in scan_rows),
            "scan_completion_tokens": sum(row["completion_tokens"] for row in scan_rows),
            "informative_count": sum(row["is_informative"] for row in scan_rows),
            "noninformative_count": sum(not row["is_informative"] for row in scan_rows),
            "required_pool_size": pool_size,
            "scan_trajectories_used_for_training": False,
        }
        _write_json(output_dir / "summary.json", negative_summary)
        raise RuntimeError(str(error)) from error
    _write_json(output_dir / "probe_ids.json", selection["probe"])
    _write_jsonl(output_dir / "informative_pool.jsonl", selection["informative_pool"])
    _write_jsonl(output_dir / "random_pool.jsonl", selection["random_pool"])
    summary = {
        "status": "passed",
        "frozen_protocol_identity": protocol_identity(protocol),
        "medqa_train_size": len(samples),
        "scan_samples": len(scan_rows),
        "scan_rollouts": sum(row["num_rollouts"] for row in scan_rows),
        "scan_completion_tokens": sum(row["completion_tokens"] for row in scan_rows),
        "informative_count": sum(row["is_informative"] for row in scan_rows),
        "noninformative_count": sum(not row["is_informative"] for row in scan_rows),
        "pool_size": pool_size,
        "informative_candidates_after_probe": selection["informative_candidates_after_probe"],
        "eligible_count": selection["eligible_count"],
        "informative_pool_size": len(selection["informative_pool_ids"]),
        "random_pool_size": len(selection["random_pool_ids"]),
        "pool_overlap_count": selection["pool_overlap_count"],
        "pool_overlap_ratio": selection["pool_overlap_ratio"],
        "probe": selection["probe"],
        "scan_trajectories_used_for_training": False,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/difficulty.yaml"))
    args = parser.parse_args()
    print(json.dumps(run(args.config), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
