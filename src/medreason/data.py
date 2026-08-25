"""Dataset normalization and deterministic Phase 1 preparation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from medreason.prompt import encoded_length, render_prompt, render_sft_completion, validate_generation_context


LABELS = tuple("ABCDEFGHIJ")
GOLD_LEAKAGE_RE = re.compile(
    r"^\s*(?:Ans\.\s*is|Answer:|Correct\s+answer\s+is|The\s+correct\s+option\s+is)\s*['\"]?([A-D])['\"]?\s*[.:;-]?\s*",
    flags=re.IGNORECASE,
)


def validate_sample(sample: Mapping[str, Any], require_explanation: bool = False) -> None:
    required = {"id", "source", "split", "question", "options", "answer", "explanation", "subject"}
    missing = required - sample.keys()
    if missing:
        raise ValueError(f"sample is missing fields: {sorted(missing)}")
    if not isinstance(sample["id"], str) or not sample["id"].strip():
        raise ValueError("sample id must be a non-empty string")
    if not isinstance(sample["question"], str) or not sample["question"].strip():
        raise ValueError("question must be a non-empty string")
    options = sample["options"]
    if not isinstance(options, dict) or len(options) < 2:
        raise ValueError("options must be an ordered dictionary with at least two entries")
    expected_labels = list(LABELS[: len(options)])
    if list(options) != expected_labels or any(not isinstance(value, str) or not value.strip() for value in options.values()):
        raise ValueError(f"options must have contiguous labels {expected_labels} and non-empty text")
    if sample["answer"] not in options:
        raise ValueError("answer must be one of the option labels")
    if require_explanation and (
        not isinstance(sample["explanation"], str) or not sample["explanation"].strip()
    ):
        raise ValueError("SFT sample explanation must be non-empty")


def clean_gold_leakage(explanation: str, gold_label: str) -> str:
    """Remove only an explicit gold-label phrase at the start of an explanation."""
    match = GOLD_LEAKAGE_RE.match(explanation)
    if match is None or match.group(1).upper() != gold_label:
        return explanation.strip()
    return explanation[match.end() :].strip()


def normalize_medmcqa(row: Mapping[str, Any], split: str) -> dict[str, Any]:
    cop = row["cop"]
    if isinstance(cop, str) and cop.lower() in "abcd":
        answer = cop.upper()
    elif isinstance(cop, int) and 0 <= cop < 4:
        answer = LABELS[cop]
    else:
        raise ValueError(f"invalid MedMCQA cop: {cop!r}")
    sample = {
        "id": str(row["id"]),
        "source": "medmcqa",
        "split": split,
        "question": str(row["question"]).strip(),
        "options": {label: str(row[key]).strip() for label, key in zip(LABELS[:4], ("opa", "opb", "opc", "opd"))},
        "answer": answer,
        "explanation": clean_gold_leakage(str(row.get("exp") or ""), answer),
        "subject": str(row.get("subject_name") or "").strip() or None,
    }
    validate_sample(sample)
    return sample


def _stable_id(source: str, split: str, index: int, question: str) -> str:
    digest = hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]
    return f"{source}:{split}:{index}:{digest}"


def normalize_medqa(row: Mapping[str, Any], split: str, index: int) -> dict[str, Any]:
    raw_options = row["options"]
    options = dict(raw_options) if isinstance(raw_options, Mapping) else dict(raw_options)
    sample = {
        "id": _stable_id("medqa", split, index, str(row["question"])),
        "source": "medqa",
        "split": split,
        "question": str(row["question"]).strip(),
        "options": {str(label).upper(): str(text).strip() for label, text in options.items()},
        "answer": str(row["answer_idx"]).upper(),
        "explanation": None,
        "subject": str(row.get("meta_info") or "").strip() or None,
    }
    validate_sample(sample)
    if len(sample["options"]) != 4:
        raise ValueError("MedQA source must contain exactly four options")
    return sample


def normalize_mmlu_pro(row: Mapping[str, Any], split: str, index: int) -> dict[str, Any]:
    options = {LABELS[i]: str(text).strip() for i, text in enumerate(row["options"])}
    answer = str(row.get("answer") or LABELS[int(row["answer_index"])]).upper()
    sample = {
        "id": f"mmlu_pro:{split}:{row.get('question_id', index)}",
        "source": "mmlu_pro",
        "split": split,
        "question": str(row["question"]).strip(),
        "options": options,
        "answer": answer,
        "explanation": None,
        "subject": str(row["category"]).strip(),
    }
    validate_sample(sample)
    return sample


def exact_dedup_key(sample: Mapping[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
    return sample["question"], tuple(sample["options"].items())


def exact_deduplicate(samples: Iterable[dict[str, Any]], forbidden_keys: set | None = None) -> list[dict[str, Any]]:
    seen = set() if forbidden_keys is None else set(forbidden_keys)
    result = []
    for sample in samples:
        key = exact_dedup_key(sample)
        if key not in seen:
            seen.add(key)
            result.append(sample)
    return result


def stratified_sample(samples: Sequence[dict[str, Any]], size: int, seed: int, field: str = "subject") -> list[dict[str, Any]]:
    """Proportionally sample strata using largest-remainder allocation."""
    if size > len(samples):
        raise ValueError(f"requested {size} samples from only {len(samples)} rows")
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[sample[field]].append(sample)
    quotas = {key: size * len(group) / len(samples) for key, group in groups.items()}
    counts = {key: math.floor(quota) for key, quota in quotas.items()}
    remaining = size - sum(counts.values())
    order = sorted(groups, key=lambda key: (-(quotas[key] - counts[key]), str(key)))
    for key in order[:remaining]:
        counts[key] += 1

    rng = random.Random(seed)
    selected = []
    for key in sorted(groups, key=str):
        group = list(groups[key])
        rng.shuffle(group)
        selected.extend(group[: counts[key]])
    rng.shuffle(selected)
    return selected


def split_medmcqa_validation(
    sft_candidates: Sequence[dict[str, Any]],
    eval_candidates: Sequence[dict[str, Any]],
    dev_size: int,
    seed: int,
    forbidden_eval_keys: set,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select SFT dev under SFT filters; keep final-eval candidates under MCQA validity only."""
    sft_candidates = exact_deduplicate(sft_candidates)
    if len(sft_candidates) < dev_size:
        raise ValueError(f"not enough SFT-eligible MedMCQA validation rows: {len(sft_candidates)} < {dev_size}")
    rng = random.Random(seed)
    shuffled_sft = list(sft_candidates)
    rng.shuffle(shuffled_sft)
    sft_dev = shuffled_sft[:dev_size]

    excluded = set(forbidden_eval_keys)
    excluded.update(exact_dedup_key(sample) for sample in sft_dev)
    final_eval_candidates = exact_deduplicate(eval_candidates, forbidden_keys=excluded)
    rng.shuffle(final_eval_candidates)
    return sft_dev, final_eval_candidates


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_distribution_id(spec: Mapping[str, Any]) -> str:
    """Return the repository that actually hosts downloadable dataset files."""
    return str(spec.get("distribution_id", spec["id"]))


def prepare_data(config_path: Path) -> dict[str, Any]:
    """Download, normalize, split, length-filter, deduplicate, and write Phase 1 data."""
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = config["seed"]
    specs = config["datasets"]
    hub = HfApi()
    resolved_revisions = {
        name: hub.dataset_info(dataset_distribution_id(spec), revision=spec["revision"]).sha
        for name, spec in specs.items()
    }
    output_dir = Path(config["output_dir"])
    tokenizer_cfg = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_cfg["name_or_path"],
        revision=tokenizer_cfg["revision"],
        trust_remote_code=tokenizer_cfg["trust_remote_code"],
    )
    if tokenizer.eos_token is None:
        raise ValueError("tokenizer must define eos_token")

    medmcqa_spec = specs["medmcqa"]
    raw_train = load_dataset(
        medmcqa_spec["id"], revision=resolved_revisions["medmcqa"], split=medmcqa_spec["splits"]["train"]
    )
    raw_validation = load_dataset(
        medmcqa_spec["id"], revision=resolved_revisions["medmcqa"], split=medmcqa_spec["splits"]["validation"]
    )
    sft_candidates = []
    for row in raw_train:
        if row["choice_type"] != "single":
            continue
        try:
            sample = normalize_medmcqa(row, "train")
        except ValueError:
            continue
        if not sample["explanation"]:
            continue
        if encoded_length(tokenizer, sample["explanation"]) > config["lengths"]["explanation_max_tokens"]:
            continue
        sequence = render_prompt(sample) + render_sft_completion(sample, tokenizer.eos_token)
        if encoded_length(tokenizer, sequence) <= config["lengths"]["sft_max_tokens"]:
            sft_candidates.append(sample)
    sft_candidates = exact_deduplicate(sft_candidates)
    sft_train = stratified_sample(sft_candidates, config["samples"]["sft_train"], seed)

    validation_candidates = []
    validation_sft_candidates = []
    for row in raw_validation:
        if row["choice_type"] != "single":
            continue
        try:
            sample = normalize_medmcqa(row, "validation")
        except ValueError:
            continue
        validation_candidates.append(sample)
        if sample["explanation"] and encoded_length(tokenizer, sample["explanation"]) <= config["lengths"]["explanation_max_tokens"]:
            sequence = render_prompt(sample) + render_sft_completion(sample, tokenizer.eos_token)
            if encoded_length(tokenizer, sequence) <= config["lengths"]["sft_max_tokens"]:
                validation_sft_candidates.append(sample)
    dev_size = config["samples"]["sft_dev"]
    eval_size = config["samples"]["medmcqa_eval"]
    sft_dev, medmcqa_eval_candidates = split_medmcqa_validation(
        validation_sft_candidates,
        validation_candidates,
        dev_size,
        seed,
        forbidden_eval_keys={exact_dedup_key(sample) for sample in sft_train},
    )

    medqa_spec = specs["medqa"]
    medqa_splits = {}
    for split, source_split in medqa_spec["splits"].items():
        raw = load_dataset(
            dataset_distribution_id(medqa_spec),
            medqa_spec.get("config_name"),
            revision=resolved_revisions["medqa"],
            split=source_split,
        )
        normalized = [normalize_medqa(row, split, index) for index, row in enumerate(raw)]
        medqa_splits[split] = exact_deduplicate(normalized)

    training_keys = {exact_dedup_key(sample) for sample in (*sft_train, *medqa_splits["train"])}
    medmcqa_eval = exact_deduplicate(medmcqa_eval_candidates, forbidden_keys=training_keys)[:eval_size]
    if len(medmcqa_eval) != eval_size:
        raise ValueError("not enough MedMCQA final-eval rows after cross-dataset train/eval exact dedup")
    medqa_splits["test"] = exact_deduplicate(medqa_splits["test"], forbidden_keys=training_keys)

    mmlu_spec = specs["mmlu_pro"]
    raw_mmlu = load_dataset(
        mmlu_spec["id"], revision=resolved_revisions["mmlu_pro"], split=mmlu_spec["splits"]["test"]
    )
    mmlu_health = [
        normalize_mmlu_pro(row, "test", index)
        for index, row in enumerate(raw_mmlu)
        if str(row["category"]).lower() == mmlu_spec["category"].lower()
    ]
    mmlu_health = exact_deduplicate(mmlu_health)
    mmlu_health = exact_deduplicate(mmlu_health, forbidden_keys=training_keys)

    context_limit = int(getattr(tokenizer, "model_max_length"))
    completion_cap = config["lengths"]["frozen_max_completion_length"]
    for split_rows in (*medqa_splits.values(), mmlu_health):
        for sample in split_rows:
            validate_generation_context(sample, tokenizer, completion_cap, context_limit)

    outputs = {
        "sft_train.jsonl": sft_train,
        "sft_dev.jsonl": sft_dev,
        "medmcqa_eval.jsonl": medmcqa_eval,
        "medqa_train.jsonl": medqa_splits["train"],
        "medqa_dev.jsonl": medqa_splits["dev"],
        "medqa_test.jsonl": medqa_splits["test"],
        "mmlu_pro_health_test.jsonl": mmlu_health,
    }
    for filename, rows in outputs.items():
        _write_jsonl(output_dir / filename, rows)
    ids = [sample["id"] for rows in outputs.values() for sample in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("internal sample IDs are not globally unique")
    summary = {
        "seed": seed,
        "requested_dataset_revisions": {name: spec["revision"] for name, spec in specs.items()},
        "resolved_dataset_revisions": resolved_revisions,
        "dataset_distributions": {name: dataset_distribution_id(spec) for name, spec in specs.items()},
        "dataset_configs": {name: spec.get("config_name") for name, spec in specs.items()},
        "eligibility_counts": {
            "medmcqa_train_sft": len(sft_candidates),
            "medmcqa_validation_sft": len(validation_sft_candidates),
            "medmcqa_validation_mcqa": len(validation_candidates),
        },
        "counts": {filename: len(rows) for filename, rows in outputs.items()},
        "prompt_serialization": "plain_text_v1",
        "tokenizer_revision": tokenizer_cfg["revision"],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    print(json.dumps(prepare_data(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
