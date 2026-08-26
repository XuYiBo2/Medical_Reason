"""Re-evaluate an existing SFT adapter with predefined generation-only diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from medreason.data import configure_huggingface_downloads
from medreason.train_sft import _reload_adapter, _write_json, evaluate_dev


def select_generation_candidate(results: list[dict[str, Any]], format_rate_gate: float) -> float | None:
    """Select the first predefined penalty that passes the format gate."""
    for result in results:
        if result["metrics"]["format_rate"] >= format_rate_gate:
            return result["repetition_penalty"]
    return None


def run(config_path: Path, adapter_dir: Path, output_dir: Path) -> dict[str, Any]:
    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    configure_huggingface_downloads()
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"SFT adapter not found: {adapter_dir}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = _reload_adapter(config, adapter_dir, is_trainable=False)
    raw_dev = load_dataset("json", data_files=config["data"]["dev_path"], split="train")
    samples = list(raw_dev)

    results = []
    for penalty in config["evaluation"]["repetition_penalty_diagnostic"]:
        variant_name = f"repetition_penalty_{penalty:.2f}".replace(".", "p")
        metrics = evaluate_dev(
            model,
            tokenizer,
            samples,
            config,
            output_dir / variant_name,
            generation_overrides={"repetition_penalty": penalty},
        )
        results.append({"repetition_penalty": penalty, "metrics": metrics})
        torch.cuda.empty_cache()

    gate = config["evaluation"]["format_rate_gate"]
    selected = select_generation_candidate(results, gate)
    summary = {
        "adapter_dir": str(adapter_dir),
        "format_rate_gate": gate,
        "candidate_order": config["evaluation"]["repetition_penalty_diagnostic"],
        "results": results,
        "selected_repetition_penalty": selected,
        "status": "passed" if selected is not None else "no_candidate_passed",
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/sft/adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sft/reeval"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.adapter_dir, args.output_dir), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

