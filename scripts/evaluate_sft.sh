#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET=1
uv run python -m medreason.evaluate_sft \
  --config configs/sft.yaml \
  --adapter-dir outputs/sft/adapter \
  --output-dir outputs/sft/reeval

