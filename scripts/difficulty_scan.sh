#!/usr/bin/env bash
set -euo pipefail

export HF_HUB_DISABLE_XET=1
uv run python -m medreason.difficulty --config configs/difficulty.yaml
