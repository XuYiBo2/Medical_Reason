#!/usr/bin/env bash
set -euo pipefail

uv run python -m medreason.train_full_grpo --config configs/full_grpo.yaml --compare-budgets
