#!/usr/bin/env bash
set -euo pipefail

uv run python -m medreason.freeze_protocol --config configs/grpo.yaml
