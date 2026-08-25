#!/usr/bin/env bash
set -euo pipefail

uv run python -m medreason.phase0 qlora-smoke \
  --config configs/phase0.yaml \
  --output-dir outputs/phase0/qlora_smoke

