#!/usr/bin/env bash
set -euo pipefail

# Public Hub files must remain downloadable without Xet/CAS authentication.
export HF_HUB_DISABLE_XET=1

uv run python -m medreason.data --config configs/data.yaml
