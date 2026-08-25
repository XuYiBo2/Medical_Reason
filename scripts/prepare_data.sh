#!/usr/bin/env bash
set -euo pipefail

uv run python -m medreason.data --config configs/data.yaml

