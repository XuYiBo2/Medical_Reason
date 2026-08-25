#!/usr/bin/env bash
set -euo pipefail

uv run python -m medreason.phase0 api-check --output outputs/phase0/api_check.json

