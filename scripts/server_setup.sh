#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --locked --extra server-cu124

echo "Environment created. Run commands through uv:"
echo "  uv run python scripts/select_bfcl_subset.py --per-category 1"
echo "  uv run agent-serving-study list"
