#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install bfcl-eval "sglang[all]"

echo "Environment created. Verify CUDA/SGLang compatibility, then run:"
echo "  source .venv/bin/activate"
echo "  python scripts/select_bfcl_subset.py"
echo "  agent-serving-study list"

