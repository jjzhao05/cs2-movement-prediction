#!/bin/bash
# One-time setup on a Delta login node: a venv with CUDA PyTorch and the training dependencies.
#
#     bash delta/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source delta/config.sh

module load "$PYTHON_MODULE" 2>/dev/null || echo "module $PYTHON_MODULE not found; using $(command -v python3)"
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
    || { echo "python3 >= 3.10 needed: set PYTHON_MODULE in delta/config.sh"; exit 1; }

[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip
pip install torch --index-url "$TORCH_INDEX"
pip install -r delta/requirements.txt
python -c 'import torch, numpy, scipy, sklearn; print("torch", torch.__version__, "CUDA build", torch.version.cuda)'
echo "venv ready: $VENV"
