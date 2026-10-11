#!/bin/bash
# Provision a freshly rented GPU box (vast.ai, PyTorch template, A100 40GB+).
#
#   git clone <repo> && cd <repo>/extsum
#   cp .env.example .env && nano .env        # HF_TOKEN (+ OPENAI_* if needed)
#   bash deploy/vast/setup.sh
#
# Installs the package with the modern vLLM (needed for gemma-4 / qwen-3.5),
# fetches the sentence-tokenizer data and prints the GPU. Safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# vast.ai images ship a venv that non-login shells do not activate.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f /venv/main/bin/activate ]; then
    # shellcheck disable=SC1091
    . /venv/main/bin/activate
fi

if [ ! -f "$ROOT/.env" ]; then
    echo "No .env found. Create it first: cp .env.example .env  (then add HF_TOKEN)" >&2
    exit 1
fi

pip install -U pip
pip install -e ".[vllm,endpoint]"

# vLLM upgrades torch (and its CUDA build) but leaves the image's torchaudio
# behind; transformers then imports the stale torchaudio and vLLM fails with
# "PyTorch and TorchAudio were compiled with different CUDA versions". Nothing
# here uses audio, so drop it.
pip uninstall -y torchaudio 2>/dev/null || true

python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

python - <<'PY'
import torch, vllm
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | vllm {vllm.__version__}")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"gpu: {p.name}  {p.total_memory / 1e9:.0f}GB")
PY

echo
echo "ready. Typical next steps:"
echo "  extsum list"
echo "  EXPERIMENTS=billsum bash deploy/vast/sweep.sh            # all local models"
echo "  extsum run -e billsum -m qwen35_4b --limit 5              # 5-document smoke test"
