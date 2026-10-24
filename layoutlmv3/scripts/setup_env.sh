#!/usr/bin/env bash
# Bootstrap the LayoutLMv3 environment on a fresh Linux GPU machine, following layoutlmv3/README.md.
#
# Deviations from the README (on purpose):
#   * clones YOUR fork (set REPO_URL) instead of microsoft/unilm
#   * installs torch BEFORE requirements.txt (timm would otherwise pull a different torch first)
#   * pins protobuf==3.20.3 (tensorboard 2.7 / datasets 1.15 break with protobuf>=4)
#
# Target GPUs: Ampere (A100 / A6000 / RTX 3090 / A10 ...). torch 1.10+cu111 ships kernels up to sm_86
# and has NO native kernels for Ada (sm_89) or Hopper (sm_90).
#
# Usage:  bash setup_env.sh            (run from anywhere)
#         REPO_URL=... WORKDIR=/workspace bash setup_env.sh
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/pn-minh/unilm.git}"
WORKDIR="${WORKDIR:-$HOME}"
ENV_NAME="${ENV_NAME:-layoutlmv3}"

# 1. conda (install Miniconda if missing)
set +u
if ! command -v conda >/dev/null 2>&1; then
  if [ ! -d "$HOME/miniconda3" ]; then
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
  fi
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

# Accept Anaconda ToS on newer Miniconda releases (conda >= 25.x)
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

# 2. env (README: python 3.7)
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y --name "$ENV_NAME" python=3.7
fi
conda activate "$ENV_NAME"
set -u

# 3. code
mkdir -p "$WORKDIR"
if [ ! -d "$WORKDIR/unilm/.git" ]; then
  git clone "$REPO_URL" "$WORKDIR/unilm"
fi
cd "$WORKDIR/unilm/layoutlmv3"

# 4. torch / torchvision (README: 1.10.0+cu111 / 0.11.1+cu111)
pip install "setuptools==59.5.0" "wheel<0.40.0"
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# 5. python deps
DS_BUILD_OPS=0 pip install -r requirements.txt
pip install "protobuf==3.20.3" "setuptools==59.5.0" "huggingface-hub==0.4.0"

# 6. detectron2 (only needed for PubLayNet / object_detection; prebuilt wheel for cu111 + torch1.10)
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html

# 7. this repo (provides the `layoutlmft` package)
pip install -e .

# 8. smoke test
python - <<'EOF'
import torch, transformers, datasets, timm
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
print("transformers", transformers.__version__, "| datasets", datasets.__version__, "| timm", timm.__version__)
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0), "| capability", torch.cuda.get_device_capability(0))
    print("arch list in this torch build:", torch.cuda.get_arch_list())
    x = torch.randn(64, 64, device="cuda"); print("matmul ok:", (x @ x).sum().item() == (x @ x).sum().item())
import layoutlmft
from layoutlmft.models.layoutlmv3 import LayoutLMv3ForTokenClassification  # noqa: F401
import detectron2; print("detectron2", detectron2.__version__)
EOF
echo "Done. Activate with: conda activate $ENV_NAME   (repo: $WORKDIR/unilm/layoutlmv3)"

