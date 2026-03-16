#!/bin/bash
# =============================================================================
# setup_vllm_venv.sh  —  建立評估腳本所使用的 vLLM 虛擬環境。
#
# 用法：  bash scripts/setup_vllm_venv.sh
#
# 此虛擬環境將使用 uv + Python 3.12 建立於 <repo-root>/vllm-venv，
# 並安裝包含 CUDA 附加套件的 vLLM。在提交任何 SLURM 評估任務前，請先執行此腳本一次。
# =============================================================================

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM_VENV="${WORK_DIR}/vllm-venv"

# 確保可使用使用者本機二進位檔 (uv)
export PATH="$HOME/.local/bin:$HOME/miniconda3/bin:$HOME/miniconda3/condabin:$PATH"

# 如果尚未安裝 uv，則進行安裝
if ! command -v uv &>/dev/null; then
    echo "[設定] 找不到 uv — 正在透過 pip 安裝 ..."
    pip install --user uv
fi

echo "[設定] 正在 ${VLLM_VENV} 建立 vllm-venv (Python 3.12) ..."
uv venv --python 3.12 "${VLLM_VENV}"

echo "[設定] 正在安裝 vLLM (這可能需要幾分鐘) ..."
uv pip install --python "${VLLM_VENV}" vllm

echo "[設定] 正在驗證安裝 ..."
"${VLLM_VENV}/bin/python" -c "import vllm; print(f'  vLLM {vllm.__version__} 正常')"
"${VLLM_VENV}/bin/python" -c "import torch;  print(f'  PyTorch {torch.__version__} 正常')"
"${VLLM_VENV}/bin/python" -c "import ray;    print(f'  Ray {ray.__version__} 正常')"

echo "[設定] 完成。vllm-venv 已在 ${VLLM_VENV} 準備就緒"
