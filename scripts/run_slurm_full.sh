#!/bin/bash
#SBATCH --job-name=nemotron-nano-full   # <= 請改成你的 job name (識別用)
#SBATCH --partition=normal              #    全批次請使用 normal
#SBATCH --nodes=4                       # <= 請改成你的節點數量，normal partition 最少 2 個節點
#SBATCH --ntasks-per-node=1             #    每台節點的 task 數量
#SBATCH --gres=gpu:H200:8               #    每台節點的 GPU 數量
#SBATCH --cpus-per-task=104             #    每台節點的 CPU 數量
#SBATCH --mem=0                         #    設置為該節點所有記憶體
#SBATCH --time=2-00:00:00               # <= 請改成你的時間限制
#SBATCH --output=logs/slurm_full_%j.out #    請改成你的輸出日誌路徑
#SBATCH --error=logs/slurm_full_%j.err  #    請改成你的錯誤日誌路徑
#SBATCH --account=YOUR_ACCOUNT          #    請改成你的帳號或計畫代號

set -euo pipefail

# ==============================================================================
# 使用者定義輸入 (User defined Inputs) - 可透過執行腳本時傳入參數覆蓋
# ------------------------------------------------------------------------------
# 1. MODEL_NAME: 欲進行評測的模型名稱。可以是 Hugging Face 上的 Repo ID 
#                (例如 "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16") 或本地端絕對路徑。
MODEL_NAME=${1:-"nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"}

# 2. TP_SIZE (Tensor Parallel Size): 決定模型要在單一節點內切分給「幾張 GPU」同時運算。
#                                    若模型超過單卡 VRAM，請調高此數值。
TP_SIZE=${2:-1}

# 3. PP_SIZE (Pipeline Parallel Size): 管線平行處理大小。一般針對超大模型才需設定，預設為 1 即可。
PP_SIZE=${3:-1}

# 4. HF_UPLOAD_REPO: Hugging Face 上用來存放評測結果資料集的 Repo 路徑。
#                    請確保系統環境已登入 Hugging Face (執行過 `hf auth login`) 且具備寫入權限。
HF_UPLOAD_REPO=${4:-"whats2000/nemotron-nano-eval-logs-and-scores"}

# 5. 生成控制參數 (Generation Parameters):
#                MAX_TOKENS: 必須小於 MAX_MODEL_LEN (因為要預留輸入 prompt 的空間)
MAX_TOKENS=${5:-16384}
THINKING_START_TAG=${6:-"<think>"}
THINKING_END_TAG=${7:-"</think>"}

# 6. 伺服器啟動參數 (Server Parameters):
#    MAX_MODEL_LEN: 模型最大上下文長度 (會影響 vLLM 分配的 GPU 記憶體量)
MAX_MODEL_LEN=${8:-32768}

# 7. HF_VARIANT: 上傳到 Hugging Face 時的 variant 名稱，用於在同一個 Repo 下區分不同評測條件。
#               例如："default"、"high", "low" 等表示不同評測條件
HF_VARIANT=${9:-"default"}

# 8. 工作目錄 (Working Directory): 專案程式碼的根目錄絕對路徑。
#    留空則自動偵測，如自動偵測失敗再手動填寫。
WORK_DIR=""                                    # <= 選填，留空自動偵測

# ------------------------------------------------------------------------------
# ⚠️ 注意 (Notice):
# 其他如資料集設定 (dataset_paths)、System Prompt 或是較少更動的評測選項，
# 仍需要直接去修改 yaml 配置檔 (configs/config_slurm_full.yaml)。
# ==============================================================================

# ==============================================================================
# 自動偵測 WORK_DIR（請勿修改）
# ------------------------------------------------------------------------------
if [ -z "${WORK_DIR}" ]; then
    if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
        # SLURM 環境：使用 sbatch 命令下達時由 SLURM 記錄的路徑
        WORK_DIR="${SLURM_SUBMIT_DIR}"
    elif [ -n "${BASH_SOURCE[0]:-}" ] && [ "${BASH_SOURCE[0]}" != "$0" -o -f "${BASH_SOURCE[0]}" ]; then
        # 本機執行：從腳本本身位置向上一層推導
        WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        WORK_DIR="$(dirname "${WORK_DIR}")"
    else
        echo "ERROR: 無法自動偵測 WORK_DIR，請在上方使用者定義區填寫專案絕對路徑。" >&2
        exit 1
    fi
fi
# ==============================================================================

mkdir -p logs results

# 確保運算節點上有使用者本地二進制檔 (uv, hf) 於路徑中
export PATH="$HOME/.local/bin:$HOME/miniconda3/bin:$HOME/miniconda3/condabin:$PATH"

cd "${WORK_DIR}"

# 將 HF/模型快取導向 /work 以避免 /home 空間配額耗盡
export HF_HOME="/work/${USER}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export UV_CACHE_DIR="/work/${USER}/.cache/uv"
mkdir -p "${HF_HOME}"

# Token: 透過 sbatch --export 傳入；若無則自動讀取 token 快取檔案
if [[ -z "${HF_TOKEN:-}" ]]; then
    HF_TOKEN="$(cat "${HF_HOME}/token" 2>/dev/null || cat ~/.cache/huggingface/token 2>/dev/null || true)"
fi
: "${HF_TOKEN:?ERROR: HF_TOKEN not set. Run: hf auth login}"

export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"

VLLM_VENV="${WORK_DIR}/vllm-venv"
if [[ ! -x "${VLLM_VENV}/bin/python" ]]; then
    echo "ERROR: vllm venv not found at ${VLLM_VENV}" >&2
    exit 1
fi

# 計算動態變數
NNODES=${SLURM_NNODES:-1}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-8}
INSTANCES_PER_NODE=$(( GPUS_PER_NODE / (TP_SIZE * PP_SIZE) ))
DP_SIZE=$(( NNODES * INSTANCES_PER_NODE ))

# 環境資訊
GPU_MODEL=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | xargs)
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
GPU_MEMORY_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader -i 0 | awk '{print $1}')
GPU_MEMORY=$(echo "scale=1; $GPU_MEMORY_MIB / 1024" | bc 2>/dev/null || awk "BEGIN {printf \"%.1f\", $GPU_MEMORY_MIB/1024}")
CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i 0 | xargs)
PYTHON_VERSION=$(uv run python --version | awk '{print $2}')
TORCH_VERSION=$("${VLLM_VENV}/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")

# ==============================================================================
# 準備資料集 (Dataset Preparation)
# ------------------------------------------------------------------------------
DATASET_DIR="/work/${USER}/datasets"
mkdir -p "${DATASET_DIR}"

if [[ ! -d "${DATASET_DIR}/ikala__tmmluplus" ]]; then
    echo "正在下載 tmmluplus (test) ..."
    uv run twinkle-eval --download-dataset ikala/tmmluplus \
        --dataset-split test \
        --output-dir "${DATASET_DIR}"
fi

if [[ ! -d "${DATASET_DIR}/lianghsun__tw-legal-benchmark-v1" ]]; then
    echo "正在下載 tw-legal-benchmark-v1 (train) ..."
    uv run twinkle-eval --download-dataset lianghsun/tw-legal-benchmark-v1 \
        --dataset-split train \
        --output-dir "${DATASET_DIR}"
fi

if [[ ! -d "${DATASET_DIR}/cais__mmlu" ]]; then
    echo "正在下載 mmlu (test) ..."
    uv run twinkle-eval --download-dataset cais/mmlu \
        --dataset-split test \
        --output-dir "${DATASET_DIR}"
fi
# ==============================================================================

# 建立動態配置檔
CONFIG_RUN="configs/run_full_${SLURM_JOB_ID:-test}.yaml"
cp configs/config_slurm_full.yaml $CONFIG_RUN

# 確保 REPEAT_RUNS 佔位符被正確處理 (如果存在)
sed -i "s|PLACEHOLDER_MODEL_NAME|${MODEL_NAME}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_DATASET_DIR|${DATASET_DIR}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_GPU_MODEL|${GPU_MODEL}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_GPU_COUNT|${GPU_COUNT}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_GPU_MEMORY_GB|${GPU_MEMORY}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_CUDA_VERSION|${CUDA_VERSION}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_DRIVER_VERSION|${DRIVER_VERSION}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_TP_SIZE|${TP_SIZE}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_PP_SIZE|${PP_SIZE}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_PYTHON_VERSION|${PYTHON_VERSION}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_TORCH_VERSION|${TORCH_VERSION}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_NNODES|${NNODES}|g" $CONFIG_RUN
sed -i "s|PLACEHOLDER_REPEAT_RUNS|5|g" $CONFIG_RUN # Full script defaults to 5
sed -i "s|PLACEHOLDER_MAX_MODEL_LEN|${MAX_MODEL_LEN}|g" $CONFIG_RUN

if [ "$THINKING_START_TAG" = "null" ]; then
    sed -i "s|PLACEHOLDER_THINKING_START_TAG|null|g" $CONFIG_RUN
else
    sed -i "s|PLACEHOLDER_THINKING_START_TAG|\"${THINKING_START_TAG}\"|g" $CONFIG_RUN
fi

if [ "$THINKING_END_TAG" = "null" ]; then
    sed -i "s|PLACEHOLDER_THINKING_END_TAG|null|g" $CONFIG_RUN
else
    sed -i "s|PLACEHOLDER_THINKING_END_TAG|\"${THINKING_END_TAG}\"|g" $CONFIG_RUN
fi

sed -i "s|PLACEHOLDER_MAX_TOKENS|${MAX_TOKENS}|g" $CONFIG_RUN


echo "============================================="
echo " 開始完整生產評測 "
echo " 模型 (MODEL): $MODEL_NAME"
echo " 策略 (STRATEGY): TP=$TP_SIZE | PP=$PP_SIZE | 總節點數=$NNODES"
echo " 計算得出每節點實例數: $INSTANCES_PER_NODE (總平行工作數 DP=$DP_SIZE)"
echo "============================================="

# 在背景啟動多個本地 vLLM 伺服器
WRAPPER_SCRIPT="run_node_wrapper_${SLURM_JOB_ID:-test}.sh"
cat << 'EOF' > $WRAPPER_SCRIPT
#!/bin/bash
export INSTANCES_PER_NODE=$7
export RANK_OFFSET=$(( SLURM_NODEID * INSTANCES_PER_NODE ))
export WORLD_SIZE=$(( SLURM_NNODES * INSTANCES_PER_NODE ))
VLLM_VENV_PATH="${6}"

echo "[節點 $SLURM_NODEID] 開始啟動 $INSTANCES_PER_NODE 個 vLLM 實例..."

for i in $(seq 0 $((INSTANCES_PER_NODE - 1))); do
    GLOBAL_RANK=$(( RANK_OFFSET + i ))
    START_GPU=$(( i * $2 * $3 ))
    END_GPU=$(( START_GPU + $2 * $3 - 1 ))
    GPU_LIST=$(seq -s ',' $START_GPU $END_GPU)
    PORT=$(( 8000 + i ))
    BASE_URL="http://localhost:${PORT}/v1"
    
    LOCAL_CONFIG="${4%.yaml}_rank${GLOBAL_RANK}.yaml"
    cp "${4}" "$LOCAL_CONFIG"
    sed -i "s|PLACEHOLDER_BASE_URL|${BASE_URL}|g" "$LOCAL_CONFIG"
    
    (
        export CUDA_VISIBLE_DEVICES=$GPU_LIST
        export RANK=$GLOBAL_RANK
        export WORLD_SIZE=$WORLD_SIZE
        export TORCHINDUCTOR_CACHE_DIR="/tmp/inductor_${GLOBAL_RANK}"
        export TRITON_CACHE_DIR="/tmp/triton_${GLOBAL_RANK}"
        export VLLM_CACHE_ROOT="/tmp/vllm_cache_${GLOBAL_RANK}"
        export XDG_CACHE_HOME="/tmp/xdg_cache_${GLOBAL_RANK}"
        export TIKTOKEN_RS_CACHE_DIR="/tmp/tiktoken_rs_${GLOBAL_RANK}"
        export TMPDIR="/tmp/vllm_tmp_${GLOBAL_RANK}"

        mkdir -p "$TMPDIR" "$TIKTOKEN_RS_CACHE_DIR"
        # 加入啟動延遲錯開，防止動態通訊埠分配產生競爭 (Race Condition)
        # vLLM 內部會同時使用 `get_open_port()` 尋找 NCCL 初始化通訊埠，
        # 如果在同一個毫秒啟動，會導致多個實例拿到同一個通訊埠而引發 NCCL error。
        sleep $(( i * 10 ))

        MAX_VLLM_RETRIES=3
        VLLM_START_OK=0

        for attempt in $(seq 1 $MAX_VLLM_RETRIES); do
            echo "[節點 $SLURM_NODEID | Rank $GLOBAL_RANK] 啟動 vLLM (第 $attempt/$MAX_VLLM_RETRIES 次, GPUs: $GPU_LIST, Port: $PORT)..."
            "${VLLM_VENV_PATH}/bin/python" -m vllm.entrypoints.openai.api_server \
                --model "${1}" \
                --tensor-parallel-size "${2}" \
                --pipeline-parallel-size "${3}" \
                --port $PORT \
                --gpu-memory-utilization 0.90 \
                --max-model-len "${5}" \
                --limit-mm-per-prompt '{"image": 1}' \
                --trust-remote-code > "/tmp/vllm_${GLOBAL_RANK}.log" 2>&1 &
            VLLM_PID=$!

            WAIT_DEADLINE=$(( $(date +%s) + 3600 ))
            VLLM_READY=0
            while [ $(date +%s) -lt $WAIT_DEADLINE ]; do
                if ! kill -0 $VLLM_PID 2>/dev/null; then
                    echo "[節點 $SLURM_NODEID | Rank $GLOBAL_RANK] vLLM 程序已意外終止 (第 $attempt 次)。"
                    break
                fi
                if curl -s "${BASE_URL}/models" > /dev/null 2>&1; then
                    VLLM_READY=1
                    break
                fi
                sleep 5
            done

            if [ $VLLM_READY -eq 1 ]; then
                VLLM_START_OK=1
                break
            fi

            echo "[節點 $SLURM_NODEID | Rank $GLOBAL_RANK] vLLM 第 $attempt 次啟動失敗，正在終止程序..."
            kill $VLLM_PID 2>/dev/null || true
            wait $VLLM_PID 2>/dev/null || true

            if [ $attempt -lt $MAX_VLLM_RETRIES ]; then
                echo "[節點 $SLURM_NODEID | Rank $GLOBAL_RANK] 等待 30 秒後重試..."
                sleep 30
            fi
        done

        if [ $VLLM_START_OK -eq 0 ]; then
            echo "[節點 $SLURM_NODEID | Rank $GLOBAL_RANK] vLLM 重試 ${MAX_VLLM_RETRIES} 次均失敗，取消整個 SLURM Job 以防止其他節點無效 hanging..."
            # copy the vLLM log for inspection
            mkdir -p logs
            cp "/tmp/vllm_${GLOBAL_RANK}.log" "logs/vllm_node${SLURM_NODEID}_rank${GLOBAL_RANK}.log" 2>/dev/null || true
            scancel "${SLURM_JOB_ID}"
            exit 1
        fi

        echo "[節點 $SLURM_NODEID | Rank $GLOBAL_RANK] vLLM 就緒。開始評測 (本機寫入)..."
        uv run twinkle-eval --config "$LOCAL_CONFIG" --export json
        
        echo "[節點 $SLURM_NODEID | Rank $GLOBAL_RANK] 評測完成。"
        kill $VLLM_PID
        rm -f "$LOCAL_CONFIG"
    ) &
done

wait
EOF
chmod +x $WRAPPER_SCRIPT

echo "透過 srun 部署任務到各個節點..."

# 在啟動所有 worker 之前，由 head node 統一記錄並匯出固定時間戳。
# SLURM 會自動將已匯出的環境變數傳遞給所有 srun task，
# 確保所有 node/rank 使用相同的時間戳，避免跨分鐘邊界造成合併失敗。
RUN_TIMESTAMP=$(date +"%Y%m%d_%H%M")
export TWINKLE_EVAL_RUN_TIMESTAMP="${RUN_TIMESTAMP}"
echo "本次評測時間戳 (RUN_TIMESTAMP): ${RUN_TIMESTAMP}"

START_TIME=$(date +%s)

# srun 會自動在每一個分配到的節點上執行封裝腳本並設定 SLURM_NODEID
# --wait=0: 不在某一個 task 提前完成時終止其他仍在執行的 tasks
srun --ntasks="${NNODES}" --ntasks-per-node=1 --nodes="${NNODES}" --wait=0 \
    bash $WRAPPER_SCRIPT "${MODEL_NAME}" "${TP_SIZE}" "${PP_SIZE}" "${CONFIG_RUN}" "${MAX_MODEL_LEN}" "${VLLM_VENV}" "${INSTANCES_PER_NODE}" || true

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "完整評測完成，共耗時 $DURATION 秒。"

rm -f $CONFIG_RUN $WRAPPER_SCRIPT

# ==========================================
# 合併評測結果與上傳
# ==========================================
echo "等待檔案系統同步..."
sleep 5

# 直接使用本次 job 開始時統一設定的 RUN_TIMESTAMP，不再動態掃描。
# 這避免了多節點在不同分鐘啟動時產生不同時間戳、導致合併遺漏碎片的問題。
if [ -n "${RUN_TIMESTAMP:-}" ]; then
    # 清點實際寫出的 JSON shard 數是否符合預期 rank 數
    # 多節點腳本：預期 DP_SIZE (= NNODES × INSTANCES_PER_NODE) 個 shard
    EXPECTED_SHARDS=${DP_SIZE}
    # 同時支援兩種檔案格式：
    #   多 rank： results_{ts}_node*_rank*.json  (world_size > 1)
    #   單 rank： results_{ts}.json              (world_size = 1)
    ACTUAL_SHARDS=$(find results/ -maxdepth 1 -name "results_${RUN_TIMESTAMP}_node*_rank*.json" 2>/dev/null | wc -l)
    if [ "${ACTUAL_SHARDS}" -eq 0 ]; then
        # 回落檢查單節點無後綴格式
        if [ -f "results/results_${RUN_TIMESTAMP}.json" ]; then
            ACTUAL_SHARDS=1
        fi
    fi

    echo "=========================================="
    echo "Shard 清點：預期 ${EXPECTED_SHARDS} 個，實際找到 ${ACTUAL_SHARDS} 個 (Timestamp: ${RUN_TIMESTAMP})"

    if [ "${ACTUAL_SHARDS}" -eq 0 ]; then
        echo "❌ 找不到任何 shard 檔案，可能所有節點均評測失敗，中止合併。"
        exit 1
    elif [ "${ACTUAL_SHARDS}" -lt "${EXPECTED_SHARDS}" ]; then
        echo "⚠️  警告：僅收到 ${ACTUAL_SHARDS}/${EXPECTED_SHARDS} 個 shard，部分節點/rank 可能失敗，結果將不完整。"
    else
        echo "✅ Shard 數量符合預期，開始合併與上傳..."
    fi

    if [ -n "$HF_UPLOAD_REPO" ] && [ "$HF_UPLOAD_REPO" != "your-org/eval-logs" ]; then
        uv run twinkle-eval --finalize-results "${RUN_TIMESTAMP}" --hf-repo-id "${HF_UPLOAD_REPO}" --hf-variant "${HF_VARIANT}"
    else
        uv run twinkle-eval --finalize-results "${RUN_TIMESTAMP}"
    fi
else
    echo "⚠️ 找不到任何可供處理的評測結果，請確認各節點評測是否已完成並寫入 results/ 目錄。"
fi
