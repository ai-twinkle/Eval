#!/usr/bin/env bash
# Local Breeze-ASR Evaluation — 在 jaren-x570 本地推論
#
# 前置條件：
#   1. breeze_asr_server.py 已部署到 jaren-x570
#   2. 模型已下載（首次啟動會自動從 HuggingFace 下載）
#
# 使用方式：
#   bash scripts/run_local_asr_eval.sh        # 跑全部（ASR-25 + ASR-26）
#   bash scripts/run_local_asr_eval.sh asr25  # 只跑 Breeze-ASR-25
#   bash scripts/run_local_asr_eval.sh asr26  # 只跑 Breeze-ASR-26

set -e
cd "$(dirname "$0")/.."

MODEL_FILTER="${1:-all}"
LINUX_HOST="${ASR_LINUX_HOST:?ERROR: set ASR_LINUX_HOST to your GPU server IP/hostname}"
ASR25_PORT=8765
ASR26_PORT=8766

wait_for_server() {
  local url="$1"
  local label="$2"
  echo "  Waiting for $label..."
  for i in $(seq 1 60); do
    if curl -sf "$url/health" >/dev/null 2>&1; then
      echo "  ✅ $label is ready"
      return 0
    fi
    sleep 5
  done
  echo "  ❌ $label did not start within 5 minutes"
  return 1
}

run_config() {
  local config="$1"
  local label="$2"
  echo ""
  echo "═══════════════════════════════════════════════════════"
  echo "  ▶  $label"
  echo "     config: $config"
  echo "═══════════════════════════════════════════════════════"
  maiagent-eval --config "$config" --export json
}

# ── Breeze-ASR-25 ─────────────────────────────────────────────────────────────
if [[ "$MODEL_FILTER" == "asr25" || "$MODEL_FILTER" == "all" ]]; then
  echo ""
  echo "▶ 啟動 Breeze-ASR-25 server on $LINUX_HOST:$ASR25_PORT ..."
  ssh "$LINUX_HOST" "
    pkill -f 'breeze_asr_server.py.*8765' 2>/dev/null || true
    sleep 1
    ASR_MODEL=MediaTek-Research/Breeze-ASR-25 \
    ASR_MODEL_CACHE=/opt/models/hub/llm/mediatek \
    nohup python3 ~/breeze_asr_server.py --port $ASR25_PORT \
      > /tmp/breeze_asr25.log 2>&1 &
    echo started
  "
  wait_for_server "http://$LINUX_HOST:$ASR25_PORT" "Breeze-ASR-25"

  run_config "configs/local/breeze_asr25_nan.yaml"   "Breeze-ASR-25 × 台語（Local）"
  run_config "configs/local/breeze_asr25_zh_tw.yaml" "Breeze-ASR-25 × 國語（Local）"

  echo "  Stopping Breeze-ASR-25 server..."
  ssh "$LINUX_HOST" "pkill -f 'breeze_asr_server.py.*8765' 2>/dev/null || true"
fi

# ── Breeze-ASR-26 ─────────────────────────────────────────────────────────────
if [[ "$MODEL_FILTER" == "asr26" || "$MODEL_FILTER" == "all" ]]; then
  echo ""
  echo "▶ 啟動 Breeze-ASR-26 server on $LINUX_HOST:$ASR26_PORT ..."
  ssh "$LINUX_HOST" "
    pkill -f 'breeze_asr_server.py.*8766' 2>/dev/null || true
    sleep 1
    ASR_MODEL=MediaTek-Research/Breeze-ASR-26 \
    ASR_MODEL_CACHE=/opt/models/hub/llm/mediatek \
    nohup python3 ~/breeze_asr_server.py --port $ASR26_PORT \
      > /tmp/breeze_asr26.log 2>&1 &
    echo started
  "
  wait_for_server "http://$LINUX_HOST:$ASR26_PORT" "Breeze-ASR-26"

  run_config "configs/local/breeze_asr26_nan.yaml"   "Breeze-ASR-26 × 台語（Local）"
  run_config "configs/local/breeze_asr26_zh_tw.yaml" "Breeze-ASR-26 × 國語（Local）"

  echo "  Stopping Breeze-ASR-26 server..."
  ssh "$LINUX_HOST" "pkill -f 'breeze_asr_server.py.*8766' 2>/dev/null || true"
fi

echo ""
echo "✅  Local 評測完成！結果在 results/ 目錄"
