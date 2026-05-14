#!/usr/bin/env bash
# MaiAgent ASR Evaluation — 完整跑所有模型
#
# 前置條件：
#   export NCHC_RAP_API_KEY=<your-nchc-key>
#   export DASHSCOPE_API_KEY=<your-dashscope-key>
#   uv sync  # 安裝依賴
#
# 使用方式：
#   bash scripts/run_asr_eval.sh
#   bash scripts/run_asr_eval.sh nan    # 只跑台語評測
#   bash scripts/run_asr_eval.sh zh     # 只跑國語評測

set -e
cd "$(dirname "$0")/.."

# ── 環境檢查 ─────────────────────────────────────────────────────────────────
if [[ -z "$NCHC_RAP_API_KEY" ]]; then
  echo "❌  缺少 NCHC_RAP_API_KEY，請先 export NCHC_RAP_API_KEY=<key>"
  exit 1
fi
if [[ -z "$DASHSCOPE_API_KEY" ]]; then
  echo "❌  缺少 DASHSCOPE_API_KEY，請先 export DASHSCOPE_API_KEY=<key>"
  exit 1
fi

LANG_FILTER="${1:-all}"

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

# ── Step 0: 建立 100 筆子集 ──────────────────────────────────────────────────
echo "▶ 建立 tat_s2st 100-sample 子集..."
python3 scripts/make_asr_subset.py \
  --input datasets/tat_s2st/test.jsonl \
  --output datasets/tat_s2st/test_100.jsonl \
  --n 100

# ── 台語評測（tat_s2st × 100 筆）───────────────────────────────────────────
if [[ "$LANG_FILTER" == "nan" || "$LANG_FILTER" == "all" ]]; then
  run_config "configs/maiagent/breeze_asr26_nan.yaml"     "Breeze-ASR-26 × 台語"
  run_config "configs/maiagent/whisper_lv3_nan.yaml"      "Whisper-LV3 × 台語（基準線）"
  run_config "configs/maiagent/qwen3_asr_nan.yaml"        "Qwen3-ASR × 台語"
fi

# ── 國語評測（cv_zh_tw × test.jsonl，需先下載資料集）─────────────────────────
if [[ "$LANG_FILTER" == "zh" || "$LANG_FILTER" == "all" ]]; then
  if [[ ! -f "datasets/cv_zh_tw/test.jsonl" ]]; then
    echo ""
    echo "⚠️  cv_zh_tw 資料集尚未下載，跳過國語評測"
    echo "   下載方式："
    echo "   uv run python -c \"from datasets import load_dataset; ds = load_dataset('mozilla-foundation/common_voice_17_0', 'zh-TW', split='test'); ds.to_json('datasets/cv_zh_tw/test.jsonl')\""
  else
    # 建立 100 筆子集
    python3 scripts/make_asr_subset.py \
      --input datasets/cv_zh_tw/test.jsonl \
      --output datasets/cv_zh_tw/test_100.jsonl \
      --n 100

    run_config "configs/maiagent/breeze_asr25_zh_tw.yaml"  "Breeze-ASR-25 × 國語"
    run_config "configs/maiagent/breeze_asr26_zh_tw.yaml"  "Breeze-ASR-26 × 國語"
    run_config "configs/maiagent/whisper_lv3_zh_tw.yaml"   "Whisper-LV3 × 國語（基準線）"
    run_config "configs/maiagent/qwen3_asr_zh_tw.yaml"     "Qwen3-ASR × 國語"
  fi
fi

echo ""
echo "✅  評測完成！結果在 results/ 目錄"
echo "   查看摘要：ls -lt results/ | head -10"
