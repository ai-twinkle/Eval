# ASR -- Automatic Speech Recognition Evaluation

## 來源

ASR 評測涵蓋四個核心 benchmark：

| Benchmark | Paper / 來源 | 資料集連結 |
|-----------|-------------|-----------|
| LibriSpeech | Panayotov et al., 2015 ([arXiv:1512.02913](https://arxiv.org/abs/1512.02913)) | [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) |
| Aishell-1 | Bu et al., 2017 ([arXiv:1709.05522](https://arxiv.org/abs/1709.05522)) | [carlot/AIShell](https://huggingface.co/datasets/carlot/AIShell) |
| Fleurs | Conneau et al., 2023 ([arXiv:2205.12446](https://arxiv.org/abs/2205.12446)) | [google/fleurs](https://huggingface.co/datasets/google/fleurs) |
| Common Voice | Ardila et al., 2020 ([arXiv:1912.06670](https://arxiv.org/abs/1912.06670)) | [mozilla-foundation/common_voice_17_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) |

參考框架：
- OpenAI Whisper ([Radford et al., 2022](https://arxiv.org/abs/2212.04356))
- Qwen2-Audio ([Chu et al., 2024](https://arxiv.org/abs/2407.10759))

## 目的

衡量語音辨識模型將音訊轉錄為文字的準確度。主要指標：

- **WER (Word Error Rate)**: 用於英文等以空格分詞的語言
- **CER (Character Error Rate)**: 用於中文、日文、韓文等無空格分詞的語言

## Leaderboard

- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

## 實作說明

### 架構

```
evaluation_method: "asr"
    |
    v
ASRExtractor (pass-through)  +  ASRScorer (WER/CER via jiwer)
```

### 支援的 API 模式

| 模式 | Config `llm_api.type` | 適用模型 | API Endpoint |
|------|----------------------|---------|-------------|
| Whisper API | `whisper` | Whisper, faster-whisper, Groq | `/v1/audio/transcriptions` |
| Chat Completions | `openai` | Qwen2-Audio, GPT-4o | `/v1/chat/completions` |

### WhisperModel

新增 `WhisperModel` 繼承 `LLM` ABC，透過 `/v1/audio/transcriptions` endpoint 進行語音轉錄。
`call()` 接收音檔路徑，將轉錄結果包裝為 `ChatCompletion` 格式回傳。

### ASRExtractor

Pass-through extractor：LLM 輸出即為轉錄文字，不需進一步解析。
設定 `uses_audio = True` 讓 Evaluator 走音檔評測路徑。

### ASRScorer

- 依語言自動選擇 WER 或 CER（可透過 `asr_metric` 強制指定）
- Text normalization pipeline：NFKC、lowercase、remove punctuation
- `score()` 回傳 bool（exact match，用於框架 accuracy 計算）
- `score_full()` 回傳完整 WER/CER 數值（記錄於 JSONL 詳細結果）

### Optional Dependencies

```bash
pip install twinkle-eval[asr]
# 安裝 jiwer（WER/CER 計算）
```

## 分數對比

測試條件：
- 模型：Breeze-ASR-25（Whisper API）
- 資料集：Common Voice 24.0 TW（繁體中文），50 筆
- 日期：2026-04-07

| 指標 | Twinkle Eval | jiwer 直接計算 | 差異 |
|------|-------------|---------------|------|
| CER | 3.80% | 3.70% | 0.10% |
| WER | 34.00% | 34.00% | 0.00% |
| Exact Match | 74.0% | — | — |

CER 的微小差異（0.10%）來自 Twinkle Eval 使用 per-sample 平均（macro average），
而 jiwer 直接計算為 corpus-level（micro average，以總字元數加權）。
兩者皆為正確的計算方式，差異在統計方法而非實作錯誤。

> 注：50 筆樣本屬小型子集（< 50 筆），依 CLAUDE.md 6.3 節容差標準僅作 sanity check。

## 速度對比

測試條件：
- 模型：Breeze-ASR-25（Whisper API）
- 資料集：Common Voice 24.0 TW（繁體中文）
- 硬體：單機，透過 HTTPS 呼叫遠端 API

| 方式 | 樣本數 | 總耗時 | 每筆耗時 | 加速倍率 |
|------|--------|--------|---------|---------|
| Twinkle Eval（並行） | 50 | 9.5s | 0.19s | 7.5x |
| Sequential baseline | 10 | 14.2s | 1.42s | 1.0x |

並行評測在 ASR 場景下仍有顯著加速效果。
實際加速倍率取決於網路頻寬（音檔上傳）和 API 端點的並行處理能力。

## 授權資訊

| 資料集 | 授權 |
|--------|------|
| LibriSpeech | CC-BY-4.0 |
| Aishell-1 | Apache-2.0 |
| Fleurs | CC-BY-4.0 |
| Common Voice | CC0-1.0 |
