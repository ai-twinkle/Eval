# ASR -- Automatic Speech Recognition Evaluation

## 來源

ASR 評測涵蓋四個核心 benchmark：

| Benchmark | 語言 | Paper / 來源 | 資料集連結 | 下載方式 |
|-----------|------|-------------|-----------|----------|
| LibriSpeech | 英文 | Panayotov et al., 2015 ([arXiv:1512.02913](https://arxiv.org/abs/1512.02913)) | [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) | `download_librispeech_en_100.py` |
| CommonVoice zh-TW | 繁體中文（台灣）| Ardila et al., 2020 ([arXiv:1912.06670](https://arxiv.org/abs/1912.06670)) | [JacobLinCool/common-voice-17-zh-TW](https://huggingface.co/datasets/JacobLinCool/common-voice-17-zh-TW)¹ | `download_cv_zh_tw_100.py` |
| FLEURS cmn_hans_cn | 普通話（大陸）² | Conneau et al., 2022 ([arXiv:2205.12446](https://arxiv.org/abs/2205.12446)) | [google/fleurs](https://huggingface.co/datasets/google/fleurs) | `download_fleurs_cmn_hans_cn.py` |
| TAT-S2ST | 台語（閩南語）| NYCU SARC ([官方頁面](https://sites.google.com/nycu.edu.tw/sarc/tat_s2st_benchmark)) | [Google Drive](https://docs.google.com/folderview?id=1z9Mj42f6BHuxXDvmXxZGk2fLOfG7J55S) | `prepare_tat_s2st.py`³ |

¹ Mozilla Foundation 原版 `mozilla-foundation/common_voice_17_0` 需 HuggingFace 帳號申請授權；`JacobLinCool` mirror 公開可存取。  
² FLEURS 僅有 `cmn_hans_cn`（簡體中文，大陸口音）。**不存在** `cmn_hans_tw`（台灣國語）variant。  
³ TAT-S2ST 的 HuggingFace mirror `NYCU-SARC/TAT-S2ST` 已下架（404）；請從 Google Drive 下載原始資料。詳見 [`docs/datasets.md`](../datasets.md)。

參考框架：
- OpenAI Whisper ([Radford et al., 2022](https://arxiv.org/abs/2212.04356))
- Qwen2-Audio ([Chu et al., 2024](https://arxiv.org/abs/2407.10759))
- Qwen3.5-Omni ([arXiv:2509.17765](https://arxiv.org/abs/2509.17765))

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
pip install maiagent-eval[asr]
# 安裝 jiwer（WER/CER 計算）
```

## 分數對比

測試條件：
- 模型：Breeze-ASR-25（Whisper API）
- 資料集：Common Voice 24.0 TW（繁體中文），50 筆
- 日期：2026-04-07

| 指標 | MaiAgent Eval | jiwer 直接計算 | 差異 |
|------|-------------|---------------|------|
| CER | 3.80% | 3.70% | 0.10% |
| WER | 34.00% | 34.00% | 0.00% |
| Exact Match | 74.0% | — | — |

CER 的微小差異（0.10%）來自 MaiAgent Eval 使用 per-sample 平均（macro average），
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
| MaiAgent Eval（並行） | 50 | 9.5s | 0.19s | 7.5x |
| Sequential baseline | 10 | 14.2s | 1.42s | 1.0x |

並行評測在 ASR 場景下仍有顯著加速效果。
實際加速倍率取決於網路頻寬（音檔上傳）和 API 端點的並行處理能力。

## 已知限制

### TAT-S2ST：漢羅混寫造成 CER 虛高

TAT-S2ST ground truth 使用學術「漢羅」書寫（Traditional Chinese + Tailo 羅馬字混寫），
例如 `咱就kā叫做幼瓷`、`食飯kann歇晝`。所有現代 ASR 模型均輸出純漢字，導致 `substitution_rate ≈ 1.0`。
此 CER 主要反映「模型是否輸出漢羅拼音」而非語音辨識準確性，不適合與純漢字轉錄集直接比較。

### Qwen 系模型：簡繁體輸出不一致

| 模型 | 輸出字體 | OpenCC 需要？|
|------|---------|--------------|
| Breeze-ASR-25/26 | 繁體 ✅ | 不需要 |
| Qwen3-ASR-Flash (DashScope) | 簡體 ❌ | 必須加 `s2tw` |
| Qwen3-ASR-1.7B (本機 HF) | 混合 ⚠️ | 建議加 `s2tw` |
| Qwen-Omni-Turbo | 簡體（預期）⚠️ | 建議加 `s2tw` |

做 zh-TW 評測時，Qwen 系模型一律建議套 OpenCC `s2tw` 後處理，否則繁簡差異會虛增 CER。

## 授權資訊

| 資料集 | 授權 |
|--------|------|
| LibriSpeech | CC-BY-4.0 |
| CommonVoice zh-TW | CC0-1.0 |
| FLEURS cmn_hans_cn | CC-BY-4.0 |
| TAT-S2ST | 學術研究用途（NYCU SARC）|
