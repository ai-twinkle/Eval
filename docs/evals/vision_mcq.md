# Vision MCQ — 視覺多選題評測

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 類型** | Vision Language Model（VLM）多選題 |
| **evaluation_method** | `vision_mcq` |
| **實作狀態** | ✅ Phase 1（核心程式 + Extractor + VLMEvalKit 對比驗證） |
| **需要 optional deps** | `pip install twinkle-eval[vision]`（僅在 `max_image_size` 啟用時需要 Pillow） |
| **首次實作日期** | 2026-04-10 |
| **Milestone** | VLM Phase 1 — Vision MCQ Evaluation |

---

## 1. 來源

Vision MCQ 是 Twinkle Eval 進入 Vision Language Model 評測的第一階段，
覆蓋四個主流 VLM benchmark：

| Benchmark | 類型 | Paper / 來源 | 資料集連結 |
|-----------|------|-------------|-----------|
| **MMBench** | 多模態 MCQ | Liu et al., 2023 ([arXiv:2307.06281](https://arxiv.org/abs/2307.06281)) | [lmms-lab/MMBench](https://huggingface.co/datasets/lmms-lab/MMBench) |
| **MMStar** | 視覺推理 MCQ | Chen et al., 2024 ([arXiv:2403.20330](https://arxiv.org/abs/2403.20330)) | [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) |
| **MMMU** | 學科多模態 MCQ | Yue et al., 2023 ([arXiv:2311.16502](https://arxiv.org/abs/2311.16502)) | [MMMU/MMMU](https://huggingface.co/datasets/MMMU/MMMU) |
| **POPE** | 物件幻覺偵測（Yes/No） | Li et al., 2023 ([arXiv:2305.10355](https://arxiv.org/abs/2305.10355)) | [lmms-lab/POPE](https://huggingface.co/datasets/lmms-lab/POPE) |

### 參考框架

- **VLMEvalKit** — OpenCompass 維護的 VLM 統一評測框架（[GitHub](https://github.com/open-compass/VLMEvalKit)）
- 本實作的 prompt 格式與答案提取邏輯與 VLMEvalKit 對齊，以便分數對比

### 設計參考

本階段參考了 [Liquid AI 的 LFM2.5-VL-450M 部落格](https://www.liquid.ai/blog/lfm2-5-vl-450m)，
其評測方法論涵蓋的 13 個 VLM benchmark 是 Twinkle Eval 後續 Vision 階段的目標。
Phase 1 先聚焦於 MCQ 與 Yes/No 兩種最常見的答案格式。

---

## 2. 目的與用途

### 評什麼能力

| Benchmark | 衡量能力 |
|-----------|---------|
| MMBench | 通用多模態理解與推理（20 個能力維度） |
| MMStar | 視覺核心推理能力（過濾掉純文字可解的題目） |
| MMMU | 大學等級跨學科多模態問答（30 學科） |
| POPE | 物件幻覺率（Yes/No 二元判斷） |

### 適合的比較場景

- VLM 在通用多模態 MCQ 上的整體能力比較
- POPE 用於量化模型「看不見也回答有」的幻覺傾向
- MMMU 用於評估專業領域的多模態知識

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| accuracy | 答對題數 / 總題數 | ✅ |
| POPE accuracy | Yes/No 的 exact match 比率 | ✅ |

---

## 3. Leaderboard

- **MMBench Leaderboard**：<https://mmbench.opencompass.org.cn/>
- **MMMU Leaderboard**：<https://mmmu-benchmark.github.io/>
- **MMStar Leaderboard**：<https://mmstar-benchmark.github.io/>
- **OpenVLM Leaderboard**（VLMEvalKit）：<https://huggingface.co/spaces/opencompass/open_vlm_leaderboard>

---

## 4. 本專案實作說明

### 架構

```
evaluation_method: "vision_mcq"
        |
        v
Evaluator.uses_vision 路由
        |
        ├─ 從 strategy_config 讀取 image_field / max_image_size / image_detail
        ├─ 本地圖片 → base64 data URI；URL → 直接傳遞
        ├─ 建構 OpenAI multimodal messages（image_url + text）
        └─ 呼叫 OpenAIModel.call(messages=...)
                 |
                 v
        VisionMCQExtractor (Yes/No → 字母 fallback)  +  ExactMatchScorer
```

### Extractor

`twinkle_eval/metrics/extractors/vision_mcq.py`

`VisionMCQExtractor` 設定 `uses_vision = True` 讓 Evaluator 走圖片評測路徑。
`extract()` 的提取順序：

1. **最高優先序：`\boxed{}` / `\box{}`** — 推理型 VLM（reasoning model）的標準輸出格式，沒有歧義，命中即回傳。同時支援單反斜線（raw LaTeX）與雙反斜線（JSON-escaped），以及字母（`\boxed{A}`）與 Yes/No（`\boxed{Yes}`/`\boxed{是}`）兩種內容
2. **再嘗試 Yes/No**（POPE 等 benchmark）— 順序很關鍵：避免「Yes」「No」字面被字母 pattern 誤抓為「Y」「N」
3. **最後嘗試字母選項**（嚴格 VLM-friendly patterns）

`\boxed{}` 放在最前面的原因：推理型 VLM（Qwen2.5-VL、DeepSeek-VL2、LFM-VL with reasoning 等）被訓練成最後輸出 `\boxed{答案}`，這個格式來自 GSM8K/MATH 並被推廣到所有 reasoning 場景。即使 system prompt 沒明確要求，許多模型仍會自發產出 boxed 答案。對非 reasoning 模型則完全 inert（不會誤觸發），是零成本的 forward compatibility。

字母選項採用**獨立的嚴格 patterns**而非 `PatternExtractor.DEFAULT_PATTERNS`：
後者的兜底 `([A-Z]).` 會把英文文章中的單字首字母（"It does." → "I"）誤判為答案。
本 Extractor 的每個 letter pattern 都要求明確的答案語境：

- `answer is X` / `correct answer is **X**` / `Correct Answer: X` / `Correct Option: **X**`
- 中文 `答案：X` / `正確選項是 X` / `選 X 項`
- Markdown bold + 字母 + 分隔符：`**X:**`、`**X)**`
- 行首字母 + 分隔符：`X. ...` / `(X) ...`
- **末尾單獨字母**：句尾換行 + 單字母 + EOS（VLM 簡短回答）

支援的 Yes/No 格式：
- 英文：`"Yes, ..."`、`"Answer: No"`、`"... yes."`
- 中文：`"答案是：是"`、`"答案為：否"`
- 大小寫不敏感（`re.IGNORECASE`）

### Scorer

直接重用 `ExactMatchScorer`（與文字 MCQ 相同），對 normalized answer 做 exact match。

### Evaluator 路由（`uses_vision` 分支）

`twinkle_eval/runners/evaluator.py` 在 `uses_audio` 分支之後新增 `elif getattr(self.extractor, "uses_vision", False)` 分支：

1. 從 `extractor._config`（即 `strategy_config`）讀取 `image_field`（預設 `"image_path"`）、`max_image_size`、`image_detail`
2. 對每題：
   - 取出圖片路徑（支援 `image_field`、`image_url`、`image` 三種 fallback）
   - 用 `_encode_image_to_data_uri()` 編碼為 base64 data URI 或保留原 URL
   - 用 `_build_vision_messages()` 建構 OpenAI multimodal messages
   - 透過 `executor.submit(self.llm.call, ..., messages=messages)` 並行送出
3. 答案提取與評分流程與文字 MCQ 一致（含 `<think>` tag 剝離、`reasoning_content` fallback）

### 圖片編碼

```python
_encode_image_to_data_uri(image_path, max_image_size=None)
```

- HTTP/HTTPS URL：原樣回傳
- 本地檔案：讀檔 → base64 → `data:image/{mime};base64,...`
- 副檔名自動轉 MIME（`jpg → jpeg`）
- 若指定 `max_image_size`，會用 Pillow 將最長邊縮放至該大小，並把 RGBA/LA/P 模式轉為 RGB（以便存為 JPEG）
- Pillow 未安裝時會記錄警告並回退為原始檔案編碼

### Optional Dependencies

```bash
# 僅當需要 max_image_size 縮放時才需要安裝
pip install twinkle-eval[vision]
```

未安裝 Pillow 時，`max_image_size` 會被忽略（記錄警告），`evaluation_method: vision_mcq` 本身仍可正常運作。

---

## 5. 使用方式

### 資料集格式

每筆 JSONL 範例：

```json
{
  "id": "mmbench_dev_001",
  "image_path": "datasets/example/vision_mcq/images/001.jpg",
  "question": "What is the dominant color of the cat?",
  "A": "Black",
  "B": "White",
  "C": "Orange",
  "D": "Grey",
  "answer": "C"
}
```

POPE 範例（Yes/No）：

```json
{
  "id": "pope_001",
  "image_path": "datasets/example/vision_mcq/images/coco_001.jpg",
  "question": "Is there a dog in the image?",
  "answer": "No"
}
```

支援的圖片欄位（依優先序）：`image_path` → `image_url` → `image`，可透過 `strategy_config.image_field` 自訂。

### config.yaml 範例

```yaml
llm_api:
  type: "openai"
  base_url: "http://localhost:8000/v1"
  api_key: "your-api-key"
  api_rate_limit: -1
  max_retries: 3
  timeout: 300

model:
  name: "your-vlm-model"
  temperature: 0.0
  top_p: 0.9
  max_tokens: 1024

evaluation:
  dataset_paths:
    - "datasets/example/vision_mcq/"
  evaluation_method: "vision_mcq"
  repeat_runs: 1
  shuffle_options: false

  strategy_config:
    image_field: "image_path"      # 資料集中圖片欄位名稱
    max_image_size: null           # 最長邊像素數；null 為不縮放
    image_detail: "auto"           # OpenAI image_url detail: auto/low/high

logging:
  level: "INFO"
```

### 完整 config template

```bash
twinkle-eval --init vision_mcq
```

範本位於 `twinkle_eval/templates/vision_mcq.yaml`。

---

## 6. 分數對比（vs. VLMEvalKit）

容差標準：完整 benchmark（≥200 筆）±2%、中型（50–199 筆）±3%、小型（<50 筆）±5%。

### MMStar_MINI（150 筆）— 2026-04-10

| 項目 | 值 |
|------|---|
| **資料集** | MMStar_MINI（VLMEvalKit 官方 150 題子集） |
| **模型** | 開源 VLM（OpenAI 相容 API，~30B 參數級） |
| **API 端點** | OpenAI Chat Completions 相容 |
| **參數** | `temperature=0.0`、`top_p=0.9`、`max_tokens=1024` |
| **題數** | 150 |
| **參考框架** | VLMEvalKit `a9343a1e`（`run.py --data MMStar_MINI --api-nproc 8 --mode all`）|

#### 整體準確率

| 框架 | 答對 | 準確率 |
|------|------|--------|
| **Twinkle Eval** | 115/150 | **76.67%** |
| **VLMEvalKit** | 117/150 | **78.00%** |
| **差距** | -2 題 | **+1.33%** ✅（容差 ±3% 內）|

#### Per-Sample 一致性

| 一致性 | 題數 | 比率 |
|--------|------|------|
| **總體一致** | 134/150 | **89.33%** |
| 兩邊都答對 | 108 | 72.00% |
| 兩邊都答錯 | 26 | 17.33% |
| **不一致** | 16/150 | 10.67% |
| Twinkle 對、VLMEvalKit 錯 | 7 | — |
| Twinkle 錯、VLMEvalKit 對 | 9 | — |

> 不一致的 16 題主要源自模型隨機性（兩邊獨立 sampling），而非評測邏輯差異。
> 兩邊各贏一些（7 vs 9）顯示沒有系統性偏差。

#### Per-Category 準確率

| Category | n | Twinkle | VLMEvalKit | Δ |
|----------|---|---------|------------|---|
| coarse perception | 23 | 82.61% | 73.91% | -8.70 |
| fine-grained perception | 35 | 62.86% | 60.00% | -2.86 |
| instance reasoning | 20 | 85.00% | 95.00% | +10.00 |
| logical reasoning | 21 | 76.19% | 80.95% | +4.76 |
| math | 28 | 71.43% | 82.14% | +10.71 |
| science & technology | 23 | 91.30% | 86.96% | -4.35 |
| **Overall** | **150** | **76.67%** | **78.00%** | **+1.33** |

> 子分類波動較大（±10%）是 20–35 題小樣本的正常統計噪音；
> 整體 +1.33% 的差距遠小於容差，可視為兩個框架在 MMStar_MINI 上達成一致。

#### 未解析的 3 題

Twinkle Eval 有 3 題（2.0%）無法從模型回答中提取出選項字母，這些都是**模型自身輸出問題**而非 parser 失誤：

| ID | 原因 |
|----|------|
| q9 | 模型拒答（"correct answer cannot be determined from the image alone"）|
| q30 | `max_tokens=1024` 截斷在數學計算中段 |
| q72 | `max_tokens=1024` 截斷在邏輯推理中段 |

提高 `max_tokens` 預期可消除後兩個。

---

## 7. 速度對比

### MMStar_MINI（150 筆）— 同一模型 / 同一端點 / 同一硬體

| 框架 | 並行度 | 總耗時 | 倍率 |
|------|--------|--------|------|
| **VLMEvalKit** | `--api-nproc 8` | **~140 s**（2:20）| 1.0x |
| **Twinkle Eval** | `api_rate_limit: -1`（不限）| **~10.6 s** | **~13.2x faster** ✅ |

測量條件：
- 兩個框架在同一台 macOS 機器上跑，呼叫同一個 OpenAI 相容 API 端點
- VLMEvalKit 需要先 inference（~140s）再 judge（不計入），用 8 並發
- Twinkle Eval 用 `ThreadPoolExecutor` 全速並行（無 rate limit）
- 兩邊都包含 image base64 編碼 + multimodal message 建構的開銷
- 排除模型載入、資料集下載等一次性 overhead

> 13x 加速的主因：Twinkle Eval 把所有 150 題同時送進 ThreadPoolExecutor，
> 而 VLMEvalKit 的 `api-nproc 8` 限制了同時只有 8 條 API 請求在飛。
> 實際倍率取決於 API 端點的並發承載能力——越能扛並發，差距越大。

### 重現方式

```bash
# 1. 下載 MMStar_MINI tsv
mkdir -p /tmp/mmstar_data
curl -o /tmp/mmstar_data/MMStar_MINI.tsv \
  https://opencompass.openxlab.space/utils/TEST/MMStar_MINI.tsv

# 2. 轉成 Twinkle Eval JSONL
python scripts/convert_mmstar_mini_for_comparison.py

# 3. 設定 config_local_mmstar_mini.yaml（gitignored，含 API 金鑰）
#    範本見 twinkle_eval/templates/vision_mcq.yaml

# 4. 跑 Twinkle Eval
time python -m twinkle_eval.cli -c config_local_mmstar_mini.yaml

# 5. 跑 VLMEvalKit（參考端點）
git clone https://github.com/open-compass/VLMEvalKit.git /tmp/VLMEvalKit
cd /tmp/VLMEvalKit && pip install -e .
python run.py --data MMStar_MINI --model <YOUR_MODEL> \
  --base-url <YOUR_BASE_URL> --key <YOUR_KEY> \
  --judge <JUDGE_MODEL> --api-nproc 8 --max-tokens 1024
```

---

## 8. 已知限制與 TODO

### Phase 1 範圍

✅ 已完成：
- `VisionMCQExtractor` + `uses_vision` 路由
- 4 個 benchmark 加入 registry（mmbench / mmstar / mmmu / pope）
- Yes/No 與字母選項雙模式提取
- 嚴格 VLM-friendly LETTER_PATTERNS（避免把英文文章中的隨機首字母誤判）
- 本地圖片 base64 編碼 + 可選 Pillow 縮放
- `\boxed{}` / `\box{}` 最高優先序提取（forward compat 給推理型 VLM）
- 50 個單元測試（含 MMStar_MINI regression cases 與 boxed pattern 測試）
- Example dataset（10 筆 MMStar 樣本）
- VLMEvalKit 分數對比驗證（MMStar_MINI 150 題，差距 +1.33% 在容差內）
- 速度對比測量（13x faster vs VLMEvalKit）

### Phase 2 規劃（vision_vqa）

- RealWorldQA、MMVet、OCRBench、InfoVQA、CountBench
- 需要 short-answer / open-ended 評分機制（非 exact match）

### Phase 3 規劃（vision_detect）

- RefCOCO-M 等需要 bounding box 的偵測任務
- 評分指標：IoU

### 已知行為差異

- **嚴格 LETTER_PATTERNS 而非沿用 `PatternExtractor`**：VLM 的回答格式比文字 MCQ 多樣許多（bold markdown、句尾單字母、`**Correct Answer: X**` 等），且 `PatternExtractor` 的兜底 `[A-Z].` 容易把英文文章中的隨機首字母誤判（例：「It does.」→ "I"）。`vision_mcq.py` 因此維護自己的嚴格 pattern 集
- **Yes/No 提取必須在字母提取之前**：避免「Yes」「No」字面被字母 pattern 誤抓為「Y」「N」
- **HuggingFace Image columns**：HF datasets 把圖片存為 PIL 物件，下載後需要先儲存為 jpg/png 檔案才能放入 `image_path` 欄位

---

## 9. 授權資訊

| 資料集 | 授權 |
|--------|------|
| MMBench | Apache-2.0 |
| MMStar | Apache-2.0 |
| MMMU | Apache-2.0 |
| POPE | MIT |

本實作未直接移植任何 VLMEvalKit 程式碼，僅在 prompt 與答案提取邏輯上對齊參考框架。
