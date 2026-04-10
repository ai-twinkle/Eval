# Vision MCQ — 視覺多選題評測

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 類型** | Vision Language Model（VLM）多選題 |
| **evaluation_method** | `vision_mcq` |
| **實作狀態** | 🚧 Phase 1（核心程式 + Extractor，分數/速度對比待補） |
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

1. **先嘗試 Yes/No** — 這個順序很關鍵：因為 `PatternExtractor` 的兜底模式 `([A-Z]).` 會把 `"Yes,"` 誤解析為字母 `"Y"`，所以 POPE 必須先走 Yes/No 路徑
2. **再嘗試字母選項** — 重用 `PatternExtractor` 的 `DEFAULT_PATTERNS`（A/B/C/D ...）

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

> ⏳ **待補**：需在實際 VLM API 端點上跑完整 benchmark 後填入。
> 容差標準：完整 benchmark（≥200 筆）±2%、中型（50–199 筆）±3%、小型（<50 筆）±5%。

### 計畫測試矩陣

| Benchmark | 規模 | 預計使用模型 | 對比框架 |
|-----------|------|------------|---------|
| MMBench (dev en) | ~4,000 | TBD | VLMEvalKit |
| MMStar | 1,500 | TBD | VLMEvalKit |
| MMMU (val) | ~900 | TBD | VLMEvalKit |
| POPE | ~9,000 | TBD | VLMEvalKit |

---

## 7. 速度對比

> ⏳ **待補**：需在實際 VLM API 端點上完成測量。

本專案的並行 API 請求架構在 VLM 評測上預期會帶來類似 ASR（7.5x）量級的加速，
實際倍率取決於：

- VLM 模型的單次推理時間（圖片越大、越多 token，越慢）
- API 端點的並發承載能力
- 圖片上傳頻寬（base64 編碼後的 payload 較大）

---

## 8. 已知限制與 TODO

### Phase 1 範圍

✅ 已完成：
- `VisionMCQExtractor` + `uses_vision` 路由
- 4 個 benchmark 加入 registry（mmbench / mmstar / mmmu / pope）
- Yes/No 與字母選項雙模式提取
- 本地圖片 base64 編碼 + 可選 Pillow 縮放
- 28 個單元測試

⏳ 待補：
- Example dataset（10–20 筆 MMBench 樣本，需從 HF 下載）
- VLMEvalKit 分數對比驗證
- 速度對比測量

### Phase 2 規劃（vision_vqa）

- RealWorldQA、MMVet、OCRBench、InfoVQA、CountBench
- 需要 short-answer / open-ended 評分機制（非 exact match）

### Phase 3 規劃（vision_detect）

- RefCOCO-M 等需要 bounding box 的偵測任務
- 評分指標：IoU

### 已知行為差異

- **PatternExtractor 的 `[A-Z].` 兜底模式較鬆**：在 Yes/No 題目上會誤判，因此本 Extractor **必須**把 Yes/No 提取放在字母提取**之前**。這是 `vision_mcq.py` 與其他 MCQ Extractor 的關鍵差異
- **HuggingFace Image columns**：HF datasets 把圖片存為 PIL 物件，下載後需要先儲存為 jpg/png 檔案才能放入 `image_path` 欄位。`scripts/create_example_datasets.py` 後續會處理這個轉換

---

## 9. 授權資訊

| 資料集 | 授權 |
|--------|------|
| MMBench | Apache-2.0 |
| MMStar | Apache-2.0 |
| MMMU | Apache-2.0 |
| POPE | MIT |

本實作未直接移植任何 VLMEvalKit 程式碼，僅在 prompt 與答案提取邏輯上對齊參考框架。
