# RAGAS (Retrieval-Augmented Generation Assessment) Evaluation

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | RAGAS (Retrieval-Augmented Generation Assessment) |
| **evaluation_method** | `ragas`（config.yaml 中填入的值）|
| **實作狀態** | ✅ 完整實作（v1：consolidated judge prompt） |
| **需要 optional deps** | 不需要 |
| **實作日期** | 2026-03-26 |
| **實作者** | lianghsun (via Claude Code) |

---

## 1. 來源

### Paper

- **標題**：RAGAS: Automated Evaluation of Retrieval Augmented Generation
- **作者**：Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
- **發表**：EACL 2024
- **連結**：https://arxiv.org/abs/2309.15217

### 官方實作

- **Repo**：https://github.com/explodinggradients/ragas（現為 vibrantlabsai/ragas）
- **授權**：Apache 2.0
- **官方 library**：`pip install ragas`（v0.4+，依賴 langchain 生態系）
- 本專案**未依賴官方 library**，自行實作核心評分邏輯，以 Twinkle Eval 的 OpenAI-compatible LLM 作為 judge

### 資料集

- **HuggingFace**：`explodinggradients/WikiEval`（50 筆 Wikipedia QA，用於 metric correlation 驗證）
- **本專案 example**：`datasets/example/ragas/wikieval.jsonl`（10 筆，5 good + 3 ungrounded + 2 poor）

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

RAGAS 是一個 **RAG pipeline 品質評價框架**，而非固定題目的 benchmark。它評價已有的 RAG 輸出（question + response + retrieved contexts + reference answer），透過 4 個指標量化 RAG 系統的品質：

1. **Faithfulness**：回答中的聲明是否有檢索到的 context 支撐（偵測幻覺）
2. **Answer Relevancy**：回答與問題的相關性和完整度
3. **Context Precision**：檢索到的 context 是否與問題相關
4. **Context Recall**：context 是否涵蓋了 reference answer 的所有面向

### 與其他 benchmark 的關鍵差異

- **RAGAS 不是固定題目** — 使用者自帶 RAG pipeline 的輸出
- **評分需要 LLM judge** — config.yaml 中的模型作為 judge，評分本身會消耗 API calls
- **分數不可重現** — 不同 judge 模型或溫度會產生不同分數
- **資料格式不同** — 需要 `retrieved_contexts` 欄位

### 適合的比較場景

- 評估不同 RAG 架構（chunking 策略、retriever 選擇）的品質差異
- 監控 RAG pipeline 的品質退化
- 比較不同 LLM 作為 generator 時的輸出品質

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| faithfulness | 回答的聲明是否有 context 支撐 | ✅ |
| answer_relevancy | 回答與問題的相關性 | ✅ |
| context_precision | 檢索到的 context 是否相關 | ✅ |
| context_recall | context 是否涵蓋 reference 的所有面向 | ✅ |

---

## 3. Leaderboard

- **官方 Leaderboard**：無（RAGAS 是 metric framework，不是固定 benchmark）
- RAGAS 分數依賴 judge 模型選擇，不同 judge 不可直接比較

---

## 4. 本專案實作說明

### 架構：LLM-as-Judge

```
Dataset (pre-existing RAG outputs)
    ↓
Judge Prompt (pre-assembled in dataset's "question" field)
    ↓
LLM (config.yaml) acts as JUDGE
    ↓
RAGASExtractor → parses JSON scores
    ↓
RAGASScorer → threshold-based is_correct
```

config.yaml 中的 LLM 作為 judge，**不是被評測的模型**，而是評價其他模型 RAG 輸出的裁判。

### Extractor

```
twinkle_eval/metrics/extractors/ragas.py
```

從 judge LLM 的回應中提取 JSON 物件，包含 4 個 0.0–1.0 的分數。支援：
- 純 JSON 回應
- markdown code block 包裹的 JSON
- 混雜文字中的 JSON 物件

### Scorer

```
twinkle_eval/metrics/scorers/ragas.py
```

- `score()` 計算 4 個指標的平均值，若 >= `ragas_threshold`（預設 0.5）則回傳 True
- 支援 `strategy_config.ragas_threshold` 自訂閾值

### 特殊設計決策

- **Consolidated judge prompt**：將 4 個指標的評價合併為一次 LLM 呼叫，而非官方 RAGAS 的多步驟分解（claim decomposition → NLI verification）。這大幅降低 API 成本，但犧牲一些粒度
- **Pre-assembled prompts**：judge prompt 在 dataset 中預組裝，evaluator 流程不需修改
- **不依賴 ragas package**：避免引入 langchain 生態系的龐大依賴鏈
- **Judge 模型建議**：使用能力較強的模型（如 GPT-4 級別）作為 judge，以確保評分品質

### Optional Dependencies

不需要額外安裝任何套件。

---

## 5. 使用方式

### config.yaml 範例

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/ragas/"
  evaluation_method: "ragas"

  # 選填：調整 is_correct 閾值
  # strategy_config:
  #   ragas_threshold: 0.5
```

### 準備自己的 RAGAS 資料集

RAGAS 資料集的 `question` 欄位需為預組裝的 judge prompt，包含：
- 使用者問題
- RAG pipeline 的回答
- 檢索到的 context
- Reference answer

`answer` 欄位為 JSON metadata（供除錯和追蹤用）。

### 完整 config template

參見 `twinkle_eval/templates/ragas.yaml`

---

## 6. 分數對比（vs. 參考框架）

### 說明

RAGAS 是 metric framework，沒有固定的「正確分數」。分數對比的目的是驗證 judge prompt 能否正確區分好/壞回答。

官方 RAGAS library 使用多步驟 LLM 呼叫（claim decomposition + NLI），本專案使用 consolidated judge prompt。因此分數不會完全一致，但判別方向應一致。

### 測試環境

- **Judge 模型**：Devstral-Small-2-24B-Instruct-2512
- **資料集大小**：10 筆（5 good + 3 ungrounded + 2 poor answers from WikiEval）
- **測試日期**：2026-03-26
- **硬體**：Apple Silicon Mac（API 呼叫）

### 結果

| Answer Type | Faithfulness | Relevancy | Precision | Recall | Average |
|-------------|-------------|-----------|-----------|--------|---------|
| good (×5) | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| ungrounded (×3) | 0.00–0.80 | 0.00–0.90 | 1.00 | 1.00 | **0.50–0.93** |
| poor (×2) | 0.50 | 0.70 | 1.00 | 1.00 | **0.80** |

### 分析

- **Good answers**：4 個指標均為 1.0，judge 正確辨識出高品質回答
- **Ungrounded answers**：faithfulness 明顯下降（0.00–0.80），正確偵測到幻覺/不忠實內容
- **Poor answers**：faithfulness 和 relevancy 下降，但降幅不如 ungrounded 明顯
- **Context precision/recall**：始終為 1.0，因為 WikiEval 的 context 品質都很好（差異在 response 而非 retrieval）

Judge 成功區分了三種回答品質，驗證 consolidated prompt 的有效性。

---

## 7. 速度對比

### 測試環境

- **Judge 模型**：Devstral-Small-2-24B-Instruct-2512
- **API 端點**：遠端 LiteLLM proxy（vLLM 後端）
- **資料集大小**：10 筆
- **硬體**：Apple Silicon Mac（單機）

### 結果

| 框架 | 總耗時 | 每題平均耗時 | 並行方式 | LLM calls/題 |
|------|--------|------------|---------|-------------|
| **Twinkle Eval** | ~6 秒 | ~0.6 秒 | ThreadPoolExecutor（並行） | 1（consolidated） |
| RAGAS 官方 library | N/A | N/A | asyncio | 6–8（multi-step） |

本專案使用 consolidated judge prompt（1 次 LLM 呼叫/題），官方 RAGAS 使用多步驟分解（每題 6–8 次呼叫）。本專案的 API 成本約為官方的 1/6 到 1/8。

---

## 8. 已知限制與 TODO

- **Consolidated prompt vs multi-step**：官方 RAGAS 使用 claim decomposition + NLI verification，粒度更高。本專案 v1 使用單次 judge 呼叫，犧牲了 claim-level 的精確度。未來可考慮支援 multi-step 模式
- **Judge 品質依賴**：分數品質取決於 judge 模型的能力。建議使用 GPT-4 級別模型作為 judge
- **Context precision/recall 鑑別力不足**：WikiEval 的 context 品質都很好，導致這兩個指標始終為 1.0。需要更多 context 品質變異的資料集來測試
- **無 embedding-based metrics**：官方 RAGAS 的 answer_relevancy 使用 embedding cosine similarity，本專案使用 LLM 判斷。未來可考慮加入 embedding 支援
- **無自動化 dataset 生成工具**：使用者需自行組裝 judge prompt。未來可提供類似 `--generate-niah` 的工具
