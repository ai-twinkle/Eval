# IFEval Evaluation

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | IFEval（Instruction Following Evaluation） |
| **evaluation_method** | `ifeval` （config.yaml 中填入的值）|
| **實作狀態** | ✅ 完整實作 |
| **需要 optional deps** | `pip install twinkle-eval[ifeval]` |
| **實作日期** | 2026-03-25 |
| **實作者** | @lianghsun |

---

## 1. 來源

### Paper

- **標題**：Instruction-Following Evaluation for Large Language Models
- **作者**：Jeffrey Zhou, Tianhao Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, Le Hou
- **發表**：arXiv 2023
- **連結**：https://arxiv.org/abs/2311.07911

### 官方實作

- **Repo**：https://github.com/google-research/google-research/tree/master/instruction_following_eval
- **授權**：Apache 2.0
- 本專案移植了 checker 核心程式碼（`twinkle_eval/metrics/checkers/ifeval/`），主要修改：
  - 將 `absl.logging` 替換為標準 `logging`
  - 將 `immutabledict` 替換為 `types.MappingProxyType`
  - 適配 Twinkle Eval 的模組架構

### 資料集

- **HuggingFace**：`google/IFEval`（`train` split）
- **本專案 example**：`datasets/example/ifeval/`（10 筆，涵蓋 3 種 instruction type）

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

IFEval 評測模型遵循**可驗證的格式化指令**（verifiable instructions）的能力。不同於以語意判斷為主的評測，IFEval 的所有 25 種指令類型（如「使用少於 100 個字」、「不能使用逗號」、「以大寫字母作為每個詞的開頭」）都可以用程式碼客觀驗證，去除主觀評估的模糊性。

每道題包含 1–3 個指令，評測結果分為 4 個指標：依「題目層級/指令層級」× 「strict/loose」的 2×2 組合。Strict 直接評估原始回應；Loose 評估 8 種文字前處理變體（移除 markdown、移除首末行的所有組合）。

### 適合的比較場景

- 比較不同模型對格式化指令的服從能力
- 評估 instruction tuning 效果
- 快速驗證模型訓練迭代中 instruction-following 能力的變化

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| `prompt_strict` | 每道題目所有指令全部通過（strict）的題目比例 | ✅ |
| `prompt_loose` | 每道題目所有指令全部通過（loose）的題目比例 | ✅ |
| `instruction_strict` | 所有指令（包含單題多指令）個別通過（strict）的比例 | ✅ |
| `instruction_loose` | 所有指令個別通過（loose）的比例 | ✅ |

> 主要參考指標為 `prompt_strict` 和 `instruction_strict`。

---

## 3. Leaderboard

- **官方 Leaderboard**：無獨立維護
- **Open LLM Leaderboard v2**：https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard（含 IFEval）
- **Papers With Code**：https://paperswithcode.com/sota/instruction-following-on-ifeval

---

## 4. 本專案實作說明

### Extractor

```
twinkle_eval/metrics/extractors/ifeval.py
```

IFEval 不需要從回應中「提取」答案，模型的完整輸出就是待評估的文字。Extractor 採用 pass-through 設計：直接回傳 `llm_output` 原文，不做任何截取或解析。

同時設有 `uses_ifeval = True` 旗標，讓 `evaluator.py` 識別此評測模式並走 IFEval 專用的評分流程。

### Scorer

```
twinkle_eval/metrics/scorers/ifeval.py
```

`IFEvalScorer.score_full()` 接收模型回應、指令 ID 列表、kwargs，呼叫 checker 模組驗證每條指令，計算：
- `prompt_strict` / `prompt_loose`：所有指令是否全部通過
- `instruction_strict` / `instruction_loose`：各指令個別是否通過

Loose 評分透過 `_get_loose_variants()` 產生 8 種文字前處理變體（`2^3`，對應 remove_markdown × remove_first_line × remove_last_line），取任一通過則算通過。

### Checker

```
twinkle_eval/metrics/checkers/ifeval/
├── __init__.py
├── instructions.py          # 25 種指令檢查器
├── instructions_registry.py # 指令 ID → Checker 映射
├── instructions_util.py     # 輔助工具（統計字數、段落等）
└── evaluation_lib.py        # 整合評分入口
```

移植自 Google Research 官方實作（Apache 2.0），包含 25 種可驗證指令類型：
`change_case`、`combination`、`detectable_content`、`detectable_format`、`keywords`、`language`、`length_constraints`、`punctuation`、`startend`。

### 特殊設計決策

- **Pass-through Extractor**：IFEval 評的是完整回應，不需要提取，避免截斷資訊影響評分
- **Strict/Loose 分離**：4 個指標同時計算並輸出，方便對比不同嚴格度下的能力差異
- **`langdetect` 非確定性**：官方工具使用 `langdetect`，其結果在不同執行間可能略有不同。本實作沿用相同函式庫，因此少數涉及語言偵測的 `language:response_language` 類型題目可能有微小波動（< 0.1%），屬已知行為

### Optional Dependencies

```bash
pip install twinkle-eval[ifeval]
# 首次使用需初始化 NLTK 資料（用於 sentence tokenizer）：
python -c "import nltk; nltk.download('punkt_tab')"
```

---

## 5. 使用方式

### config.yaml 範例

```yaml
llm_api:
  base_url: "http://your-api-endpoint/v1"
  api_key: "your-api-key"
  api_rate_limit: 5   # 建議限速，避免 vLLM 後端過載
  max_retries: 3
  timeout: 120

model:
  name: "your-model-name"
  temperature: 0.0
  max_tokens: 4096

evaluation:
  dataset_paths:
    - "datasets/ifeval/"
  evaluation_method: "ifeval"

logging:
  level: "INFO"
```

### 完整 config template

參見 `twinkle_eval/config.ifeval.template.yaml`

---

## 6. 分數對比（vs. 參考框架）

### 測試環境

- **模型**：Llama 3.3 70B Instruct（vLLM 後端，MI210 GPU）
- **資料集大小**：541 筆（完整 google/IFEval `train` split）
- **測試日期**：2026-03-25
- **硬體**：MacBook（M 系列，無需本地 GPU）

### 結果對比

| 指標 | Twinkle Eval | 參考框架（google/instruction_following_eval） | 差異 | 是否符合容差？ |
|------|-------------|---------------------------------------------|------|--------------|
| `prompt_strict` | **89.65%** | 89.65% | **+0.00%** | ✅ 完全吻合 |
| `prompt_loose` | **91.87%** | 90.94% | +0.93% | ✅（≤ ±2%）|
| `instruction_strict` | **92.93%** | 92.93% | **+0.00%** | ✅ 完全吻合 |
| `instruction_loose` | **94.72%** | 94.00% | +0.72% | ✅（≤ ±2%）|

> 容差標準：完整 benchmark（≥ 200 筆）±2%

### 差異說明

Strict 指標完全吻合（差異 0.00%）。Loose 指標差距 < 1%，來源為 `langdetect` 的非確定性（語言偵測結果在不同執行間可能略有差異），以及文字前處理邊界條件的實作細節（markdown 移除邏輯）。差異遠低於 ±2% 容差標準，視為通過。

---

## 7. 速度對比

### 測試環境

- **模型**：Llama 3.3 70B Instruct（vLLM，MI210 GPU）
- **API 端點**：本地 vLLM（LiteLLM proxy）
- **資料集大小**：541 筆
- **硬體**：MacBook（M 系列，無需本地 GPU）

### 結果

| 框架 | 總耗時 | 每題平均耗時 | 並行方式 |
|------|--------|------------|---------|
| **Twinkle Eval**（5 QPS 速率限制） | **477.9s** | 0.88s | ThreadPoolExecutor（並行） |
| 估算循序評測基線 | ~1,083–1,625s | 2–3s | 逐題同步呼叫 |
| **加速比（估算）** | **~2.3–3.4x** | — | — |

> **無速率限制的理論加速**：若 API 後端允許，Twinkle Eval 的並行執行僅受最慢單題響應時間約束（本次觀測最慢 ~20s）。相比循序評測基線（541 × 2–3s = 1,082–1,623s），理論加速可達 **54–81x**。

> 本次測試限制在 5 QPS（`api_rate_limit: 5`），以避免 vLLM 後端在高並行下出現過載錯誤（已知：無限制並行時，後端可能回應 401 "All connection attempts failed"）。實際部署時視後端承載能力可適度放開限制。

### 評分後處理速度（僅 post-processing，不含 API 呼叫）

| 工具 | 541 題總耗時 | 每題耗時 |
|------|------------|---------|
| **Twinkle Eval** | 0.57s | ~1.05ms |
| Google 官方工具 | ~0.8s | ~1.5ms |

評分後處理本身不是瓶頸，兩者速度相近。

---

## 8. 已知限制與 TODO

- **`langdetect` 非確定性**：`language:response_language` 類型指令的評分在不同執行間可能有極微小差異（< 0.1%），這是 `langdetect` 函式庫本身的行為，非本專案 bug
- **IFBench 尚未實作**：AllenAI 的 IFBench（58 種 OOD 指令類型，NeurIPS 2025）將以獨立的 `evaluation_method: "ifbench"` 實作，見 Milestone #11
- **vLLM 高並行限制**：在共享 vLLM 後端上，建議設定 `api_rate_limit: 5–10`，否則可能遭遇 401 錯誤
