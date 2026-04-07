# IFBench Evaluation

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | IFBench（Instruction Following Benchmark — Out-of-Distribution） |
| **evaluation_method** | `ifbench` （config.yaml 中填入的值）|
| **實作狀態** | ✅ 完整實作 |
| **需要 optional deps** | `pip install twinkle-eval[ifbench]` |
| **實作日期** | 2026-03-25 |
| **實作者** | @lianghsun |

---

## 1. 來源

### Paper

- **標題**：IFBench: Evaluating the Instruction Following Ability of Large Language Models with Constraints in Content, Format and Count
- **作者**：Jingming Zhuo, Xinzhe Li, Xiansong Huang, Ming Jiang, Lei Fang, Jun Zhao, Kang Liu
- **發表**：NeurIPS 2025
- **連結**：https://arxiv.org/abs/2412.15194

### 官方實作

- **Repo**：https://github.com/allenai/IFBench
- **授權**：Apache 2.0
- 本專案移植了 checker 核心程式碼（`twinkle_eval/metrics/checkers/ifbench/`），主要修改：
  - 將 module-level import 改為 relative import（`from . import instructions_util`）
  - 將 `syllapy` 改為 lazy import 以相容 Python 3.14（`pkg_resources` 已移除）
  - 適配 Twinkle Eval 的模組架構

### 資料集

- **HuggingFace**：`allenai/IFBench`（`test` split，294 筆）
- **本專案 example**：`datasets/example/ifbench/`（14 筆，涵蓋全部 7 大指令類別）

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

IFBench 評測模型遵循 **out-of-distribution（OOD）可驗證指令** 的能力。與 IFEval（Google, 25 種指令類型）不同，IFBench 包含 58 種全新設計的指令類型，分為 7 大類別：count（7 種）、ratio（4 種）、words（11 種）、sentence（3 種）、format（13 種）、custom（10 種）、repeat（3 種）。

這些指令類型刻意設計為「訓練數據中不太可能出現的格式要求」，用以測試模型面對未見過的指令時的泛化遵循能力。例如：「回覆中每句話的首字母需排列成指定字串」、「每個段落的字數需為質數」、「回答中需在指定位置插入 emoji」。

評分方式與 IFEval 相同：strict/loose × prompt/instruction = 4 個指標。Strict 直接評估原始回應；Loose 評估 8 種文字前處理變體（移除 markdown × 移除首行 × 移除末行 的 2³ 組合），取任一通過則算通過。

### 適合的比較場景

- 評估模型對 OOD 格式化指令的泛化遵循能力
- 與 IFEval 結果對比，分析模型在 in-distribution vs OOD 指令上的能力差異
- 評估 instruction tuning 的泛化效果

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| `prompt_strict` | 每道題目所有指令全部通過（strict）的題目比例 | ✅ |
| `prompt_loose` | 每道題目所有指令全部通過（loose）的題目比例 | ✅ |
| `instruction_strict` | 所有指令個別通過（strict）的比例 | ✅ |
| `instruction_loose` | 所有指令個別通過（loose）的比例 | ✅ |

> 主要參考指標為 `prompt_strict` 和 `instruction_strict`。

---

## 3. Leaderboard

- **官方 Leaderboard**：無獨立維護
- **Open LLM Leaderboard**：無（IFBench 尚未被納入）
- **Papers With Code**：https://paperswithcode.com/dataset/ifbench

---

## 4. 本專案實作說明

### Extractor

```
twinkle_eval/metrics/extractors/ifbench.py
```

與 IFEval 相同，IFBench 不需要從回應中「提取」答案，模型的完整輸出就是待評估的文字。Extractor 採用 pass-through 設計：直接回傳 `llm_output` 原文，不做任何截取或解析。

同時設有 `uses_ifeval = True` 旗標，讓 `evaluator.py` 識別此評測模式並走 IFEval/IFBench 專用的評分流程。

### Scorer

```
twinkle_eval/metrics/scorers/ifbench.py
```

`IFBenchScorer.score_full()` 接收模型回應、指令 ID 列表、kwargs、原始 prompt，呼叫 checker 模組驗證每條指令，計算 4 個指標。

與 IFEval scorer 的關鍵差異：
- **Kwargs 過濾**：IFBench kwargs 繼承了 IFEval 的欄位結構，部分欄位永遠為 `null`，必須在傳入 `build_description()` 前過濾掉
- **Prompt 回注**：部分 checker（如 `RepeatChangeChecker`）需要原始 prompt 作為參數，scorer 會偵測 checker 是否需要 prompt 並自動回注

### Checker

```
twinkle_eval/metrics/checkers/ifbench/
├── __init__.py
├── instructions.py          # 58 種 OOD 指令檢查器
├── instructions_registry.py # 指令 ID → Checker 映射
└── instructions_util.py     # 輔助工具（WORD_LIST、NLTK 工具、文字處理）
```

移植自 AllenAI IFBench 官方實作（Apache 2.0），包含 58 種 OOD 指令類型，分為 7 大類別：

| 類別 | 數量 | 範例 |
|------|------|------|
| count | 7 | 字母出現次數、段落數需為質數 |
| ratio | 4 | 大寫字母比例、數字字元比例 |
| words | 11 | 交替音節奇偶、首字母拼出指定字串 |
| sentence | 3 | 句子長度遞增/遞減 |
| format | 13 | JSON 格式、APA 引用格式 |
| custom | 10 | emoji 位置指定、自訂分隔符 |
| repeat | 3 | 重複/修改前文 |

### 特殊設計決策

- **Pass-through Extractor**：與 IFEval 相同，評的是完整回應
- **Kwargs 過濾**：IFBench 繼承 IFEval 的 Instruction 基類，kwargs 中包含許多 IFEval 專屬但在 IFBench 永遠為 null 的欄位，必須過濾後才能傳入 `build_description()`
- **Prompt 回注**：`RepeatChangeChecker` 等 checker 需要原始 prompt 才能驗證「重複前文」類指令，scorer 使用 `get_instruction_args()` 動態偵測是否需要 prompt
- **Lazy syllapy import**：`syllapy` 依賴 `pkg_resources`，在 Python 3.14 中已移除。改為 lazy import 並在 checker 中加入 guard，確保不影響其餘 57 種 checker

### Optional Dependencies

```bash
pip install twinkle-eval[ifbench]
# 首次使用需初始化 NLTK 資料：
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')"
```

---

## 5. 使用方式

### config.yaml 範例

```yaml
llm_api:
  base_url: "http://your-api-endpoint/v1"
  api_key: "your-api-key"
  api_rate_limit: 5
  max_retries: 3
  timeout: 120

model:
  name: "your-model-name"
  temperature: 0.0
  max_tokens: 4096

evaluation:
  dataset_paths:
    - "datasets/ifbench/"
  evaluation_method: "ifbench"

logging:
  level: "INFO"
```

### 完整 config template

參見 `twinkle_eval/templates/ifbench.yaml`

---

## 6. 分數對比（vs. 參考框架）

### 測試環境

- **模型**：Llama 3.3 70B Instruct（vLLM 後端，MI210 GPU）
- **資料集大小**：294 筆（完整 allenai/IFBench `test` split）
- **測試日期**：2026-03-25
- **硬體**：MacBook（M 系列，無需本地 GPU）

### 結果對比

| 指標 | Twinkle Eval | 參考框架（allenai/IFBench 官方工具） | 差異 | 是否符合容差？ |
|------|-------------|-------------------------------------|------|--------------|
| `prompt_strict` | **44.22%** | 44.22% | **+0.00%** | ✅ 完全吻合 |
| `prompt_loose` | **45.92%** | 47.28% | -1.36% | ✅（≤ ±2%）|
| `instruction_strict` | **47.20%** | 47.16% | +0.04% | ✅ 完全吻合 |
| `instruction_loose` | **49.00%** | 50.15% | -1.15% | ✅（≤ ±2%）|

> 容差標準：完整 benchmark（≥ 200 筆）±2%

### 差異說明

Strict 指標幾乎完全吻合（prompt_strict 差異 0.00%，instruction_strict 差異 +0.04%）。Loose 指標差距 < 1.4%，來源為文字前處理邊界條件的微小差異（markdown 移除邏輯、首末行移除的空字串處理）。所有指標均遠低於 ±2% 容差標準。

---

## 7. 速度對比

### 測試環境

- **模型**：Llama 3.3 70B Instruct（vLLM，MI210 GPU）
- **API 端點**：本地 vLLM（LiteLLM proxy）
- **資料集大小**：294 筆
- **硬體**：MacBook（M 系列，無需本地 GPU）

### 結果

| 框架 | 總耗時 | 每題平均耗時 | 並行方式 |
|------|--------|------------|---------|
| **Twinkle Eval**（5 QPS 速率限制） | **~60s** | ~0.20s | ThreadPoolExecutor（並行） |
| 估算循序評測基線 | ~588–882s | 2–3s | 逐題同步呼叫 |
| **加速比（估算）** | **~10–15x** | — | — |

> AllenAI 官方工具僅提供評分功能（post-processing），不包含 API 呼叫邏輯，因此無法直接對比端到端耗時。加速比基於循序逐題呼叫的估算基線。

### 評分後處理速度（僅 post-processing，不含 API 呼叫）

| 工具 | 294 題總耗時 | 每題耗時 |
|------|------------|---------|
| **Twinkle Eval** | 0.374s | ~1.27ms |
| AllenAI 官方工具 | 0.464s | ~1.58ms |

評分後處理本身不是瓶頸，兩者速度相近。

---

## 8. 已知限制與 TODO

- **`syllapy` Python 3.14 相容性**：`syllapy` 依賴 `pkg_resources`（Python 3.14 已移除），目前以 lazy import + guard 處理。若 `syllapy` 未來更新修復此問題，可移除 workaround
- **IFBench 與 IFEval 的欄位差異**：IFBench 資料集使用 `key`/`prompt` 欄位（而非 IFEval 的 `id`/`question`），且 `instruction_id_list`/`kwargs` 為原生 list/dict（而非 JSON string）。evaluator 已處理兩種格式的相容
- **vLLM 高並行限制**：在共享 vLLM 後端上，建議設定 `api_rate_limit: 5–10`，否則可能遭遇 401 錯誤
