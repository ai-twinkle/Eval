# BFCL Evaluation（Berkeley Function-Calling Leaderboard）

> 使用 `bfcl_fc`（tool call）或 `bfcl_prompt`（prompting）評測方法。

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | BFCL（Berkeley Function-Calling Leaderboard） |
| **evaluation_method** | `bfcl_fc`（原生 tool call）/ `bfcl_prompt`（prompting 模式） |
| **實作狀態** | ✅ Phase 1 完整實作（v1/v2 single-turn + v3 multi-turn mock），🚧 Phase 2 待實作（state-based evaluation） |
| **需要 optional deps** | `pip install twinkle-eval[tool]`（jsonschema） |
| **實作者** | Twinkle AI Team |

---

## 1. 來源

### Paper

- **標題**：Berkeley Function Calling Leaderboard
- **作者**：Fanjia Yan et al.
- **連結**：https://arxiv.org/abs/2402.15671
- **官方 Repo**：https://github.com/ShishirPatil/gorilla（Apache 2.0）

### 資料集

- **HuggingFace**：`gorilla-llm/Berkeley-Function-Calling-Leaderboard`
- **本專案 example**：
  - `datasets/example/bfcl/`（23 筆，含 simple / multiple / parallel / live_simple / multi_turn）
  - `datasets/example/bfcl_v1/`（15 筆）
  - `datasets/example/bfcl_v2/`（5 筆）
  - `datasets/example/bfcl_v3/`（3 筆）

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

BFCL 衡量模型的 function calling 能力：
- **Simple**：單一函式呼叫
- **Multiple**：從多個候選函式中選擇正確的一個
- **Parallel**：一次呼叫多個函式
- **Live Simple**：真實世界 API 的函式呼叫
- **Multi-turn**：多輪對話中的連續函式呼叫

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| Accuracy | 函式名稱與參數完全正確的比例 | ✅ |

---

## 3. Leaderboard

- **官方 Leaderboard**：https://gorilla.cs.berkeley.edu/leaderboard.html

---

## 4. 本專案實作說明

### 兩種模式

| 模式 | evaluation_method | 說明 |
|------|-------------------|------|
| Tool Call | `bfcl_fc` | 使用 OpenAI tools API，模型原生回傳 tool_calls |
| Prompting | `bfcl_prompt` | 將函式定義注入 system prompt，模型以文字回傳 JSON |

### Extractor

- `ToolCallExtractor`（bfcl_fc）：從 ChatCompletion.tool_calls 提取
- `BFCLPromptExtractor`（bfcl_prompt）：從文字回應中解析 JSON

### Scorer

`BFCLScorer`：驗證函式名稱與參數是否與 ground truth 一致，支援 JSON Schema 驗證。

### 已知限制

- Phase 2（state-based evaluation with domain simulators）尚未實作（見 Issue #42）
- Multi-turn 目前使用 mock tool results，非真正的狀態追蹤

---

## 5. 使用方式

### config.yaml 範例

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/bfcl/simple/"
  evaluation_method: "bfcl_fc"
```

### 完整 config template

參見 `twinkle_eval/templates/bfcl.yaml`
