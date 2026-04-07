# Regex Match Evaluation（可設定正則提取 + 字串比對）

> 使用 `regex_match` 評測方法，以可設定的正則表達式提取完整答案字串，再做 exact string match。
> BBH (BIG-Bench Hard) 為首個使用者，但本方法不綁定特定 benchmark。

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | 通用正則提取評測（首個使用者：BBH） |
| **evaluation_method** | `regex_match` |
| **實作狀態** | ✅ 完整實作 |
| **需要 optional deps** | 不需要 |
| **實作日期** | 2026-03-27 |
| **實作者** | Twinkle AI Team |

---

## 1. 來源

### 設計動機

許多 benchmark 要求模型在回應末尾以固定格式輸出答案（如 `the answer is X`），且答案可能是選項字母、布林值、數字或自由文字。現有的 `pattern`（僅抓單一字母）、`box`（僅抓 `\boxed{}`）、`custom_regex`（設計上用於選項字母）皆無法完整處理這類混合格式。

`regex_match` 提供一個**泛用的提取 + 比對框架**，讓使用者自訂提取 pattern，搭配可設定的正規化模式進行字串比對。

### BBH (BIG-Bench Hard)

- **標題**：Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them
- **作者**：Mirac Suzgun et al.
- **發表**：ACL 2023 (Findings)
- **連結**：https://arxiv.org/abs/2210.09261
- **官方 Repo**：https://github.com/suzgunmirac/BIG-Bench-Hard（MIT）
- **HuggingFace**：`maveriq/bigbenchhard`
- **規模**：6,511 題，27 個子任務
- **特色**：涵蓋三種答案格式——MC `(A)`/`(B)`、Binary `Yes`/`No`/`True`/`False`、Free-form（數字、字串序列）

### 本專案 Example 資料

| 資料集 | 路徑 | 筆數 |
|--------|------|------|
| BBH | `datasets/example/bbh/` | 15 |

涵蓋 MC (7)、Binary (4)、Free-form (4) 三種子任務類型。

---

## 2. 目的與用途

### 這個評測方法在做什麼？

`regex_match` 從 LLM 輸出中用正則表達式提取答案字串，然後與 ground truth 做精確比對。

與其他方法的差異：
- `pattern` / `box`：提取**單一選項字母**（A/B/C/D）
- `math`：提取 `\boxed{}` 內容 + **數學語意等價**比對
- `regex_match`：提取**完整答案字串** + **精確字串**比對

### 適合的比較場景

- 混合答案格式的 benchmark（如 BBH：MC + binary + free-form）
- 需要自訂提取 pattern 的評測場景
- 任何「模型輸出有固定格式結尾」的 CoT 評測

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| Accuracy | 提取的答案與 ground truth 完全一致的比例 | ✅ |

---

## 3. Leaderboard

- **BBH Papers With Code**：https://paperswithcode.com/dataset/bbh

---

## 4. 本專案實作說明

### Extractor

```
twinkle_eval/metrics/extractors/regex_match.py
```

`RegexMatchExtractor`：
- 預設 pattern：`[Tt]he answer is (.*)`
- 支援自訂 pattern（字串或列表），透過 `strategy_config.answer_pattern` 設定
- 使用 `re.IGNORECASE` 匹配
- 自動移除結尾句號（BBH target 不含句號）
- 提取**完整答案字串**（不限於單一字母）

### Scorer

```
twinkle_eval/metrics/scorers/string_match.py
```

`StringMatchScorer`：
- 可設定 `normalize_mode`：
  - `strip`（預設）：僅去除首尾空白
  - `upper`：去除首尾空白 + 轉大寫
  - `lower`：去除首尾空白 + 轉小寫
  - `none`：不做任何正規化
- 正規化後做精確字串比對

### 特殊設計決策

- **泛用設計**：不綁定 BBH，任何「固定格式結尾 + exact match」的 benchmark 都能使用
- **與 ExactMatchScorer 的差異**：`ExactMatchScorer` 固定使用 `upper` 正規化（為選擇題字母設計），`StringMatchScorer` 可設定正規化模式，適合多種答案格式
- **pattern 順序**：多個 pattern 時，依序嘗試，回傳第一個匹配結果

---

## 5. 使用方式

### config.yaml 範例（BBH）

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/bbh/"
  evaluation_method: "regex_match"
  system_prompt:
    en: "Follow each question's instructions carefully. At the end of your response, clearly state your final answer using the format: 'the answer is {your answer}'."
  strategy_config:
    answer_pattern: "the answer is (.*)"
    normalize_mode: "strip"
```

### 自訂 pattern 範例

```yaml
evaluation:
  evaluation_method: "regex_match"
  strategy_config:
    answer_pattern:
      - "Final answer: (.*)"
      - "ANSWER: (.*)"
    normalize_mode: "lower"
```

### 完整 config template

參見 `twinkle_eval/templates/regex_match.yaml`

---

## 6. 分數對比（vs. 參考框架）

### 測試環境

- **模型**：Devstral-Small-2-24B-Instruct-2512（vLLM backend via LiteLLM proxy）
- **資料集**：BBH 6 個子任務 × 250 題 = 1,500 題
  - MC: disambiguation_qa, date_understanding
  - Binary: boolean_expressions, navigate
  - Free-form: multistep_arithmetic_two, object_counting
- **日期**：2026-03-28
- **硬體**：API 呼叫（不需本地 GPU）

### 重要差異

| 項目 | Twinkle Eval | lm-evaluation-harness (v0.4.11) |
|------|-------------|--------------------------------|
| Prompting | **0-shot**（僅 system prompt） | **3-shot CoT**（含 few-shot examples） |
| System prompt | 自訂指令要求 "the answer is {X}" 格式 | 無 system prompt（BBH 內建 CoT prompt） |
| 答案提取 | `RegexMatchExtractor`（多 pattern + 後處理） | 內建 `get-answer` filter |
| 溫度 | 0.0 | 0.0 |

> ⚠️ **注意**：由於 prompting 策略不同（0-shot vs. 3-shot CoT），兩者的分數**不能直接比較**。
> 此對比旨在驗證評測流程的正確性與一致性，而非逐一對齊分數。

### 逐任務分數

| 子任務 | 類型 | Twinkle Eval (0-shot) | lm-eval (3-shot CoT) | TE 無法解析率 |
|--------|------|----------------------|---------------------|-------------|
| disambiguation_qa | MC | 34.0% | 36.0% | 36.8% |
| date_understanding | MC | 61.6% | 74.4% | 6.0% |
| boolean_expressions | Binary | 18.0% | 0.0% | 50.4% |
| navigate | Binary | 36.0% | 76.0% | 61.6% |
| multistep_arithmetic_two | Free-form | 51.2% | 18.0% | 29.6% |
| object_counting | Free-form | 0.0% | 75.2% | 94.4% |
| **平均** | | **33.5%** | **47.3%** | **46.5%** |

### 分析

1. **lm-eval 3-shot CoT 優勢明顯**：navigate（76% vs 36%）、object_counting（75% vs 0%）、date_understanding（74% vs 62%），few-shot examples 顯著提升模型遵循指定格式的能力
2. **Twinkle Eval 0-shot 也有亮點**：multistep_arithmetic_two（51% vs 18%），boolean_expressions（18% vs 0%），顯示 0-shot 的 system prompt 在某些任務上效果更好
3. **Twinkle Eval 高無法解析率**：object_counting (94%)、navigate (62%)、boolean_expressions (50%) — 模型在 0-shot 下不穩定地遵循 "the answer is X" 格式，降低有效分數
4. **disambiguation_qa 兩者接近**（34% vs 36%），該任務的格式遵循較簡單

### 結論

- Twinkle Eval 的 `regex_match` 評測邏輯正確運作，能正確提取和比對 BBH 三種答案格式
- 分數差異主要來自 prompting 策略（0-shot vs 3-shot CoT），而非評測邏輯錯誤
- 若使用者需要與 lm-eval 對齊分數，應在 system prompt 中加入 few-shot examples 或調整 prompt

---

## 7. 速度對比

### 測試條件

| 項目 | Twinkle Eval | lm-evaluation-harness |
|------|-------------|----------------------|
| 題數 | 1,500 | 1,500 |
| 並行 | ThreadPoolExecutor（預設 worker 數） | num_concurrent=10, batch_size=1 |
| API | OpenAI-compatible（LiteLLM proxy → vLLM） | 同左 |
| 模型 | Devstral-Small-2-24B-Instruct-2512 | 同左 |

### 結果

| 框架 | 總耗時 | 每題平均 | 速度倍率 |
|------|--------|---------|---------|
| **Twinkle Eval** | **256 秒（4 分 16 秒）** | 0.17 秒/題 | **1.86x** |
| lm-evaluation-harness | 477 秒（7 分 57 秒） | 0.32 秒/題 | 1.0x（基準） |

> Twinkle Eval 在相同題數下比 lm-evaluation-harness 快約 **1.9 倍**。
>
> 注意：lm-eval 使用 3-shot CoT prompt（每次 API 呼叫含更多 token），可能導致其每次呼叫的回應時間較長。
> 此速度差異部分來自 prompt 長度差異（3-shot vs 0-shot），部分來自並行策略差異。

### lm-evaluation-harness 使用注意事項

在本次測試中，lm-eval v0.4.11 需要以下調整才能與 OpenAI-compatible API 搭配使用：

1. **URL 格式**：需使用完整端點 URL（`.../v1/chat/completions`），而非僅 `.../v1`
2. **`--apply_chat_template` 必需**：`openai-chat-completions` model type 要求此 flag
3. **`type` 欄位相容性**：lm-eval 在 message 中加入 `type: "text"` 欄位，部分 vLLM 後端會拒絕此欄位（需手動修補 `api_models.py`）
4. **Python 3.14 bug**：`api_models.py:545` 的 `outputs` 變數未初始化（已修補）

---

## 8. 已知限制與 TODO

- 若模型輸出中有多個 `the answer is ...`，目前取最後一個匹配（`finditer` 行為），適合 CoT 推理場景
- 0-shot prompting 下，模型可能不穩定地遵循指定的答案格式（如 BBH 的 object_counting 有 94% 無法解析），建議使用者根據模型特性調整 system prompt 或加入 few-shot examples
