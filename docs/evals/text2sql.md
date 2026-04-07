# Text-to-SQL Evaluation

> 統一的 Text-to-SQL 評測，支援 Spider 1.0、BIRD、Spider 2.0-lite 三個 benchmark。

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | Text-to-SQL（Spider 1.0 / BIRD / Spider 2.0-lite） |
| **evaluation_method** | `text2sql`（config.yaml 中填入的值）|
| **實作狀態** | ✅ 完整實作 |
| **需要 optional deps** | 不需要（使用 Python 內建 `sqlite3`） |
| **實作日期** | 2026-03-27 |
| **實作者** | lianghsun |

---

## 1. 來源

### Spider 1.0

- **標題**：Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task
- **作者**：Tao Yu et al.
- **發表**：EMNLP 2018
- **連結**：https://arxiv.org/abs/1809.08887
- **官方 Repo**：https://github.com/taoyds/spider（Apache 2.0）
- **HuggingFace**：`xlangai/spider`
- **規模**：1034 dev questions, 20 databases

### BIRD

- **標題**：Can LLM Already Serve as A Database Interface? A Big Bench for Large-Scale Database Grounded Text-to-SQL
- **作者**：Jinyang Li et al.
- **發表**：NeurIPS 2023
- **連結**：https://arxiv.org/abs/2305.03111
- **官方 Repo**：https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird（CC BY-SA 4.0）
- **規模**：1534 dev questions, 95 databases
- **特色**：提供 external knowledge（evidence）欄位，模擬真實世界的知識需求

### Spider 2.0 / Spider 2.0-lite

- **標題**：Spider 2.0: Evaluating Language Models on Real-World Enterprise Text-to-SQL Workflows
- **作者**：Fangyu Lei et al.
- **發表**：2024
- **連結**：https://arxiv.org/abs/2411.07763
- **官方 Repo**：https://github.com/xlang-ai/Spider2（Apache 2.0）
- **規模**：完整版 632 questions（BigQuery / Snowflake / PostgreSQL / SQLite），lite 版 85 SQLite-only questions

> **為什麼只支援 Spider 2.0-lite？**
>
> Spider 2.0 完整版需要 BigQuery、Snowflake、PostgreSQL 等雲端資料庫帳號才能執行
> SQL 並驗證結果。這與本專案「單機即裝即用」的設計理念衝突。因此本專案僅支援
> Spider 2.0-lite（85 題 SQLite-only 子集），可在本地端完整執行，無需任何雲端帳號。

### 本專案 Example 資料

| 資料集 | 路徑 | 筆數 | 資料庫 |
|--------|------|------|--------|
| Spider 1.0 | `datasets/example/spider/` | 10 筆 | concert_singer, pets_1 |
| BIRD | `datasets/example/bird/` | 10 筆 | california_schools, financial |
| Spider 2.0-lite | `datasets/example/spider2_lite/` | 10 筆 | book_store |

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

Text-to-SQL 評測衡量語言模型將自然語言問題轉換為可執行 SQL 查詢的能力。
這是一個實用性極強的任務：如果模型能準確地將自然語言轉為 SQL，
就能讓非技術人員直接用自然語言查詢資料庫。

三個 benchmark 各有側重：
- **Spider 1.0**：跨領域泛化能力（20 個不同領域的資料庫）
- **BIRD**：需要外部知識推理（evidence 欄位）+ 真實世界資料庫（大型、複雜 schema）
- **Spider 2.0-lite**：企業級場景的子集（更複雜的查詢模式）

### 適合的比較場景

- 比較不同 LLM 的 SQL 生成能力
- 評估模型對複雜 SQL（JOIN、GROUP BY、子查詢）的掌握程度
- 評估模型在有/無外部知識提示下的表現差異（BIRD）

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| Execution Accuracy (EX) | 預測 SQL 執行結果與 gold SQL 結果集相同 | ✅ |
| Exact Match (EM) | 正規化後的 SQL 字串完全一致 | ✅ |

> **EX vs EM**：EX 是更合理的指標，因為語義相同的 SQL 可以有不同的寫法。
> 例如 `SELECT COUNT(*) FROM t` 和 `SELECT count(1) FROM t` 在 EM 中不匹配，
> 但在 EX 中結果一致故視為正確。

---

## 3. Leaderboard

- **Spider 1.0 Leaderboard**：https://yale-lily.github.io/spider
- **BIRD Leaderboard**：https://bird-bench.github.io/
- **Spider 2.0 Leaderboard**：https://spider2-sql.github.io/
- **Papers With Code (Spider)**：https://paperswithcode.com/dataset/spider

---

## 4. 本專案實作說明

### Extractor

```
twinkle_eval/metrics/extractors/text2sql.py
```

`Text2SQLExtractor` 從 LLM 回應中提取 SQL 查詢，依序嘗試：
1. ` ```sql ... ``` ` markdown code block
2. ` ``` ... ``` ` 通用 code block（內容為 SQL）
3. 以 `SELECT` / `WITH` 等 SQL 關鍵字開頭的語句
4. Fallback：整段文字視為 SQL

提取後會清理結尾分號、合併多餘空白。

### Scorer

```
twinkle_eval/metrics/scorers/text2sql.py
```

`Text2SQLScorer` 支援兩種評分模式：

| 模式 | config 值 | 說明 |
|------|-----------|------|
| **Execution Accuracy** | `text2sql_scoring_mode: "exec"` | 對 SQLite 執行 predicted 與 gold SQL，比對結果集（預設）|
| **Exact Match** | `text2sql_scoring_mode: "em"` | 正規化 SQL 後比對字串 |

EX 模式的結果集比較邏輯：
- 忽略行順序（sorted comparison）
- 浮點數精度容差（round to 6 decimals）
- 字串不區分大小寫
- 若 predicted SQL 執行失敗（語法錯誤），視為不正確

### 特殊設計決策

- **Gold answer 格式**：JSON 字串 `{"sql": "...", "db_id": "..."}`，讓 scorer 能同時取得 SQL 和資料庫 ID
- **EX 模式 fallback**：若資料庫檔案不存在，自動 fallback 至 EM 模式
- **Read-only 執行**：SQLite 連線使用 `mode=ro` + `PRAGMA query_only = ON`，確保不會修改資料庫
- **統一 Extractor/Scorer**：三個 benchmark 共用同一組 Extractor/Scorer，僅需切換 `text2sql_db_base_path`

### Optional Dependencies

不需要。SQL 提取使用正則表達式，SQL 執行使用 Python 內建 `sqlite3` 模組。

---

## 5. 使用方式

### config.yaml 範例

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/spider/"
  evaluation_method: "text2sql"
  strategy_config:
    text2sql_scoring_mode: "exec"       # "exec" 或 "em"
    text2sql_db_base_path: "datasets/example/spider/databases"
```

### 多資料集設定

由於三個 benchmark 的資料庫位於不同路徑，建議為每個 benchmark 分別建立 config。
使用同一個 `evaluation_method: "text2sql"`，只需切換 `dataset_paths` 和 `text2sql_db_base_path`。

### 完整 config template

參見 `twinkle_eval/templates/text2sql.yaml`

---

## 6. 分數對比（vs. 參考框架）

### 測試環境

- **模型**：Devstral-Small-2-24B-Instruct-2512
- **資料集大小**：example 子集（各 10 筆）
- **測試日期**：2026-03-27
- **評分模式**：Execution Accuracy（exec）

### 結果

| 資料集 | Twinkle Eval (EX) | 題數 |
|--------|-------------------|------|
| Spider 1.0 example | **90.00%** | 10 |
| BIRD example | **60.00%** | 10 |
| Spider 2.0-lite example | **80.00%** | 10 |

> **注意**：以上為 example 子集（各 10 筆），僅作為 sanity check 確認流程可跑通。
> 不強制與參考框架進行分數對比（§6.3 容差標準：≤20 筆僅作 sanity check）。

### 參考框架說明

- **Spider 1.0 官方評測腳本**：https://github.com/taoyds/spider/blob/master/evaluation.py — 使用自定義 SQL parser 進行 AST 比對（Exact Match）+ 執行比對
- **BIRD 官方評測腳本**：https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird/llm/src — 僅使用 Execution Accuracy
- **test-suite-sql-eval**：https://github.com/taoyds/test-suite-sql-eval — Spider 的增強版評測腳本（使用 test suite 比對）

本專案的 EX 模式與官方 BIRD 評測腳本的邏輯一致（執行 SQL + 比對結果集）。

---

## 7. 速度對比

### 測試環境

- **模型**：Devstral-Small-2-24B-Instruct-2512（遠端 API）
- **資料集大小**：10 筆
- **硬體**：Apple Silicon（單機）

### 結果

| 資料集 | Twinkle Eval 總耗時 | 每題平均耗時 | 並行方式 |
|--------|-------------------|------------|---------|
| Spider 1.0 (10 筆) | ~2 秒 | ~0.2 秒 | ThreadPoolExecutor |
| BIRD (10 筆) | ~2 秒 | ~0.2 秒 | ThreadPoolExecutor |
| Spider 2.0-lite (10 筆) | ~3 秒 | ~0.3 秒 | ThreadPoolExecutor |

> 官方 Spider/BIRD 評測腳本不包含 LLM 推論部分（僅做後處理評分），
> 因此速度對比主要反映的是 Twinkle Eval 的並行 API 呼叫優勢。

---

## 8. 已知限制與 TODO

- **僅支援 SQLite**：EX 模式僅能對 SQLite 資料庫執行查詢。Spider 2.0 完整版所需的 BigQuery / Snowflake / PostgreSQL 不在支援範圍
- **EM 模式的局限**：Exact Match 對 SQL 寫法差異非常敏感（別名、語法變體、空白等），建議優先使用 EX 模式
- **結果集比較策略**：目前使用 sorted row comparison，對於某些涉及 ORDER BY 的查詢可能過於寬鬆（忽略了排序順序）
- **多資料庫 db_base_path**：目前每個 config 只能指定一個 `text2sql_db_base_path`，若需要在同一次評測中混合不同 benchmark 的資料集，需要分別建立 config
