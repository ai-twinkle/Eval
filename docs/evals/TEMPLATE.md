# {Benchmark Name} Evaluation

> **使用說明**：複製本檔案為 `docs/evals/{benchmark_name}.md`，填入所有 `{...}` 欄位，刪除本行說明。

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | {Benchmark Name} |
| **evaluation_method** | `{method_name}` （config.yaml 中填入的值）|
| **實作狀態** | ✅ 完整實作 / 🚧 Phase 1（部分） / ⏳ 待實作 |
| **需要 optional deps** | `pip install twinkle-eval[{extra}]` / 不需要 |
| **實作日期** | {YYYY-MM-DD} |
| **實作者** | {GitHub username} |

---

## 1. 來源

### Paper

- **標題**：{Paper title}
- **作者**：{Authors}
- **發表**：{Conference/Journal, Year}
- **連結**：{https://arxiv.org/abs/XXXX.XXXXX}

### 官方實作

- **Repo**：{https://github.com/org/repo}
- **授權**：{MIT / Apache 2.0 / CC BY 4.0 / ...}
- 若本專案移植了原始程式碼，請在此標注並附上移植的具體檔案

### 資料集

- **HuggingFace**：{`org/dataset_name`（`split`）}
- **直接下載**：{官方下載連結}
- **本專案 example**：`datasets/example/{benchmark_name}/`（{N} 筆，涵蓋 {describe coverage}）

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

{1–3 段說明：這個評測旨在衡量模型的哪種能力、用什麼方式衡量、數值的解讀方式}

### 適合的比較場景

- {場景一，例如：比較 instruction-following 能力}
- {場景二}

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| {metric_1} | {說明} | ✅ |
| {metric_2} | {說明} | ✅ |

---

## 3. Leaderboard

> 若此 Benchmark 有官方或社群維護的 Leaderboard，附上連結。

- **官方 Leaderboard**：{https://...} / 無
- **Open LLM Leaderboard**：{https://huggingface.co/spaces/...} / 無
- **Papers With Code**：{https://paperswithcode.com/sota/...} / 無

---

## 4. 本專案實作說明

### Extractor

```
twinkle_eval/metrics/extractors/{name}.py
```

{說明 Extractor 做了什麼，例如：pass-through、從 \boxed{} 提取、正則匹配}

### Scorer

```
twinkle_eval/metrics/scorers/{name}.py
```

{說明 Scorer 的評分邏輯，例如：exact match、fuzzy match、rule-based checker}

### Checker（若有）

```
twinkle_eval/metrics/checkers/{name}/
```

{若有獨立的 checker 模組，說明其來源和作用}

### 特殊設計決策

- {決策一，例如：為什麼選擇 pass-through 而不提取}
- {決策二}
- {若 strict/loose 或多種評分模式，說明差異}

### Optional Dependencies

```bash
pip install twinkle-eval[{extra}]
# 若需要額外初始化步驟，例如：
# python -c "import nltk; nltk.download('punkt_tab')"
```

---

## 5. 使用方式

### config.yaml 範例

```yaml
evaluation:
  dataset_paths:
    - "datasets/{benchmark_name}/"
  evaluation_method: "{method_name}"
```

### 完整 config template

參見 `twinkle_eval/config.{benchmark_name}.template.yaml`

---

## 6. 分數對比（vs. 參考框架）

> **規範**：使用相同模型、相同題目，同時跑本專案與參考框架。
> 容差標準：完整 benchmark（≥200 筆）±2%、中型（50–199 筆）±3%、小型（<50 筆）±5%。

### 測試環境

- **模型**：{model name & version}
- **資料集大小**：{N} 筆（{完整 benchmark / 子集}）
- **測試日期**：{YYYY-MM-DD}
- **硬體**：{CPU model, RAM}（無需 GPU）

### 結果對比

| 指標 | Twinkle Eval | 參考框架（{framework name}） | 差異 | 是否符合容差？ |
|------|-------------|---------------------------|------|--------------|
| {metric_1} | {X.X%} | {X.X%} | {±X.X%} | ✅ / ❌ |
| {metric_2} | {X.X%} | {X.X%} | {±X.X%} | ✅ / ❌ |

### 差異說明

{若有差異，說明原因，例如：preprocessing 不同、prompt format 不同、答案正規化邏輯差異}

---

## 7. 速度對比

> 本專案核心優勢是並行 API 請求帶來的速度提升。

### 測試環境

- **模型**：{model name}
- **API 端點**：{類型，例如：本地 vLLM / OpenAI / NVIDIA Build}
- **資料集大小**：{N} 筆
- **硬體**：{CPU model}（單機，無需 GPU）

### 結果

| 框架 | 總耗時 | 每題平均耗時 | 並行方式 |
|------|--------|------------|---------|
| **Twinkle Eval** | {X} 秒 | {X} 秒 | ThreadPoolExecutor（並行） |
| {參考框架} | {X} 秒 | {X} 秒 | {同步 / 非同步 / N workers} |
| **加速比** | **{N}x** | — | — |

{若無法直接對比，說明原因}

---

## 8. 已知限制與 TODO

- {限制一，例如：Phase 2 尚未實作（見 Issue #XX）}
- {限制二}
- {若有 known flakiness 或 checker 不完整，記錄於此}
