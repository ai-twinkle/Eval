# NIAH (Needle in a Haystack) Evaluation

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | NIAH (Needle in a Haystack) |
| **evaluation_method** | `niah`（config.yaml 中填入的值）|
| **實作狀態** | ✅ 完整實作 |
| **需要 optional deps** | 不需要 |
| **實作日期** | 2026-03-26 |
| **實作者** | lianghsun (via Claude Code) |

---

## 1. 來源

### Paper / 原始提出

NIAH 並非源自單一論文，而是由多方獨立發展的長文本檢索評測方法：

- **Kamradt Original**：Greg Kamradt, 2023. [LLMTest_NeedleInAHaystack](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)
  - 最早提出 NIAH 概念的 repo，使用 Paul Graham 文章作為 haystack
- **NeedleBench**：OpenCompass, 2024. [NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window?](https://arxiv.org/abs/2407.11963)
  - 擴展為多語言（中英文）、多 needle 類型
- **LongBench**：Bai et al., 2024. [LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding](https://arxiv.org/abs/2308.14508)
  - 包含 `passage_retrieval_zh` 子任務，為段落級檢索

### 官方實作

- **Kamradt Repo**：https://github.com/gkamradt/LLMTest_NeedleInAHaystack（MIT）
- **OpenCompass NeedleBench**：https://github.com/open-compass/opencompass（Apache 2.0）
- **LongBench**：https://github.com/THUDM/LongBench（MIT）
- 本專案未直接移植任何第三方程式碼，僅使用其公開資料集

### 資料集

- **HuggingFace (NeedleBench)**：`opencompass/NeedleBench`（`test` split）
- **HuggingFace (LongBench)**：`THUDM/LongBench`（`data.zip` 中的 `passage_retrieval_zh.jsonl`）
- **本專案 example**：
  - `datasets/example/niah/kamradt/`（10 筆，EN，context 1K–8K tokens × depth 0–100%）
  - `datasets/example/niah/needlebench/`（5 筆，3 ZH + 2 EN，4K context，50% depth）
  - `datasets/example/niah/longbench/`（5 筆，ZH 段落檢索）

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

NIAH 測試 LLM 在長文本中檢索特定事實的能力。核心設計是：

1. **Haystack**（乾草堆）：一段很長的背景文本
2. **Needle**（針）：一個短事實句子，被插入在 haystack 的某個位置
3. **Question**：詢問模型 needle 的內容

透過控制兩個維度來系統性測試：
- **Context Length**（文本長度）：從短（1K tokens）到長（128K+ tokens）
- **Needle Depth**（插入深度）：從最前面（0%）到最後面（100%）

結果通常以 2D 熱力圖呈現，橫軸為 context length，縱軸為 needle depth，顏色代表正確率。

### 適合的比較場景

- 比較不同模型對長文本的處理能力
- 評估模型在不同 context window 大小下的檢索準確度
- 找出模型的「lost in the middle」現象（中間位置正確率下降）
- 驗證模型宣稱的 context window 實際可用程度

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| accuracy (substring) | gold answer 是否出現在 response 中 | ✅ |
| accuracy (exact) | 正規化後精確比對 | ✅ |
| accuracy (f1) | token-level F1 score | ✅ |

---

## 3. Leaderboard

- **官方 Leaderboard**：無統一 leaderboard（各框架各自報告）
- **Open LLM Leaderboard**：無（NIAH 不在標準評測集中）
- **OpenCompass NeedleBench**：https://opencompass.org.cn/dataset-detail/NeedleBench

---

## 4. 本專案實作說明

### Extractor

```
twinkle_eval/metrics/extractors/niah.py
```

Pass-through：直接回傳模型原始回答，不做任何提取。NIAH 的評分是對完整 response 做 substring match，不需要提取特定格式。

### Scorer

```
twinkle_eval/metrics/scorers/niah.py
```

支援三種評分模式（透過 `strategy_config.niah_scoring_mode` 切換）：

| 模式 | 說明 | 適用場景 |
|------|------|---------|
| `substring`（預設） | gold answer 是否為 response 的子字串（不區分大小寫） | Kamradt、NeedleBench |
| `exact` | 正規化後精確比對 | LongBench passage_retrieval（段落編號） |
| `f1` | token-level F1 score，支援中文分詞 | 需要部分匹配評分的場景 |

### Generator

```
twinkle_eval/datasets/niah.py
```

提供 CLI 工具 `--generate-niah`，讓使用者用自己的文本和 needle 生成自訂 NIAH 測試集：

```bash
twinkle-eval --generate-niah \
    --haystack my_docs.txt \
    --needle "公司年度營收目標是 42 億元。" \
    --question "公司的年度營收目標是多少？" \
    --answer "42 億元" \
    --context-lengths 1024,4096,16384 \
    --needle-depths 0,25,50,75,100 \
    --output-dir datasets/my_niah/
```

### 特殊設計決策

- **Prompt 預組裝**：Generator 將 haystack + needle 組裝成完整的 question 欄位，evaluator 不需修改即可直接使用
- **Pass-through Extractor**：NIAH 回答格式自由，不需要從 `\boxed{}` 或選項中提取
- **Substring 作為預設**：大多數 NIAH 文獻使用 substring match，與 Kamradt 原始實作一致
- **中文分詞**：F1 模式內建簡易中文逐字分詞，不需額外 dependency

### Optional Dependencies

不需要額外安裝任何套件。

---

## 5. 使用方式

### config.yaml 範例

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/niah/kamradt/"
  evaluation_method: "niah"

  # 選填：切換評分模式
  # strategy_config:
  #   niah_scoring_mode: "substring"  # "substring" | "exact" | "f1"
  #   niah_f1_threshold: 0.5
```

### 完整 config template

參見 `twinkle_eval/templates/niah.yaml`

---

## 6. 分數對比（vs. 參考框架）

### 說明

NIAH 沒有標準化的評測框架可供對比。原始 Kamradt repo 為一組 Python scripts，OpenCompass 整合了 NeedleBench 但 scorer 邏輯與本專案完全一致（substring match）。因此**不存在框架間的分數差異問題**，本節僅記錄 sanity check 結果。

由於 example dataset 僅 10–20 筆（≤20 筆），依 CLAUDE.md §6.3 規範，僅作 sanity check，不強制對比。

### 測試環境

- **模型**：Devstral-Small-2-24B-Instruct-2512
- **資料集大小**：20 筆（10 kamradt + 5 needlebench + 5 longbench）
- **測試日期**：2026-03-26
- **硬體**：Apple Silicon Mac（API 呼叫，無需 GPU）

### 結果

| Dataset | Twinkle Eval (substring) | 說明 |
|---------|-------------------------|------|
| Kamradt (EN, 10 rows) | **100%** | 全部正確 |
| NeedleBench (ZH+EN, 5 rows) | **80%** | 1/5 失敗 |
| LongBench ZH (5 rows) | **100%** | 全部正確 |

### 錯誤分析

- **NeedleBench 1/5 失敗**：模型回答 "on **the** Mysterious Island" 而 gold answer 為 "on Mysterious Island"（無冠詞），多出的 "the" 導致 gold answer 的 substring match 失敗。此為 substring match 的已知邊界情況 — gold answer 是 response 的語意子集但不是字面子字串。可用 `f1` 模式緩解。

---

## 7. 速度對比

### 測試環境

- **模型**：Devstral-Small-2-24B-Instruct-2512
- **API 端點**：遠端 LiteLLM proxy（vLLM 後端）
- **資料集大小**：20 筆（3 個 NIAH 子集）
- **硬體**：Apple Silicon Mac（單機）

### 結果

| 框架 | 總耗時 | 每題平均耗時 | 並行方式 |
|------|--------|------------|---------|
| **Twinkle Eval** | ~6 秒 | ~0.3 秒 | ThreadPoolExecutor（並行） |
| Kamradt 原始 repo | N/A | N/A | 逐題同步呼叫 |

Kamradt 原始 repo 為逐題同步呼叫的 scripts，非標準框架，無法直接在相同環境執行對比。NIAH 的瓶頸在於 API 延遲而非 scoring，Twinkle Eval 的並行架構在題數較多時優勢會更明顯。

---

## 8. 已知限制與 TODO

- **Sequential-NIAH 未實作**：NeedleBench 的 Sequential-NIAH（多 needle 依序檢索）資料集尚未公開，待公開後可擴充
- **Multi-needle reasoning 未實作**：NeedleBench 的 multi-needle reasoning 子集需要更複雜的 scoring 邏輯，列為未來擴充
- **Substring match 邊界情況**：當模型回答中加入額外冠詞或空格時，可能導致 false negative。建議此類場景使用 `f1` 模式
- **無 2D 熱力圖視覺化**：目前僅輸出 accuracy 數值，未提供 context_length × needle_depth 的熱力圖。可作為未來 exporter 功能擴充
