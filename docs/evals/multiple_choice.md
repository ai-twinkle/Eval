# Multiple Choice Evaluation（選擇題評測）

> 涵蓋 `pattern`、`box`、`custom_regex` 三種評測方法，適用於 MMLU、MMLU-Pro、TMMLU+ 等選擇題 benchmark。

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | 選擇題（MMLU / MMLU-Pro / TMMLU+ 等） |
| **evaluation_method** | `box`（預設推薦）/ `pattern` / `custom_regex` |
| **實作狀態** | ✅ 完整實作（專案核心功能） |
| **需要 optional deps** | 不需要 |
| **實作者** | Twinkle AI Team |

---

## 1. 來源

### MMLU

- **標題**：Measuring Massive Multitask Language Understanding
- **作者**：Dan Hendrycks et al.
- **發表**：ICLR 2021
- **連結**：https://arxiv.org/abs/2009.03300
- **HuggingFace**：`cais/mmlu`

### MMLU-Pro

- **標題**：MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark
- **作者**：Yubo Wang et al.
- **發表**：NeurIPS 2024
- **連結**：https://arxiv.org/abs/2406.01574
- **HuggingFace**：`TIGER-Lab/MMLU-Pro`

### TMMLU+

- **標題**：TMMLU+: An Improved Traditional Chinese Evaluation Suite
- **作者**：iKala
- **HuggingFace**：`ikala/tmmluplus`

### MMLU-Redux

- **標題**：Are We Done with MMLU?
- **作者**：Aryo Pradipta Gema et al.
- **發表**：NAACL 2025
- **連結**：https://arxiv.org/abs/2406.04127
- **HuggingFace**：`edinburgh-dawg/mmlu-redux`（v1: 3,000 題 / 30 科）、`edinburgh-dawg/mmlu-redux-2.0`（v2: 5,700 題 / 57 科）
- **特色**：MMLU 的人工審核修正版。原始 MMLU 約 6.49% 的題目存在標注錯誤，本資料集提供 `error_type` 欄位標記錯誤類型，以及 `correct_answer` 欄位提供修正後的正確答案

### SuperGPQA

- **標題**：SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines
- **作者**：m-a-p Team
- **發表**：2025
- **連結**：https://arxiv.org/abs/2502.14739
- **HuggingFace**：`m-a-p/SuperGPQA`
- **規模**：26,529 題，涵蓋 13 個一級學科、72 個二級學科、285 個子學科
- **特色**：選項數 4–10 不等（A–J），類似 MMLU-Pro。含 `difficulty`（easy/middle/hard）和 `is_calculation` 欄位

### GPQA

- **標題**：GPQA: A Graduate-Level Google-Proof Q&A Benchmark
- **作者**：David Rein et al.
- **發表**：2023
- **連結**：https://arxiv.org/abs/2311.12022
- **HuggingFace**：`Idavidrein/gpqa`（gated dataset，需申請存取）
- **規模**：Diamond 198 題、Main 448 題、Extended 546 題
- **特色**：研究所等級科學問題（物理、化學、生物），人類專家正確率約 65%，非專家僅約 34%

### 本專案 Example 資料

| 資料集 | 路徑 | 筆數 |
|--------|------|------|
| MMLU | `datasets/example/mmlu/` | 20 |
| MMLU-Pro | `datasets/example/mmlu_pro/` | 20 |
| TMMLU+ | `datasets/example/tmmluplus/` | 20 |
| MMLU-Redux | `datasets/example/mmlu_redux/` | 10 |
| SuperGPQA | `datasets/example/supergpqa/` | 10 |
| GPQA Diamond | `datasets/example/gpqa/` | 10 |

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

選擇題評測衡量模型的知識廣度與推理能力。模型需從給定選項中選出正確答案。
本專案透過選項隨機排列（`shuffle_options`）消除模型對選項位置的偏好。

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| Accuracy | 正確答案的比例 | ✅ |

---

## 3. Leaderboard

- **MMLU Leaderboard**：https://paperswithcode.com/sota/multi-task-language-understanding-on-mmlu
- **Open LLM Leaderboard**：https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

---

## 4. 本專案實作說明

### 三種 Extractor

| 方法 | Extractor | 說明 |
|------|-----------|------|
| `pattern` | `PatternExtractor` | 正則表達式匹配答案字母（含中英文模式） |
| `box` | `BoxExtractor` | 提取 `\boxed{X}` 中的答案（推薦） |
| `custom_regex` | `CustomRegexExtractor` | 使用者自訂正則表達式 |

### Scorer

所有選擇題方法共用 `ExactMatchScorer`：正規化後精確比對答案字母。

### 特殊設計決策

- **`box` 模式推薦**：搭配 system prompt 指示模型使用 `\boxed{}` 格式，提取穩定度最高
- **選項隨機排列**：`shuffle_options: true` 可在每次評測時隨機排列選項順序，消除位置偏好
- **動態選項偵測**：不硬編碼 A/B/C/D，支援任意數量的選項（如 MMLU-Pro 的 A–J）

---

## 5. 使用方式

### config.yaml 範例

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/mmlu/"
  evaluation_method: "box"
  system_prompt:
    en: "Please read the question carefully and select the best answer. Present your final answer in the format \\boxed{answer}, e.g., \\boxed{A}."
  shuffle_options: true
```

### 完整 config template

參見 `twinkle_eval/templates/multiple_choice.yaml`
