# Math Evaluation（數學評測）

> 使用 `math` 評測方法，適用於 GSM8K、AIME 2025 等數學 benchmark。

---

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | 數學評測（GSM8K / AIME 2025 等） |
| **evaluation_method** | `math` |
| **實作狀態** | ✅ 完整實作 |
| **需要 optional deps** | `pip install twinkle-eval[math]`（mathruler, sympy, pylatexenc） |
| **實作者** | cyc00518 |

---

## 1. 來源

### GSM8K

- **標題**：Training Verifiers to Solve Math Word Problems
- **作者**：Karl Cobbe et al.
- **發表**：2021
- **連結**：https://arxiv.org/abs/2110.14168
- **HuggingFace**：`openai/gsm8k`

### AIME 2025

- **標題**：American Invitational Mathematics Examination 2025
- **HuggingFace**：`MathArena/aime_2025`

### 本專案 Example 資料

| 資料集 | 路徑 | 筆數 |
|--------|------|------|
| GSM8K | `datasets/example/gsm8k/` | 20 |
| AIME 2025 | `datasets/example/aime2025/` | 30（全題組） |

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

數學評測衡量模型的數學推理能力。GSM8K 為國小/國中程度應用題，AIME 為高中競賽等級。
模型需要展示解題過程並給出最終數值答案。

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| Accuracy | 最終答案正確的比例 | ✅ |

---

## 3. Leaderboard

- **GSM8K Leaderboard**：https://paperswithcode.com/sota/arithmetic-reasoning-on-gsm8k

---

## 4. 本專案實作說明

### Extractor

```
twinkle_eval/metrics/extractors/math.py
```

`MathExtractor` 從 LLM 回應中提取 `\boxed{}` 內的數學答案，支援巢狀括號。

### Scorer

```
twinkle_eval/metrics/scorers/math.py
```

`MathRulerScorer` 使用 [MathRuler](https://github.com/mathruler/mathruler) 進行數學等價比較，
支援分數、根號、科學記號等不同表示法的正規化與比對。

### Optional Dependencies

```bash
pip install twinkle-eval[math]
# 安裝 mathruler, sympy, pylatexenc
```

---

## 5. 使用方式

### config.yaml 範例

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/gsm8k/"
  evaluation_method: "math"
  system_prompt:
    en: "Please solve the problem step by step. Present your final answer in \\boxed{answer} format."
```

### 完整 config template

參見 `twinkle_eval/templates/math.yaml`
