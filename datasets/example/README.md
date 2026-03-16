# 範例評測資料集

用於快速驗證 Twinkle Eval 設定是否正確，以及除錯用途。每個資料集為原始 benchmark 的子集。

## 資料集清單

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `gsm8k/` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 20 | `math` | 國小/國中程度數學應用題 |
| `aime2025/` | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) | 30 | `math` | AIME 2025 競賽題（全題組） |
| `tmmluplus/` | [ikala/tmmluplus](https://huggingface.co/datasets/ikala/tmmluplus) | 20 | `box` | 繁體中文選擇題（economics × 10、basic_medical_science × 10） |
| `mmlu/` | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | 20 | `box` | 英文選擇題 A–D（high_school_mathematics × 10、high_school_computer_science × 10）|
| `mmlu_pro/` | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | 20 | `box` | 英文多選項選擇題 A–J（選項數 9–10 不固定） |

## 快速開始

### 選擇題（box 模式）

```yaml
# config.yaml
evaluation:
  dataset_paths:
    - "datasets/example/tmmluplus/"
    - "datasets/example/mmlu/"
    - "datasets/example/mmlu_pro/"
  evaluation_method: box
  system_prompt:
    zh: "請仔細閱讀以下問題，並從選項中選出最正確的答案。請將最終答案以 \\boxed{答案} 的格式呈現，例如 \\boxed{A}。"
    en: "Please read the question carefully and select the best answer. Present your final answer in the format \\boxed{answer}, e.g., \\boxed{A}."
  datasets_prompt_map:
    "datasets/example/mmlu/": "en"
    "datasets/example/mmlu_pro/": "en"
```

### 數學題（math 模式）

需先安裝：`pip install twinkle-eval[math]`

```yaml
# config.yaml
evaluation:
  dataset_paths:
    - "datasets/example/gsm8k/"
    - "datasets/example/aime2025/"
  evaluation_method: math
  system_prompt:
    en: "Please solve the problem step by step. Present your final answer in \\boxed{answer} format."
```

### MMLU-Pro（多選項，A–J）

```yaml
# config.yaml
evaluation:
  dataset_paths:
    - "datasets/example/mmlu_pro/"
  evaluation_method: box
  system_prompt:
    en: "Please read the question carefully and select the best answer. Present your final answer in the format \\boxed{answer}, e.g., \\boxed{A}."
  datasets_prompt_map:
    "datasets/example/mmlu_pro/": "en"
```

### 混合模式（dataset_overrides）

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/tmmluplus/"
    - "datasets/example/gsm8k/"
  evaluation_method: box
  system_prompt:
    zh: "請將最終答案以 \\boxed{答案} 格式呈現。"
    en: "Present your final answer in \\boxed{answer} format."
  dataset_overrides:
    "datasets/example/gsm8k/":
      evaluation_method: math
```

## 資料格式

**選擇題（tmmluplus、mmlu、mmlu_pro）**
```json
{"question": "...", "A": "...", "B": "...", "C": "...", "answer": "A"}
```
`mmlu` 保留原始 `choices` list + 整數 `answer`，由 Twinkle Eval 自動正規化。
`mmlu_pro` 已展開為 `A`–`J` 具名欄位。

**數學題（gsm8k、aime2025）**
```json
{"question": "...", "answer": "42"}
```

## 重新生成

```bash
python scripts/create_example_datasets.py
```
