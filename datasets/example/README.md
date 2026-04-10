# 範例評測資料集

用於快速驗證 Twinkle Eval 設定是否正確，以及除錯用途。每個資料集為原始 benchmark 的子集。

## 資料集清單

### 選擇題

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `tmmluplus/` | [ikala/tmmluplus](https://huggingface.co/datasets/ikala/tmmluplus) | 20 | `box` | 繁體中文選擇題（economics × 10、basic_medical_science × 10） |
| `mmlu/` | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | 20 | `box` | 英文選擇題 A–D（high_school_mathematics × 10、high_school_computer_science × 10）|
| `mmlu_pro/` | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) | 20 | `box` | 英文多選項選擇題 A–J（選項數 9–10 不固定） |
| `mmlu_redux/` | [edinburgh-dawg/mmlu-redux](https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux) | 10 | `box` | MMLU 修正版 — 人工審核並修正標注錯誤（anatomy × 5、college_mathematics × 5） |
| `supergpqa/` | [m-a-p/SuperGPQA](https://huggingface.co/datasets/m-a-p/SuperGPQA) | 10 | `box` | 研究所等級跨領域 QA — 285 個子學科、4–10 個選項（10 個不同學科各 1 題） |
| `gpqa/` | [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) | 10 | `box` | 研究所等級科學 QA（GPQA Diamond split）— 物理、化學、生物（gated dataset） |

### 數學

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `gsm8k/` | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | 20 | `math` | 國小/國中程度數學應用題 |
| `aime2025/` | [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025) | 30 | `math` | AIME 2025 競賽題（全題組） |

### Function Calling (BFCL)

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `bfcl/` | [gorilla-llm/Berkeley-Function-Calling-Leaderboard](https://huggingface.co/gorilla-llm) | 23 | `bfcl_fc` | BFCL 多種類型（simple / multiple / parallel / live_simple / multi_turn） |
| `bfcl_v1/` | 同上 | 15 | `bfcl_fc` | BFCL v1 子集（simple / multiple / parallel） |
| `bfcl_v2/` | 同上 | 5 | `bfcl_fc` | BFCL v2 live_simple 子集 |
| `bfcl_v3/` | 同上 | 3 | `bfcl_fc` | BFCL v3 multi_turn 子集 |

### Instruction Following

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `ifeval/` | [google/IFEval](https://huggingface.co/datasets/google/IFEval) | 10 | `ifeval` | Google IFEval — 25 種可驗證指令的遵循度評測 |
| `ifbench/` | [Yale-LILY/IFBench](https://huggingface.co/datasets/Yale-LILY/IFBench) | 14 | `ifbench` | IFBench — 語言學約束指令遵循評測（音節、詞頻、語音等） |

### 長文本 / 大海撈針 (NIAH)

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `niah/kamradt/` | Kamradt NeedleInAHaystack | 10 | `niah` | 英文大海撈針（不同 context 長度與 needle 深度） |
| `niah/needlebench/` | [NeedleBench](https://github.com/open-compass/opencompass) | 5 | `niah` | 中文大海撈針 |
| `niah/longbench/` | [THUDM/LongBench](https://huggingface.co/datasets/THUDM/LongBench) | 5 | `niah` | LongBench 段落檢索（中文） |

### RAG 評測 (RAGAS)

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `ragas/` | [explodinggradients/WikiEval](https://huggingface.co/datasets/explodinggradients/WikiEval) | 10 | `ragas` | LLM-as-judge 評估 RAG pipeline 品質（faithfulness / answer_relevancy / context_precision / context_recall） |

### Text-to-SQL

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `spider/` | [Spider 1.0](https://yale-lily.github.io/spider) | 10 | `text2sql` | 跨領域 Text-to-SQL（concert_singer、pets_1 兩個 SQLite DB） |
| `bird/` | [BIRD](https://bird-bench.github.io/) | 10 | `text2sql` | 含外部知識的 Text-to-SQL（california_schools、financial 兩個 SQLite DB） |
| `spider2_lite/` | [Spider 2.0-lite](https://spider2-sql.github.io/) | 10 | `text2sql` | 企業級 Text-to-SQL 的 SQLite 子集（book_store DB）。僅支援 lite 版（85 題 SQLite-only），完整版需 BigQuery/Snowflake 雲端帳號 |

### ASR（語音辨識）

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `asr/` | [OKHand/Clean_Common_Voice_Speech_24.0-TW](https://huggingface.co/datasets/OKHand/Clean_Common_Voice_Speech_24.0-TW) | 10 | `asr` | 繁體中文語音辨識（CER），含 WAV 音檔 + reference transcription |

### Regex Match（BBH 等混合格式）

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `bbh/` | [BIG-Bench-Hard](https://github.com/suzgunmirac/BIG-Bench-Hard) | 15 | `regex_match` | BBH 27 子任務取樣 — MC `(A)`/`(B)` + Binary `Yes`/`No` + Free-form（數字、字串序列） |

### Vision MCQ（視覺多選題，VLM）

| 目錄 | 來源 | 題數 | 評測方法 | 說明 |
|------|------|------|----------|------|
| `vision_mcq/` | [Lin-Chen/MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar) | 10 | `vision_mcq` | VLM 多模態選擇題 A–D，含 MMStar coarse/fine perception 與 instance reasoning 範例（內含 jpg 圖片） |

## 快速開始

### 選擇題（box 模式）

```yaml
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
evaluation:
  dataset_paths:
    - "datasets/example/gsm8k/"
    - "datasets/example/aime2025/"
  evaluation_method: math
  system_prompt:
    en: "Please solve the problem step by step. Present your final answer in \\boxed{answer} format."
```

### Function Calling（bfcl_fc 模式）

需先安裝：`pip install twinkle-eval[tool]`

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/bfcl/simple/"
  evaluation_method: bfcl_fc
```

### Instruction Following（ifeval 模式）

需先安裝：`pip install twinkle-eval[ifeval]`

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/ifeval/"
  evaluation_method: ifeval
```

### 大海撈針（niah 模式）

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/niah/kamradt/"
  evaluation_method: niah
```

### RAGAS（ragas 模式）

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/ragas/"
  evaluation_method: ragas
```

### Text-to-SQL（text2sql 模式）

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/spider/"
  evaluation_method: text2sql
  strategy_config:
    text2sql_scoring_mode: "exec"
    text2sql_db_base_path: "datasets/example/spider/databases"
```

### Regex Match（regex_match 模式，BBH）

```yaml
evaluation:
  dataset_paths:
    - "datasets/example/bbh/"
  evaluation_method: regex_match
  system_prompt:
    en: "Follow each question's instructions carefully. Think step by step. You MUST end your response with exactly this format: 'the answer is {your answer}' where {your answer} is the option like (A), (B), etc., or the exact value like Yes, No, True, False, valid, invalid, or the computed result."
```

### Vision MCQ（vision_mcq 模式，VLM）

需要 vision-capable 的 OpenAI 相容 API 端點（如 vLLM + Qwen2-VL、OpenAI GPT-4o、NVIDIA Build VLM）。
若需要圖片縮放，先安裝：`pip install twinkle-eval[vision]`

```yaml
llm_api:
  type: "openai"
  base_url: "http://localhost:8000/v1"
  api_key: "your-api-key"

model:
  name: "your-vlm-model"
  max_tokens: 1024

evaluation:
  dataset_paths:
    - "datasets/example/vision_mcq/"
  evaluation_method: vision_mcq
  strategy_config:
    image_field: "image_path"      # 圖片欄位名稱
    max_image_size: null           # 最長邊像素數；null 為不縮放
    image_detail: "auto"           # auto / low / high
```

### ASR（asr 模式）

需先安裝：`pip install twinkle-eval[asr]`

```yaml
llm_api:
  type: "whisper"                    # 或 "openai"（多模態模型）
  base_url: "http://localhost:8000/v1"
  api_key: "your-api-key"

evaluation:
  dataset_paths:
    - "datasets/librispeech/"
  evaluation_method: asr
  strategy_config:
    asr_language: "en"
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

**Function Calling（bfcl）**
```json
{"question": "[{\"role\": \"user\", \"content\": \"...\"}]", "functions": "[...]", "answer": "..."}
```

**Instruction Following（ifeval、ifbench）**
```json
{"question": "...", "instruction_id_list": "[...]", "kwargs": "[...]"}
```

**大海撈針（niah）**
```json
{"question": "...", "answer": "the secret number is 42"}
```

**RAGAS**
```json
{"question": "<pre-assembled judge prompt>", "answer": "<metadata JSON>"}
```

**Text-to-SQL（spider、bird、spider2_lite）**
```json
{"question": "<schema + question prompt>", "answer": "{\"sql\": \"SELECT ...\", \"db_id\": \"...\"}",  "db_id": "..."}
```

**ASR（librispeech、aishell1）**
```json
{"audio_path": "path/to/audio.wav", "answer": "reference transcription text"}
```

**Regex Match / BBH（bbh）**
```json
{"id": "bbh_task_0", "question": "...", "answer": "(B)", "subtask": "disambiguation_qa"}
```
`answer` 可為選項 `(A)`/`(B)`、布林值 `Yes`/`No`/`True`/`False`、數字 `24`、或字串序列 `barn damp dot`。

**Vision MCQ（vision_mcq）**
```json
{"id": "mmbench_001", "image_path": "datasets/example/vision_mcq/images/001.jpg", "question": "What is the dominant color of the cat?", "A": "Black", "B": "White", "C": "Orange", "D": "Grey", "answer": "C"}
```
POPE Yes/No 格式：
```json
{"id": "pope_001", "image_path": "datasets/example/vision_mcq/images/coco_001.jpg", "question": "Is there a dog in the image?", "answer": "No"}
```
圖片欄位名稱可透過 `strategy_config.image_field` 自訂（預設 `image_path`），支援本地路徑或 HTTP/HTTPS URL。

## 重新生成

```bash
python scripts/create_example_datasets.py
```
