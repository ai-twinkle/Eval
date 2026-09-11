# config.yaml 完整參照

各 `evaluation_method` 的必填欄位與 `strategy_config` 參數對照。
欄位名稱一律 snake_case；新欄位都有預設值，舊 config 不會因新版本報錯。

## 通用區塊

### `llm_api`

| 欄位 | 預設 | 說明 |
|------|------|------|
| `base_url` | —（必填） | OpenAI 相容端點，含 `/v1` |
| `api_key` | —（必填） | 本地服務隨便填；**儲存結果時會自動移除** |
| `type` | `openai` | `openai` / `whisper`（ASR 專用） |
| `api_rate_limit` | `-1` | 每秒請求數，`-1` 為不限 |
| `max_retries` | `3` | 失敗重試次數 |
| `timeout` | `600` | 單次請求逾時（秒） |
| `disable_ssl_verify` | `false` | 自簽憑證環境設 `true` |

### `model`

| 欄位 | 預設 |
|------|------|
| `name` | —（必填），會寫進結果路徑 |
| `temperature` | `0.0` |
| `top_p` | `0.9` |
| `max_tokens` | `4096` |
| `frequency_penalty` | `0.0` |
| `presence_penalty` | `0.0` |
| `extra_body` | `{}` — 傳給 API 的額外參數，如 `{"chat_template_kwargs": {...}}` |

### `evaluation`

| 欄位 | 預設 | 說明 |
|------|------|------|
| `dataset_paths` | —（必填） | list，即使只有一個路徑 |
| `evaluation_method` | —（必填） | 見下方各方法 |
| `system_prompt` | — | `{zh: ..., en: ...}`；`box` / `math` 會用 |
| `system_prompt_enabled` | `true` | |
| `datasets_prompt_map` | `{}` | `{"datasets/mmlu/": "en"}` 指定資料集用哪種語言 |
| `repeat_runs` | `1` | >1 時計算平均與標準差 |
| `shuffle_options` | `false` | 選項隨機排列，消除位置偏好 |
| `samples_per_question` | `1` | 每題採樣數（pass@k 用） |
| `pass_k` | `1` | |
| `strategy_config` | `{}` | 傳給 Extractor / Scorer 的參數 |
| `dataset_overrides` | `{}` | 依資料集路徑前綴覆寫設定，見下 |

### `dataset_overrides`

同一次 run 內對不同資料集套不同設定，key 是路徑前綴：

```yaml
evaluation:
  dataset_overrides:
    "datasets/gsm8k/":
      system_prompt_enabled: false
      temperature: 0.6
    "datasets/aime2025/":
      samples_per_question: 8
      pass_k: 4
```

可覆寫：`evaluation_method`、`system_prompt_enabled`、`samples_per_question`、
`pass_k`、`repeat_runs`、`shuffle_options`，以及 model 參數
`temperature` / `top_p` / `max_tokens` / `frequency_penalty` / `presence_penalty`。

### `logging`

`level`: `DEBUG` / `INFO` / `WARNING` / `ERROR`

---

## 各方法專屬設定

### `pattern` — 正則比對（選擇題通用）

```yaml
evaluation:
  evaluation_method: "pattern"
  strategy_config:
    patterns:            # 可選，覆寫預設的中英文模式
      - "答案[是為：:]\\s*([A-Z])"
```

### `box` — 提取 `\boxed{}`

**必須設 `system_prompt`**，否則模型不會用 `\boxed{}` 格式，全部抓不到。

```yaml
evaluation:
  evaluation_method: "box"
  system_prompt:
    zh: "請將最終答案放在 \\boxed{} 中。"
    en: "Put your final answer in \\boxed{}."
  strategy_config:
    patterns: [...]      # 可選
```

### `logit` — 比較選項 log-probability

不生成文字，改用 `/v1/completions` 的 echo 模式逐選項算 log-likelihood。
端點必須支援 completions API 與 `logprobs`。適合多選項題，完全不依賴輸出格式。

```yaml
evaluation:
  evaluation_method: "logit"
```

### `math` — 數學推理

```yaml
evaluation:
  evaluation_method: "math"
  system_prompt:
    zh: "請逐步推理，並將最終答案放在 \\boxed{} 中。"
```

需 `pip install twinkle-eval[math]`（mathruler、sympy、pylatexenc）。

### `custom_regex` — 自訂正則

**必須**設 `strategy_config.patterns`，否則初始化就會失敗。

```yaml
evaluation:
  evaluation_method: "custom_regex"
  strategy_config:
    patterns:
      - "final answer:\\s*([A-Z])"
```

### `regex_match` — 自由格式字串比對（BBH）

> ⚠️ **`system_prompt` 對這個方法不生效**（見本檔開頭「`evaluation`」一節與
> [#144](https://github.com/ai-twinkle/Eval/issues/144)）。`models/openai.py` 的白名單只認
> `box` / `math`。輸出格式要求必須寫進資料集的 `question` 欄位，寫在 config 裡不會送出。
> 專案的 `templates/regex_match.yaml` 目前仍有這個誤導。

```yaml
evaluation:
  evaluation_method: "regex_match"
  # 注意：這裡不放 system_prompt——設了也不會送出
  strategy_config:
    answer_pattern: [...]        # 可選，覆寫預設
    normalize_mode: "strip"      # strip / upper / lower / none（非法值會 raise ValueError）
```

### `ifeval` / `ifbench` — 指令遵循

```yaml
evaluation:
  evaluation_method: "ifeval"
  dataset_paths: ["datasets/ifeval/"]
```

資料集欄位：`id`、`question`（IFBench 用 `prompt`）、`instruction_id_list`、`kwargs`。
兩者都吃 JSON string 與原生 list/dict。

```bash
pip install twinkle-eval[ifeval]     # langdetect, nltk
pip install twinkle-eval[ifbench]    # emoji, syllapy, nltk
python -c "import nltk; nltk.download('punkt_tab')"
```

輸出四個指標：`prompt_strict` / `prompt_loose` / `instruction_strict` / `instruction_loose`。
`accuracy` 等同 `prompt_strict`。

### `bfcl_fc` / `bfcl_prompt` — 函式呼叫

- `bfcl_fc`：把 function 定義轉成 OpenAI `tools` 送出，讀 `message.tool_calls`。端點必須支援 tool calling。
- `bfcl_prompt`：把 function 定義注入 system prompt，從文字回應解析。

```yaml
evaluation:
  evaluation_method: "bfcl_fc"
  dataset_paths:
    - "datasets/bfcl/simple/"
    - "datasets/bfcl/multiple/"
    - "datasets/bfcl/parallel/"
  repeat_runs: 1          # ground truth 固定，跑 1 次即可
```

資料集欄位：`question`（JSON messages）、`functions`（JSON）、`answer`。
需 `pip install twinkle-eval[tool]`。

### `niah` — 長文本大海撈針

```yaml
evaluation:
  evaluation_method: "niah"
  strategy_config:
    niah_scoring_mode: "substring"   # substring / exact / f1（LongBench passage_retrieval_zh 用 exact）
    niah_f1_threshold: 0.5
```

### `ragas` — RAG 品質

```yaml
evaluation:
  evaluation_method: "ragas"
  strategy_config:
    ragas_threshold: 0.5
```

### `text2sql` — Text-to-SQL

```yaml
evaluation:
  evaluation_method: "text2sql"
  strategy_config:
    text2sql_scoring_mode: "exec"                        # exec / em（只有 "em" 走 Exact Match，
                                                         # 其他任何值都落進 exec 分支，無驗證）
    text2sql_db_base_path: "datasets/spider/databases"   # exec 模式必填
    text2sql_timeout: 30
```

`exec` 模式會實際執行 SQL 比對結果集，需要資料庫檔案在 `text2sql_db_base_path` 底下。

> ⚠️ 下列任一情況會**靜默回退成 Exact Match**且不留 log：`db_base_path` 為空、
> 題目缺 `db_id`、對應的 `.sqlite` 不存在、或 gold SQL 執行失敗。
> 分數看起來偏低時，先確認資料庫檔案路徑正確。

### `asr` — 語音辨識

兩條路徑：Whisper API（`llm_api.type: whisper`）或 Chat Completions 多模態。

```yaml
llm_api:
  type: "whisper"
evaluation:
  evaluation_method: "asr"
  strategy_config:
    asr_language: "zh"           # 語言代碼
    asr_metric: "auto"           # auto（依語言選）/ wer / cer
    remove_punctuation: true
    to_lower: true
    normalize_unicode: true      # NFKC
```

資料集欄位：`audio_path`（或 `question`）、`answer`。
需 `pip install twinkle-eval[asr]`（jiwer）。額外輸出 `avg_wer` / `avg_cer`。

### `vision_mcq` — 視覺多選題

```yaml
evaluation:
  evaluation_method: "vision_mcq"
  strategy_config:
    image_field: "image_path"    # 圖片路徑/URL 的欄位名
    max_image_size: null         # 最長邊像素；null 不縮放（縮放需 Pillow）
    image_detail: "auto"         # auto / low / high
```

支援本地檔案（自動 base64 data URI，magic bytes 偵測 MIME）與 http(s) URL（直接傳遞）。
單張圖片上限 50 MB。需 `pip install twinkle-eval[vision]`（僅縮放需要）。

支援字母答案（A–Z）與 Yes/No 二元答案（POPE 等幻覺偵測），優先解析 `\boxed{}`。

---

## 輸出格式

```yaml
# CLI: --export json csv html excel google_sheets
# 注意：excel 需要 openpyxl，但它目前未被宣告為依賴，乾淨環境會在評測跑完後才失敗
#       見 https://github.com/ai-twinkle/Eval/issues/147
```

`results_{timestamp}.json` 必要欄位：`timestamp`、`config`（已移除 api_key）、
`dataset_results`、`duration_seconds`。

`eval_results_{timestamp}_run{N}.jsonl` 每行實際欄位：`question_id`、`sample_id`、
`question`、`correct_answer`、`predicted_answer`、`is_correct`、`llm_output`、
`llm_reasoning_output`、`usage_*`。各路徑另有增補欄位：`logit` 有 `logprob_scores`、
`ifeval` / `ifbench` 有 `prompt_strict` / `prompt_loose` / `instruction_strict` /
`instruction_loose`、`vision_mcq` 有 `image_path`、`asr` 有 `wer` / `cer`。
寫入一律 append 模式，確保多檔多 run 累積不遺失。

> ⚠️ CLAUDE.md §10 規定每行要有 `timestamp` 與 `file`，但實作並未寫出這兩個欄位。
> `file` 缺失導致 `--resume` 無法運作（它以 `{file}|{question_id}` 為 key），見 [#145](https://github.com/ai-twinkle/Eval/issues/145)。
> 既有欄位名稱不得修改（向下相容）。
