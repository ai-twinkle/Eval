---
name: run-eval
description: 用 Twinkle Eval 跑一次評測——建立 config.yaml、下載或指定資料集、驗證設定、執行並讀結果。當使用者說「跑 benchmark」「跑評測」「建 config」「evaluate 這個模型」「twinkle-eval 怎麼跑」「評測結果怎麼看」「為什麼分數是 0」時使用。涵蓋 15 種 evaluation_method 的 config 差異、27 個內建可下載 benchmark、--validate / --dry-run / --resume，以及用 unparsed_rate 診斷 extractor 失效。
---

# 跑一次評測

## 前提：本專案不啟動模型服務

Twinkle Eval 只做一件事——**拿題目去呼叫已經在外部運行的 OpenAI 相容端點**。
模型要自己先起好（vLLM、Ollama、OpenAI、NVIDIA Build 都行），再把 `base_url` 填進 config。

端點沒回應時，本專案的正確行為是依 `max_retries` 重試後報錯退出，**不會**也**不該**嘗試
重啟服務（CLAUDE.md 原則 G）。

## 1. 產生 config

```bash
twinkle-eval --init                  # 列出全部 11 個範本
twinkle-eval --init multiple_choice  # 產生 configs/multiple_choice.yaml
twinkle-eval --init all              # 全部產生到 configs/
```

範本存放在 `twinkle_eval/templates/`，`--init` 直接掃該目錄。

### config 骨架

```yaml
llm_api:
  base_url: "http://localhost:8000/v1"   # 必填
  api_key: "EMPTY"                       # 必填（本地 vLLM 隨便填）
  api_rate_limit: -1                     # QPS，-1 為不限
  max_retries: 3
  timeout: 600
  disable_ssl_verify: false

model:
  name: "my-model"                       # 必填，會寫進結果路徑與紀錄
  temperature: 0.0
  top_p: 0.9
  max_tokens: 4096
  extra_body:                            # 傳給 API 的額外參數

evaluation:
  dataset_paths:                         # 必填，list（即使只有一個）
    - "datasets/example/tmmluplus/"
  evaluation_method: "box"               # 必填，見下表
  repeat_runs: 1                         # >1 時算平均與標準差
  shuffle_options: false                 # 選項隨機排列

logging:
  level: "INFO"
```

**每個 evaluation_method 的必填欄位與 `strategy_config` 參數不同**，
完整對照見 `references/config-reference.md`。

### 挑 evaluation_method

| 方法 | 用在 | 備註 |
|------|------|------|
| `pattern` | 選擇題，通用首選 | 正則比對，含中英文預設模式 |
| `box` | 選擇題，推理模型 | 提取 `\boxed{}` / `\box{}`；**需要 `system_prompt`** |
| `logit` | 多選項題 | 比較各選項 log-probability，不依賴輸出格式；選項數不限 |
| `math` | 數學推理 | `\boxed{}` + MathRuler；需 `[math]` |
| `regex_match` | BBH 之類自由格式 | ⚠️ `system_prompt` **不生效**，見下方說明 |
| `custom_regex` | 自訂格式 | 必須設 `strategy_config.patterns` |
| `ifeval` / `ifbench` | 指令遵循 | 需 `[ifeval]` / `[ifbench]` + nltk 資料 |
| `bfcl_fc` / `bfcl_prompt` | 函式呼叫 | FC 走 tools API，prompt 走注入 |
| `niah` | 長文本大海撈針 | |
| `ragas` | RAG 品質 | |
| `text2sql` | Text-to-SQL | 需設 `text2sql_db_base_path` |
| `asr` | 語音辨識 | 需 `[asr]`；`llm_api.type: whisper` 或多模態 |
| `vision_mcq` | 視覺多選題 | 需 `[vision]`（縮放用） |

> ⚠️ **`evaluation.system_prompt` 只對 `box` 與 `math` 生效。**
> `models/openai.py` 的 `_build_messages()` 以白名單決定是否送出 system message
> （`method in {"box", "math"}`），其他方法即使在 config 設了 `system_prompt` 也不會進 request。
> `regex_match` 這類需要指定輸出格式的方法，格式要求必須寫進資料集的 `question` 欄位裡。
> （專案的 `templates/regex_match.yaml` 目前有同樣的誤導，見 [#144](https://github.com/ai-twinkle/Eval/issues/144)）

```bash
twinkle-eval --list-strategies   # 執行期確認可用方法
```

## 2. 準備資料集

```bash
twinkle-eval --download-dataset list          # 列出 27 個內建 benchmark
twinkle-eval --download-dataset mmlu          # 短名稱
twinkle-eval --download-dataset tmmluplus gsm8k
twinkle-eval --download-dataset all
```

內建短名稱：`mmlu` `mmlu_pro` `mmlu_redux` `tmmluplus` `supergpqa` `gpqa` `formosa_bench`
`gsm8k` `aime2025` `bbh` `ifeval` `ifbench` `bfcl` `needlebench` `longbench` `wikieval`
`librispeech` `aishell1` `fleurs` `common_voice` `mmbench` `mmstar` `mmmu` `pope`
`spider` `bird` `spider2_lite`

`gpqa` 是 gated dataset，會互動式要求 HuggingFace token。

**先用 example 資料集驗證流程跑得通**再下載完整 benchmark：
`datasets/example/` 底下每個 benchmark 都有 10–30 筆的子集，跑一次只要幾秒。

## 3. 本機 config 的命名規則（重要）

含真實 API 金鑰的 config **絕對不得 commit**（CLAUDE.md 原則 E）。這三種前綴/後綴已被
`.gitignore` 全局排除，本機測試一律用其中之一：

```
config_local_*.yaml
config_test_*.yaml
*.local.yaml
```

寫入任何含金鑰的 config 前，**先確認該路徑已在 `.gitignore` 中**。不確定就檢查：

```bash
git check-ignore -v config_local_myrun.yaml   # 有輸出 = 已被忽略
git diff --staged | grep -i "api_key"         # commit 前確認
```

## 4. 執行

```bash
twinkle-eval --validate --config config_local_myrun.yaml   # 只驗設定與資料集，不呼叫 API
twinkle-eval --dry-run  --config config_local_myrun.yaml   # 顯示評測計畫，不呼叫 API
twinkle-eval --config config_local_myrun.yaml              # 正式跑
twinkle-eval --config config_local_myrun.yaml --export json csv html excel
```

**永遠先跑 `--validate` 再跑正式評測。** 資料集路徑錯、缺必填欄位、格式不符都會在這一步
抓到，省下一整輪 API 費用與時間。

中斷後續跑（⚠️ 目前失效，見 [#145](https://github.com/ai-twinkle/Eval/issues/145)）：

```bash
twinkle-eval --resume 20260825_1430 --config config_local_myrun.yaml
```

## 5. 讀結果

```
results/
├── results_{timestamp}.json                 # 整體摘要
└── eval_results_{timestamp}_run{N}.jsonl    # 各題明細（append 模式）
```

摘要含 `dataset_results`（各資料集的 `average_accuracy`、`average_std`、
`average_pass_at_k`、`total_unparsed_count`）與 `duration_seconds`。
`config` 欄位是**移除 api_key 後**的設定。

明細每行含 `question_id` / `sample_id` / `question` / `correct_answer` /
`predicted_answer` / `is_correct` / `llm_output` / `llm_reasoning_output` / token 用量。
各路徑另有增補欄位（`logit` 的 `logprob_scores`、`ifeval` 的四個指標、`vision_mcq` 的
`image_path`、`asr` 的 `wer` / `cer`）。

> ⚠️ 實際輸出**沒有** `file` 與 `timestamp` 欄位（CLAUDE.md §10 的規範與實作不符）。
> 這也讓 `--resume` 目前無法運作：它以 `{file}|{question_id}` 作為已完成紀錄的 key，
> `file` 缺失使得 key 變成 `|{idx}`，與比對端的 `{檔案路徑}|` 永遠不匹配，
> 結果是一題都不會跳過、整輪重跑並 append 出重複列。見 [#145](https://github.com/ai-twinkle/Eval/issues/145)。

```bash
# 快速看正確率
jq -r '.dataset_results | to_entries[] | "\(.key): \(.value.average_accuracy)"' \
  results/results_*.json

# 撈出所有答錯的題目
jq -c 'select(.is_correct == false) | {question_id, predicted_answer, correct_answer}' \
  results/eval_results_*_run0.jsonl | head
```

## 診斷：分數異常低

先看 **`unparsed_rate`**——這是判斷「模型答錯」還是「extractor 沒抓到」的關鍵。

```bash
# 資料集層級（摘要 JSON 用 average_unparsed_rate，per-file 才叫 unparsed_rate）
jq -r '.dataset_results | to_entries[] | "\(.key): \(.value.average_unparsed_rate)"' \
  results/results_*.json
```

| 症狀 | 多半是 |
|------|--------|
| `unparsed_rate` 高（>10%） | extractor 沒對上輸出格式 |
| `unparsed_rate` ≈ 0 但分數低 | 模型是真的答錯 |
| 全部 0 分且無 unparsed | ground truth 欄位或正規化對不上 |
| 「所有資料集評測均失敗」 | 資料集路徑、格式，或 API 端點問題 |

extractor 沒抓到時，撈幾筆 `llm_output` 出來看實際格式：

```bash
jq -r 'select(.predicted_answer == null) | .llm_output' \
  results/eval_results_*_run0.jsonl | head -3
```

常見成因：

- **`box` 方法但沒設 `system_prompt`** → 模型不知道要用 `\boxed{}`，自然抓不到
- **推理模型的 think tag** → evaluator 會剝離完整的 `<think>...</think>` / `<reason>` /
  `<reasoning>` 標籤對；只有結尾 tag 而無開頭 tag 會被視為格式不合格而原樣保留
- **`content` 為 null**（vLLM `skip_special_tokens=true`）→ 會回退讀 `reasoning`，
  再回退 `reasoning_content`（vLLM 0.18+ 改名，兩者都支援）
- **選項超過 4 個**（MMLU-Pro A–J、SuperGPQA）→ 確認 extractor 支援多字母選項

## 效能

並行度由 `ThreadPoolExecutor` 預設值決定，用 `llm_api.api_rate_limit` 節流：

- 本地 vLLM：`-1`（不限）
- 有 QPS 限制的商用 API：設成實際上限，否則會大量 429

`repeat_runs > 1` 會線性放大時間，但能得到標準差——量化模型穩定性時才開。
