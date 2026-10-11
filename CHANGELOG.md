# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> ⚠️ **有 API 失敗的評測，分數會改變。** 失敗的題目（重試耗盡）現在計為答錯並計入分母，
> 先前則是不計入分母而使分數偏高。IFEval 的 instruction-level 指標與 ASR 的 WER / CER
> 同樣計入失敗題。
>
> ⚠️ **logprobs（`logit`）評測遇到端點不支援 echo / logprobs 時會改為報錯**：
> 先前每題都回報 unparsed、整檔以 0% 正確率計入結果；現在視為 API 失敗，整檔拋出
> `EvaluationError` 並排除在 `repeat_runs` 平均之外。
>
> ⚠️ **Parquet / CSV 資料集的自由作答題分數可能變動。** answer 正規化改為所有檔案格式一致：
> 只有當 answer 是 1–2 個字母、且對應到該題實際存在的選項鍵時才去空白並轉大寫。先前
> Parquet / CSV 會把所有 answer 轉大寫並去空白，破壞了數學（`\frac` → `\FRAC`）、SQL、
> 逐字稿等 gold answer；JSON / JSONL 則完全不正規化。選擇題分數不受影響（scorer 本身會正規化）。
>
> ⚠️ **結果檔名的時間戳改為秒級**：`results_{timestamp}.json` 與
> `eval_results_{timestamp}_run{N}.jsonl` 的 `{timestamp}` 由 `%Y%m%d_%H%M` 改為
> `%Y%m%d_%H%M%S`。依檔名格式解析結果的下游工具需調整；舊格式的結果仍可用 `--resume` 續跑。

### Added
- **OpenAI Responses API 後端**（`llm_api.type: "openai_responses"`）：呼叫 `/v1/responses`，
  於 `call()` 內轉換為 ChatCompletion 格式，所有評測路徑不需修改。支援多模態輸入、
  reasoning summary（對應 `message.reasoning`）、tool calls；`num_samples > 1` 以多次請求合併；
  `max_tokens` 對應 `max_output_tokens`，frequency / presence penalty 不支援而忽略；
  `finish_reason` 依回應狀態對應（截斷為 `length`、tool calls 為 `tool_calls`）。
  reasoning summary 需在 `extra_body` 設定 `reasoning.summary` 才會回傳
- **`--convert-to-excel JSONL_FILE`**：將逐題結果 JSONL 轉為 `.xlsx`，常用欄位在前、
  各 benchmark 的額外欄位依出現順序附加；處理 Excel 儲存格長度上限與非法控制字元，
  `=` 開頭的內容一律以純文字寫入（避免模型輸出在開檔時被當成公式執行）。
  需安裝新的可選依賴群組 `twinkle-eval[excel]`（openpyxl）
- `llm_api.max_workers`（可選，預設 `null`；必須是正整數或 `null`，由 `--validate` 檢查）：
  控制並行請求的執行緒數，各範本已加入此欄位；`llm_api.type` 也加入各範本以標示可選的後端
- `model.temperature` / `model.top_p` 可設為 `null`：不送出該參數、交由伺服器決定
  （OpenAI o 系列等推理模型不接受 temperature）。Responses API 的 reasoning 設定
  （如 `reasoning: {effort: ..., summary: "auto"}`）透過 `model.extra_body` 傳入
- 分散式腳本（`scripts/run_slurm_*.sh`）匯出的 `TWINKLE_EVAL_RUN_TIMESTAMP` 現在會被 runner
  採用，各節點寫入同一組檔名，之後即可用該時間戳執行 `--finalize-results`
- 多資料集評測時，每完成一個資料集就印出累計成績

### Changed
- **evaluator 依 Extractor 的 capability flag 分派評測管線**：`evaluate_file()` 拆成七條
  獨立的 `_run_*_pipeline()`，以 `Evaluator._PIPELINES` 資料驅動分派；統計與逐題紀錄
  集中於 `_EvalAccumulator`。新增評測路徑時只需註冊，不需修改 `evaluate_file()`。
  在沒有 API 失敗、沒有 `--resume` 的情況下，七條路徑送出的 prompt、JSONL 內容與 metrics
  與先前相同（JSONL 多了 `file` 欄位）
- JSONL 每列新增 `file` 欄位（來源資料集檔案路徑），作為 `--resume` 比對已完成題目的依據
- **`--validate` 會對 API 端點試打一次**：結構驗證通過後，及早回報連線 / API 金鑰 / 模型名稱
  錯誤並以 exit code 1 結束。試打使用 30 秒 timeout 且不重試；`logit` 方法改以 completions
  logprobs 試打，音檔評測略過；`--dry-run` 仍不連網
- pandas / pyarrow / HF `datasets` 改為在函式內延遲載入，純 JSONL 評測不再負擔其啟動時間
- HTML 報表的逐題渲染去重複，改以 list 組裝字串
- 套件全面以 black / isort 格式化（不改變行為）

### Fixed
- **單題 API 失敗不再毀掉整檔結果，也不再灌水分數**：失敗的 sample 計為答錯，
  `failed_count` 寫入逐檔 metrics 與 `results_*.json`（`failed_count`、`total_failed_count`）。
  失敗題不寫入 JSONL，`--resume` 會重新評測它們
- **整檔 API 全部失敗時拋出 `EvaluationError`**，而不是回報 `accuracy = 0` 並被計入
  `repeat_runs` 的平均
- **logprobs 路徑任一選項失敗即整題視為失敗**：`score_continuation()` 出錯時回傳 `-inf`，
  先前會在其餘選項中挑答案並照常計分
- **`--resume` 先前從未跳過任何題目**：讀取已完成紀錄時比對的 `file` 欄位，evaluator 從未寫入。
  現在已完成的題目會跳過，其結果回填進統計，summary 涵蓋整個資料集；回填涵蓋
  `samples_per_question > 1` 的每個 sample，以及 IFEval（`prompt_loose`、`instruction_*`）與
  ASR（WER / CER）的附加指標。缺少 `file` 欄位的舊版結果列無法比對，會提示並重新評測；
  讀到紀錄卻沒有任何一筆對應到評測檔案時（例如資料集路徑寫法改變）也會提示
- `scorer.score_full` 的 introspection 延遲到 IFEval 路徑首次使用時才進行
- `shuffle_options` 只套用在選擇題路徑（logprobs / vision / text）

- **`--download-dataset` 在指定的 split 不存在時改為下載所有 split**：先前假設一定有 `test`
  split，split 名稱自訂的資料集會完全下載失敗；多個 split 也不再互相覆蓋同名 parquet 檔
- 開啟 Google Drive 上傳時，本次評測的 log 檔一定會被上傳：先前以 runner 的時間戳比對 log
  檔名，而 log 檔的時間戳在 import 時決定，兩者不在同一秒（秒級時間戳下）就會漏傳

- **BFCL FC（`bfcl_fc`）遇到不合法的 tool call 參數不再中止整個檔案**：先前一題的 arguments
  不是合法 JSON（或是 `5`、`null` 這類非物件的 JSON）就會拋出例外，該檔案已完成的題目也
  一併遺失。現在該題計為無法解析，原始 tool calls 保存在 JSONL 新增的 `tool_calls` 欄位；
  arguments 缺少、為空字串或只有空白時視為沒有參數。`BFCLScorer` 遇到非物件參數時判為不符
  （`bfcl_prompt` 路徑同樣受保護）
- **BFCL FC 保存推理內容**：`llm_reasoning_output` 先前固定為 null，現在與其他路徑一樣
  讀取 `message.reasoning` / `reasoning_content`
- **`--download-dataset bfcl` 可以使用了**：HF repo 只有原始的 `BFCL_v3_*.json`，`load_dataset`
  無法讀取，下載一律失敗。現在逐檔下載題目與答案並以 `merge_bfcl_files` 合併為
  `datasets/bfcl/{category}.jsonl`（8 個單輪 AST 類別，共 2,350 題）。HF repo 版本鎖定為
  已驗證的 revision；任一類別下載失敗時不會留下只有部分類別的目錄

### Removed
- `twinkle_eval.core.Registry`：匯出但從未被使用（各工廠有自己的註冊表）
- `core/validators.py` 的 `validate_dataset_content` 與 `RuntimeValidator`：沒有任何呼叫端，
  且前者硬編碼 A/B/C/D 選項標籤（違反原則 C）

### 注意
- JSONL 仍在**整個檔案評測完成後**才寫入。若在檔案評測途中中斷（Ctrl-C、OOM、被搶佔），
  該檔案已完成的題目不會留下紀錄，`--resume` 會從頭評測該檔案；question-level 續跑只涵蓋
  已寫出的檔案與 API 失敗的題目
- 以 `--resume` 續跑時，若剩下的題目全部 API 失敗，因已有回填的結果，不會拋出例外，
  而是以失敗題計為答錯產生結果（`failed_count` 會反映）

## [2.10.0] - 2026-09-15

> ⚠️ **視覺評測的分數會上升。** `mmbench`、`mmstar`、`mmmu`、`pope`、`vistw_mcq`
> 都受 #166 的修復影響——先前抓不到的答案現在讀得到了。與舊版分數不可直接比較。

### Added
- **VisTW 繁體中文台灣在地視覺語言評測**（Milestone #23，arXiv 2503.10427，CC BY 4.0）。
  這是本專案第一個繁中視覺評測——既有的 MMBench、MMStar、MMMU、POPE 全是英文或簡體中文為主。
  - **MCQ 子集**（21 學科）：沿用既有的 `vision_mcq`，不新增評測方法。兩個協定範本：
    `vistw_mcq.yaml`（要求 `\boxed{}`，本專案協定）與 `vistw_mcq_native.yaml`（官方
    `BASELINE_PROMPT` 逐字，用於對比 leaderboard）
  - **Dialogue 子集**：新增 `vistw_dialogue`（生成）與 `vistw_judge`（LLM judge 0–10 分）
    兩個評測方法。兩階段都走既有的 evaluator 路徑，evaluator 未改動
  - `scripts/build_vistw_judge_dataset.py`、example 資料集、`docs/evals/vistw.md`

### Fixed
- **`VisionMCQExtractor` 抓不到 `答案: $A`**（#166）：官方 VisTW 的 prompt 教模型輸出
  「答案: $字母」，模型照抄那個 `$`。更嚴重的是抓不到之後會退而用較寬鬆的 pattern
  **抓到推理文字中的其他字母，把正確作答判成答錯**——那比 unparsed 危險，因為它不會
  出現在 `unparsed_rate` 裡。現比照官方在比對前剝掉 `$`。
  實測 253 題：75.89%（11 題 unparsed）→ **81.82%（0 題 unparsed）**。
- **`--benchmark` 因缺少 `import copy` 而 `NameError` 崩潰**（#161/#162）：2.9.0 移除
  `main.py` 的 runner class 時把 `import copy` 一併帶走，而 `--benchmark` 的金鑰清理
  還在用它。

### Changed
- VisTW 範本的 `max_tokens` 2048 → 8192。推理型 VLM（如 gemma 4）會在寫出答案前用掉
  大量 token，實測 2048 時 19% 因截斷而 unparsed、分數低估 19 個百分點；不設上限則會
  推理到代理層逾時。

### 對比驗證
與官方實作（TMMMU-Benchmark/evaluation）在相同模型、相同端點、相同 253 題下對比：
**Twinkle Eval 81.01% vs 官方 80.56%，差 0.45 個百分點**，在 §6.3 的 ±2% 容差內。
完整方法與四項對齊修改記錄於 `docs/evals/vistw.md` §6。

## [2.9.0] - 2026-09-13

> ⚠️ **本版會改變評測分數。** 修正了題目 metadata 欄位被送進 prompt 的問題（#143）。
> 含 `id` / `domain` / `discipline` / `subfield` / `category` 等欄位的資料集
> （GPQA、SuperGPQA、MMLU-Redux、vision_mcq）分數預期會**下降**，因為先前的 prompt
> 夾帶了學科分類，等同免費提示。與舊版分數不可直接比較，請重跑基準。

### Added
- `twinkle_eval.datasets.index_to_label()`：`_index_to_label()` 提升為公開 API（保留舊名稱別名）
- `runners.evaluator.detect_option_keys()`：動態偵測選項鍵，支援任意數量選項
- `runners.evaluator.build_question_text()`：統一三條路徑的題目組建邏輯
- `runners.evaluator.describe_dropped_fields()`：每個檔案提示一次哪些非選項欄位不會進入
  prompt，讓「作答必需的欄位（hint、context）被靜默丟棄」變成可觀察，而非無聲的分數下降
- `tests/test_shuffle_options.py`：38 個測試，涵蓋選項偵測、重排、prompt 組建、語意標籤保護
  與 runner 單一實作

### Fixed
- **題目 metadata 被當成選項送進 prompt**（#143）：文字與 logit 路徑以
  「排除 `question` / `answer` 後其餘欄位全印」的方式組題目，導致 `id`、`domain`、
  `discipline`、`category` 等欄位進入 prompt。這既是雜訊，學科分類欄位更等同答題提示。
  **`shuffle_options` 預設為 `false`，預設路徑即受影響**，此行為自 2.8.0 以前即存在。
  現改為：偵測得到選項鍵時只列出選項；偵測不到時（Text-to-SQL 等）維持列出其餘欄位，
  因為 `db_id` / `evidence` 本來就該進 prompt。vision 路徑一併統一（原本漏排除 `category`）。
  **僅解決選擇題路徑**——沒有選項鍵的資料集（NIAH 洩漏 `needle_depth`、RAGAS 洩漏
  `answer_type`、BBH 洩漏 `subtask`）仍有同類問題，需要區分「哪些非選項欄位該進 prompt」，
  另行追蹤於 #146。
- **選項重排硬編碼 A/B/C/D**（#140）：`shuffle_question_options()` 的註解宣稱動態偵測，
  實作仍寫死 `["A","B","C","D"]`（違反 §2 原則 C）。MMLU-Pro（A–J）、SuperGPQA（4–10 選項）
  開啟 `shuffle_options: true` 時，E 之後的選項被靜默丟棄；正解落在 E–J 時 `answer` 無法回填，
  該題直接被跳過。現以 `detect_option_keys()` 動態偵測，並對未從 A 起始的選項鍵
  （如 T/F、Y/N）提供單字元回退。這類標籤帶有語意，**不會被重排**——重排會讓 `T` 指向「否」，
  使正確作答被判為錯。偵測結果少於兩個選項時視為沒有選項，避免單一 `A` 欄位讓其餘欄位
  （如 `context`）被丟棄。
- **選項重排丟棄非選項欄位**（#141）：重排時從空 dict 重建題目，`image_path` / `id` 等欄位
  全部遺失。vision 路徑在重排後才讀圖片欄位，導致 `vision_mcq` + `shuffle_options: true`
  每題都被 skip、**評測到零題**。現以 `dict(question_data)` 為基底，只覆寫選項鍵與 `answer`。
- 選項重排改以「原始選項鍵」而非選項文字定位正解，修正兩個選項文字相同時 `answer`
  指向錯誤選項的問題。
- 無法在選項鍵中定位正解時原樣回傳不重排（取代原本回填失敗後拋 KeyError 被跳過的行為），
  並降為 warning——這是資料格式問題，不是執行錯誤。

### Changed
- **`TwinkleEvalRunner` 去重**（#142）：`main.py` 與 `runners/standard.py` 各有一份實作且已分歧
  （相似度約 68%）。`main.py` 那份是實際被使用的、且是功能超集（多了 `--resume` 的
  `completed_records` 參數），`runners/standard.py` 則是無人 import 的死程式碼。
  註：`--resume` 本身目前是壞的（JSONL 缺 `file` 欄位，見 #145），此處只是忠實保留既有行為。
  現以 `main.py` 的實作為準搬入 `runners/standard.py`，`main.py` 改為重新匯出，符合 §4
  「main.py 不實作具體評測邏輯」。`twinkle_eval`、`twinkle_eval.main`、`twinkle_eval.runners`、
  `twinkle_eval.runners.standard` 四條 import 路徑皆維持可用（原則 F），行為不變。
- `results_{timestamp}.json` 的 `accuracy_std` 在 `repeat_runs: 1` 時由 `0` 改為 `0.0`
  （搬移時採用明確的 `float()` 轉型）。JSON number 語意相同，下游 parser 不受影響。

<!-- 以下來自 #160（system_prompt），與本版一同釋出 -->
### Fixed
- **`evaluation.system_prompt` 只對 `box` 與 `math` 生效**（#144）：`_build_messages()` 以
  `method in {"box", "math"}` 的硬編碼白名單決定是否送出，其餘方法即使設了也不進 request。
  `templates/regex_match.yaml` 這個官方範本自己就設了一段指定輸出格式的 prompt 卻從未送出，
  使用者照範本設定反而得到爆高的 `unparsed_rate` 且無從得知原因。
  改為：只要設了非空 `system_prompt` 且 `system_prompt_enabled` 為真就送出。
- **vision 路徑在任何設定下都送不出 system prompt**：它不經過 `_build_messages()`，
  evaluator 自己組 multimodal messages 直接傳給 `call()`。`_build_vision_messages()`
  新增 optional 的 `system_prompt` 參數。
- `templates/regex_match.yaml` 補上 `datasets_prompt_map`。該範本只定義 `en` 鍵，而
  `prompt_lang` 預設為 `zh`，所以即使移除白名單，prompt 仍然送不出去。

### Changed
- **設了 `system_prompt` 的非 box/math 設定，現在該 prompt 會真的送達模型**，分數會變動。
  這是修復而非副作用——那些設定本來就預期 prompt 會被送出。`regex_match` 使用者應預期
  `unparsed_rate` 下降。
- **`box` / `math` 在未設 `system_prompt` 時不再送出 `content` 為空字串的 system message。**
  舊版會送出空的 system block，多數 chat template 仍會渲染它，因此這可能使既有的
  box / math 分數有小幅變動。
- system prompt 的語言解析改為區分「鍵不存在」與「鍵存在但為空」：前者依序回退 `zh`、
  再回退唯一已定義的語言；後者視為該語言明確不要 prompt，**不跨語言回退**。
  先前的 `or` 串接會讓 `{zh: "中文", en: null}` 搭配英文資料集收到中文 prompt。

## [2.8.1] - 2026-09-11

本版為 PR #136（`Fix/audit bugfix batch`，作者 @dave-apmic）前半段的獨立修復批次，
內容為不改變評測分數的 bug fix。原 PR 的其餘部分（evaluator 重構、question-level
resume、新 feature）另行審查。

### Fixed
- **API 金鑰寫入 benchmark 結果檔**（原則 E 破口）：`--benchmark` 的輸出路徑未經
  `_prepare_config_for_saving()` 清理，完整的 `llm_api.api_key` 會被寫進
  `benchmark_results_*.json`。同時修正 `_prepare_config_for_saving()` 會就地刪除
  `self.config["llm_instance"]` 的問題——那讓同一個 runner 無法重複執行
  （第二次 `run_evaluation()` 會 `KeyError`）。
- **`finalize` 刪除合併後的 JSONL**（原則 D 資料遺失）：rank0 的 shard 路徑與合併輸出
  路徑相同，shard 清理會把剛合併好的結果檔一併刪掉。
- **HTML exporter 在 `usage_total_tokens` 為 `None` 時崩潰**（`TypeError: int + NoneType`）。
  同時修正 `llm_resoning_output` 的拼字，使推理輸出能正確顯示——writer 寫出的一直是
  正確拼字的鍵，exporter 讀的是一個從不存在的鍵。
- **HTML 報告未轉義模型輸出**：`question`、`correct_answer`、`predicted_answer`、
  `llm_output`、`reasoning` 現在都經過 `html.escape()`。先前模型回應中若含 `<script>`
  或任何標籤，會破壞報告版面或直接注入頁面。
- **`cli.py` 的 `sys.path` hack 遮蔽 HuggingFace `datasets` 套件**：`import datasets` 會
  解析到專案內的 `twinkle_eval/datasets/`，造成循環 import 錯誤。
- **text2sql 的 SQL 執行逾時從未生效**：`execute_sql()` 收到 `text2sql_timeout` 後並未實際套用到 sqlite。
- **gated dataset 檢查的運算子優先序錯誤**：`A and B or C` 導致任何含 `403` 的錯誤都被
  當成 gated dataset 而靜默略過。
- **`logs/` 目錄在每次 CLI 呼叫時都被建立**（改為延遲初始化），以及 log 檔名的同分鐘碰撞
  （時間戳加到秒）。

### Changed
- **`--dry-run` 與評測啟動不再進行 Google 服務的連線檢查**。原本這些網路呼叫發生在
  `ConfigurationManager.load_config()`，違反 §4「config.py 不做 API 呼叫」與 §12
  「`--dry-run` 不呼叫 API」。憑證檔案的格式驗證（存在、JSON 合法、必要欄位、
  `type == "service_account"`）全部保留。
  ⚠️ 代價：原本在評測開始前就會擋下的「資料夾不存在或未共享」診斷（含 Service Account
  email 與三步解法）不再出現，該失敗改為在評測結束的上傳階段才以 log 呈現。
- **Google Sheets 匯出移除 `API_金鑰` 欄位**（30 → 29 欄）。
  ⚠️ 既有試算表的歷史列仍保有截斷的金鑰，且在新表頭下會位移一欄，建議封存或清空舊表。
- **`--download-dataset` 對非 gated 的 403 錯誤現在會以 exit 1 結束**，先前會靜默略過。
- **text2sql 的 EX 評分現在真的會在 `text2sql_timeout`（預設 30 秒）中止**。
  ⚠️ 若 gold SQL 本身逾時，該題會退回 Exact Match 評分，可能使 text2sql 分數有小幅變動。

### 注意
- 秒級時間戳**只套用到 log 檔名**。`results_{timestamp}.json` 與
  `eval_results_{timestamp}_run{N}.jsonl` 仍為分鐘精度，同分鐘啟動兩次評測會出問題，
  而且兩個檔案的失效方式不同：

  | 檔案 | 同分鐘第二次執行 |
  |------|----------------|
  | `results_{timestamp}.json` | **被覆蓋**，第一次的結果消失 |
  | `eval_results_{timestamp}_run{N}.jsonl` | **累加**（append 模式），兩次的紀錄混在同一檔 |

  JSONL 的情況更麻煩：資料沒有遺失，但 `results_*.json` 的 `individual_runs.results`
  仍指向那個檔案，任何從 JSONL 重算正確率的下游工具都會**重複計數**。
  runner 端的時間戳變更屬於 PR #136 後半段（輸出檔名變更需依 §7 先行討論）。

## [2.8.0] - 2026-04-10

### Added
- **VLM Phase 1 — Vision MCQ 視覺多選題評測**（Milestone #22，PR #134）：
  - `VisionMCQExtractor`：支援字母答案（A–Z）與 Yes/No 二元答案（POPE 等幻覺偵測），優先解析 `\boxed{}` / `\box{}`（推理型 VLM 標準輸出格式），以 `findall + 取最後一個 match` 策略處理 VLM 回顯選項列表後再給答案的常見情境
  - Evaluator 新增 `uses_vision` 路由：`_encode_image_to_data_uri()` 以 magic-byte 偵測 MIME（PNG/JPEG/GIF/WebP/BMP），`os.path.realpath` 解析 symlink，50MB 大小上限保護
  - 支援本地檔案（base64 data URI）與 HTTP/HTTPS URL（直接傳遞）
  - 4 個 vision benchmark：MMBench、MMStar、MMMU、POPE
  - Example dataset：10 筆 MMStar 樣本（`datasets/example/vision_mcq/`，含 jpg 圖片）
  - `docs/evals/vision_mcq.md`：含分數對比與速度對比
  - 61 個 vision_mcq 測試（`tests/test_vision_mcq.py`）
  - Optional dependency `vision = ["Pillow>=10.0.0"]`（圖片驗證/縮放）

### Changed
- `datasets/file.py`：多模態附帶資源（圖片、音檔、影片）改為統計後一次性 `log_info`，避免逐檔 warn 噪音
- CLAUDE.md §13 新增「強制 Reviewer Agent」規定：所有 coding agent 在任何 PR push 之前，必須先 spawn 獨立的 reviewer agent 檢查 diff，blocker 必須先處理才能 push

## [2.7.1] - 2026-04-09

### Fixed
- **vLLM 0.18+ reasoning 欄位相容性**（PR #127）：vLLM 0.18+ 將 `reasoning_content` 改名為 `reasoning`，新增 `_get_reasoning_text()` helper 優先讀取 `reasoning`，`None` 時才回退 `reasoning_content`，相容 vLLM <0.13 / 0.13.x / >=0.18 三種版本

### Changed
- CLAUDE.md 新增 bug fix 必須立即推 PATCH 版號的規範

## [2.7.0] - 2026-04-08

### Added
- **ASR 語音辨識評測**（Milestone #21）：`WhisperModel` 支援 `/v1/audio/transcriptions` Whisper API 與 Chat Completions 多模態兩種路徑；`ASRExtractor` + `ASRScorer` 自動依語言選擇 WER（英文）或 CER（中文）；jiwer 作為 optional dependency（`pip install twinkle-eval[asr]`）
- **4 個 ASR Benchmark**：LibriSpeech（英文）、Aishell-1（中文）、Fleurs（102 語言）、Common Voice 17.0（多語言），總計 23 個可下載 benchmark
- Evaluator 新增音檔評測路徑，支援 `uses_audio` flag 自動分流
- ASR config template（`twinkle_eval/templates/asr.yaml`）
- Example dataset：10 筆 Common Voice TW 繁體中文音檔（`datasets/example/asr/`）
- `docs/evals/asr.md`：含分數對比（CER 3.80% vs jiwer 3.70%）與速度對比（7.5x 並行加速）
- 61 個 ASR 測試（`tests/test_asr.py`）

### Changed
- README.md 新增 ASR 區塊（23 benchmarks、9 大評測類型）

## [2.6.0] - 2026-04-07

### Added
- **Benchmark Download Registry**（Milestone #20）：`twinkle_eval/benchmarks.py` 集中管理 19 個評測資料集，`--download-dataset` 支援短名稱（`mmlu`、`gsm8k`）、`all`（全部下載）、`list`（列出可用）、向下相容 HuggingFace ID；支援 GitHub-based 下載（BIRD、Spider 2.0-lite、LongBench）；gated dataset（GPQA）互動式 HF token 提示與批次摘要
- **CLI Enhancement**（Milestone #19）：
  - `--init` 重構：支援 `--init`（列出範本）、`--init <name>`（單一）、`--init all`（全部），自動掃描 `templates/` 目錄
  - `--dry-run`：載入設定檔與資料集，顯示評測計畫但不呼叫 API
  - `--validate`：僅驗證設定檔格式與資料集路徑
  - `--resume TIMESTAMP`：從中斷點繼續評測，跳過已完成的檔案
- **Regex Match 評測方法**（Milestone #18）：`RegexMatchExtractor` + `StringMatchScorer`，BBH 為首個 use case，含 example dataset、docs、66 tests

### Changed
- 設定檔範本從 `twinkle_eval/config.*.template.yaml` 搬入 `twinkle_eval/templates/*.yaml`，刪除舊 `config.template.yaml`
- CLAUDE.md 新增 CHANGELOG 規則：每次 bump 版本號必須同步更新 CHANGELOG

## [1.4.1] - 2026-03-17

### Changed
- 更新 CHANGELOG.md 與專案文件，準備 PyPI 發布

## [1.4.0] - 2026-03-16

### Added
- 基於 Logit 的評測策略（`evaluation_method: logit`，closes #7）：透過 `/v1/completions` 搭配 `echo=True` 計算 log P(choice | context) 為每個選項評分；相容 vLLM 及 lm-evaluation-harness MMLU 模板格式；支援任意選項數量（如 MMLU-Pro 的 A–J）
- `OpenAIModel.score_continuation()`：透過 `ThreadPoolExecutor` 並行計算每個選項的對數概率；`logprob_scores` 字典會寫入 JSONL 輸出供除錯使用
- 在 `datasets/example/` 下新增範例評測子集，可快速驗證設定而無需下載完整資料集（closes #24）：gsm8k（20 題）、AIME 2025（30 題）、TMMLU+（20 題）、MMLU（20 題）、MMLU-Pro（20 題）
- `scripts/create_example_datasets.py`：維護者用於從 HuggingFace 重新生成範例子集的腳本

## [1.3.0] - 2026-03-16

### Added
- Slurm 多節點分散式評測支援：每個節點/rank 輸出獨立的 shard JSONL，避免並行寫入衝突
- `twinkle-eval --finalize-results <timestamp>`：自動合併分散式碎片並重新計算評測指標
- `twinkle-eval --hf-repo-id` / `--hf-variant`：評測完成後自動上傳結果至 Hugging Face dataset repo
- `pip install twinkle-eval[slurm]` optional extras：`huggingface-hub` 不再強制安裝；未安裝時呼叫上傳功能會拋出清楚的提示訊息
- `dataset.py`：自動正規化 MMLU HuggingFace 格式（`choices` list + 整數 `answer`）為 A/B/C/D 具名欄位格式，支援超過 4 個選項
- `twinkle_eval/finalize.py`：碎片合併邏輯（含備援路徑、清理機制）
- `twinkle_eval/hf_uploader.py`：HuggingFace Dataset 上傳服務
- `scripts/`：可直接 sbatch 的 Slurm 腳本（測試版與完整生產版）
- `configs/`：Slurm 評測設定範例
- `SLURM_README.md`：分散式評測操作說明
- `distributed` config 區段：自動從 `WORLD_SIZE`/`RANK` 環境變數讀取分散式設定

## [1.2.0] - 2026-03-16

### Added
- 數學評測策略（`MathExtractionStrategy`）：從 `\boxed{}` 提取答案，並使用 mathruler 進行語意等價判斷，支援巢狀大括號、LaTeX 大小寫正規化、逗號分隔解集合的無序比對
- `pip install twinkle-eval[math]` optional extras：數學功能所需的 `mathruler`、`sympy`、`pylatexenc` 不再強制安裝；未安裝時選用 `evaluation_method: math` 會拋出清楚的提示訊息
- `dataset_overrides` config：可針對特定資料集路徑覆蓋 `evaluation_method`、`system_prompt_enabled`、`samples_per_question`、`pass_k`、`repeat_runs`、`shuffle_options` 及模型參數
- `samples_per_question` 與 `pass@k`：單題可產生多個樣本並計算 pass@k 指標
- `system_prompt_enabled` config 欄位：可全域停用或啟用 system prompt
- `EvaluationStrategy.normalize_answer()` 與 `is_correct()` 方法：讓各策略自訂答案正規化與等價判斷邏輯

### Changed
- `evaluate_file()` 回傳值由 `(path, accuracy, results_path)` 改為 `(path, metrics_dict, results_path)`，`metrics_dict` 包含 `accuracy`、`pass_at_k`、`pass_metric`、`pass_k`
- `models.py`：`call()` 新增 `eval_method`、`system_prompt_enabled`、`num_samples`、`model_overrides` 可選參數；`math` 方法與 `box` 方法同樣使用 system prompt
- `main.py`：每個資料集使用獨立的 `Evaluator` 實例，支援 per-dataset 策略切換

## [1.1.6] - 2026-03-16

### Fixed
- Unified reasoning output parsing: auto-detect and strip complete inline `<think>`/`<reason>`/`<reasoning>` tag blocks; fallback to `reasoning_content` when `content` is null

## [1.1.5] - 2026-03-16

### Fixed
- Fallback to `reasoning_content` when `message.content` is null, preventing silent zero-accuracy on reasoning models

## [1.1.4] - 2026-03-16

### Fixed
- JSONL per-question detail file no longer overwritten when evaluating multiple files in the same dataset directory (`'w'` → `'a'` mode)

## [1.1.3] - 2026-03-16

### Fixed
- `get_info()` raised `NameError: __email__ is not defined`
- Synced `__version__` in `__init__.py` with `pyproject.toml`
- Evaluation no longer silently exits with code 0 when all datasets fail; raises `EvaluationError` with a clear message

### Added
- pytest infrastructure (`tests/` directory) with regression tests for all fixed issues
- Added Thomas Liang and Ren-Di Wu as authors/maintainers

## [1.1.3] - 2025-02-03

### Added
- PyPI publishing guide for maintainers

### Changed
- Updated README.md with enhanced documentation
- Project metadata updates in pyproject.toml

## [1.1.2] - 2025-01-XX

### Added
- Google Drive integration for uploading log and result files
- Google Sheets integration for exporting evaluation results
- Support for service account and OAuth authentication for Google services
- Configuration validation for Google services
- Multiple file upload functionality based on start time

### Changed
- Evaluation result storage format updated to JSONL for better data streaming
- Enhanced file upload functionality to support multiple log and result files

### Fixed
- Typo corrections in evaluators module
- Handling for missing reasoning content in evaluation responses

## [1.1.0] - 2025-01-XX

### Added
- Benchmark testing functionality with configurable parameters
- Performance metrics calculation and summary display
- HTML export functionality for converting JSON results to HTML format
- Environment configuration parameters (GPU info, system info)
- Support for `extra_body` parameter in API configuration
- Download datasets from HuggingFace Hub functionality
- Dataset information retrieval commands
- Support for Apache Arrow (`.arrow`) file format
- Debug guide and test scripts for VSCode debugging
- CLI commands for listing available LLMs, strategies, and exporters
- Docker support with Dockerfile and .dockerignore
- DevContainer configuration for consistent development environment

### Changed
- Major refactor: reorganized codebase into modular `twinkle_eval` package
- Updated dataset download to save as Parquet format by default
- Improved progress bar descriptions (changed to "評測題庫中")
- Version bump to 1.1.0 with updated author contact information
- Enhanced result exporters to include environment configuration
- Optimized configuration handling for better serialization
- Improved security by removing sensitive information from exported results

### Removed
- Legacy files: old `data_loader.py`, `evaluator.py`, `llm_api.py`, `main.py`
- Old `config.py` in favor of new configuration system
- `requirements.txt` in favor of pyproject.toml dependency management

### Fixed
- Removed non-serializable object instances from configuration
- Enhanced handling of sensitive information in configuration

## [1.0.0] - 2024-XX-XX

### Added
- Initial release of Twinkle Eval
- OpenAI-compatible API support for LLM evaluation
- Multi-format dataset support (CSV, JSON, JSONL, Parquet, TSV)
- Pattern-based answer extraction strategy
- Box-based answer extraction strategy (LaTeX `\box{}` format)
- Custom regex strategy for answer extraction
- Option shuffling to prevent position bias
- Multiple evaluation runs with statistical analysis (mean, std)
- Rate limiting and parallel execution with ThreadPoolExecutor
- Automatic retry logic for API failures
- Results export in JSON, CSV, HTML, and JSONL formats
- YAML-based configuration system
- Comprehensive logging with UTF-8 encoding for Chinese characters
- Progress tracking with tqdm
- CLI interface with multiple commands
- Configuration template initialization (`--init`)
- Support for multiple datasets in single evaluation run
- Detailed per-question results tracking

### Features
- Factory pattern for pluggable components (LLM, Strategy, Exporter)
- Graceful error handling with custom exception hierarchy
- Support for Traditional Chinese and English prompts
- Configurable model parameters (temperature, top_p, max_tokens, etc.)
- Per-dataset and overall accuracy statistics

---

## Release Notes

### Performance Highlights

Twinkle Eval achieves **up to 17x faster** evaluation compared to iKala/ievals through:
- Parallel API call execution with ThreadPoolExecutor
- Efficient rate limiting without blocking
- Optimized dataset loading and processing

### Supported Datasets

- **TMMLU+**: [ikala/tmmluplus](https://huggingface.co/datasets/ikala/tmmluplus)
- **tw-legal**: [lianghsun/tw-legal-benchmark-v1](https://huggingface.co/datasets/lianghsun/tw-legal-benchmark-v1)
- **MMLU**: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu)
- Any custom dataset following the required format

### Migration Notes

#### From 1.0.x to 1.1.x
- Configuration format remains backward compatible
- New optional Google services configuration section
- JSONL export format added for detailed results
- Environment configuration section is optional

#### From pre-1.0 to 1.0.x
- Complete codebase refactor - direct upgrade not supported
- Configuration file format changed to YAML
- New modular architecture with factory patterns

---

## Links

- [GitHub Repository](https://github.com/ai-twinkle/Eval)
- [PyPI Package](https://pypi.org/project/twinkle-eval/)
- [Documentation](https://github.com/ai-twinkle/Eval#readme)
- [Bug Reports](https://github.com/ai-twinkle/Eval/issues)
- [Discord Community](https://discord.gg/Cx737yw4ed)

---

[1.1.3]: https://github.com/ai-twinkle/Eval/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/ai-twinkle/Eval/compare/v1.1.0...v1.1.2
[1.1.0]: https://github.com/ai-twinkle/Eval/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/ai-twinkle/Eval/releases/tag/v1.0.0
