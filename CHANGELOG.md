# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.9.0] - 2026-08-25

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
