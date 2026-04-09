# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
