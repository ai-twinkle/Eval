# Twinkle Eval — 專案規範手冊（CLAUDE.md）

> **本文件是所有開發者與 coding agent 的強制閱讀文件。**
> 在修改任何程式碼之前，必須先完整閱讀本文件，確保所有變更符合本專案的設計理念與規範。
> 若提議的變更與本文件有所衝突，**必須先至 GitHub 開立 Issue 進行討論**，而非直接提交 PR 要求合入。

---

## 目錄

1. [專案定位與設計理念](#1-專案定位與設計理念)
2. [核心設計原則（不得違反）](#2-核心設計原則不得違反)
3. [架構總覽](#3-架構總覽)
4. [模組職責邊界](#4-模組職責邊界)
5. [擴充規範——如何正確新增功能](#5-擴充規範如何正確新增功能)
6. [新增評測 Benchmark 的完整規範](#6-新增評測-benchmark-的完整規範)
7. [必須先開 Issue 的情況](#7-必須先開-issue-的情況)
8. [程式碼風格規範](#8-程式碼風格規範)
9. [設定檔規範（config.yaml）](#9-設定檔規範configyaml)
10. [輸出格式規範](#10-輸出格式規範)
11. [依賴管理規範](#11-依賴管理規範)
12. [CLI 設計規範](#12-cli-設計規範)
13. [提交 PR 前的 Checklist](#13-提交-pr-前的-checklist)
14. [專案現況快照](#14-專案現況快照)
15. [貢獻者](#15-貢獻者)

---

## 1. 專案定位與設計理念

### 1.1 誕生背景

2025 年初，推理模型（reasoning model）開始大量出現。這類模型在輸出正式答案之前，會先產生大量的推理過程（chain-of-thought），導致每次 API 呼叫的回應時間遠高於傳統模型。

然而，當時現有的評測框架（如 iKala/ievals）採用**同步、逐題呼叫**的設計，面對推理模型時評測時間會等比例放大，一個完整的 benchmark 動輒耗費數小時。對於正在進行模型訓練迭代的團隊而言，這意味著每次訓練完都需要等待過久才能得到評測反饋，白白浪費 GPU 運行時間與開發週期。

**Twinkle Eval 因此而生**：以並行 API 請求為核心手段，讓評測速度不再是訓練迭代的瓶頸。實測相比 iKala/ievals 快 9–17 倍，使團隊能夠快速取得評測結果、驅動下一輪訓練決策。

### 1.2 核心設計哲學

**「輕量、單機、即裝即用」是這個專案的根本設計方向。**

本專案從一開始就以「`pip install twinkle-eval` 後，在單台機器上即可執行完整評測」為設計約束。這個約束是刻意的：

- **不依賴叢集基礎設施**：不需要 SLURM、Kubernetes、或任何分散式排程系統才能運作
- **不需要特殊硬體**：評測本身不需要 GPU，只需要能呼叫 API 的網路環境
- **降低使用門檻**：任何人在任何環境（本機、Colab、CI/CD）都能直接執行

多節點分散式評測（如 SLURM 支援）若作為 PR 提交，**定位是擴充功能（extension）**，不得影響單機執行路徑的正確性與簡潔性。核心程式碼必須在不依賴任何分散式元件的情況下完整運作。

### 1.3 關鍵設計邊界：本專案永遠不啟動模型端點

**Twinkle Eval 不負責啟動、部署、或管理任何 LLM 服務。**

本專案的職責範圍嚴格限定於：
> 「拿著評測題目，去呼叫**已經在外部運行的** API 端點，取得回答，並計算評測指標。」

使用者需要自行在外部啟動模型服務（vLLM、Ollama、OpenAI、NVIDIA Build 等），再將端點的 `base_url` 填入 `config.yaml`，Twinkle Eval 才開始工作。

這個邊界意味著：
- 本專案程式碼中**不得出現**任何啟動 `vllm serve`、`ollama run` 或其他模型服務的邏輯
- 本專案不管理模型的生命週期（啟動、關閉、重啟）
- 若 API 端點無回應，本專案的責任是報錯退出，而非嘗試修復或重啟服務

任何試圖在程式碼內部啟動模型服務的 PR，在未開 Issue 討論並取得 maintainer 明確同意前，不得合入。

### 1.4 核心目標

- **高效**：以並行請求大幅縮短評測時間，讓評測不再是訓練迭代的瓶頸
- **客觀**：透過選項隨機排列，消除模型對選項位置的偏好（參考 [Changing Answer Order Can Decrease MMLU Accuracy](https://arxiv.org/html/2406.19470v1)）
- **穩定性量化**：支援多次執行並計算標準差，反映模型一致性
- **易擴充**：模組化設計，讓新增 LLM 後端、評測策略、輸出格式都不需修改核心邏輯
- **API 相容優先**：以 OpenAI 相容 API 為統一介面，不綁定特定模型或服務商

### 1.5 本專案不是什麼

- 一個模型訓練框架
- 一個模型部署或服務管理工具
- 一個資料標注工具
- 一個需要叢集才能運作的分散式系統（單機是 first-class citizen）

**在新增任何功能之前，請先確認該功能是否符合上述定位與邊界。**

---

## 2. 核心設計原則（不得違反）

以下原則反映專案的根本設計決策，**任何違反這些原則的 PR 在開 Issue 討論並取得 maintainer 共識前，不得合入**。

### 原則 A：工廠模式 + 策略模式是唯一的擴充路徑

本專案使用三大工廠類別作為擴充點：

| 工廠 | 對應介面 | 負責建立 |
|------|----------|----------|
| `LLMFactory` | `LLM`（ABC） | LLM 後端實作 |
| `EvaluationStrategyFactory` | `EvaluationStrategy`（ABC） | 答案提取策略 |
| `ResultsExporterFactory` | （抽象基底） | 結果輸出格式 |

**規則**：
- 新增 LLM 後端 → 繼承 `LLM`，實作 `call()` 和 `validate_config()`，向 `LLMFactory` 註冊
- 新增評測策略 → 繼承 `EvaluationStrategy`，實作 `extract_answer()` 和 `get_strategy_name()`，向 `EvaluationStrategyFactory` 註冊
- 新增輸出格式 → 繼承對應基底類別，向 `ResultsExporterFactory` 註冊
- **禁止**在 `evaluators.py`、`main.py` 等核心流程中用 `if/elif` 判斷具體類型，應改用工廠或策略物件

### 原則 B：配置驅動（Config-Driven），禁止硬編碼行為

所有可變行為（評測方法、系統提示詞、資料集路徑、模型參數等）都必須透過 `config.yaml` 控制，不得在程式碼中硬編碼預設值或行為分支。

**禁止**：
```python
# 錯誤：硬編碼行為
if model_name == "gpt-4":
    do_something_special()
```

**允許**：
```python
# 正確：透過 config 控制
extra_body = model_config.get("extra_body", {})
```

### 原則 C：選項標籤不得硬編碼為 A/B/C/D

歷史上 A/B/C/D 被 hardcode，這已被識別為設計缺陷（PR #17 修正中）。未來所有涉及選項的邏輯必須動態偵測選項鍵，支援任意數量的選項：

```python
# 錯誤：硬編碼
for key in ["A", "B", "C", "D"]:
    ...

# 正確：動態偵測
option_keys = [k for k in question_data if k.isupper() and len(k) <= 2]
```

### 原則 D：評測結果必須確保不遺失

評測結果輸出時，**必須使用 append 模式**（`'a'`）而非 overwrite 模式（`'w'`）寫入 JSONL 檔案，以確保多檔、多 run 的結果都能正確累積。（此問題已記錄於 PR #17）

### 原則 E：API 金鑰絕對不得出現在輸出、日誌或 Git 歷史中

**兩條硬性規定，違反任一都是嚴重問題：**

1. **程式碼輸出**：在儲存結果或輸出日誌前，必須呼叫 `_prepare_config_for_saving()` 移除敏感資訊。任何新增的輸出路徑都必須遵守這個規則。

2. **Git commit**：含有真實 API 金鑰的 config 檔案**絕對不得 commit 到 Git**。
   - 本機測試用的 config 檔案（如 `config_test_*.yaml`、`config_local_*.yaml`、`config_*.local.yaml`）必須列入 `.gitignore`
   - **命名規範**：本機 config 檔案一律使用以下前綴或後綴之一，這些檔案已全局被 `.gitignore` 排除：
     - `config_local_*.yaml`
     - `config_test_*.yaml`（測試用）
     - `*.local.yaml`
   - 若不確定某個 config 是否含有 API 金鑰，在 commit 前先執行 `git diff --staged | grep -i "api_key"` 確認
   - coding agent 在建立任何含有 API 金鑰的 config 檔案時，**必須先確認該路徑已在 `.gitignore` 中**，然後才能寫入

### 原則 F：Backward Compatibility 優先

除非有充分理由並取得 maintainer 共識，新功能不得破壞：
- 現有 `config.yaml` 格式（舊設定檔應仍可正常運作）
- 現有 CLI 介面（現有命令列選項的行為不得改變）
- 現有輸出檔案格式（`results_{timestamp}.json` 和 `eval_results_{timestamp}_run{N}.jsonl` 的結構）

### 原則 G：本專案不啟動任何模型服務

本專案的職責是「呼叫已在外部運行的 API 端點」，程式碼中**絕對不得**出現：
- 以 subprocess、os.system 或任何方式執行 `vllm serve`、`ollama run`、`python -m ...` 等啟動模型服務的指令
- 監控、重啟、或管理外部模型服務進程的邏輯
- 假設自己有權限操作模型所在機器的任何邏輯

若 API 端點無回應或回傳錯誤，本專案的正確行為是：**依設定的 `max_retries` 重試後回報錯誤，絕不嘗試自行修復服務**。

### 原則 H：單機執行是 first-class citizen，分散式是可選擴充

本專案必須在**不依賴任何叢集基礎設施**的情況下完整運作。任何新增功能都不得讓「單機執行」的路徑變得更複雜或增加額外必要依賴。

分散式 / 多節點相關功能（如 SLURM 腳本）屬於擴充功能，應以**完全可選**的方式實作：
- 不引入新的 required dependency
- 不修改現有核心模組（`evaluators.py`、`main.py` 等）的主要邏輯
- 單機使用者若從未接觸分散式功能，應感受不到任何差異

---

## 3. 架構總覽

```
twinkle_eval/
├── __init__.py             # 套件入口，定義公開 API（__all__）
├── cli.py                  # CLI 入口點（entry point: twinkle-eval 命令）
├── main.py                 # TwinkleEvalRunner 主流程 + argparse 定義
│                           # ↑ 控制器層，協調各模組
├── config.py               # ConfigurationManager：載入、驗證、解析 YAML
├── templates/              # 設定檔範本（`--init` 時產生）
│   ├── multiple_choice.yaml
│   ├── math.yaml
│   ├── regex_match.yaml
│   └── ...                 # 每個 evaluation method 一個範本
│
├── models.py               # LLM 抽象層
│   ├── LLM（ABC）          # → call(), validate_config()
│   ├── OpenAIModel         # 目前唯一實作，OpenAI 相容格式
│   └── LLMFactory          # 工廠，用 register_llm() 擴充
│
├── dataset.py              # 資料集載入
│   ├── Dataset             # 載入 CSV/JSON/JSONL/Parquet，迭代題目
│   ├── find_all_evaluation_files()  # 遞迴找指定目錄下的所有評測檔
│   └── download_huggingface_dataset() / list_huggingface_dataset_info()
│
├── evaluators.py           # 評測核心
│   ├── RateLimiter         # 控制 API 呼叫速率
│   └── Evaluator           # 並行評測（ThreadPoolExecutor），對接 LLM + Strategy
│
├── evaluation_strategies.py # 答案提取策略
│   ├── EvaluationStrategy（ABC）  # → extract_answer(), get_strategy_name()
│   ├── PatternMatchingStrategy    # 正則表達式匹配（預設含中英文模式）
│   ├── BoxExtractionStrategy      # 提取 \box{} / \boxed{}
│   ├── CustomRegexStrategy        # 自訂正則
│   └── EvaluationStrategyFactory  # 工廠，用 register_strategy() 擴充
│
├── results_exporters.py    # 結果輸出
│   └── ResultsExporterFactory     # 工廠，支援 json/csv/html/google_sheets
│
├── validators.py           # 輸入驗證（config & dataset）
├── exceptions.py           # 自訂例外類別（繼承自 TwinkleEvalError）
├── google_services.py      # Google Drive / Sheets 整合（可選功能）
├── benchmark.py            # LLM 效能基準測試（BenchmarkRunner）
└── logger.py               # 日誌工具（log_info, log_error 等）
```

### 資料流向

```
config.yaml
    ↓ ConfigurationManager.load_config()
config dict（含 llm_instance、evaluation_strategy_instance）
    ↓
TwinkleEvalRunner.run_evaluation()
    ↓
Evaluator.evaluate_file()  ←── Dataset（逐題迭代）
    ↓ (ThreadPoolExecutor 並行)
LLM.call()  →  API response
    ↓
EvaluationStrategy.extract_answer()  →  predicted answer
    ↓
比對 correct_answer  →  accuracy
    ↓
results/eval_results_{timestamp}_run{N}.jsonl（append 模式）
    ↓
ResultsExporter.export()  →  results/results_{timestamp}.json
```

---

## 4. 模組職責邊界

每個模組有明確的單一職責，**禁止跨越邊界**：

| 模組 | 職責 | 禁止做的事 |
|------|------|-----------|
| `config.py` | 載入 & 驗證 YAML，建立 llm/strategy 實例 | 不做 API 呼叫、不做評測邏輯 |
| `models.py` | 封裝 LLM API 呼叫 | 不做答案解析、不做資料集處理 |
| `dataset.py` | 載入資料集、格式正規化 | 不做 API 呼叫、不做評分 |
| `evaluators.py` | 協調並行評測流程、計算 accuracy | 不直接解析答案（交給 strategy）|
| `evaluation_strategies.py` | 從 LLM 輸出文字提取答案 | 不做 API 呼叫、不讀檔案 |
| `results_exporters.py` | 將結果字典序列化為各種格式 | 不做評測邏輯、不修改結果 |
| `main.py` | 組裝流程、定義 CLI 參數 | 不實作具體的評測或解析邏輯 |
| `validators.py` | 驗證 config 結構與資料集格式 | 不修改 config 或資料集 |

---

## 5. 擴充規範——如何正確新增功能

### 5.1 新增 LLM 後端

```python
# 1. 在 models.py 中繼承 LLM
class MyNewModel(LLM):
    def validate_config(self) -> bool:
        # 驗證必要的 config 欄位
        ...
        return True

    def call(self, question_text: str, prompt_lang: str = "zh") -> ChatCompletion:
        # 呼叫 API 並回傳 OpenAI ChatCompletion 格式
        ...

# 2. 向工廠註冊
LLMFactory.register_llm("my_backend", MyNewModel)
```

**注意**：`call()` 的回傳值必須相容於 `ChatCompletion` 格式，`evaluators.py` 依賴這個介面。若目標 API 格式不同，應在 `call()` 內部轉換，而非修改 `evaluators.py`。

### 5.2 新增評測策略

```python
# 1. 在 evaluation_strategies.py 中繼承 EvaluationStrategy
class MyStrategy(EvaluationStrategy):
    def get_strategy_name(self) -> str:
        return "my_strategy"  # 對應 config.yaml 中的 evaluation_method

    def extract_answer(self, llm_output: str) -> Optional[str]:
        # 從 llm_output 中提取答案字母，回傳 None 表示無法提取
        ...

# 2. 向工廠註冊
EvaluationStrategyFactory.register_strategy("my_strategy", MyStrategy)
```

### 5.3 新增輸出格式

繼承對應的 Exporter 基底類別，向 `ResultsExporterFactory` 註冊，並更新 `cli.py` 的 `--export` choices（若為公開格式）。

### 5.4 新增 CLI 參數

- 在 `main.py` 的 `create_cli_parser()` 中新增 `add_argument()`
- 在 `main()` 中加入對應的處理邏輯
- 更新 `templates/*.yaml` 中的說明（若涉及新 config 欄位）
- 更新 README 的「命令列選項」段落

---

## 6. 新增評測 Benchmark 的完整規範

每次新增一個評測 benchmark（如 IFEval、BFCL、RAGAS 等），必須依照以下流程進行，**缺一不可**。

### 6.0 開始前：建立 Milestone 與 Issues

**在寫任何一行程式碼之前**，必須先完成 GitHub 上的規劃工作：

1. **建立 Milestone**：在 GitHub 建立一個對應的 Milestone，標題格式為 `{Benchmark Name} — {一句話說明}`
   - 例如：`IFEval — Instruction Following Evaluation`
   - Milestone description 應包含：benchmark 來源、目的、預計實作範圍

2. **開立 Issues 並掛上 Milestone**：將以下每個步驟（6.1–6.5）各開一個 Issue：
   - 每個 Issue 必須掛上 **`type: feature`** label
   - 每個 Issue 必須附屬在該 benchmark 的 Milestone 下
   - Issue 標題格式：`feat({benchmark}): {步驟描述}`

3. **不得跳過**：沒有對應 Milestone 和 Issues 的 benchmark 實作，不得開 PR

**標準 Issue 結構（一個 benchmark 共需開 6 個 Issues）：**

| Issue | 標題範例 | 對應步驟 |
|-------|---------|--------|
| 1 | `feat(ifeval): prepare example dataset from HuggingFace` | 6.1 資料集 |
| 2 | `feat(ifeval): implement Extractor + Scorer + Checkers` | 6.2 實作 |
| 3 | `feat(ifeval): score comparison vs reference framework` | 6.3 分數對比 |
| 4 | `feat(ifeval): speed benchmark vs reference framework` | 6.4 速度對比 |
| 5 | `feat(ifeval): write docs/evals/{name}.md` | 6.5 文件 |
| 6 | `feat(ifeval): write tests/test_{name}.py` | 6.6 測試 |

### 6.0.1 合入（Merge）的前提條件

**一個 benchmark 的 PR 必須等到以下全部完成才可合入 main：**

- [ ] 所有 6 個 Issues 已 close（或在同一 PR 中解決）
- [ ] `docs/evals/{benchmark_name}.md` 已完整填寫（包含分數對比與速度對比）
- [ ] 分數誤差符合第 6.3 節的容差標準
- [ ] 前一個優先級更高的 benchmark 若尚未完成，此 benchmark 不得合入

**Benchmark 開發優先順序原則**：先完成再開始下一個。若 IFEval 尚未完成（含比較報告），不得開始 IFBench 的實作。

### 6.1 提供評測集來源與 example 樣本

- 明確記錄資料集來源（paper、官方 repo、HuggingFace dataset ID）
- 從官方來源取出有代表性的樣本放入 `datasets/example/{benchmark_name}/`
  - 樣本數建議：**10–20 筆**，涵蓋該 benchmark 的主要題型分佈
  - 樣本需為可直接執行的完整格式（含 `id`、`question`、答案欄位等）
  - 若 benchmark 有子類別（如 BFCL 的 simple/multiple/parallel），每個子類別至少 2–3 筆
- `datasets/example/{benchmark_name}/` 必須能讓任何人在不下載完整資料集的情況下跑通完整流程

### 6.2 依參考框架重構為本專案架構

若有參考既有框架（如 lm-evaluation-harness、DeepEval、RAGAS 官方實作）：

- **不得直接複製整個框架**，必須萃取核心評分邏輯，以最小依賴實作
- 若需移植第三方程式碼（如 Google IFEval checkers），必須：
  - 在 `docs/evals/{benchmark_name}.md` 記錄授權資訊（Apache 2.0 / MIT 等）
  - 明確標注「移植自 {原始來源}」，並在對應的 Python 檔案頂部加入 attribution 注解
  - 若原始程式碼使用不可用的依賴（如 `absl`、`immutabledict`），必須替換為標準庫等價物
- 實作架構必須遵循本專案的 Extractor/Scorer 模式（第 5.2 節）
- 若需要 optional dependency，加入 `pyproject.toml` 的 `[project.optional-dependencies]` 並附上安裝指引

### 6.3 與原始框架進行分數對比驗證

新增 benchmark 後，**必須**使用相同模型、相同題目，同時跑本專案與參考框架，並在 `docs/evals/{benchmark_name}.md` 記錄對比結果。

**分數容差標準：**

| 資料集大小 | 可接受誤差 | 說明 |
|-----------|----------|------|
| ≥ 200 筆 | ±2% | 完整 benchmark，誤差應極小 |
| 50–199 筆 | ±3% | 中型子集，允許略高統計波動 |
| < 50 筆 | ±5% | 小型子集，波動較大屬正常 |
| ≤ 20 筆（example） | 僅作 sanity check | 不強制對比，僅確認流程可跑通 |

若分數超出容差，必須調查原因（通常為：preprocessing 差異、prompt 格式差異、答案正規化邏輯差異），並在文件中說明差異原因或修正方式。

### 6.4 記錄評測速度對比

本專案核心優勢是速度。每個新增的 benchmark 必須記錄：

- **本專案（單機）**：總耗時、並行 worker 數、模型名稱
- **參考框架（若有）**：同等硬體、同等題數下的耗時
- 記錄位置：`docs/evals/{benchmark_name}.md` 的「速度對比」段落

若對比框架不支援特定模型或無法在同等環境執行，可註明「無法直接對比」並說明原因。

### 6.5 建立評測文件（docs/evals/{benchmark_name}.md）

每個 benchmark 必須在 `docs/evals/` 下建立一個對應的文件，使用 `docs/evals/TEMPLATE.md` 作為模板，包含：

- **來源**：原始 paper（附 DOI/arXiv 連結）、官方 repo、資料集連結
- **目的**：這個評測在衡量什麼能力、適合用於哪些比較場景
- **Leaderboard**（若有）：附上官方 leaderboard 連結
- **實作說明**：Extractor/Scorer 設計、特殊邏輯說明、optional deps
- **分數對比**：本專案 vs. 參考框架（含模型名稱、資料集規模、日期）
- **速度對比**：單機測速結果

文件模板位於 `docs/evals/TEMPLATE.md`。

### 6.5.1 更新 datasets/example/README.md

每次新增或修改 benchmark 的 example 資料集後，**必須**同步更新 `datasets/example/README.md`：

- 在「資料集清單」表格中新增該 benchmark 的條目（目錄、來源、題數、評測方法、說明）
- 在「快速開始」段落中新增對應的 config.yaml 範例
- 在「資料格式」段落中新增該 benchmark 的 JSONL 格式說明

此 README 是使用者了解所有 example 資料集的入口，**缺少更新會導致 PR 不符合合入條件**。

### 6.6 建立測試檔案（tests/test_{benchmark_name}.py）

每個 benchmark **必須**在 `tests/` 下建立對應的 pytest 測試檔案，至少涵蓋以下測試類別：

- **Extractor**：`get_name()` 回傳正確名稱、`extract()` pass-through 行為、`uses_ifeval` 旗標（若適用）
- **Scorer**：`get_name()` 回傳正確名稱、`normalize()` 行為、`score()` 的正確/錯誤/空值/無效 ground truth 場景、`score_full()` 回傳結構驗證
- **Checker / 評分邏輯**：選取 3–5 種具代表性的 instruction/rule type，各寫一個 pass 和一個 fail 的 case
- **Checker Registry**（若有）：驗證所有指令 ID 都已註冊、數量正確、所有 category 都存在
- **PRESETS 註冊**：確認 `PRESETS["{benchmark_name}"]` 存在且對應正確的 Extractor/Scorer class
- **Example Dataset**：檔案存在、格式正確（必要欄位齊全、筆數符合預期、涵蓋所有子類別）
- **Edge cases**：空回應、None 值、kwargs 中包含 null 值的過濾（若適用）

命名慣例：`tests/test_{benchmark_name}.py`，class 名稱使用 `TestXxx` 格式。

**此測試檔案必須能在不呼叫任何外部 API 的情況下通過**（pure unit test）。提交 PR 前必須：
1. 執行 `python3 -m pytest tests/test_{benchmark_name}.py -v` 確認新增測試全部通過
2. 執行 `python3 -m pytest tests/ -v` 確認**完整測試套件**沒有因本次變更而產生新的失敗

---

## 7. 必須先開 Issue 的情況

以下情況**不得直接提交 PR**，必須先在 GitHub 開立 Issue 說明理由、設計方案、影響範圍，取得至少一位 maintainer（teds-lin 或 lianghsun）的明確同意後，才能開始實作：

| 類別 | 具體情況 | 原因 |
|------|----------|------|
| 🔴 Breaking Change | 修改現有 `config.yaml` 必填欄位的名稱或結構 | 影響所有現有使用者 |
| 🔴 Breaking Change | 修改 `results_{timestamp}.json` 或 JSONL 的輸出結構 | 影響下游分析工具 |
| 🔴 Breaking Change | 修改現有 CLI 選項的行為（包含 flag 名稱、預設值） | 影響所有腳本整合 |
| 🟡 架構變更 | 修改 `LLM`、`EvaluationStrategy` 等 ABC 介面 | 影響所有現有實作 |
| 🟡 架構變更 | 在核心流程（evaluators.py、main.py）新增大量邏輯 | 可能破壞關注點分離 |
| 🟡 依賴變更 | 新增 required dependency（非 optional） | 影響所有使用者的安裝體積 |
| 🟡 依賴變更 | 升級已有依賴到 major version | 可能引入不相容問題 |
| 🔵 大型功能 | 新增評測資料集類型或 benchmark 流程 | 需確認與現有格式相容 |
| 🔵 大型功能 | 新增第三方服務整合（如 Slurm、HuggingFace、資料庫） | 需確認選用/必用策略 |
| 🔵 大型功能 | 修改 `repeat_runs`、`shuffle_options` 等核心評測行為 | 影響評測結果的可重現性 |

**例外**：以下情況可直接提交 PR（但 PR 描述仍需清楚說明動機）：
- 修復已有 Issue 記錄的 bug，且修復範圍與 Issue 描述相符
- 純文件更新（README、CHANGELOG、docstring）
- 新增測試案例
- 程式碼風格修正（不改變行為）
- 修復明顯的 typo 或 NameError

---

## 8. 程式碼風格規範

### 工具設定

| 工具 | 設定 |
|------|------|
| formatter | Black，line-length=100 |
| import sorter | isort，profile=black |
| type checker | mypy（strict mode） |
| linter | flake8 |

提交前執行：
```bash
black twinkle_eval/
isort twinkle_eval/
flake8 twinkle_eval/
mypy twinkle_eval/
```

### 語言規範

- **程式碼邏輯**（變數名、函式名、class 名、commit message）：英文
- **使用者面對的輸出**（CLI 訊息、日誌、錯誤提示）：繁體中文
- **docstring**：中英文皆可，但同一個函式只用一種語言，不混用
- **YAML 設定值**：視語境（中文 prompt 用中文，config 欄位名用英文）

### 例外處理規範

- 使用 `exceptions.py` 中的自訂例外，不得直接 `raise Exception("...")`
- 在模組邊界捕捉例外並記錄到 `logger`，再 re-raise 或回傳預設值
- **不得吞掉例外（`except: pass`）**，至少要 `log_error()`

```python
# 正確
try:
    result = some_operation()
except DatasetError as e:
    log_error(f"資料集處理失敗: {e}")
    raise

# 錯誤
try:
    result = some_operation()
except:
    pass
```

### Type Annotations

所有新增的函式都必須有完整的 type annotation（mypy strict 要求）：

```python
def evaluate_file(self, file_path: str, timestamp: str, prompt_lang: str = "zh") -> tuple[str, float, str]:
    ...
```

---

## 9. 設定檔規範（config.yaml）

### 現有結構（不得重新命名現有欄位）

```yaml
llm_api:
  base_url: "http://..."      # 必填
  api_key: "..."              # 必填，儲存結果時必須移除
  api_rate_limit: 2           # QPS，-1 為不限
  max_retries: 5
  timeout: 600
  disable_ssl_verify: false

model:
  name: "model-name"          # 必填，用於結果路徑與記錄
  temperature: 0.0
  top_p: 0.9
  max_tokens: 4096
  frequency_penalty: 0.0
  presence_penalty: 0.0
  extra_body: null            # 可選，傳遞 API 額外參數

evaluation:
  dataset_paths:              # 必填，list 格式（即使只有一個路徑）
    - "datasets/dataset1/"
  evaluation_method: "box"   # 必填："pattern" | "box" | "custom_regex"
  system_prompt:             # box method 必填
    zh: "..."
    en: "..."
  datasets_prompt_map:       # 可選，指定特定資料集使用哪種語言的 system_prompt
    "datasets/mmlu/": "en"
  repeat_runs: 5             # 可選，預設 1
  shuffle_options: true      # 可選，預設 false

logging:
  level: "INFO"              # DEBUG | INFO | WARNING | ERROR
```

### 新增欄位的規範

- 所有新欄位**必須有預設值**（`config.get("field", default)`），確保舊設定檔不報錯
- 必填的新欄位需同時更新 `validators.py` 加入驗證，並更新 `templates/*.yaml`
- 欄位名稱使用 snake_case

---

## 10. 輸出格式規範

### 輸出檔案路徑規則

```
results/
├── results_{timestamp}.json           # 整體評測摘要（每次 run 一個）
└── eval_results_{timestamp}_run{N}.jsonl  # 各題詳情（每個 run 一個，N 從 0 開始）
```

### `results_{timestamp}.json` 必要欄位

```json
{
  "timestamp": "20250314_1158",
  "config": { ... },          // 移除 api_key 後的設定
  "dataset_results": { ... }, // 各資料集結果
  "duration_seconds": 0.0
}
```

### `eval_results_{timestamp}_run{N}.jsonl` 每行格式

每行是一個獨立的 JSON 物件，包含：
- `timestamp`
- `file`（資料集路徑）
- `question_id`
- `question`
- `correct_answer`
- `predicted_answer`
- `is_correct`

**禁止**修改這些欄位名稱（向下相容）。新增欄位需確保不會讓舊版解析程式出錯。

---

## 11. 依賴管理規範

- 依賴定義在 `pyproject.toml` 的 `[project.dependencies]`
- 選用功能（如 Google Services、HuggingFace 上傳）應考慮放在 `[project.optional-dependencies]` 或在 import 時加 try/except
- **新增任何 required dependency 之前必須開 Issue**
- 版本限制使用 `>=` 而非 `==`，避免過度鎖定（`numpy ~=2.3.0` 是例外，有特殊相容性需求）
- 開發工具（pytest、black 等）放在 `[project.optional-dependencies] dev`

### 版本同步規則

每次修改 `pyproject.toml` 的 `version` 欄位，**必須同步修改 `twinkle_eval/__init__.py` 的 `__version__`**（目前兩者不一致，待修正）。

### Release 與 Tag 規範

**以下兩種情況必須推版號：**

1. **Milestone 完成**：當一個 GitHub Milestone 的所有 Issue 都已 close 時，bump MINOR 版號
2. **Bug fix 合入**：當 bug fix PR 合入 main 時，必須立即 bump PATCH 版號（第三碼）

**推版號時，必須完成以下所有步驟：**

1. Bump 版本號（`pyproject.toml` + `twinkle_eval/__init__.py`，兩者必須同步）
2. 更新 `CHANGELOG.md`，記錄本次版本的變更內容（新功能、Bug fix、Breaking change 等）
3. 建立 Git tag 並發佈 GitHub Release（`gh release create`）
4. 若為 Milestone 完成，同時 close 該 Milestone

**CHANGELOG 規則**：每次 bump 版本號時，**必須**同步更新 `CHANGELOG.md`。未更新 CHANGELOG 的版本 bump 不得合入。

**版本號遵循 [Semantic Versioning](https://semver.org/)（MAJOR.MINOR.PATCH）：**

| 變更類型 | 版本位 | 判斷標準 | 範例 |
|---------|--------|---------|------|
| **MAJOR** | `X.0.0` | Breaking change：config 格式不相容、CLI 行為改變、輸出結構變更、ABC 介面修改 | 1.0.0 → 2.0.0（模組化架構重構） |
| **MINOR** | `x.Y.0` | 新功能：新增 benchmark、新增 evaluation method、新增 exporter、新增 optional dependency group | 2.0.0 → 2.1.0（新增 IFEval + IFBench） |
| **PATCH** | `x.y.Z` | Bug fix、文件修正、效能優化、不影響使用者介面的內部重構 | 2.1.0 → 2.1.1（修正 scorer 邊界條件） |

**Bug fix 的版號更新不得延遲**：bug fix PR 合入後，必須在同一次工作流程中完成 PATCH 版號 bump、CHANGELOG 更新、Git tag 與 GitHub Release。不得等到下一個 Milestone 才一起推。

**多個 Milestone 同時完成時**（例如同一個 PR 關閉了多個 Milestone），只需 bump 一次版本、發佈一個 Release，Release notes 中列出所有完成的 Milestone。

**Release 標題格式**：`v{VERSION} — {一句話摘要}`
- 範例：`v2.1.0 — IFEval & IFBench Instruction-Following Evaluation`
- 範例：`v2.7.1 — Fix vLLM 0.18+ reasoning field compatibility`

**Tag 命名**：`v{VERSION}`（例如 `v2.1.0`），必須指向 main branch 上 version bump commit。

---

## 12. CLI 設計規範

- 所有新增的 CLI 功能必須有對應的 `--help` 說明文字
- 功能性命令（如 `--benchmark`、`--download-dataset`）不需要 `--config` 即可執行時，需能獨立運作
- 若新功能需要 `--config`，必須在 config 缺失時給出清楚的錯誤提示
- 多個相關選項應使用統一的前綴（如 `--benchmark-*`、`--hf-*`）
- 回傳值：成功 `return 0`，失敗 `return 1`，使用者中斷 `return 130`

### 現有 CLI 選項一覽

| 選項 | 說明 |
|------|------|
| `--config`, `-c` | 指定設定檔路徑（預設 `config.yaml`） |
| `--init [TEMPLATE]` | 產生設定檔範本。不帶參數列出所有可用範本，`--init <name>` 產生單一範本，`--init all` 產生全部 |
| `--validate` | 僅驗證設定檔與資料集格式是否正確（不呼叫 API） |
| `--dry-run` | 載入設定檔與資料集，顯示評測計畫但不呼叫 API |
| `--resume TIMESTAMP` | 從指定時間戳記的中斷點繼續評測（跳過已完成的題目） |
| `--export` | 輸出格式（json / csv / html / google_sheets） |
| `--list-llms` | 列出可用的 LLM 類型 |
| `--list-strategies` | 列出可用的評測策略 |
| `--list-exporters` | 列出可用的輸出格式 |
| `--benchmark` | 執行 LLM 效能基準測試 |
| `--download-dataset` | 從 HuggingFace Hub 下載資料集 |

### 設定檔範本管理

所有設定檔範本集中放在 `twinkle_eval/templates/` 目錄下，以 `{evaluation_method}.yaml` 命名。新增 benchmark 時只需將範本檔放入此目錄，`--init` 會自動掃描並列出。

```bash
twinkle-eval --init                  # 列出所有可用範本
twinkle-eval --init multiple_choice  # 產生 configs/multiple_choice.yaml
twinkle-eval --init all              # 產生全部範本到 configs/
```

---

## 13. 提交 PR 前的 Checklist

**coding agent 在提交任何 PR 之前，必須自行完成以下檢查**：

### 設計合規性
- [ ] 變更符合「核心設計原則」（第 2 節）
- [ ] 若為擴充功能，使用了正確的工廠/策略模式（第 5 節）
- [ ] 若為需要先開 Issue 的情況（第 6 節），確認 Issue 已存在且有 maintainer 回應

### 相容性
- [ ] 舊的 `config.yaml` 仍可正常運作（不得有新必填欄位且沒有預設值）
- [ ] 現有 CLI 選項行為沒有改變
- [ ] 輸出檔案格式沒有 breaking change

### 程式碼品質
- [ ] 通過 `black`、`isort`、`flake8` 檢查
- [ ] 所有新函式都有 type annotation
- [ ] 沒有 `except: pass` 或未記錄的例外吞噬
- [ ] API 金鑰等敏感資訊不會出現在輸出或日誌中
- [ ] 選項標籤沒有硬編碼 A/B/C/D

### 測試
- [ ] **必須執行完整測試套件** `python3 -m pytest tests/ -v`，確認沒有因本次變更而導致新的測試失敗
- [ ] 若有既存的失敗測試（如版本不一致、缺少本機 fixture），需確認這些失敗在本次變更**之前**就已存在，並在 PR 描述中說明
- [ ] 若本次新增了 benchmark，`tests/test_{name}.py` 必須存在且全部通過（第 6.6 節）

### 文件
- [ ] PR 描述清楚說明：動機、修改內容、測試方式
- [ ] 若新增 config 欄位，已更新 `templates/*.yaml`
- [ ] 若新增 CLI 選項，已更新 README

### 新增 Benchmark 專屬（若本 PR 新增評測方法，以下全部必須完成）
- [ ] `datasets/example/{name}/` 已建立，含 10–20 筆代表性樣本
- [ ] 若移植第三方程式碼，已在 Python 檔案頂部加入 attribution 注解
- [ ] `docs/evals/{name}.md` 已建立（使用 `docs/evals/TEMPLATE.md`）
- [ ] 文件中已記錄資料集來源（paper、repo、HuggingFace ID）
- [ ] 文件中已記錄與參考框架的**分數對比**（含模型名稱、資料集規模）
  - 分數誤差符合第 6.3 節容差標準（完整 benchmark ±2%，中型 ±3%，小型 ±5%）
- [ ] 文件中已記錄**速度對比**（本專案單機 vs. 參考框架）
- [ ] `tests/test_{name}.py` 已建立並通過（第 6.6 節），涵蓋 Extractor、Scorer、Checker、Registry、Example Dataset
- [ ] `datasets/example/README.md` 已更新，包含新資料集的條目、config 範例、格式說明（第 6.5.1 節）

### 強制 Reviewer Agent（給 coding agent 的硬性規定）

任何 coding agent（含 Claude Code、Codex 等）在 push PR、或在 PR 上 push 新 commit 之前，**必須**先 spawn 一個獨立的 reviewer agent 對自己的變更做 code review。這條規則沒有例外，包含：

- 第一次開 PR 之前
- 後續每一次 push 新 commit 之前（不論變更大小）
- 即使 coding agent 自認改動很單純、很有信心也一樣

**做法**：使用 `Agent` 工具，通常選 `general-purpose` 或更專業的 reviewer subagent，給它：

1. 完整的 diff（`git diff main...HEAD` 或對應分支的 base）
2. 變更動機（為什麼這樣改、要解決什麼問題）
3. 明確要求：找 bug、找架構違反 CLAUDE.md 原則之處、找邊界條件、找測試漏洞，回報 blockers / should-fix / nits 三層分級

**規則**：

- Reviewer agent 是**獨立第二意見**，coding agent 必須把它的回報當作真實的審查結論面對，不得忽略
- 若 reviewer 回報 blocker，**必須在 push 前修掉**（或在無法修的情況下與使用者確認），不得帶 blocker 上 PR
- Should-fix 與 nits 至少要在 commit message 或 PR 描述中明確列出處置方式（修了 / 開 follow-up issue / 解釋為何不修）
- Coding agent 不得「review 自己」充當這一步——必須是另一個 spawn 出來的 agent，因為 coding agent 對自己剛寫的程式碼有確認偏差

**執行 review 的時機**：

- ✅ Push 之前（強制）
- ✅ 大範圍重構告一段落時（即使還沒要 push）
- ❌ 每次小 edit 之後（不需要，太頻繁會浪費 context）

這條規則的目的是彌補 coding agent 看不到自己 blind spot 的弱點。歷史上多次發生 coding agent 自信滿滿 push 出去後才被使用者或其他 reviewer 找到明顯 bug 的情況，這條規則就是為了把這個 loop 內建到開發流程裡。

### 衝突確認
確認本 PR 的修改是否與以下開放中 PR 有衝突：

| 開放 PR | 主要修改模組 |
|---------|------------|
| #8  (math eval) | `evaluation_strategies.py`、`evaluators.py`、`dataset.py` |
| #9  (rate limit) | `evaluators.py`、JSONL 輸出 |
| #15 (HF upload) | `cli.py`、`main.py`、`pyproject.toml` |
| #17 (MMLU fix)  | `dataset.py`、`evaluators.py` |
| #18 (reasoning token) | `evaluation_strategies.py`、`evaluators.py` |
| #19 (Slurm + HF) | `cli.py`、`main.py`、scripts/ |

若有衝突，PR 描述中必須說明如何解決。

---

## 14. 專案現況快照

### 基本資訊

- **Repo**: https://github.com/ai-twinkle/Eval
- **套件名稱**: `twinkle-eval`（PyPI）
- **版本**: `1.1.2`（pyproject.toml）/ `1.1.0`（`__init__.py`，不一致待修正）
- **授權**: MIT
- **Python**: ≥3.11（pyproject.toml）/ classifiers 含 3.10（有矛盾，待修正）

### 主要依賴

| 套件 | 用途 |
|------|------|
| openai ≥1.93.0 | OpenAI 相容 API |
| pandas ≥2.3.0 | 資料集處理 |
| numpy ~2.3.0 | 統計計算 |
| datasets ≥3.2.0 | HuggingFace 資料集下載 |
| fastparquet | Parquet 格式讀取 |
| google-api-python-client | Google Drive / Sheets |
| pyyaml | YAML 解析 |
| httpx | HTTP 請求 |
| tqdm | 進度條 |

### 已知問題（需處理）

| 嚴重度 | 問題 | 對應 Issue/PR |
|--------|------|--------------|
| 🔴 Bug | `__email__` 未定義，呼叫 `get_info()` 會 NameError | #10 |
| 🔴 Bug | 單一 `dataset_paths` 時評測靜默失敗，無任何提示 | #6 |
| 🔴 Bug | `evaluators.py` 多檔評測結果被覆蓋（`'w'` 應改 `'a'`） | PR #17 |
| 🟡 Bug | `main.py` 相對 import 在直接執行時會 ImportError | #11、PR #12 |
| 🟡 Bug | NVIDIA GPT-OSS API 回應格式差異導致 NoneType 錯誤 | #4 |
| 🟡 Bug | inline think/reasoning token 干擾答案解析 | PR #18 |
| 🟡 不一致 | pyproject.toml version=1.1.2 vs `__init__.py` version=1.1.0 | — |
| 🟡 不一致 | requires-python=3.11 vs classifiers 含 3.10 | — |
| 🟡 設計缺陷 | `evaluators.py` 選項 hardcode A/B/C/D | PR #17 修正中 |
| 🔵 Feature | HuggingFace Dataset 上傳（`--hf-repo-id`） | #14、PR #15 |
| 🔵 Feature | Slurm 多節點分散式評測 | PR #19 |
| 🔵 Feature | Math 評測策略（`\boxed{}` + MathRuler）| PR #8 |
| 🔵 Feature | Logit-based 評測支援 | #7 |
| 🔵 文件 | PyPI 正式發布準備（CHANGELOG、雙語 README）| #13 |

### 開放 PR 與建議合併順序

| PR | 類型 | 摘要 | 作者 | 建議優先序 |
|----|------|------|------|----------|
| #17 | Bug Fix | MMLU 正規化 + 結果累積修正 + 動態選項 | whats2000 | 1（最高） |
| #18 | Bug Fix | inline reasoning token 解析修正 | whats2000 | 2 |
| #12 | Bug Fix | 相對 import → 絕對 import | viiccwen | 3 |
| #9  | Bug Fix | Rate limit 修正、JSONL 詳細資訊 | dave-apmic | 4 |
| #15 | Feature | HuggingFace Dataset 上傳 | whats2000 | 5 |
| #8  | Feature | Math 評測 + per-dataset override + pass@k | cyc00518 | 6 |
| #19 | Feature | Slurm 分散式評測 + HF 整合 | whats2000 | 7（最後） |

**注意**：PR #15 與 #19 均實作 HF 上傳功能，合入 #15 前需與 #19 協調，避免重複實作。

---

## 15. 貢獻者

| GitHub | 名稱 | 角色 |
|--------|------|------|
| teds-lin | Teds Lin | Maintainer |
| lianghsun | Liang Hsun Huang | Maintainer |
| cyc00518 | Min Yi Chen | Contributor |
| k1dav / dave-apmic | Dave Sung | Contributor |
| whats2000 | whats2000 | External Contributor |
| viiccwen | Vic Wen | External Contributor |

Issue 討論請 tag maintainer（@teds-lin 或 @lianghsun）。
