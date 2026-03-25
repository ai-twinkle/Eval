# Contributing to Twinkle Eval

感謝你對 Twinkle Eval 的貢獻！在開始之前，請先完整閱讀 **[CLAUDE.md](./CLAUDE.md)**，
這是本專案所有開發者與 coding agent 的強制遵守規範。

---

## 快速開始

### 環境設定

```bash
git clone https://github.com/ai-twinkle/Eval.git
cd Eval

# 建立虛擬環境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安裝開發依賴
pip install -e ".[dev]"

# 可選：若要開發特定 benchmark
pip install -e ".[math]"    # Math eval
pip install -e ".[ifeval]"  # IFEval（需額外初始化 NLTK）
pip install -e ".[tool]"    # BFCL function calling
```

### 執行測試

```bash
pytest tests/
```

### 程式碼風格

提交前請執行：

```bash
black twinkle_eval/
isort twinkle_eval/
flake8 twinkle_eval/
```

---

## 貢獻類型

### 🐛 回報 Bug

1. 搜尋 [Issues](https://github.com/ai-twinkle/Eval/issues) 確認尚未被回報
2. 使用 Bug Report 模板開立 Issue
3. 提供：可重現步驟、預期行為、實際行為、環境資訊

### 💡 功能建議

1. 先確認是否屬於 [必須先開 Issue 討論的情況](./CLAUDE.md#7-必須先開-issue-的情況)
2. 在 Issue 中說明：動機、設計方案、影響範圍
3. 等待至少一位 maintainer（@teds-lin 或 @lianghsun）的明確同意後再開始實作

### 🔧 提交 PR

1. Fork 本 repo，從 `main` 建立新 branch
   - 命名規範：`feat/{feature-name}`、`fix/{issue-id}`、`docs/{topic}`
2. 實作並撰寫測試
3. 確認通過 PR 前的 [Checklist](./CLAUDE.md#13-提交-pr-前的-checklist)
4. 開立 PR，描述清楚動機、修改內容、測試方式

---

## 新增評測 Benchmark

新增一個 benchmark 需要完成五個步驟，詳見 **[CLAUDE.md 第 6 節](./CLAUDE.md#6-新增評測-benchmark-的完整規範)**。

簡要清單：

1. **資料集來源 + example 樣本** → `datasets/example/{name}/`（10–20 筆）
2. **依本專案架構實作** → Extractor + Scorer（+ Checker 若需要）
3. **分數對比驗證** → 與參考框架比較，誤差需符合容差標準
4. **速度對比** → 記錄單機執行時間 vs. 參考框架
5. **評測文件** → `docs/evals/{name}.md`（使用 [`docs/evals/TEMPLATE.md`](./docs/evals/TEMPLATE.md)）

---

## 架構規範

| 我想做的事 | 正確做法 | 參考 |
|-----------|---------|------|
| 新增 LLM 後端 | 繼承 `LLM` ABC，向 `LLMFactory` 註冊 | CLAUDE.md §5.1 |
| 新增評測策略 | 繼承 `Extractor` + `Scorer`，向 `PRESETS` 登錄 | CLAUDE.md §5.2 |
| 新增輸出格式 | 繼承 Exporter 基底，向 `ResultsExporterFactory` 註冊 | CLAUDE.md §5.3 |
| 新增 CLI 參數 | 在 `main.py` 的 `create_cli_parser()` 添加 | CLAUDE.md §5.4 |
| 新增 Benchmark | 遵循第 6 節完整規範 | CLAUDE.md §6 |

---

## 不接受的 PR 類型

以下類型的 PR **在未開 Issue 討論並取得 maintainer 同意前不會被合入**：

- 修改現有 `config.yaml` 必填欄位名稱或結構（breaking change）
- 修改 CLI 選項行為或輸出檔案格式（breaking change）
- 在程式碼中啟動任何模型服務（違反核心設計邊界）
- 新增 required dependency（影響所有使用者）
- 新增 Benchmark 但缺少分數對比或評測文件

---

## 聯絡 Maintainers

| GitHub | 角色 |
|--------|------|
| @teds-lin | Maintainer |
| @lianghsun | Maintainer |

Issue 討論請 tag maintainer。一般問題也可直接在 Issue 留言，我們會盡快回覆。
