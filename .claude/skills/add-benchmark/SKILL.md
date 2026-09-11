---
name: add-benchmark
description: 為 Twinkle Eval 新增一個評測 benchmark（IFEval、BFCL、RAGAS、Text2SQL、Vision MCQ 之類）。涵蓋 CLAUDE.md §6 的完整強制流程：先建 Milestone 與 6 個 Issue、準備 example dataset、實作 Extractor + Scorer 並註冊 PRESETS、與參考框架做分數與速度對比、撰寫 docs/evals/{name}.md、更新 datasets/example/README.md、建立 tests/test_{name}.py。當使用者說「新增 benchmark」「加一個評測方法」「支援 XXX 評測」「實作 evaluation_method」時使用。
---

# 新增評測 Benchmark

CLAUDE.md §6 規定的流程，**缺一步就不得合入 main**。這份 skill 是那份規範的可執行版本。

## 動手前先確認

- [ ] 這是**新的評測方法**（新的 Extractor/Scorer 配對），不是既有方法的新資料集。
      若只是新資料集且能沿用既有 `evaluation_method`，走 `twinkle_eval/benchmarks.py`
      的下載註冊表即可，不需要跑本流程。
- [ ] 上一個 benchmark 已經完成（含比較報告）。§6.0.1 明訂：**先完成再開始下一個**。

## 第 0 步：先開 GitHub，再寫程式

**沒有 Milestone 和 Issues 的 benchmark 實作不得開 PR。**

```bash
gh api repos/ai-twinkle/Eval/milestones -f title="{Benchmark} — {一句話說明}" \
  -f description="來源：{paper/repo}｜目的：{評什麼}｜範圍：{預計實作到哪}"
```

接著開 6 個 Issue，每個都要掛 `type: feature` label 與該 Milestone：

| # | 標題格式 | 對應 |
|---|---------|------|
| 1 | `feat({name}): prepare example dataset from HuggingFace` | §6.1 |
| 2 | `feat({name}): implement Extractor + Scorer (+ Checkers)` | §6.2 |
| 3 | `feat({name}): score comparison vs reference framework` | §6.3 |
| 4 | `feat({name}): speed benchmark vs reference framework` | §6.4 |
| 5 | `feat({name}): write docs/evals/{name}.md` | §6.5 |
| 6 | `feat({name}): write tests/test_{name}.py` | §6.6 |

```bash
gh issue create --title "feat({name}): ..." --label "type: feature" --milestone "{Milestone 標題}"
```

## 第 1 步：example dataset

放在 `datasets/example/{name}/`，**10–20 筆**（§6.1 規範；既有資料集實際落在 10–30 筆，如 aime2025 收了全部 30 題），涵蓋主要題型分佈。有子類別的（如 BFCL 的
simple/multiple/parallel）每個子類別至少 2–3 筆。

要求任何人**不下載完整資料集**就能跑通完整流程。格式必須是可直接執行的完整格式
（含 `id`、`question`、答案欄位）。

`scripts/` 底下有既有的抽取腳本可以參考（`prepare_bbh_example.py`、
`create_vision_mcq_example.py`）。

資料集載入器支援 JSON / JSONL / CSV / TSV / Parquet / Arrow。`twinkle_eval/datasets/file.py` 的
`_normalize_record()` 會自動把 `{"choices": [...], "answer": 1}` 這種 HuggingFace 格式
正規化成 `{"A": ..., "B": ..., "answer": "B"}`。

## 第 2 步：實作 Extractor + Scorer

架構是 **Extractor（從輸出抽答案）+ Scorer（比對正解）** 兩個 ABC，定義在
`twinkle_eval/core/abc.py`。**不要**在 `evaluators.py` / `main.py` 加 `if/elif` 分支（原則 A）。

```python
# twinkle_eval/metrics/extractors/{name}.py
from typing import Optional

from twinkle_eval.core.abc import Extractor


class MyExtractor(Extractor):
    def get_name(self) -> str:
        return "{name}"

    def extract(self, llm_output: str) -> Optional[str]:
        """抽不到回傳 None（會被計入 unparsed_count）。"""
        ...
```

```python
# twinkle_eval/metrics/scorers/{name}.py
from twinkle_eval.core.abc import Scorer


class MyScorer(Scorer):
    def get_name(self) -> str:
        return "{name}"

    def normalize(self, answer: str) -> str:
        return answer.strip().upper()

    def score(self, predicted: str, gold: str) -> bool:
        ...
```

註冊到 `twinkle_eval/metrics/__init__.py` 的 `PRESETS`：

```python
PRESETS: Dict[str, Tuple[Type[Extractor], Type[Scorer]]] = {
    ...
    "{name}": (MyExtractor, MyScorer),
}
```

同時把類別加進該檔案的 `__all__`，並視需要加進 `twinkle_eval/__init__.py`。

### Evaluator 需要新的資料流嗎？

`runners/evaluator.py` 的 `evaluate_file()` 依 extractor 上的 flag 分流。若你的 benchmark
不是「送文字、收文字」，設對應的 class attribute：

| Flag | 路徑 | 既有使用者 |
|------|------|-----------|
| `uses_logprobs` | 逐選項算 log-likelihood，不生成 | logit |
| `uses_tool_calls` | 送 `tools`，讀 `message.tool_calls` | bfcl_fc |
| `uses_prompt_injection` | 把 function 定義注入 system prompt | bfcl_prompt |
| `uses_ifeval` | 傳 `instruction_id_list` + `kwargs` 給 checker | ifeval, ifbench |
| `uses_audio` | 送音檔（Whisper API 或多模態） | asr |
| `uses_vision` | 送圖片（base64 data URI 或 URL） | vision_mcq |
| （無） | 純文字解析 | pattern, box, math, ... |

新增第 8 條路徑等於「在核心流程新增大量邏輯」，依 §7 必須**先開 Issue 取得 maintainer 同意**。
先確認能不能用既有路徑。

### 需要多指標（不只 accuracy）？

在 Scorer 上實作 `score_full()` 回傳 dict（IFEval 的四個指標、ASR 的 WER/CER 都是這樣做）。

⚠️ **但 evaluator 目前只在 `uses_ifeval` 與 `uses_audio` 兩條路徑呼叫 `score_full()`。**
文字、logit、tool_calls、prompt_injection、vision 五條路徑不會呼叫，在那些路徑上實作了也不會生效。
若新 benchmark 需要多指標又不走這兩條路徑，得先在 evaluator 加上呼叫點——那屬於
「在核心流程新增邏輯」，依 §7 要先開 Issue。

### 移植第三方程式碼

- **不得**整份複製參考框架，只萃取核心評分邏輯
- Python 檔案頂部加 attribution 註解，`docs/evals/{name}.md` 記錄授權（Apache 2.0 / MIT）
- 原始碼用到不可用的依賴（`absl`、`immutabledict`）必須換成標準庫等價物
- 新依賴一律放 `pyproject.toml` 的 `[project.optional-dependencies]`。
  新增 **required** dependency 依 §7 必須**先開 Issue 並取得 maintainer 同意**，
  且不得讓單機執行路徑增加必要依賴（原則 H）

## 第 3 步：config 範本

`twinkle_eval/templates/{name}.yaml`。`--init` 會自動掃描這個目錄，不需要改程式碼。

範本必須含 `llm_api` / `model` / `evaluation` / `logging` 四個區塊，並在註解裡寫清楚
optional deps 的安裝指令與資料集欄位需求。參考 `templates/ifeval.yaml`。

若有 `strategy_config` 參數，在範本裡列出全部並附預設值。

## 第 4 步：分數對比（§6.3）

用**相同模型、相同題目**同時跑本專案與參考框架。

| 資料集大小 | 可接受誤差 |
|-----------|----------|
| ≥ 200 筆 | ±2% |
| 50–199 筆 | ±3% |
| < 50 筆 | ±5% |
| ≤ 20 筆（example） | 僅 sanity check，不強制對比 |

超出容差必須查明原因（通常是 preprocessing、prompt 格式、或答案正規化邏輯差異）並在文件說明。

## 第 5 步：速度對比（§6.4）

速度是本專案的核心賣點，必須記錄：本專案總耗時 / 並行 worker 數 / 模型名稱，以及參考框架在
同等硬體、同等題數下的耗時。無法直接對比就註明原因。

## 第 6 步：文件

```bash
cp docs/evals/TEMPLATE.md docs/evals/{name}.md
```

填完所有 `{...}` 欄位，特別是第 6 節（分數對比）與第 7 節（速度對比）——**這兩節空著不得合入**。

### 同時更新 datasets/example/README.md（§6.5.1）

三個地方都要改，漏一個就不符合合入條件：

1. 「資料集清單」表格新增條目（目錄、來源、題數、評測方法、說明）
2. 「快速開始」新增對應的 config.yaml 範例
3. 「資料格式」新增該 benchmark 的 JSONL 格式說明

## 第 7 步：測試（§6.6）

`tests/test_{name}.py`，**必須能在不呼叫任何外部 API 的情況下通過**（pure unit test）。
必備覆蓋：

1. **Extractor** — `get_name()`、`extract()` 行為、flag（`uses_*`）
2. **Scorer** — `get_name()`、`normalize()`、`score()` 的正確/錯誤/空值/無效 ground truth
3. **`score_full()`**（若有）— 回傳結構驗證
4. **Checker / 評分邏輯** — 挑 3–5 種代表性 rule type，各寫一個 pass 與一個 fail
5. **Checker Registry**（若有）— 所有指令 ID 都已註冊、數量正確、category 齊全
6. **PRESETS 註冊** — `PRESETS["{name}"]` 存在且對應正確的類別
7. **Example Dataset** — 檔案存在、必要欄位齊全、筆數符合、涵蓋所有子類別
8. **Edge cases** — 空回應、None、kwargs 內的 null 過濾

```bash
python3 -m pytest tests/test_{name}.py -v    # 新增測試全過
python3 -m pytest tests/ -v                  # 完整套件無新增失敗
```

既有失敗（版本不一致、缺本機 fixture）要確認在本次變更**之前**就存在，並在 PR 描述說明。

## 第 8 步：合入前檢查

§6.0.1 的合入前提，全部要打勾：

- [ ] 6 個 Issue 都已 close（或在同一 PR 解決）
- [ ] `docs/evals/{name}.md` 完整，含分數對比與速度對比
- [ ] 分數誤差符合容差標準
- [ ] `datasets/example/{name}/` 有 10–20 筆
- [ ] `datasets/example/README.md` 三處都更新
- [ ] `tests/test_{name}.py` 全過，完整套件無新增失敗
- [ ] 移植的程式碼有 attribution 註解與授權記錄
- [ ] 新依賴都在 `[project.optional-dependencies]`

### push 前：強制 reviewer agent（§13）

**這條沒有例外。** 開 PR 前、以及後續每一次 push 新 commit 前，都必須 spawn 一個獨立的
reviewer agent 審自己的 diff：

```
Agent(subagent_type="general-purpose", prompt="""
審查 {branch} 的變更。先讀 CLAUDE.md，再讀 git diff main...HEAD。
變更動機：{為什麼這樣改}
找：bug、違反 CLAUDE.md 原則之處、邊界條件、測試漏洞。
回報 BLOCKER / SHOULD-FIX / NIT 三層，每項附檔案:行號與具體觸發情境。
""")
```

- Reviewer 回報 blocker **必須在 push 前修掉**
- should-fix / nit 至少要在 PR 描述說明處置方式（修了 / 開 follow-up / 為何不修）
- 不得自己 review 自己——coding agent 對剛寫的程式碼有確認偏差

### 版號（§11）

Milestone 全部 Issue close → bump **MINOR**（新增 benchmark 屬新功能）。
同步改 `pyproject.toml` 與 `twinkle_eval/__init__.py`，更新 `CHANGELOG.md`，
建 tag 並 `gh release create`，然後 close Milestone。
