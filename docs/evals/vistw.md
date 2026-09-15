# VisTW Evaluation

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | VisTW（Vision Taiwan） |
| **evaluation_method** | `vision_mcq`（MCQ 子集）／ 待定（Dialogue 子集，見 #151） |
| **實作狀態** | ✅ MCQ 完整實作並完成對比；Dialogue 已實作（對比待補） |
| **需要 optional deps** | `pip install twinkle-eval[vision]`（僅圖片縮放需要） |
| **實作日期** | 2026-09-11 |
| **實作者** | lianghsun |

---

## 1. 來源

### Paper

- **標題**：VisTW: Benchmarking Vision-Language Models for Traditional Chinese in Taiwan
- **作者**：Zhi Rui Tam、Ya-Ting Pai、Yen-Wei Lee、Yun-Nung Chen（國立臺灣大學 MiuLab）
- **發表**：arXiv preprint, 2025-03
- **連結**：https://arxiv.org/abs/2503.10427

### 官方實作

- **Repo**：https://github.com/TMMMU-Benchmark/evaluation
- **授權**：CC BY 4.0
- 本專案未移植官方程式碼。MCQ 子集沿用既有的 `vision_mcq` 評測方法，不需要新的 Extractor / Scorer。

### 資料集

| 子集 | HuggingFace | 題數 |
|------|-------------|------|
| MCQ | [`miulab/vistw-mcq`](https://huggingface.co/datasets/miulab/vistw-mcq) | 4,770（21 學科） |
| Dialogue | [`miulab/vistw-dialogue`](https://huggingface.co/datasets/miulab/vistw-dialogue) | 141 |
| Dialogue（論文版本） | [`VisTai/vistw-dialogue-v0`](https://huggingface.co/datasets/VisTai/vistw-dialogue-v0) | 131 |

- **本專案 example**：`datasets/example/vistw_mcq/`（21 筆，21 個學科各 1 題）

---

## 2. 目的與用途

### 這個 Benchmark 在評什麼？

VisTW 評測視覺語言模型（VLM）對**繁體中文**與**台灣在地文化脈絡**的理解能力。

現有的視覺 benchmark 幾乎全部以英文或簡體中文為主——本專案既有的 MMBench、MMStar、MMMU、POPE 都是如此。這留下一個空缺：一個在英文 VQA 上表現良好的模型，未必看得懂台灣的路牌、發票、健保卡、或中小學課本裡的圖表。VisTW 針對的正是這個落差。

MCQ 子集取自 21 個學科的考試題目，題目與選項皆為繁體中文，圖片內容涵蓋圖表、電路圖、樂譜、醫學影像、地圖等。

### 適合的比較場景

- 比較 VLM 在繁體中文視覺理解上的能力，特別是針對台灣使用者的應用
- 與 MMStar / MMBench 的分數並列，觀察模型是否存在「英文視覺能力強、繁中視覺能力弱」的落差

### 指標說明

| 指標 | 說明 | 越高越好？ |
|------|------|----------|
| accuracy | 選對答案的比例（0–1） | ✅ |
| unparsed_rate | 無法從回應中提取答案的比例 | ❌（越低越好） |

---

## 3. Leaderboard

- **官方 Leaderboard**：https://github.com/TMMMU-Benchmark/evaluation（2025-04-20 更新，含 46 個模型）

---

## 4. 本專案實作說明

### MCQ 子集：沿用 `vision_mcq`，無新程式碼

VisTW-MCQ 是「圖片 + 繁中題幹 + A–D 選項 + 單一正解」，與既有的 `vision_mcq` 完全一致：

- **Extractor**：`twinkle_eval/metrics/extractors/vision_mcq.py` 的 `VisionMCQExtractor`
- **Scorer**：`twinkle_eval/metrics/scorers/exact.py` 的 `ExactMatchScorer`

`VisionMCQExtractor` 優先解析 `\boxed{}`，其次是各種字母答案格式，因此對推理型與非推理型 VLM 都適用。

**沒有新增 `PRESETS` 項目**，config 直接填 `evaluation_method: "vision_mcq"` 即可。

### Dialogue 子集：兩階段，不動 evaluator

VisTW-Dialogue 是開放式問答，由 LLM judge 給 0–10 分。實作為兩階段，兩者都走既有路徑：

| 階段 | evaluation_method | 路徑 | 做什麼 |
|------|-------------------|------|--------|
| 1 生成 | `vistw_dialogue` | 既有 `uses_vision` | 送圖片+問題，記錄自由回答 |
| 2 評分 | `vistw_judge` | 既有文字路徑 | judge 讀「問題+回答+參考答案」給 0–10 |

```bash
# 階段 1
twinkle-eval --config configs/vistw_dialogue.yaml

# 中間步驟：把生成結果併上 ground_truth，組成評分資料集
python scripts/build_vistw_judge_dataset.py \
    --generation results/eval_results_{timestamp}_run0.jsonl \
    --dataset datasets/example/vistw_dialogue/test.jsonl \
    --out datasets/example/vistw_dialogue_judge/judge.jsonl
# ⚠️ 輸出**不可**放進階段 1 的 dataset_paths 目錄——
#    下次跑階段 1 時 find_all_evaluation_files() 會把 judge.jsonl 也掃進來，
#    而它每列都沒有 image_path，會全部報錯。

# 階段 2（model.name 填 judge 模型）
twinkle-eval --config configs/vistw_judge.yaml
```

**為什麼分兩階段**：judge 呼叫因此仍然並行（階段 2 是一次完整評測，走既有的 `ThreadPoolExecutor`）。若把 judge 塞進 Scorer 的 `score_full()`，judge 會變成序列執行，而並行正是本專案的核心賣點。這也與官方的兩階段結構一致，並讓「換 judge 重評」不必重跑生成。

判斷依據是既有的 `ragas`：它是本專案唯一的 LLM-as-judge 方法，**沒有任何 `uses_*` flag**，judge 提示詞烘焙在資料集的 `question` 欄位，scorer 只負責解析。

#### 指標

| 指標 | 說明 |
|------|------|
| `accuracy` | 及格率（預設門檻 6.0，可用 `vistw_judge_pass_threshold` 調整） |
| `llm_output`（JSONL 每列） | judge 的原始回應，0–10 分只存在於此 |

> `judge_score` / `judge_parsed` 是 `VisTWJudgeScorer.score_full()` 的回傳欄位，
> 但那個方法目前不會被呼叫，**兩個欄位都不會寫進 JSONL**。見下方說明。

> ⚠️ **0–10 平均分目前不會出現在 `results_*.json`，也不在 JSONL 的欄位裡。**
>
> `score_full()` 在文字路徑沒有呼叫點，所以 `judge_score` / `judge_parsed` **不會被寫出**。
> 分數只存在於 `llm_output` 的原始 judge 回應中，需自行解析：
>
> ```python
> import glob, json, statistics as st
> from twinkle_eval.metrics.scorers.vistw_judge import VisTWJudgeScorer
>
> s = VisTWJudgeScorer()
> scores, failed = [], 0
> for f in glob.glob("results/eval_results_{timestamp}_run*.jsonl"):
>     for line in open(f, encoding="utf-8"):
>         row = json.loads(line)
>         # 用 predicted_answer 而非 llm_output：它是 pass-through extractor 的輸出，
>         # 已套過 content -> reasoning 的回退，推理型 judge 在 content=null 時才不會漏。
>         v = s.parse_score(row.get("predicted_answer") or "")
>         scores.append(v) if v is not None else (failed := failed + 1)
> if not scores:
>     print(f"沒有任何可解析的分數（failed={failed}）——檢查 judge 是否遵守輸出格式")
> else:
>     print(f"avg={st.mean(scores):.2f}/10  parsed={len(scores)}  failed={failed}")
> ```
>
> 兩層阻礙與修法追蹤於 #163。在它修好之前，**回報分數時請一併回報 `failed` 筆數**。
>
> 註：上面的片段把所有 run 攤平取平均。官方的語意是每題先平均、再跨題平均；
> `failed > 0` 時兩者會分歧（某題少了幾次評分，該題在攤平法中的權重就較低）。
> 要精確對齊官方請先依 `question_id` 分組。

> ⚠️ **`unparsed_rate` 量的不是 judge 解析失敗率。** 它只計入「回應為空」的題目。
> judge 有回應但**格式不符**（抓不到 `[評分]: N`）時，`predicted_answer` 是原文而非 `None`，
> 因此不計入 unparsed —— 但該題的 `judge_score` 是 `null`、不進平均。
>
> 所以平均分的母體可能小於題數，而**沒有任何 metrics 欄位會透露這件事**。
> 請務必用上面的片段確認 `parsed` 與 `total` 的差距，並在回報分數時一併說明。
> 解析失敗的題目**不會被給預設分**——默默給一個中間值會讓分數全面失真且無跡可循。

官方對每個回答評分 5 次（temperature 0.7）取平均；本專案以 `repeat_runs: 5` 達成。

> ⚠️ **先確認端點真的有套用 temperature，否則多次評分沒有意義。**
>
> 實測某個 vLLM 0.26 後端（2026-09-13）：同一 prompt 在 temperature 0.7 與 1.5 下
> 連送 5 次，回應**完全相同**；12 題各評 5 次的標準差是 **0.00**。
> 該端點忽略 temperature，於是「5 次投票取平均」等於跑 5 次相同計算，
> 只是讓 API 用量變 5 倍。
>
> 這也讓「以 judge 自身變異當雜訊下界」的做法失效——那個下界會是 0，
> 看起來任何微小差異都顯著。做 #152 的分數對比前，請先用同一 prompt
> 連送數次確認端點確實有隨機性。

#### 評分指南的移植

`scripts/build_vistw_judge_dataset.py` 的 `JUDGE_PROMPT` 移植自官方
`simplevals/prompts.py` 的 `HUMAN_GUIDELINE`（CC BY 4.0），保留 0–10 的六級描述，
並明確要求以 `[評分]: N` 輸出，供 `VisTWJudgeScorer` 解析。

### 資料集轉換

`scripts/create_vistw_mcq_example.py` 從 HuggingFace 取樣並把圖片落地為 jpg。

兩個需要注意的地方：

1. **`qid` 在跨學科之間不唯一**（多個學科的第一題 `qid` 都是 `0`），因此 example 的 `id` 使用 `{subject}_{qid}` 以保證唯一，否則圖片檔名會互相覆蓋。
2. **答案分佈需要刻意平衡**。若直接取每科第一題，答案會嚴重偏向 A（實測 21 題中 17 題是 A），一個永遠回答 A 的模型就能拿到 81%，使這份 example 失去 sanity check 的作用。腳本改為每科取 6 題候選，再挑選使 A/B/C/D 盡量平均。

### Optional Dependencies

```bash
pip install twinkle-eval[vision]   # Pillow，僅在設定 max_image_size 需要縮放時才用到
```

---

## 5. 使用方式

### 下載完整資料集

```bash
twinkle-eval --download-dataset vistw_mcq
```

> ⚠️ 下載產出的是 parquet，其中 `image` 是 HuggingFace 的 Image struct，**不是檔案路徑**。
> vision 路徑需要本地檔案路徑或 http(s) URL，因此下載後必須先把圖片落地為檔案
> （做法參考 `scripts/create_vistw_mcq_example.py`）。
> 這是既有限制，MMBench / MMStar / MMMU / POPE 都一樣。

### config.yaml 範例

```yaml
llm_api:
  base_url: "http://localhost:8000/v1"
  api_key: "EMPTY"

model:
  name: "your-vlm"
  temperature: 0.0
  max_tokens: 2048

evaluation:
  dataset_paths:
    - "datasets/example/vistw_mcq/"     # 先用 example 驗證流程
  evaluation_method: "vision_mcq"
  shuffle_options: false        # ⚠️ 不可設為 true，見 §8
  strategy_config:
    image_field: "image_path"
    max_image_size: null                 # 需要縮圖時填最長邊像素數
    image_detail: "auto"

logging:
  level: "INFO"
```

### 兩個評測協定

本 benchmark 提供兩個範本，**分數不可互相比較**：

| 範本 | 用途 | prompt | 抽取 |
|------|------|--------|------|
| `vistw_mcq.yaml` | 本專案預設，評測模型用 | 要求 `\boxed{}` | `\boxed{}` 優先 |
| `vistw_mcq_native.yaml` | 與官方 leaderboard 對比用 | 逐字取自官方 `BASELINE_PROMPT`（`答案: $字母`） | regex `答案:` / `Answer:` |

```bash
twinkle-eval --init vistw_mcq          # 本專案協定
twinkle-eval --init vistw_mcq_native   # 官方相容協定
```

即使用 native 範本，仍與官方有一處**刻意保留**的差異：官方在兩段 regex 都失敗時會
**呼叫 LLM 當 parser** 把選項抽出來，本專案不實作這一段。

理由是它會把「模型答得多爛」與「parser 多會猜」混在一起，牴觸 §1.4 的客觀與可重現目標。
代價是我們的 unparsed 被判錯、官方被救回來，因此本專案分數會**系統性偏低**。

> **回報 VisTW 分數時必須一併回報 `unparsed_rate`**（`results_*.json` 的
> `average_unparsed_rate`）。若它明顯大於 0，分數差異的主因就是抽取協定而非模型能力。
> 這是本專案對這個差異的處理方式：不隱藏，而是讓它可見。

---

## 6. 分數對比（vs. 官方實作）

### 測試環境

| 欄位 | 內容 |
|------|------|
| **模型** | `gemma-4-31B-it`（推理型 VLM，思考輸出在 `reasoning_content`） |
| **端點** | 自架 OpenAI 相容 API（LiteLLM），兩邊使用**同一個端點** |
| **資料集** | VisTW-MCQ 253 題，21 學科按 test split 大小比例抽樣（每科 7–26 題） |
| **答案分佈** | A 69 / B 61 / C 63 / D 60（接近均勻） |
| **參考框架** | [TMMMU-Benchmark/evaluation](https://github.com/TMMMU-Benchmark/evaluation) |
| **測試日期** | 2026-09-15 |
| **硬體** | 單機，無 GPU（評測本身不需要） |

253 題 ≥ 200，依 §6.3 套用 **±2% 容差**。

### 結果

| 指標 | Twinkle Eval | 官方實作 | 差異 | 符合容差？ |
|------|-------------|---------|------|----------|
| accuracy | **81.01%** | 80.56% | **+0.45%** | ✅ |
| unparsed | 0 題 | 官方以 LLM-as-parser 補救，未單獨統計 | — | — |
| 錯誤數 | 0 | 0 | — | — |

**差異 0.45 個百分點，遠在 ±2% 容差內。**

### 為了對比而對官方實作做的四項修改

§6.3 要求相同模型、相同題目，因此必須讓兩邊條件對齊。逐項記錄：

1. **取消註解 `max_tokens` 與 `temperature`**（`llms/oai_chat.py:30-33`）。官方原本把這兩個參數註解掉、跑伺服器預設值；而該端點沒有有效預設上限，推理型模型會一路輸出到代理層逾時（實測 HTTP 524，125 秒）。
2. **每科只取前 N 題**（`simplevals/eval.py`），對齊本專案的 253 題子集。
3. **手動建立 `execution_results/` 目錄** —— 官方在寫入前不會自行建立，缺目錄會直接 `FileNotFoundError`。
4. **分科逐一執行** —— 官方以 `load_dataset` 一次載入整個 split（含圖片），記憶體需求遠高於本專案的逐題編碼；測試機器記憶體不足以一次跑完 21 科。

### 這次對比揪出的 bug

對比的價值不只在確認一致性。分科比較時 `medical` 出現 36.4 個百分點的缺口（本專案 63.6%、官方 100%），追查後發現是**本專案的真 bug**：

官方 prompt 教模型輸出 `答案: $字母`，模型照抄那個 `$`，而 `VisionMCQExtractor` 抓不到 —— 更糟的是它會退而抓到推理文字中的其他字母，**把正確作答判成答錯**。官方的 `normalize_response()` 在 regex 前會剝掉 `$`，本專案原本沒有這一步。

修復前後（同一批 253 筆回應重新計分）：

| | accuracy | unparsed |
|---|---|---|
| 修復前 | 75.89% | 11 題（4.3%） |
| 修復後 | 81.82% | 0 題 |

11 題 unparsed **全部**源自此 bug，而非模型答不出來。已修（#166）。

**21 題的 example 從未暴露這個問題，是 253 題的對比子集才揪出來的。**

### 關於抽取協定差異

官方採三段式：prompt 限定格式 → regex → **LLM-as-parser fallback**。本專案刻意不實作第三段（理由見 §8）。

原本預期這會讓本專案系統性偏低，實測結果相反 —— 修好 extractor 後本專案略高於官方。原因是官方用另一個模型去「猜」regex 抓不到的題目，猜的不一定對；而 regex 正確時直接讀到正解，不需要猜。

這支持了本專案的設計取向：**與其用 LLM 補洞，不如把 `unparsed_rate` 誠實暴露出來**——它本身就是指向 extractor 缺陷的訊號，這次正是它把 #166 指出來的。

---

## 7. 速度對比

### 本專案

| 欄位 | 內容 |
|------|------|
| 題數 | 253 |
| **總耗時** | **90 秒**（約 2.8 題/秒） |
| 並行方式 | `ThreadPoolExecutor`，`api_rate_limit: -1`（不限） |
| 模型 | `gemma-4-31B-it`，`max_tokens: 8192` |
| 錯誤數 | 0 |

### 官方實作：無法直接對比

依 §6.4 註明原因，不以失真的數字充數。

官方以 `load_dataset` 一次載入整個 split（含圖片）到記憶體，需求遠高於本專案的逐題編碼。測試機器記憶體不足以一次跑完 21 科，只能**分科逐一執行**，因此拿不到可比的單次總耗時。

另外要澄清一點：**官方也有並行**（`ThreadPoolExecutor`，預設 `max_workers=16`），並非序列執行。README 上「比 ievals 快 9–17 倍」的對比對象是同步框架，與此不是同一回事。若要與官方做有意義的速度對比，應在記憶體充足的環境下，讓兩邊使用相同的 worker 數重跑。

### 附帶觀察：`max_tokens` 對推理型 VLM 的影響

| `max_tokens` | accuracy | unparsed |
|---|---|---|
| 2048 | 57.14% | 19.0% |
| 8192 | 76.19% | 4.8% |
| 不設 | — | 推理到代理層逾時（HTTP 524，125 秒） |

（21 題 example，#166 修復前）。設太小會讓分數嚴重低估，不設會逾時。範本已設為 8192。

---

## 8. 已知限制與 TODO

### ⚠️ 抽取協定與官方不同，分數目前不可直接比較

這是本 benchmark 最重要的限制。官方的 `simplevals/utils.py` 用**三段式串接**抽答案：

| 階段 | 官方做法 | 本專案 |
|------|---------|--------|
| 0 | system prompt 指定輸出格式：`答案: $字母` | ❌ 送不出（見下） |
| 1 | regex `答案\s*:\s*(\w+)` | ✅ 有多組 regex |
| 2 | regex `Answer\s*:\s*(\w+)` | ✅ |
| 3 | **呼叫 LLM 當 parser** 抽出選項 | ❌ 無 fallback，直接判 unparsed |

官方在 regex 之前還會跑 `normalize_response()` 剝掉 markdown 與 LaTeX（`**`、`$\boxed{`、`}$`、`\mathrm{`），抽出後用 `normalize_extracted_answer()` 把阿拉伯文、孟加拉文、全形日文的 A–D 正規化。

三個落差疊加的後果：

1. **我們送不出格式指示。** 依 #144，`evaluation.system_prompt` 只對 `box` 與 `math` 生效，`vision_mcq` 設了也不會進 request。官方靠 prompt 把輸出釘死，所以他們的 regex 只認 `答案:` 就夠；我們是在模型自由發揮的輸出上做 regex。
2. **我們沒有 LLM fallback。** 官方 regex 抓不到就交給 LLM parser，實際 unparsed 率趨近 0；我們抓不到就判錯。
3. **結果是系統性低估**，且低估幅度取決於模型的措辭習慣，使模型之間的比較失真。

因此 **#152 的分數對比在 #144 解決之前沒有意義**，這比 metadata 洩漏（#143）更根本。

### ⚠️ 繁中措辭抽取缺口（#156）

`VisionMCQExtractor` 的中文 pattern 要求「選」後面必須跟著「項」或句號：

| 回應 | 結果 |
|------|------|
| `我選擇 B` / `選 A` / `綜合以上，選 B` | `None` ❌ |
| `故選 C。` / `選 A 項` | ✅ |
| `答案是 A` / `正確答案為 A` | ✅ |

VisTW 是本 repo 唯一**預期模型以繁體中文作答**的 benchmark，所以這不是邊緣情況——一個回答「我選擇 X」的 zh-TW VLM 在這份 example 上會拿 **0/21**，分數反映的是措辭而非正確率。追蹤於 #156。

### ⚠️ 推理型 VLM 需要很大的 `max_tokens`

gemma 4 這類把思考放在 `reasoning_content` 的模型，會在寫出 `\boxed{}` 之前用掉大量 token。
實測 `gemma-4-31B-it` 跑 21 題 example：

| `max_tokens` | 正確率 | unparsed |
|---|---|---|
| 2048 | 57.14% | 19.0% |
| 8192 | **76.19%** | 4.8% |
| 不設 | — | 推理到代理層逾時（HTTP 524，125 秒） |

**差 19 個百分點，全部來自截斷而非模型能力。** 分數異常低時請先看 `unparsed_rate`：
若明顯大於 0，多半是這個原因而非 extractor 失效。

範本已設為 8192。

### ⚠️ 不可開啟 shuffle_options

`shuffle_question_options()` 會把題目重建為 `{question, A–D, answer}`，**丟掉 `image_path`**，導致每一題都以「缺少圖片欄位」被跳過、評測到零題。這是既有問題（#141），修復在 `fix/shuffle-options-and-runner-dedup` 分支上，合入前請維持 `shuffle_options: false`。

### 其他

- **Dialogue 子集尚未實作**（#151）。本文件目前只涵蓋 MCQ。
- **Dialogue 子集的分數對比尚未進行**。MCQ 已完成（見 §6），Dialogue 需要一個 judge 模型與官方相同才有意義，官方用 gemini-2.0-flash。
- **Example 只有 21 筆**，每科 1 題。足以驗證流程與 extractor，不足以反映真實分數。
- 21 個學科的題目難度差異大（`mathematics`、`structural_engineering` 等需要精確讀圖與計算），單一總分會掩蓋分科差異。建議搭配 `subject` 欄位做分科統計。

---

## 附註：與 #143 的關係

VisTW-MCQ 的原始資料集帶有 `source`、`stats` 等 metadata 欄位。本專案的 example 轉換時已剔除，只保留 `id` 與 `subject`。

即便如此，在 #143 修復合入之前，`id` 與 `subject` 仍會被 evaluator 當成選項渲染進 prompt。實測送出的題目長這樣：

```
下圖顯示何種組織病變？
subject: veterinary_medicine     ← 學科提示
A: 腎類澱粉變性
...
```

只有 `subject` 會洩漏——vision 路徑已明確排除 `id`（`evaluator.py` 的 `k not in [..., "id"]`），所以 `id` 不在 prompt 裡。

也就是說，**從這份 example 的 JSONL 移除 `subject` 欄位就能完全關閉洩漏**，不需要等 #143。我們刻意保留它，是為了做分科統計（見 §8），代價是在 #143 合入前接受這個提示。若你要用這份資料看分數而非驗證流程，先把 `subject` 拿掉。

`id` 編碼了學科（`veterinary_medicine_0`）但不進 prompt，因此不受影響。

**因此：這份 example 目前只適合驗證流程是否跑通，不適合用來看分數。分數對比（#152）必須在 #143 合入之後進行**，否則測得的數字無法與官方實作比較。

使用完整資料集（`miulab/vistw-mcq`）時問題更明顯——上游還帶有 `source` 與 `stats` 欄位。
