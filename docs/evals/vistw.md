# VisTW Evaluation

## 概覽

| 欄位 | 內容 |
|------|------|
| **Benchmark 名稱** | VisTW（Vision Taiwan） |
| **evaluation_method** | `vision_mcq`（MCQ 子集）／ 待定（Dialogue 子集，見 #151） |
| **實作狀態** | 🚧 Phase 1（MCQ 完成，Dialogue 待實作） |
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

### Dialogue 子集：待實作

VisTW-Dialogue 是開放式問答，由 LLM judge 給 0–10 分。這需要「視覺 + LLM-as-judge」的組合，而本專案目前沒有——`vision_mcq` 是視覺但用 exact match，`ragas` 是 judge 但純文字。

設計討論見 #151，其中包含 evaluator 路由的選項與取捨（judge 呼叫能否並行是關鍵考量）。

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

> ⏳ **待補**——追蹤於 #152。

需使用相同模型、相同題目同時跑本專案與 https://github.com/TMMMU-Benchmark/evaluation。

容差標準（§6.3）：完整 benchmark（≥200 筆）±2%、中型（50–199 筆）±3%、小型（<50 筆）±5%。MCQ 有 4,770 題，建議取 ≥200 題子集以套用 ±2%。

`datasets/example/vistw_mcq/` 只有 21 筆，依 §6.3 僅作 sanity check，不強制對比。

---

## 7. 速度對比

> ⏳ **待補**——追蹤於 #153。

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

### ⚠️ 不可開啟 shuffle_options

`shuffle_question_options()` 會把題目重建為 `{question, A–D, answer}`，**丟掉 `image_path`**，導致每一題都以「缺少圖片欄位」被跳過、評測到零題。這是既有問題（#141），修復在 `fix/shuffle-options-and-runner-dedup` 分支上，合入前請維持 `shuffle_options: false`。

### 其他

- **Dialogue 子集尚未實作**（#151）。本文件目前只涵蓋 MCQ。
- **分數與速度對比尚未進行**（#152、#153）。依 §6.0.1，這兩節補齊前本 benchmark 不算完成。
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
