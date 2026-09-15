"""把 VisTW-Dialogue 階段 1 的生成結果組成階段 2 的評分資料集。

VisTW-Dialogue 是兩階段評測：

    階段 1  evaluation_method: vistw_dialogue   送圖片+問題，模型自由作答
    階段 2  evaluation_method: vistw_judge      judge 讀「問題+回答+參考答案」給 0–10

本腳本是中間那一步：讀階段 1 的 ``results/eval_results_*.jsonl``，
併上原始資料集的參考答案（``answer`` 欄位），產生階段 2 的資料集——其 ``question``
欄位是填好的 judge 提示詞。

judge 提示詞移植自 VisTW 官方實作 ``simplevals/prompts.py`` 的 HUMAN_GUIDELINE
（https://github.com/TMMMU-Benchmark/evaluation，CC BY 4.0）。

用法（省略 --generation 會自動取 results/ 下最新的一個）：
    python scripts/build_vistw_judge_dataset.py \
        --dataset datasets/example/vistw_dialogue/test.jsonl \
        --out datasets/example/vistw_dialogue_judge/judge.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

#: 移植自官方 simplevals/prompts.py 的 HUMAN_GUIDELINE（CC BY 4.0）。
#: 結尾要求以 [評分]: N 輸出，VisTWJudgeScorer 依此解析。
JUDGE_PROMPT = """[Question]
{question}

[Assistant Response]
{response}

[Ground Truth]
{ground_truth}

請根據使用者詢問的問題 [Question] 與正確答案 [Ground Truth]，去評價助手的回覆 [Assistant Response]，評分依照下方的評價指導手冊。

# 評分標註指南

## 評分範圍（0-10 分）

- **10 分 完美**：完全準確無誤、回答問題的所有部分、清晰且條理分明、提供有幫助的補充說明
- **8-9 分 非常好**：有些微錯誤或遺漏、主要重點都有涵蓋、組織良好
- **6-7 分 良好**：有一些小錯誤、大部分重點都有提到、組織尚可
- **4-5 分 普通**：有數個錯誤、遺漏一些重點、說明不夠完整
- **2-3 分 不佳**：有許多錯誤、遺漏重要資訊、組織不清楚
- **0-1 分 不及格**：大部分或完全錯誤、未回答問題重點

請先簡短說明評分理由，最後**必須**以下列格式輸出分數：

[評分]: N
"""


def load_jsonl(path: Path) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--generation",
        help="階段 1 的 eval_results_*.jsonl；省略時自動取 results/ 下最新的一個",
    )
    ap.add_argument(
        "--dataset", required=True, help="原始 VisTW-Dialogue 資料集（參考答案在 answer 欄位）"
    )
    ap.add_argument("--out", required=True, help="階段 2 資料集的輸出路徑")
    args = ap.parse_args()

    if not args.generation:
        candidates = sorted(
            Path("results").glob("eval_results_*_run0.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            ap.error(
                "找不到 results/eval_results_*_run0.jsonl，請先執行階段 1，或用 --generation 指定"
            )
        args.generation = str(candidates[0])
        print(f"自動選用最新的生成結果：{args.generation}")

    gen_path = Path(args.generation)
    if not gen_path.exists():
        ap.error(
            f"找不到 {gen_path}。"
            "若是從文件複製指令，記得把 {timestamp} 換成實際的檔名，"
            "或直接省略 --generation 讓腳本自動選最新的一個。"
        )

    source = load_jsonl(Path(args.dataset))
    # 階段 1 的 question_id 是資料集中的索引（evaluator 以 enumerate 產生）
    by_index: Dict[int, dict] = {i: row for i, row in enumerate(source)}

    generated = load_jsonl(gen_path)
    records, skipped = [], 0

    for g in generated:
        idx = g.get("question_id")
        src = by_index.get(idx)
        if src is None:
            skipped += 1
            continue

        response = g.get("predicted_answer") or g.get("llm_output") or ""
        if not str(response).strip():
            # 階段 1 沒產出回答的題目無從評分，跳過並回報，不要當成 0 分
            skipped += 1
            continue

        records.append(
            {
                "question": JUDGE_PROMPT.format(
                    question=src["question"],
                    response=response,
                    ground_truth=src["answer"],
                ),
                # ⚠️ 這裡**只能**有 question 與 answer。
                # 文字路徑的 build_question_text() 沒有 exclude 參數，
                # question/answer 以外的欄位會全部被印進 prompt——包括 id——
                # 而那會把 JUDGE_PROMPT 結尾的「[評分]: N」格式指令擠離結尾，
                # 使 judge 更容易忽略它。實測曾因此在 prompt 尾端出現
                # 「id: ...\noriginal_question: ...」。除錯資訊請另存 sidecar。
                "answer": "",
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"完成：{out}（{len(records)} 筆）")
    if skipped:
        print(f"⚠️  跳過 {skipped} 筆（對不上原資料集，或階段 1 未產出回答）")
        print("   這些題目不會出現在階段 2，因此不會被算成 0 分——")
        print("   但也代表最終平均分的母體比原資料集小，回報時請一併說明。")


if __name__ == "__main__":
    main()
