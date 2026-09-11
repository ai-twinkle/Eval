"""建立 Twinkle Eval VisTW-MCQ 範例資料集（用於除錯與快速驗證）。

從 HuggingFace 下載 VisTW-MCQ 的子集，把圖片以 jpg 形式落地到本機，
產生 JSONL 供 evaluation_method: vision_mcq 使用。

VisTW（arXiv 2503.10427，NTU MiuLab，CC BY 4.0）是繁體中文與台灣在地文化
脈絡的視覺語言評測。MCQ 子集涵蓋 21 個學科的圖片選擇題。

兩個不明顯但必要的處理：

1. **id 必須加上學科前綴。** 上游的 ``qid`` 在跨學科之間會重複（多個學科的
   第一題 ``qid`` 都是 0），若直接拿來當 id 與圖片檔名，21 筆會塌縮成少數
   幾筆、圖片互相覆蓋。

2. **答案分佈必須刻意平衡。** 若每科都取第一題，實測 21 題中有 17 題答案是
   A——一個永遠回答 A 的模型就能拿 81%，這份 example 便失去 sanity check 的
   作用（它存在的目的正是抓出 extractor 失效）。因此每科取數題候選，再挑選
   使 A/B/C/D 盡量平均。

用法：
    python scripts/create_vistw_mcq_example.py

產出：
    datasets/example/vistw_mcq/test.jsonl                  (21 筆，每科 1 題)
    datasets/example/vistw_mcq/images/{subject}_{qid}.jpg  (對應圖片檔)
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any, Dict, List

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "example" / "vistw_mcq"
HF_DATASET = "miulab/vistw-mcq"

#: VisTW-MCQ 的 21 個學科，與 HuggingFace 的 config 名稱一致。
SUBJECTS = [
    "accounting",
    "arts",
    "biology",
    "chemistry",
    "chinese_literature",
    "dentistry",
    "electronic_circuits",
    "fundamentals_of_physical_therapy",
    "geography",
    "mathematics",
    "mechanics",
    "medical",
    "music",
    "natural_science",
    "navigation",
    "pharmaceutical_chemistry",
    "physics",
    "sociology",
    "statistics",
    "structural_engineering",
    "veterinary_medicine",
]

#: 每科抓幾題作為候選。取 1 題會讓答案嚴重偏向 A（見模組 docstring），
#: 取 6 題足以在 21 科之間湊出接近平均的 A/B/C/D 分佈。
CANDIDATES_PER_SUBJECT = 6


def select_balanced(
    pools: Dict[str, List[Dict[str, Any]]],
    subjects: List[str],
) -> Dict[str, Dict[str, Any]]:
    """每科各選一題，貪婪地優先挑選目前最稀少的答案。

    同分時取候選池中較前面的題目，確保結果可重現。

    Args:
        pools:    學科 → 候選題目列表。
        subjects: 依序處理的學科名稱。

    Returns:
        學科 → 選中的題目。
    """
    counts: collections.Counter = collections.Counter()
    chosen: Dict[str, Dict[str, Any]] = {}

    for subject in subjects:
        pool = pools[subject]
        best = min(
            range(len(pool)),
            key=lambda i: (counts[str(pool[i]["answer"]).strip().upper()], i),
        )
        chosen[subject] = pool[best]
        counts[str(pool[best]["answer"]).strip().upper()] += 1

    return chosen


def main() -> None:
    from datasets import load_dataset

    images_dir = EXAMPLE_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pools: Dict[str, List[Dict[str, Any]]] = {}
    for subject in SUBJECTS:
        print(f"下載 {subject} ...")
        ds = load_dataset(HF_DATASET, subject, split="test")
        n = min(CANDIDATES_PER_SUBJECT, len(ds))
        pools[subject] = [ds[i] for i in range(n)]

    chosen = select_balanced(pools, SUBJECTS)

    records = []
    for subject in SUBJECTS:
        row = chosen[subject]
        # qid 在跨學科之間不唯一，必須加上學科前綴
        uid = f"{subject}_{row['qid']}"
        image_path = images_dir / f"{uid}.jpg"

        image = row["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(image_path, format="JPEG", quality=90)

        # 只保留評測需要的欄位。上游的 source / stats 不帶入——它們會被
        # evaluator 當成選項渲染進 prompt（見 #143）。
        records.append(
            {
                "id": uid,
                "subject": subject,
                "image_path": f"datasets/example/vistw_mcq/images/{uid}.jpg",
                "question": row["question"],
                "A": row["A"],
                "B": row["B"],
                "C": row["C"],
                "D": row["D"],
                "answer": str(row["answer"]).strip().upper(),
            }
        )

    out = EXAMPLE_DIR / "test.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dist = collections.Counter(r["answer"] for r in records)
    print(f"\n完成：{out}（{len(records)} 筆，{len(SUBJECTS)} 個學科）")
    print(f"答案分佈：{dict(sorted(dist.items()))}")
    print(f"圖片：{images_dir}")


if __name__ == "__main__":
    main()
