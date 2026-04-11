"""把 MMStar_MINI.tsv 轉成 Twinkle Eval vision_mcq JSONL（與 VLMEvalKit 對齊）。

VLMEvalKit 用 MMStar_MINI 作為官方小子集（150 題、6 個 category），
本腳本把同一份 tsv 轉成我們的 JSONL 格式，圖片從 base64 解碼成 jpg。
產出資料的 ``id`` 欄位使用 MMStar 原始 ``index`` 字串，方便與 VLMEvalKit
的輸出 xlsx 用 index 做 per-sample join。

執行前請確保 tsv 已下載：
    curl -o /tmp/mmstar_data/MMStar_MINI.tsv \
        https://opencompass.openxlab.space/utils/TEST/MMStar_MINI.tsv

用法：
    python scripts/convert_mmstar_mini_for_comparison.py

產出：
    datasets/comparison/mmstar_mini/test.jsonl
    datasets/comparison/mmstar_mini/images/{index}.jpg
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

TSV_PATH = Path("/tmp/mmstar_data/MMStar_MINI.tsv")
OUT_DIR = Path(__file__).resolve().parent.parent / "datasets" / "comparison" / "mmstar_mini"


def main() -> None:
    if not TSV_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {TSV_PATH}，請先執行：\n"
            f"  mkdir -p /tmp/mmstar_data && curl -o {TSV_PATH} "
            f"https://opencompass.openxlab.space/utils/TEST/MMStar_MINI.tsv"
        )

    df = pd.read_csv(TSV_PATH, sep="\t")
    print(f"載入 {len(df)} 題")

    images_dir = OUT_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for _, row in df.iterrows():
        idx = int(row["index"])
        img_b64 = row["image"]
        img_bytes = base64.b64decode(img_b64)

        # MMStar_MINI 的圖片都是 jpeg
        img_filename = f"{idx}.jpg"
        img_path = images_dir / img_filename
        with open(img_path, "wb") as f:
            f.write(img_bytes)

        record = {
            "id": str(idx),
            "image_path": f"datasets/comparison/mmstar_mini/images/{img_filename}",
            "question": str(row["question"]),
            "A": str(row["A"]),
            "B": str(row["B"]),
            "C": str(row["C"]),
            "D": str(row["D"]),
            "answer": str(row["answer"]).strip().upper(),
            "category": str(row["category"]),
            "l2_category": str(row["l2_category"]),
            "bench": str(row["bench"]),
        }
        records.append(record)

    jsonl_path = OUT_DIR / "test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  已寫入 {len(records)} 題 → {jsonl_path}")
    print(f"  已寫入 {len(records)} 張圖片 → {images_dir}/")


if __name__ == "__main__":
    main()
