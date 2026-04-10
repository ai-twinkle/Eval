"""建立 Twinkle Eval Vision MCQ 範例資料集（用於除錯）。

從 HuggingFace 下載 MMStar 的子集，並把圖片以 jpg 形式落地到本機，
產生 JSONL 供 evaluation_method: vision_mcq 使用。

用法：
    python scripts/create_vision_mcq_example.py

產出：
    datasets/example/vision_mcq/test.jsonl          (10 筆 MMStar 樣本)
    datasets/example/vision_mcq/images/{id}.jpg     (對應圖片檔)
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "example" / "vision_mcq"
N_SAMPLES = 10

# 解析 MMStar 題目中的 inline 選項：
#   "Question text\nOptions: A: foo, B: bar, C: baz, D: qux"
# 切出乾淨的問題與 A/B/C/D 欄位。
_OPTIONS_HEADER = re.compile(r"\n?Options?\s*:\s*", re.IGNORECASE)
_OPTION_SPLIT = re.compile(r",?\s*(?=[A-Z]:\s)")


def parse_question_and_options(raw_question: str) -> tuple[str, dict[str, str]]:
    """從 MMStar 題目字串中分離出問題與 A/B/C/D 選項。

    若無法解析選項，回傳 (原始問題, 空字典)。
    """
    parts = _OPTIONS_HEADER.split(raw_question, maxsplit=1)
    if len(parts) != 2:
        return raw_question.strip(), {}

    question_clean = parts[0].strip()
    options_str = parts[1].strip()

    options: dict[str, str] = {}
    for chunk in _OPTION_SPLIT.split(options_str):
        chunk = chunk.strip().rstrip(",.")
        m = re.match(r"^([A-Z]):\s*(.+)$", chunk, re.DOTALL)
        if m:
            options[m.group(1)] = m.group(2).strip().rstrip(",")

    return question_clean, options


def create_mmstar_example() -> None:
    print(f"\n建立 Vision MCQ 範例資料集 → {EXAMPLE_DIR}")

    from huggingface_hub import hf_hub_download
    import pandas as pd
    from PIL import Image

    print("  下載 mmstar.parquet ...")
    fp = hf_hub_download(
        repo_id="Lin-Chen/MMStar",
        filename="mmstar.parquet",
        repo_type="dataset",
    )

    df = pd.read_parquet(fp)
    print(f"  完整資料集: {len(df)} 筆")

    images_dir = EXAMPLE_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    sample = df.head(N_SAMPLES)

    for _, row in sample.iterrows():
        idx = int(row["index"])
        sample_id = f"mmstar_{idx:04d}"

        # 1. 解析題目與選項
        raw_question = str(row["question"])
        question_clean, options = parse_question_and_options(raw_question)
        if not options:
            print(f"  ⚠️  跳過 {sample_id}：無法解析選項")
            continue

        # 2. 儲存圖片
        img_bytes = row["image"]
        try:
            img = Image.open(io.BytesIO(img_bytes))
        except Exception as e:
            print(f"  ⚠️  跳過 {sample_id}：圖片解碼失敗 {e}")
            continue

        ext = (img.format or "JPEG").lower()
        if ext == "jpeg":
            ext = "jpg"
        image_filename = f"{sample_id}.{ext}"
        image_abspath = images_dir / image_filename
        with open(image_abspath, "wb") as f:
            f.write(img_bytes)

        # 3. 組裝 record（image_path 用相對於 datasets/example/vision_mcq/ 的路徑）
        record = {
            "id": sample_id,
            "image_path": f"datasets/example/vision_mcq/images/{image_filename}",
            "question": question_clean,
            **options,
            "answer": str(row["answer"]).strip().upper(),
            "category": str(row.get("category", "")),
        }
        records.append(record)

    # 4. 寫 JSONL
    jsonl_path = EXAMPLE_DIR / "test.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  ✅ 已儲存 {len(records)} 題 → {jsonl_path}")
    print(f"  ✅ 已儲存 {len(records)} 張圖片 → {images_dir}/")


if __name__ == "__main__":
    create_mmstar_example()
    print("\n✅ Vision MCQ 範例資料集建立完成！")
