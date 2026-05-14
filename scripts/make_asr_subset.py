#!/usr/bin/env python3
"""
從現有 JSONL 評測集取前 N 筆建立子集，用於快速 API 評測（節省 token）。

使用方式：
  python scripts/make_asr_subset.py --input datasets/tat_s2st/test.jsonl \
    --output datasets/tat_s2st/test_100.jsonl --n 100
  python scripts/make_asr_subset.py --input datasets/cv_zh_tw/test.jsonl \
    --output datasets/cv_zh_tw/test_100.jsonl --n 100
"""

import argparse
import json
import random
from pathlib import Path


def make_subset(input_path: str, output_path: str, n: int, seed: int = 42) -> None:
    records = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if len(records) <= n:
        subset = records
        print(f"⚠️  原始資料集 {len(records)} 筆 ≤ {n}，輸出全部")
    else:
        random.seed(seed)
        subset = random.sample(records, n)
        print(f"從 {len(records)} 筆中隨機取 {n} 筆（seed={seed}）")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in subset:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"✅  已寫入 {output_path}（{len(subset)} 筆）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="建立 ASR 評測子集")
    parser.add_argument("--input", required=True, help="原始 JSONL 路徑")
    parser.add_argument("--output", required=True, help="輸出 JSONL 路徑")
    parser.add_argument("--n", type=int, default=100, help="取樣筆數（預設 100）")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子（預設 42）")
    args = parser.parse_args()
    make_subset(args.input, args.output, args.n, args.seed)
