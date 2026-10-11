"""NIAH (Needle in a Haystack) 測試集生成器。

提供 CLI 工具讓使用者用自己的文本、needle 和參數組合，
生成自訂的 NIAH 測試集 JSONL 檔案。

Usage:
    twinkle-eval --generate-niah \\
        --haystack my_docs.txt \\
        --needle "公司年度營收目標是 42 億元。" \\
        --question "公司的年度營收目標是多少？" \\
        --answer "42 億元" \\
        --context-lengths 1024,4096,16384 \\
        --needle-depths 0,25,50,75,100 \\
        --output datasets/my_niah/
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from twinkle_eval.core.logger import log_error, log_info


def _read_haystack(haystack_path: str) -> str:
    """讀取 haystack 文本檔案。支援 .txt 和多個檔案的目錄。"""
    path = Path(haystack_path)

    if path.is_file():
        with open(path, encoding="utf-8") as f:
            return f.read()

    if path.is_dir():
        texts = []
        for txt_file in sorted(path.glob("*.txt")):
            with open(txt_file, encoding="utf-8") as f:
                texts.append(f.read())
        if not texts:
            raise FileNotFoundError(f"目錄 {path} 中沒有 .txt 檔案")
        return "\n\n".join(texts)

    raise FileNotFoundError(f"找不到 haystack 來源: {haystack_path}")


def _insert_needle(
    haystack: str,
    needle: str,
    context_length_chars: int,
    depth_percent: float,
) -> str:
    """將 needle 插入 haystack 的指定深度位置。

    Args:
        haystack: 完整的背景文本
        needle: 要藏入的事實/句子
        context_length_chars: 目標 context 的字元數（不含 needle）
        depth_percent: 插入深度（0.0=最前面, 1.0=最後面）

    Returns:
        組裝好的 context 文字（haystack + needle）
    """
    # 確保 haystack 足夠長
    if len(haystack) < context_length_chars:
        # 重複 haystack 直到夠長
        repeats = (context_length_chars // len(haystack)) + 1
        haystack = (haystack + "\n\n") * repeats

    needle_pos = int(context_length_chars * depth_percent)

    before = haystack[:needle_pos]
    after = haystack[needle_pos:context_length_chars]

    return before + "\n" + needle + "\n" + after


def generate_niah_dataset(
    haystack_path: str,
    needle: str,
    question: str,
    answer: str,
    context_lengths: list[int],
    needle_depths: list[float],
    output_dir: str,
    language: str = "en",
    chars_per_token: int = 4,
    prompt_template: Optional[str] = None,
) -> str:
    """生成 NIAH 測試集 JSONL。

    Args:
        haystack_path: haystack 文本檔案或目錄路徑
        needle: 要藏入的事實/句子
        question: 對應的提問
        answer: ground truth 答案
        context_lengths: context 長度列表（以 token 為單位）
        needle_depths: 插入深度列表（0-100 的百分比）
        output_dir: 輸出目錄
        language: 語言代碼（"en" 或 "zh"）
        chars_per_token: 每個 token 的平均字元數（英文≈4, 中文≈2）
        prompt_template: 自訂 prompt 模板，用 {context} 和 {question} 佔位符

    Returns:
        輸出檔案路徑
    """
    if prompt_template is None:
        prompt_template = (
            "You are given a long document below. "
            "Read it carefully and answer the question at the end.\n\n"
            "<document>\n{context}\n</document>\n\n"
            "Question: {question}\n"
            "Answer the question based only on the information in the document above."
        )

    haystack = _read_haystack(haystack_path)
    log_info(f"Haystack 長度: {len(haystack)} 字元 (≈{len(haystack) // chars_per_token} tokens)")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "niah_generated.jsonl")

    rows = []
    for ctx_len in context_lengths:
        ctx_chars = ctx_len * chars_per_token
        for depth in needle_depths:
            depth_ratio = depth / 100.0
            context = _insert_needle(haystack, needle, ctx_chars, depth_ratio)
            full_question = prompt_template.format(context=context, question=question)

            row = {
                "id": f"niah_{language}_{ctx_len}t_d{int(depth)}",
                "question": full_question,
                "answer": answer,
                "context_length": ctx_len,
                "needle_depth": depth_ratio,
                "language": language,
                "source": "custom",
            }
            rows.append(row)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    file_size = os.path.getsize(output_path)
    log_info(f"已生成 {len(rows)} 筆 NIAH 測試案例 → {output_path} " f"({file_size / 1024:.1f} KB)")
    return output_path
