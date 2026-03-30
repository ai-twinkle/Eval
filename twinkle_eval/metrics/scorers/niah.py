"""NIAH (Needle in a Haystack) Scorer。

支援三種評分模式：
- substring: 檢查 ground truth 是否為 response 的子字串（Kamradt / NeedleBench）
- exact: 精確比對（LongBench passage_retrieval_zh 的段落編號）
- f1: token-level F1 score（中文分詞後比對）

預設使用 substring match。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from twinkle_eval.core.abc import Scorer


def _normalize_text(text: str) -> str:
    """正規化文字：去除多餘空白、轉小寫。"""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _normalize_reference_label(text: str) -> str:
    """正規化段落/編號標籤，避免 `段落9` 與 `段落 9` 類型差異造成誤判。"""
    text = _normalize_text(text)
    text = re.sub(r"(段落)\s+(\d+)", r"\1\2", text)
    text = re.sub(r"(paragraph)\s+(\d+)", r"\1\2", text)
    return text


def _tokenize_chinese(text: str) -> list[str]:
    """簡易中文分詞：逐字拆分中文字元，英文按空白分詞。"""
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                tokens.extend("".join(buf).split())
                buf = []
            tokens.append(ch)
        else:
            buf.append(ch)
    if buf:
        tokens.extend("".join(buf).split())
    return tokens


def compute_f1(predicted: str, gold: str) -> float:
    """計算 token-level F1 score，支援中文。"""
    pred_tokens = _tokenize_chinese(_normalize_text(predicted))
    gold_tokens = _tokenize_chinese(_normalize_text(gold))

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0

    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def substring_match(predicted: str, gold: str) -> bool:
    """檢查 gold 是否為 predicted 的子字串（不區分大小寫）。"""
    return _normalize_text(gold) in _normalize_text(predicted)


class NIAHScorer(Scorer):
    """NIAH Scorer — 預設使用 substring match。

    config 中可透過 `niah_scoring_mode` 切換：
    - "substring"（預設）：ground truth 是否出現在 response 中
    - "exact"：精確比對（正規化後）
    - "f1"：token-level F1，閾值由 `niah_f1_threshold` 控制（預設 0.5）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.scoring_mode: str = cfg.get("niah_scoring_mode", "substring")
        self.f1_threshold: float = cfg.get("niah_f1_threshold", 0.5)

    def get_name(self) -> str:
        return "niah"

    def normalize(self, answer: Any) -> str:
        """正規化 ground truth。"""
        if answer is None:
            return ""
        return str(answer).strip()

    def score(self, predicted: Any, gold: Any) -> bool:
        """依 scoring_mode 評分。"""
        if not predicted or not gold:
            return False

        pred_str = str(predicted)
        gold_str = str(gold)

        if self.scoring_mode == "exact":
            return _normalize_reference_label(pred_str) == _normalize_reference_label(gold_str)
        elif self.scoring_mode == "f1":
            return compute_f1(pred_str, gold_str) >= self.f1_threshold
        else:
            # Default: substring
            return substring_match(pred_str, gold_str)
