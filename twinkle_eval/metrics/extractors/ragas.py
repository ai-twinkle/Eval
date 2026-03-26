"""RAGAS Extractor。

從 judge LLM 的回應中提取 4 個 RAGAS 指標分數（JSON 格式）。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from twinkle_eval.core.abc import Extractor


# RAGAS judge 應回傳的 4 個指標
RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _extract_json(text: str) -> Optional[Dict[str, float]]:
    """從 LLM 回應中提取 JSON 物件。

    支援：
    - 純 JSON 回應
    - ```json ... ``` 包裹的回應
    - 混雜文字中的 JSON 物件
    """
    if not text:
        return None

    # 嘗試直接解析
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 嘗試從 markdown code block 中提取
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 嘗試找第一個 JSON 物件
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


class RAGASExtractor(Extractor):
    """RAGAS Extractor — 從 judge LLM 回應中提取 4 個指標分數。"""

    def get_name(self) -> str:
        return "ragas"

    def extract(self, raw: Optional[Any]) -> Optional[Dict[str, float]]:
        """提取 RAGAS 指標分數。

        Returns:
            包含 4 個指標分數的 dict，或 None（解析失敗時）。
            每個分數為 0.0–1.0 的 float。
        """
        if raw is None:
            return None

        text = str(raw)
        parsed = _extract_json(text)
        if parsed is None:
            return None

        # 驗證並正規化分數
        scores: Dict[str, float] = {}
        for metric in RAGAS_METRICS:
            val = parsed.get(metric)
            if val is None:
                return None
            try:
                score = float(val)
                scores[metric] = max(0.0, min(1.0, score))
            except (TypeError, ValueError):
                return None

        return scores
