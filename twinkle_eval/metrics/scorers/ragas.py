"""RAGAS Scorer。

評價 LLM-as-judge 回傳的 4 個 RAGAS 指標分數：
- faithfulness: 回答中的聲明是否有 context 支撐
- answer_relevancy: 回答與問題的相關性
- context_precision: 檢索到的 context 是否相關
- context_recall: context 是否涵蓋 reference answer 的所有面向

每個指標為 0.0–1.0 的連續分數。整體 is_correct 由平均分數
是否超過閾值決定（預設 0.5）。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from twinkle_eval.core.abc import Scorer

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class RAGASScorer(Scorer):
    """RAGAS Scorer — 解讀 judge LLM 的 4 個指標分數。

    config 可設定：
    - ``ragas_threshold``（float, 預設 0.5）：平均分數的 is_correct 閾值
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.threshold: float = cfg.get("ragas_threshold", 0.5)

    def get_name(self) -> str:
        return "ragas"

    def normalize(self, answer: Any) -> Any:
        """RAGAS 的 gold answer 為 JSON metadata，不做正規化。"""
        return answer

    def score(self, predicted: Any, gold: Any) -> bool:
        """依 4 個指標的平均分數判斷 is_correct。

        Args:
            predicted: RAGASExtractor 回傳的 dict（4 個 float 分數），
                       或 None（judge 回應解析失敗時）。
            gold: ground truth（RAGAS 中為原始 metadata JSON）。

        Returns:
            True 若平均分數 >= threshold。
        """
        if not isinstance(predicted, dict):
            return False

        scores = []
        for metric in RAGAS_METRICS:
            val = predicted.get(metric)
            if val is None:
                return False
            scores.append(float(val))

        avg = sum(scores) / len(scores)
        return avg >= self.threshold
