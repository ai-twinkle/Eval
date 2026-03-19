"""精確字串比對評分器。"""

from typing import Any, Dict, Optional

from twinkle_eval.core.abc import Scorer


class ExactMatchScorer(Scorer):
    """精確字串比對評分器。

    正規化後比對（轉大寫、去除首尾空白）。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def get_name(self) -> str:
        return "exact_match"

    def normalize(self, answer: str) -> str:
        """正規化答案：轉大寫並去除首尾空白。"""
        return answer.strip().upper()

    def score(self, predicted: str, gold: str) -> bool:
        """判斷預測答案是否與正解完全相符。"""
        return predicted == gold
