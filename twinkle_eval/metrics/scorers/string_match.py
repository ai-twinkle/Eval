"""可設定正規化模式的字串比對評分器。"""

from typing import Any, Dict, Optional

from twinkle_eval.core.abc import Scorer


class StringMatchScorer(Scorer):
    """可設定正規化模式的精確字串比對評分器。

    與 ExactMatchScorer（固定 upper + strip）不同，本 Scorer 支援
    多種正規化模式，適用於答案格式多樣的 benchmark（如 BBH）。

    Config 欄位（皆為可選）：
        normalize_mode (str): 正規化模式，可選值：
            - ``"strip"``（預設）：僅去除首尾空白
            - ``"upper"``：去除首尾空白 + 轉大寫（等同 ExactMatchScorer）
            - ``"lower"``：去除首尾空白 + 轉小寫
            - ``"none"``：不做任何正規化
    """

    VALID_MODES = ("strip", "upper", "lower", "none")

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._mode: str = self._config.get("normalize_mode", "strip")
        if self._mode not in self.VALID_MODES:
            raise ValueError(
                f"normalize_mode '{self._mode}' 不支援。可用模式: {self.VALID_MODES}"
            )

    def get_name(self) -> str:
        return "string_match"

    def normalize(self, answer: str) -> str:
        """依設定模式正規化答案。"""
        s = str(answer)
        if self._mode == "none":
            return s
        s = s.strip()
        if self._mode == "upper":
            return s.upper()
        if self._mode == "lower":
            return s.lower()
        return s  # "strip"

    def score(self, predicted: str, gold: str) -> bool:
        """判斷正規化後的預測答案是否與正解完全相符。"""
        return predicted == gold
