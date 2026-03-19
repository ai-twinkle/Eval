"""數學語意等價評分器（基於 mathruler）。"""

import re
from typing import Any, Callable, Dict, List, Optional

from twinkle_eval.core.abc import Scorer


class MathRulerScorer(Scorer):
    """數學語意等價評分器。

    使用 mathruler.grader.grade_answer 進行語意等價判斷，並補強大小寫與排列差異。
    需要安裝：pip install twinkle-eval[math]
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        try:
            from mathruler.grader import grade_answer  # type: ignore[import]

            self._grade_answer: Callable[[str, str], bool] = grade_answer
        except ImportError:
            raise ImportError(
                "數學評測策略需要安裝額外套件，請執行：\n"
                "  pip install twinkle-eval[math]"
            )

    def get_name(self) -> str:
        return "math_ruler"

    def normalize(self, answer: str) -> str:
        """數學答案維持原格式，僅去除首尾空白。"""
        return str(answer).strip()

    def score(self, predicted: str, gold: str) -> bool:
        """用 mathruler 進行語意等價判斷，並補強大小寫與排列差異。"""
        if not predicted or not gold:
            return False
        if self._grade_answer(predicted, gold):
            return True
        return self._post_check_equivalence(predicted, gold)

    def _post_check_equivalence(self, predicted: str, gold: str) -> bool:
        """補強 mathruler 漏網的大小寫 / 逗號分隔順序差異。"""
        normalized_pred = self._normalize_latex_commands(predicted)
        normalized_gold = self._normalize_latex_commands(gold)

        if self._grade_answer(normalized_pred, normalized_gold):
            return True
        return self._compare_as_unordered_list(normalized_pred, normalized_gold)

    def _normalize_latex_commands(self, expr: str) -> str:
        """將 LaTeX 指令轉小寫，例如 \\FRAC -> \\frac。"""
        lowered = re.sub(r"\\[A-Za-z]+", lambda m: m.group(0).lower(), expr)
        lowered = re.sub(r"\\mbox\{([^}]+)\}", lambda m: m.group(1), lowered)
        return lowered.lower()

    def _compare_as_unordered_list(self, predicted: str, gold: str) -> bool:
        """允許逗號分隔解集合忽略順序比對，例如 1,-2 等價 -2,1。"""
        pred_items = self._split_simple_commas(predicted)
        gold_items = self._split_simple_commas(gold)

        if not pred_items or not gold_items or len(pred_items) != len(gold_items):
            return False

        used = [False] * len(gold_items)
        for pred_item in pred_items:
            matched = False
            for idx, gold_item in enumerate(gold_items):
                if used[idx]:
                    continue
                if self._grade_answer(pred_item, gold_item):
                    used[idx] = True
                    matched = True
                    break
            if not matched:
                return False
        return True

    def _split_simple_commas(self, expr: str) -> List[str]:
        """拆解簡單逗號分隔元素（跳過含 LaTeX 排版的表達式）。"""
        stripped = expr.strip()
        if "\\\\" in stripped or "\\begin" in stripped or "\\end" in stripped:
            return []
        if len(stripped) >= 2 and stripped[0] in "([{<" and stripped[-1] in ")]}>":
            stripped = stripped[1:-1]
        parts = [p.strip() for p in stripped.split(",") if p.strip()]
        return parts if len(parts) > 1 else []
