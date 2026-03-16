"""
測試數學評測策略（MathExtractionStrategy）的核心行為。
注意：本測試不需要安裝 mathruler，使用 mock 代替。
"""

import pytest
from unittest.mock import MagicMock, patch


def _make_math_strategy():
    """建立一個 MathExtractionStrategy，其中 grade_answer 被 mock。
    使用 __new__ 跳過 __init__，不需要安裝 mathruler。
    """
    from twinkle_eval.evaluation_strategies import MathExtractionStrategy

    strategy = MathExtractionStrategy.__new__(MathExtractionStrategy)
    strategy.config = {}
    # 用簡單的完全比對模擬 grade_answer
    strategy._grade_answer = lambda a, b: str(a).strip() == str(b).strip()
    return strategy


class TestMathExtractionStrategyBoxed:
    """extract_answer 從 \\boxed{} 提取答案"""

    def test_extracts_simple_boxed(self):
        strategy = _make_math_strategy()
        assert strategy.extract_answer(r"答案是 \boxed{42}") == "42"

    def test_extracts_last_boxed_when_multiple(self):
        strategy = _make_math_strategy()
        assert strategy.extract_answer(r"首先 \boxed{10} 最後 \boxed{42}") == "42"

    def test_extracts_fraction_in_boxed(self):
        strategy = _make_math_strategy()
        assert strategy.extract_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_fallback_to_last_line_when_no_boxed(self):
        strategy = _make_math_strategy()
        result = strategy.extract_answer("計算過程...\n最終答案是 42")
        assert result == "最終答案是 42"

    def test_returns_none_for_empty_input(self):
        strategy = _make_math_strategy()
        assert strategy.extract_answer("") is None

    def test_returns_none_for_none_input(self):
        strategy = _make_math_strategy()
        assert strategy.extract_answer(None) is None


class TestMathExtractionStrategyIsCorrect:
    """is_correct 語意等價判斷"""

    def test_exact_match(self):
        strategy = _make_math_strategy()
        assert strategy.is_correct("42", "42") is True

    def test_mismatch(self):
        strategy = _make_math_strategy()
        assert strategy.is_correct("42", "43") is False

    def test_none_predicted_returns_false(self):
        strategy = _make_math_strategy()
        assert strategy.is_correct(None, "42") is False

    def test_none_gold_returns_false(self):
        strategy = _make_math_strategy()
        assert strategy.is_correct("42", None) is False


class TestMathExtractionStrategyNormalize:
    """normalize_answer 維持原始格式"""

    def test_strips_whitespace(self):
        strategy = _make_math_strategy()
        assert strategy.normalize_answer("  42  ") == "42"

    def test_keeps_latex_intact(self):
        strategy = _make_math_strategy()
        assert strategy.normalize_answer(r"\frac{1}{2}") == r"\frac{1}{2}"


class TestMathImportError:
    """未安裝 mathruler 時應拋出清楚的 ImportError"""

    def test_import_error_message(self):
        import sys
        # 確保 mathruler 不在已載入模組中
        for key in list(sys.modules.keys()):
            if "mathruler" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"mathruler": None, "mathruler.grader": None}):
            from twinkle_eval.evaluation_strategies import MathExtractionStrategy

            with pytest.raises(ImportError, match="pip install twinkle-eval\\[math\\]"):
                MathExtractionStrategy()
