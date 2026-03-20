"""
測試數學評測策略（MathExtractor + MathRulerScorer）的核心行為。
注意：本測試不需要安裝 mathruler，使用 mock 代替。
"""

import pytest
from unittest.mock import MagicMock, patch


def _make_math_extractor():
    """建立一個 MathExtractor，使用 __new__ 跳過 __init__。"""
    from twinkle_eval.metrics.extractors.math import MathExtractor

    extractor = MathExtractor.__new__(MathExtractor)
    extractor._config = {}
    return extractor


def _make_math_scorer():
    """建立一個 MathRulerScorer，其中 grade_answer 被 mock。
    使用 __new__ 跳過 __init__，不需要安裝 mathruler。
    """
    from twinkle_eval.metrics.scorers.math import MathRulerScorer

    scorer = MathRulerScorer.__new__(MathRulerScorer)
    scorer._config = {}
    # 用簡單的完全比對模擬 grade_answer
    scorer._grade_answer = lambda a, b: str(a).strip() == str(b).strip()
    return scorer


class TestMathExtractorBoxed:
    """extract 從 \\boxed{} 提取答案"""

    def test_extracts_simple_boxed(self):
        extractor = _make_math_extractor()
        assert extractor.extract(r"答案是 \boxed{42}") == "42"

    def test_extracts_last_boxed_when_multiple(self):
        extractor = _make_math_extractor()
        assert extractor.extract(r"首先 \boxed{10} 最後 \boxed{42}") == "42"

    def test_extracts_fraction_in_boxed(self):
        extractor = _make_math_extractor()
        assert extractor.extract(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"

    def test_fallback_to_last_line_when_no_boxed(self):
        extractor = _make_math_extractor()
        result = extractor.extract("計算過程...\n最終答案是 42")
        assert result == "最終答案是 42"

    def test_returns_none_for_empty_input(self):
        extractor = _make_math_extractor()
        assert extractor.extract("") is None

    def test_returns_none_for_none_input(self):
        extractor = _make_math_extractor()
        assert extractor.extract(None) is None


class TestMathRulerScorerIsCorrect:
    """score 語意等價判斷"""

    def test_exact_match(self):
        scorer = _make_math_scorer()
        assert scorer.score("42", "42") is True

    def test_mismatch(self):
        scorer = _make_math_scorer()
        assert scorer.score("42", "43") is False

    def test_none_predicted_returns_false(self):
        scorer = _make_math_scorer()
        assert scorer.score(None, "42") is False

    def test_none_gold_returns_false(self):
        scorer = _make_math_scorer()
        assert scorer.score("42", None) is False


class TestMathRulerScorerNormalize:
    """normalize 維持原始格式"""

    def test_strips_whitespace(self):
        scorer = _make_math_scorer()
        assert scorer.normalize("  42  ") == "42"

    def test_keeps_latex_intact(self):
        scorer = _make_math_scorer()
        assert scorer.normalize(r"\frac{1}{2}") == r"\frac{1}{2}"


class TestMathImportError:
    """未安裝 mathruler 時應拋出清楚的 ImportError"""

    def test_import_error_message(self):
        import sys
        # 確保 mathruler 不在已載入模組中
        for key in list(sys.modules.keys()):
            if "mathruler" in key:
                del sys.modules[key]

        with patch.dict("sys.modules", {"mathruler": None, "mathruler.grader": None}):
            from twinkle_eval.metrics.scorers.math import MathRulerScorer

            with pytest.raises(ImportError, match="pip install twinkle-eval\\[math\\]"):
                MathRulerScorer()
