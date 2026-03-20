"""Tests for BoxExtractor and PatternExtractor option range coverage.

Regression test for the [A-D] hardcoding bug: extractors must correctly extract
answers for options beyond D (E–J), as required by datasets like MMLU-Pro.
"""
import pytest

from twinkle_eval.metrics.extractors.box import BoxExtractor as BoxExtractionStrategy
from twinkle_eval.metrics.extractors.pattern import PatternExtractor as PatternMatchingStrategy


class TestBoxExtractionStrategyOptionRange:
    """BoxExtractionStrategy 應支援 A–Z 的任意選項，不限於 A–D。"""

    def setup_method(self):
        self.extractor = BoxExtractionStrategy()

    @pytest.mark.parametrize("option", list("ABCDEFGHIJ"))
    def test_extracts_boxed_option(self, option):
        r"""\\boxed{X} 對 A–J 所有選項都應能正確提取。"""
        assert self.extractor.extract(f"\\boxed{{{option}}}") == option

    @pytest.mark.parametrize("option", list("ABCDEFGHIJ"))
    def test_extracts_box_option(self, option):
        r"""\\box{X} 對 A–J 所有選項都應能正確提取。"""
        assert self.extractor.extract(f"\\box{{{option}}}") == option

    def test_extracts_from_multiline_output(self):
        r"""實際模型輸出格式：推導過程後以 $$\n\\boxed{J}\n$$ 結尾。"""
        llm_output = "The answer is option J.\n\n$$\n\\boxed{J}\n$$"
        assert self.extractor.extract(llm_output) == "J"

    def test_does_not_extract_lowercase(self):
        """小寫選項不應被提取（選項標籤統一為大寫）。"""
        result = self.extractor.extract("\\boxed{j}")
        # extract() 本身不應匹配小寫
        assert result != "j"


class TestPatternMatchingStrategyOptionRange:
    """PatternMatchingStrategy 應支援 A–Z 的任意選項，不限於 A–D。"""

    def setup_method(self):
        self.extractor = PatternMatchingStrategy()

    @pytest.mark.parametrize("option", list("ABCDEFGHIJ"))
    def test_extracts_answer_colon(self, option):
        """「答案：X」格式對 A–J 所有選項都應能正確提取。"""
        assert self.extractor.extract(f"答案：{option}") == option

    @pytest.mark.parametrize("option", list("ABCDEFGHIJ"))
    def test_extracts_correct_answer_is(self, option):
        """「correct answer is:\\nX.」格式對 A–J 所有選項都應能正確提取。"""
        assert self.extractor.extract(f"correct answer is:\n{option}.") == option
