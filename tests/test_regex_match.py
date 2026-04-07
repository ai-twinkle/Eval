"""tests/test_regex_match.py — RegexMatchExtractor + StringMatchScorer 測試。"""

import json
import os

import pytest

from twinkle_eval.metrics.extractors.regex_match import RegexMatchExtractor
from twinkle_eval.metrics.scorers.string_match import StringMatchScorer
from twinkle_eval.metrics import PRESETS, create_metric_pair


# ──────────────────────────────────────────────────────────────────────
# RegexMatchExtractor
# ──────────────────────────────────────────────────────────────────────


class TestRegexMatchExtractorBasic:
    """基本屬性測試。"""

    def test_get_name(self) -> None:
        ext = RegexMatchExtractor()
        assert ext.get_name() == "regex_match"

    def test_default_pattern(self) -> None:
        ext = RegexMatchExtractor()
        assert len(ext.patterns) >= 4
        assert r"[Tt]he answer is" in ext.patterns[0]

    def test_custom_pattern_string(self) -> None:
        ext = RegexMatchExtractor({"answer_pattern": r"result: (.*)"})
        assert ext.patterns == [r"result: (.*)"]

    def test_custom_pattern_list(self) -> None:
        patterns = [r"answer: (.*)", r"result: (.*)"]
        ext = RegexMatchExtractor({"answer_pattern": patterns})
        assert ext.patterns == patterns


class TestRegexMatchExtractorExtract:
    """答案提取測試。"""

    def test_bbh_mc_format(self) -> None:
        ext = RegexMatchExtractor()
        output = "Let me think step by step... the answer is (B)."
        assert ext.extract(output) == "(B)"

    def test_bbh_mc_uppercase(self) -> None:
        ext = RegexMatchExtractor()
        output = "After reasoning, The answer is (C)."
        assert ext.extract(output) == "(C)"

    def test_bbh_binary_yes(self) -> None:
        ext = RegexMatchExtractor()
        output = "Based on the analysis, the answer is Yes."
        assert ext.extract(output) == "Yes"

    def test_bbh_binary_false(self) -> None:
        ext = RegexMatchExtractor()
        output = "Evaluating the expression, the answer is False."
        assert ext.extract(output) == "False"

    def test_bbh_binary_invalid(self) -> None:
        ext = RegexMatchExtractor()
        output = "The argument is flawed, so the answer is invalid."
        assert ext.extract(output) == "invalid"

    def test_bbh_integer(self) -> None:
        ext = RegexMatchExtractor()
        output = "Computing step by step: the answer is 24."
        assert ext.extract(output) == "24"

    def test_bbh_negative_integer(self) -> None:
        ext = RegexMatchExtractor()
        output = "the answer is -50."
        assert ext.extract(output) == "-50"

    def test_bbh_word_sorting(self) -> None:
        ext = RegexMatchExtractor()
        output = "Sorting alphabetically, the answer is barn damp dot."
        assert ext.extract(output) == "barn damp dot"

    def test_bbh_dyck_languages(self) -> None:
        ext = RegexMatchExtractor()
        output = "the answer is ] ]."
        assert ext.extract(output) == "] ]"

    def test_trailing_period_removed(self) -> None:
        ext = RegexMatchExtractor()
        output = "the answer is (A)."
        assert ext.extract(output) == "(A)"

    def test_no_trailing_period(self) -> None:
        ext = RegexMatchExtractor()
        output = "the answer is (A)"
        assert ext.extract(output) == "(A)"

    def test_no_match(self) -> None:
        ext = RegexMatchExtractor()
        output = "I'm not really sure about this question"
        assert ext.extract(output) is None

    def test_empty_string(self) -> None:
        ext = RegexMatchExtractor()
        assert ext.extract("") is None

    def test_none_input(self) -> None:
        ext = RegexMatchExtractor()
        assert ext.extract(None) is None  # type: ignore[arg-type]

    def test_custom_pattern_extraction(self) -> None:
        ext = RegexMatchExtractor({"answer_pattern": r"ANSWER:\s*(.*)"})
        output = "ANSWER: 42"
        assert ext.extract(output) == "42"

    def test_multiple_patterns_fallback(self) -> None:
        ext = RegexMatchExtractor(
            {"answer_pattern": [r"Final answer: (.*)", r"the answer is (.*)"]}
        )
        output = "the answer is Yes."
        assert ext.extract(output) == "Yes"

    def test_multiple_patterns_first_match(self) -> None:
        ext = RegexMatchExtractor(
            {"answer_pattern": [r"Final answer: (.*)", r"the answer is (.*)"]}
        )
        output = "Final answer: No"
        assert ext.extract(output) == "No"

    def test_correct_answer_is_pattern(self) -> None:
        ext = RegexMatchExtractor()
        output = "The correct answer is: (B) 12/25/1937"
        assert ext.extract(output) == "(B)"

    def test_correct_option_is_pattern(self) -> None:
        ext = RegexMatchExtractor()
        output = "The correct option is (D) three."
        assert ext.extract(output) == "(D)"

    def test_final_answer_pattern(self) -> None:
        ext = RegexMatchExtractor()
        output = "### Final Answer:\nThe argument is **invalid**."
        result = ext.extract(output)
        assert result is not None

    def test_the_final_answer_is_pattern(self) -> None:
        ext = RegexMatchExtractor()
        output = "The final answer is 42."
        assert ext.extract(output) == "42"

    def test_case_insensitive_matching(self) -> None:
        ext = RegexMatchExtractor()
        output = "THE ANSWER IS (D)."
        assert ext.extract(output) == "(D)"

    def test_last_match_when_multiple_occurrences(self) -> None:
        """多行情況下取最後一個 match，因為模型可能在推理中修正答案。"""
        ext = RegexMatchExtractor()
        output = "the answer is (A).\nWait, the answer is (B)."
        assert ext.extract(output) == "(B)"


# ──────────────────────────────────────────────────────────────────────
# StringMatchScorer
# ──────────────────────────────────────────────────────────────────────


class TestStringMatchScorerBasic:
    """基本屬性測試。"""

    def test_get_name(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.get_name() == "string_match"

    def test_default_mode_is_strip(self) -> None:
        scorer = StringMatchScorer()
        assert scorer._mode == "strip"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="normalize_mode"):
            StringMatchScorer({"normalize_mode": "banana"})


class TestStringMatchScorerNormalize:
    """正規化模式測試。"""

    def test_strip_mode(self) -> None:
        scorer = StringMatchScorer({"normalize_mode": "strip"})
        assert scorer.normalize("  (B)  ") == "(B)"

    def test_upper_mode(self) -> None:
        scorer = StringMatchScorer({"normalize_mode": "upper"})
        assert scorer.normalize("  yes  ") == "YES"

    def test_lower_mode(self) -> None:
        scorer = StringMatchScorer({"normalize_mode": "lower"})
        assert scorer.normalize("  Yes  ") == "yes"

    def test_none_mode(self) -> None:
        scorer = StringMatchScorer({"normalize_mode": "none"})
        assert scorer.normalize("  Yes  ") == "  Yes  "

    def test_non_string_input(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.normalize(42) == "42"  # type: ignore[arg-type]


class TestStringMatchScorerScore:
    """評分邏輯測試。"""

    def test_exact_match(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("(B)", "(B)") is True

    def test_mismatch(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("(A)", "(B)") is False

    def test_binary_match(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("Yes", "Yes") is True

    def test_binary_case_mismatch_strip_mode(self) -> None:
        scorer = StringMatchScorer({"normalize_mode": "strip"})
        # strip 模式不改大小寫，所以 yes != Yes
        assert scorer.score("yes", "Yes") is False

    def test_binary_case_match_lower_mode(self) -> None:
        scorer = StringMatchScorer({"normalize_mode": "lower"})
        # evaluator 會先 normalize 再 score，模擬真實流程
        pred = scorer.normalize("yes")
        gold = scorer.normalize("Yes")
        assert scorer.score(pred, gold) is True

    def test_integer_match(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("24", "24") is True

    def test_integer_mismatch(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("25", "24") is False

    def test_word_sorting_match(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("barn damp dot", "barn damp dot") is True

    def test_word_sorting_mismatch(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("barn dot damp", "barn damp dot") is False

    def test_dyck_languages_match(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("] ]", "] ]") is True

    def test_empty_predicted(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("", "(B)") is False

    def test_both_empty(self) -> None:
        scorer = StringMatchScorer()
        assert scorer.score("", "") is True


# ──────────────────────────────────────────────────────────────────────
# PRESETS 註冊
# ──────────────────────────────────────────────────────────────────────


class TestRegexMatchPresets:
    """PRESETS 註冊驗證。"""

    def test_regex_match_in_presets(self) -> None:
        assert "regex_match" in PRESETS

    def test_preset_extractor_class(self) -> None:
        ext_cls, _ = PRESETS["regex_match"]
        assert ext_cls is RegexMatchExtractor

    def test_preset_scorer_class(self) -> None:
        _, scorer_cls = PRESETS["regex_match"]
        assert scorer_cls is StringMatchScorer

    def test_create_metric_pair(self) -> None:
        ext, scorer = create_metric_pair("regex_match")
        assert isinstance(ext, RegexMatchExtractor)
        assert isinstance(scorer, StringMatchScorer)

    def test_create_metric_pair_with_config(self) -> None:
        ext, scorer = create_metric_pair(
            "regex_match",
            {"answer_pattern": r"result: (.*)", "normalize_mode": "lower"},
        )
        assert ext.patterns == [r"result: (.*)"]
        assert scorer._mode == "lower"


# ──────────────────────────────────────────────────────────────────────
# Example Dataset 驗證
# ──────────────────────────────────────────────────────────────────────

EXAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "datasets",
    "example",
    "bbh",
)
EXAMPLE_FILE = os.path.join(EXAMPLE_DIR, "bbh.jsonl")


class TestBBHExampleDataset:
    """BBH example dataset 格式驗證。"""

    def test_file_exists(self) -> None:
        assert os.path.isfile(EXAMPLE_FILE), f"Example file not found: {EXAMPLE_FILE}"

    def test_jsonl_valid(self) -> None:
        with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    pytest.fail(f"Line {i} is not valid JSON")

    def test_required_fields(self) -> None:
        required = {"id", "question", "answer", "subtask"}
        with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                missing = required - set(row.keys())
                assert not missing, f"Line {i} missing fields: {missing}"

    def test_sample_count(self) -> None:
        with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
        assert count == 15, f"Expected 15 samples, got {count}"

    def test_mc_samples_present(self) -> None:
        """MC 子任務存在且 answer 格式為 (X)。"""
        mc_subtasks = set()
        with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                if row["answer"].startswith("(") and row["answer"].endswith(")"):
                    mc_subtasks.add(row["subtask"])
        assert len(mc_subtasks) >= 5, f"Expected ≥5 MC subtasks, got {mc_subtasks}"

    def test_binary_samples_present(self) -> None:
        """Binary 子任務存在。"""
        binary_answers = {"Yes", "No", "True", "False", "valid", "invalid"}
        binary_count = 0
        with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                if row["answer"] in binary_answers:
                    binary_count += 1
        assert binary_count >= 3, f"Expected ≥3 binary samples, got {binary_count}"

    def test_freeform_samples_present(self) -> None:
        """Free-form 子任務存在（數字或空格分隔字串）。"""
        freeform_subtasks = {
            "multistep_arithmetic_two",
            "object_counting",
            "word_sorting",
            "dyck_languages",
        }
        found = set()
        with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                if row["subtask"] in freeform_subtasks:
                    found.add(row["subtask"])
        assert len(found) >= 3, f"Expected ≥3 free-form subtasks, got {found}"

    def test_all_three_types_covered(self) -> None:
        """確認三種類型都有。"""
        binary_answers = {"Yes", "No", "True", "False", "valid", "invalid"}
        has_mc = False
        has_binary = False
        has_freeform = False

        freeform_subtasks = {
            "multistep_arithmetic_two",
            "object_counting",
            "word_sorting",
            "dyck_languages",
        }

        with open(EXAMPLE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                ans = row["answer"]
                if ans.startswith("(") and ans.endswith(")"):
                    has_mc = True
                elif ans in binary_answers:
                    has_binary = True
                elif row["subtask"] in freeform_subtasks:
                    has_freeform = True

        assert has_mc, "No MC samples found"
        assert has_binary, "No binary samples found"
        assert has_freeform, "No free-form samples found"


# ──────────────────────────────────────────────────────────────────────
# 端對端整合測試（不呼叫 API）
# ──────────────────────────────────────────────────────────────────────


class TestRegexMatchIntegration:
    """Extractor + Scorer 整合測試。"""

    @pytest.fixture
    def pipeline(self):
        ext, scorer = create_metric_pair("regex_match")
        return ext, scorer

    def test_bbh_mc_correct(self, pipeline) -> None:
        ext, scorer = pipeline
        output = "Let me think... the answer is (B)."
        predicted = ext.extract(output)
        gold = "(B)"
        norm_pred = scorer.normalize(predicted)
        norm_gold = scorer.normalize(gold)
        assert scorer.score(norm_pred, norm_gold) is True

    def test_bbh_mc_wrong(self, pipeline) -> None:
        ext, scorer = pipeline
        output = "the answer is (A)."
        predicted = ext.extract(output)
        gold = "(B)"
        norm_pred = scorer.normalize(predicted)
        norm_gold = scorer.normalize(gold)
        assert scorer.score(norm_pred, norm_gold) is False

    def test_bbh_binary_correct(self, pipeline) -> None:
        ext, scorer = pipeline
        output = "the answer is No."
        predicted = ext.extract(output)
        gold = "No"
        norm_pred = scorer.normalize(predicted)
        norm_gold = scorer.normalize(gold)
        assert scorer.score(norm_pred, norm_gold) is True

    def test_bbh_integer_correct(self, pipeline) -> None:
        ext, scorer = pipeline
        output = "the answer is 24."
        predicted = ext.extract(output)
        gold = "24"
        norm_pred = scorer.normalize(predicted)
        norm_gold = scorer.normalize(gold)
        assert scorer.score(norm_pred, norm_gold) is True

    def test_bbh_freeform_correct(self, pipeline) -> None:
        ext, scorer = pipeline
        output = "the answer is syndrome therefrom."
        predicted = ext.extract(output)
        gold = "syndrome therefrom"
        norm_pred = scorer.normalize(predicted)
        norm_gold = scorer.normalize(gold)
        assert scorer.score(norm_pred, norm_gold) is True

    def test_extraction_failure_returns_none(self, pipeline) -> None:
        ext, scorer = pipeline
        output = "I'm not sure about this one."
        predicted = ext.extract(output)
        assert predicted is None
