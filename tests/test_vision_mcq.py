"""Tests for Vision MCQ (VLM 視覺多選題) evaluation method."""

from twinkle_eval.benchmarks import BENCHMARK_REGISTRY
from twinkle_eval.metrics import PRESETS, create_metric_pair
from twinkle_eval.metrics.extractors.vision_mcq import VisionMCQExtractor
from twinkle_eval.metrics.scorers.exact import ExactMatchScorer


# ---------------------------------------------------------------------------
# VisionMCQExtractor
# ---------------------------------------------------------------------------


class TestVisionMCQExtractor:
    def setup_method(self) -> None:
        self.extractor = VisionMCQExtractor()

    def test_get_name(self) -> None:
        assert self.extractor.get_name() == "vision_mcq"

    def test_uses_vision_flag(self) -> None:
        assert self.extractor.uses_vision is True

    def test_uses_logprobs_flag(self) -> None:
        # vision_mcq 走文字回應，不走 logprobs
        assert self.extractor.uses_logprobs is False

    def test_uses_audio_flag(self) -> None:
        assert getattr(self.extractor, "uses_audio", False) is False

    # ── Yes/No 提取（POPE 等 benchmark）─────────────────────────────────────
    def test_extract_yes_simple(self) -> None:
        assert self.extractor.extract("Yes, the object is present.") == "Yes"

    def test_extract_no_simple(self) -> None:
        assert self.extractor.extract("No, there is no cat in the image.") == "No"

    def test_extract_yes_with_answer_prefix(self) -> None:
        assert self.extractor.extract("Answer: Yes") == "Yes"

    def test_extract_no_with_answer_prefix(self) -> None:
        assert self.extractor.extract("The answer is no.") == "No"

    def test_extract_yes_chinese(self) -> None:
        assert self.extractor.extract("答案是：是") == "Yes"

    def test_extract_no_chinese(self) -> None:
        assert self.extractor.extract("答案為：否") == "No"

    def test_extract_case_insensitive(self) -> None:
        assert self.extractor.extract("YES, I can see it.") == "Yes"

    def test_extract_yes_at_end(self) -> None:
        assert self.extractor.extract("Looking at the image carefully... yes.") == "Yes"

    # ── 字母選項提取（MMBench / MMStar / MMMU / BLINK）──────────────────────
    def test_extract_letter_chinese(self) -> None:
        assert self.extractor.extract("答案：A") == "A"

    def test_extract_letter_in_sentence(self) -> None:
        assert self.extractor.extract("After analysis the answer is A.") == "A"

    def test_extract_letter_at_start(self) -> None:
        assert self.extractor.extract("B. The dog is sitting") == "B"

    def test_extract_letter_d(self) -> None:
        assert self.extractor.extract("答案: D") == "D"

    # ── VLM 真實輸出格式（regression cases）──────────────────────────────────
    def test_extract_bold_letter_with_colon(self) -> None:
        # gemma 等 VLM 常見輸出："The correct answer is **D: A man with...**"
        text = "The correct answer is **D: A man with a diagram of his face**."
        assert self.extractor.extract(text) == "D"

    def test_extract_bold_letter_after_newline(self) -> None:
        # "The most appropriate choice is:\n**A: engaged**"
        text = "The most appropriate choice among the options provided is:\n**A: engaged**"
        assert self.extractor.extract(text) == "A"

    def test_extract_does_not_match_random_word_capital(self) -> None:
        # 修正前：PatternExtractor 的 [A-Z]. 兜底會抓到 "It does" → "I"
        text = (
            "Based on the visual cues, the cat has a neutral expression. "
            "It does not show distress. The answer is **A**."
        )
        result = self.extractor.extract(text)
        assert result == "A", f"應抓到 A，而非 {result!r}"

    def test_extract_correct_option_is(self) -> None:
        text = "Based on the image, the correct option is:\n\n**D: The suitcase is beneath the book**"
        assert self.extractor.extract(text) == "D"

    def test_extract_parenthesized_letter(self) -> None:
        assert self.extractor.extract("After analysis, the answer is (B).") == "B"

    # ── MMStar_MINI regression cases（gemma 4 31B 實際輸出格式）─────────────
    def test_extract_correct_answer_bold_eos(self) -> None:
        # "...The mean of Data Set A is **11**.\n\nCorrect Answer: **B**"
        text = "The mean of Data Set A is **11**.\n\nCorrect Answer: **B**"
        assert self.extractor.extract(text) == "B"

    def test_extract_bold_correct_answer_full(self) -> None:
        # "...= 9.\n\n**Correct Answer: A**"
        text = "Final result = 9.\n\n**Correct Answer: A**"
        assert self.extractor.extract(text) == "A"

    def test_extract_correct_option_plain_eos(self) -> None:
        # "...= 40°.\n\nCorrect Option: C"
        text = "After computing the angles we get $\\angle CAB = 40°$.\n\nCorrect Option: C"
        assert self.extractor.extract(text) == "C"

    def test_extract_correct_option_bold_eos(self) -> None:
        # "...explanation.\n\nCorrect Option: **B**"
        text = "Looking at the chart values carefully.\n\nCorrect Option: **B**"
        assert self.extractor.extract(text) == "B"

    def test_extract_bare_letter_at_end(self) -> None:
        # "...The 3rd number is 9.\n\nB"
        text = "Counting from the left, the 3rd number is 9.\n\nB"
        assert self.extractor.extract(text) == "B"

    def test_extract_bare_letter_at_end_with_period(self) -> None:
        # "...That makes 3 groups.\n\nC."
        text = "That makes 3 groups.\n\nC."
        assert self.extractor.extract(text) == "C"

    def test_extract_bold_correct_answer_d(self) -> None:
        text = "After analyzing all options carefully.\n\n**Correct Answer: D**"
        assert self.extractor.extract(text) == "D"

    # ── Edge cases ──────────────────────────────────────────────────────────
    def test_extract_empty_string(self) -> None:
        assert self.extractor.extract("") is None

    def test_extract_whitespace_only(self) -> None:
        assert self.extractor.extract("   ") is None

    def test_extract_none(self) -> None:
        assert self.extractor.extract(None) is None

    # ── strategy_config 透過 Extractor.__init__ 傳遞 ────────────────────────
    def test_config_storage(self) -> None:
        cfg = {
            "image_field": "image_url",
            "max_image_size": 512,
            "image_detail": "low",
        }
        extractor = VisionMCQExtractor(cfg)
        assert extractor._config["image_field"] == "image_url"
        assert extractor._config["max_image_size"] == 512
        assert extractor._config["image_detail"] == "low"


# ---------------------------------------------------------------------------
# PRESETS registration
# ---------------------------------------------------------------------------


class TestVisionMCQPresets:
    def test_preset_registered(self) -> None:
        assert "vision_mcq" in PRESETS

    def test_preset_uses_correct_classes(self) -> None:
        extractor_cls, scorer_cls = PRESETS["vision_mcq"]
        assert extractor_cls is VisionMCQExtractor
        assert scorer_cls is ExactMatchScorer

    def test_create_metric_pair(self) -> None:
        extractor, scorer = create_metric_pair(
            "vision_mcq",
            {"image_field": "image_path", "image_detail": "auto"},
        )
        assert isinstance(extractor, VisionMCQExtractor)
        assert isinstance(scorer, ExactMatchScorer)
        assert extractor.uses_vision is True


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------


class TestVisionMCQBenchmarks:
    EXPECTED = ("mmbench", "mmstar", "mmmu", "pope")

    def test_all_benchmarks_registered(self) -> None:
        for name in self.EXPECTED:
            assert name in BENCHMARK_REGISTRY, f"benchmark '{name}' 未在 registry 中"

    def test_all_use_vision_mcq_method(self) -> None:
        for name in self.EXPECTED:
            entry = BENCHMARK_REGISTRY[name]
            assert entry["eval_method"] == "vision_mcq", (
                f"benchmark '{name}' 的 eval_method 應為 vision_mcq，"
                f"實際為 {entry['eval_method']}"
            )

    def test_all_have_required_fields(self) -> None:
        for name in self.EXPECTED:
            entry = BENCHMARK_REGISTRY[name]
            assert "source" in entry
            assert "description" in entry
            assert "license" in entry
            if entry["source"] == "huggingface":
                assert "hf_id" in entry


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


class TestVisionMCQExports:
    def test_export_from_metrics(self) -> None:
        from twinkle_eval.metrics import VisionMCQExtractor as Exported
        assert Exported is VisionMCQExtractor

    def test_export_from_top_level(self) -> None:
        from twinkle_eval import VisionMCQExtractor as Exported
        assert Exported is VisionMCQExtractor
