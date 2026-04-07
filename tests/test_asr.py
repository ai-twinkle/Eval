"""Tests for ASR (Automatic Speech Recognition) evaluation method."""

import pytest

from twinkle_eval.metrics import PRESETS, create_metric_pair
from twinkle_eval.metrics.extractors.asr import ASRExtractor
from twinkle_eval.metrics.scorers.asr import ASRScorer, _tokenize_mixed, _is_cjk_char


# ---------------------------------------------------------------------------
# ASRExtractor
# ---------------------------------------------------------------------------


class TestASRExtractor:
    def setup_method(self) -> None:
        self.extractor = ASRExtractor()

    def test_get_name(self) -> None:
        assert self.extractor.get_name() == "asr"

    def test_uses_audio_flag(self) -> None:
        assert self.extractor.uses_audio is True

    def test_uses_logprobs_flag(self) -> None:
        assert self.extractor.uses_logprobs is False

    def test_extract_passthrough(self) -> None:
        text = "這是一段測試語音"
        assert self.extractor.extract(text) == text

    def test_extract_strips_whitespace(self) -> None:
        assert self.extractor.extract("  hello world  ") == "hello world"

    def test_extract_empty_string(self) -> None:
        assert self.extractor.extract("") is None

    def test_extract_whitespace_only(self) -> None:
        assert self.extractor.extract("   ") is None

    def test_extract_none(self) -> None:
        assert self.extractor.extract(None) is None

    def test_extract_chinese(self) -> None:
        assert self.extractor.extract("今天天氣真好") == "今天天氣真好"

    def test_extract_mixed(self) -> None:
        assert self.extractor.extract("Hello 你好 World") == "Hello 你好 World"


# ---------------------------------------------------------------------------
# ASRScorer — normalize
# ---------------------------------------------------------------------------


class TestASRScorerNormalize:
    def setup_method(self) -> None:
        self.scorer = ASRScorer()

    def test_lowercase(self) -> None:
        assert self.scorer.normalize("Hello World") == "hello world"

    def test_remove_punctuation(self) -> None:
        assert self.scorer.normalize("你好，世界！") == "你好世界"

    def test_nfkc_fullwidth(self) -> None:
        # 全形 ABC → 半形 abc（NFKC + lowercase）
        assert self.scorer.normalize("ＡＢＣ") == "abc"

    def test_collapse_whitespace(self) -> None:
        assert self.scorer.normalize("a   b   c") == "a b c"

    def test_mixed_normalization(self) -> None:
        result = self.scorer.normalize("Hello, 你好！World。")
        assert result == "hello 你好world"

    def test_empty_string(self) -> None:
        assert self.scorer.normalize("") == ""

    def test_only_punctuation(self) -> None:
        assert self.scorer.normalize("，。！？") == ""

    def test_no_normalization(self) -> None:
        scorer = ASRScorer({
            "normalize_unicode": False,
            "remove_punctuation": False,
            "to_lower": False,
        })
        assert scorer.normalize("Hello, World!") == "Hello, World!"


# ---------------------------------------------------------------------------
# ASRScorer — score (exact match)
# ---------------------------------------------------------------------------


class TestASRScorerScore:
    def setup_method(self) -> None:
        self.scorer = ASRScorer()

    def test_exact_match(self) -> None:
        assert self.scorer.score("hello world", "hello world") is True

    def test_mismatch(self) -> None:
        assert self.scorer.score("hello world", "hello") is False

    def test_empty_both(self) -> None:
        assert self.scorer.score("", "") is True

    def test_empty_predicted(self) -> None:
        assert self.scorer.score("", "hello") is False

    def test_empty_gold(self) -> None:
        assert self.scorer.score("hello", "") is False


# ---------------------------------------------------------------------------
# ASRScorer — metric_name
# ---------------------------------------------------------------------------


class TestASRScorerMetricName:
    def test_default_zh_uses_cer(self) -> None:
        scorer = ASRScorer({"asr_language": "zh"})
        assert scorer.metric_name == "cer"

    def test_japanese_uses_cer(self) -> None:
        scorer = ASRScorer({"asr_language": "ja"})
        assert scorer.metric_name == "cer"

    def test_korean_uses_cer(self) -> None:
        scorer = ASRScorer({"asr_language": "ko"})
        assert scorer.metric_name == "cer"

    def test_english_uses_wer(self) -> None:
        scorer = ASRScorer({"asr_language": "en"})
        assert scorer.metric_name == "wer"

    def test_french_uses_wer(self) -> None:
        scorer = ASRScorer({"asr_language": "fr"})
        assert scorer.metric_name == "wer"

    def test_override_cer(self) -> None:
        scorer = ASRScorer({"asr_language": "en", "asr_metric": "cer"})
        assert scorer.metric_name == "cer"

    def test_override_wer(self) -> None:
        scorer = ASRScorer({"asr_language": "zh", "asr_metric": "wer"})
        assert scorer.metric_name == "wer"


# ---------------------------------------------------------------------------
# ASRScorer — score_full (requires jiwer)
# ---------------------------------------------------------------------------


class TestASRScorerScoreFull:
    @pytest.fixture(autouse=True)
    def _check_jiwer(self) -> None:
        pytest.importorskip("jiwer", reason="jiwer not installed")

    def setup_method(self) -> None:
        self.scorer_en = ASRScorer({"asr_language": "en"})
        self.scorer_zh = ASRScorer({"asr_language": "zh"})

    def test_perfect_match_en(self) -> None:
        result = self.scorer_en.score_full("Hello World", "Hello World")
        assert result["is_correct"] is True
        assert result["wer"] == 0.0
        assert result["cer"] == 0.0
        assert result["metric"] == "wer"

    def test_perfect_match_zh(self) -> None:
        result = self.scorer_zh.score_full("今天天氣真好", "今天天氣真好")
        assert result["is_correct"] is True
        assert result["cer"] == 0.0
        assert result["metric"] == "cer"

    def test_partial_match_en(self) -> None:
        result = self.scorer_en.score_full("Hello World", "Hello")
        assert result["is_correct"] is False
        assert result["wer"] > 0.0

    def test_partial_match_zh(self) -> None:
        result = self.scorer_zh.score_full("今天天氣", "今天天氣真好")
        assert result["is_correct"] is False
        assert result["cer"] > 0.0

    def test_empty_prediction(self) -> None:
        result = self.scorer_en.score_full("", "hello world")
        assert result["is_correct"] is False
        assert result["wer"] == 1.0

    def test_empty_both(self) -> None:
        result = self.scorer_en.score_full("", "")
        assert result["is_correct"] is True
        assert result["wer"] == 0.0

    def test_result_structure(self) -> None:
        result = self.scorer_en.score_full("hello", "hello")
        assert "is_correct" in result
        assert "wer" in result
        assert "cer" in result
        assert "metric" in result
        assert "metric_value" in result
        assert "predicted_normalized" in result
        assert "gold_normalized" in result

    def test_normalization_in_score_full(self) -> None:
        result = self.scorer_en.score_full("HELLO, WORLD!", "hello world")
        assert result["is_correct"] is True
        assert result["wer"] == 0.0

    def test_zh_punctuation_normalization(self) -> None:
        result = self.scorer_zh.score_full("你好，世界！", "你好世界")
        assert result["is_correct"] is True
        assert result["cer"] == 0.0


# ---------------------------------------------------------------------------
# _tokenize_mixed
# ---------------------------------------------------------------------------


class TestTokenizeMixed:
    def test_chinese_only(self) -> None:
        assert _tokenize_mixed("你好世界") == ["你", "好", "世", "界"]

    def test_english_only(self) -> None:
        assert _tokenize_mixed("hello world") == ["hello", "world"]

    def test_mixed(self) -> None:
        tokens = _tokenize_mixed("Hello 你好 World")
        assert tokens == ["Hello", "你", "好", "World"]

    def test_empty(self) -> None:
        assert _tokenize_mixed("") == []

    def test_whitespace(self) -> None:
        assert _tokenize_mixed("   ") == []


# ---------------------------------------------------------------------------
# _is_cjk_char
# ---------------------------------------------------------------------------


class TestIsCJKChar:
    def test_chinese_char(self) -> None:
        assert _is_cjk_char(ord("你")) is True

    def test_english_char(self) -> None:
        assert _is_cjk_char(ord("A")) is False

    def test_digit(self) -> None:
        assert _is_cjk_char(ord("1")) is False

    def test_cjk_punctuation(self) -> None:
        # 。is in CJK Symbols and Punctuation range
        assert _is_cjk_char(ord("。")) is True


# ---------------------------------------------------------------------------
# PRESETS 註冊
# ---------------------------------------------------------------------------


class TestASRPreset:
    def test_preset_registered(self) -> None:
        assert "asr" in PRESETS

    def test_preset_classes(self) -> None:
        extractor_cls, scorer_cls = PRESETS["asr"]
        assert extractor_cls is ASRExtractor
        assert scorer_cls is ASRScorer

    def test_create_metric_pair(self) -> None:
        extractor, scorer = create_metric_pair("asr")
        assert isinstance(extractor, ASRExtractor)
        assert isinstance(scorer, ASRScorer)

    def test_create_metric_pair_with_config(self) -> None:
        extractor, scorer = create_metric_pair("asr", {"asr_language": "en"})
        assert isinstance(scorer, ASRScorer)
        assert scorer.metric_name == "wer"


# ---------------------------------------------------------------------------
# WhisperModel 註冊
# ---------------------------------------------------------------------------


class TestWhisperModelRegistration:
    def test_factory_has_whisper(self) -> None:
        from twinkle_eval.models import LLMFactory
        assert "whisper" in LLMFactory._registry

    def test_factory_whisper_class(self) -> None:
        from twinkle_eval.models import LLMFactory, WhisperModel
        assert LLMFactory._registry["whisper"] is WhisperModel


# ---------------------------------------------------------------------------
# Benchmark Registry — ASR entries
# ---------------------------------------------------------------------------


class TestASRBenchmarkRegistry:
    def setup_method(self) -> None:
        from twinkle_eval.benchmarks import BENCHMARK_REGISTRY
        self.registry = BENCHMARK_REGISTRY

    def test_librispeech_exists(self) -> None:
        assert "librispeech" in self.registry

    def test_aishell1_exists(self) -> None:
        assert "aishell1" in self.registry

    def test_fleurs_exists(self) -> None:
        assert "fleurs" in self.registry

    def test_common_voice_exists(self) -> None:
        assert "common_voice" in self.registry

    def test_asr_benchmarks_use_asr_method(self) -> None:
        asr_benchmarks = ["librispeech", "aishell1", "fleurs", "common_voice"]
        for name in asr_benchmarks:
            assert self.registry[name]["eval_method"] == "asr", (
                f"{name} should use asr eval method"
            )

    def test_asr_benchmarks_are_huggingface(self) -> None:
        asr_benchmarks = ["librispeech", "aishell1", "fleurs", "common_voice"]
        for name in asr_benchmarks:
            assert self.registry[name]["source"] == "huggingface", (
                f"{name} should be huggingface source"
            )


# ---------------------------------------------------------------------------
# Template 檔案存在
# ---------------------------------------------------------------------------


class TestASRTemplate:
    def test_template_exists(self) -> None:
        import os
        template_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "twinkle_eval", "templates", "asr.yaml",
        )
        assert os.path.isfile(template_path), f"ASR template not found at {template_path}"
