"""Tests for NIAH (Needle in a Haystack) evaluation."""

import json
import os
import tempfile

import pytest

from twinkle_eval.metrics.extractors.niah import NIAHExtractor
from twinkle_eval.metrics.scorers.niah import (
    NIAHScorer,
    _normalize_reference_label,
    _normalize_text,
    _tokenize_chinese,
    compute_f1,
    substring_match,
)
from twinkle_eval.datasets.niah import (
    _insert_needle,
    _read_haystack,
    generate_niah_dataset,
)


# ── Extractor ─────────────────────────────────────────────────────────────────


class TestNIAHExtractor:
    def setup_method(self):
        self.extractor = NIAHExtractor({})

    def test_get_name(self):
        assert self.extractor.get_name() == "niah"

    def test_passthrough(self):
        text = "The answer is 42 billion dollars."
        assert self.extractor.extract(text) == text

    def test_passthrough_none(self):
        assert self.extractor.extract(None) is None

    def test_passthrough_empty(self):
        assert self.extractor.extract("") == ""


# ── Scorer helpers ────────────────────────────────────────────────────────────


class TestNormalizeText:
    def test_basic(self):
        assert _normalize_text("  Hello   World  ") == "hello world"

    def test_lowercase(self):
        assert _normalize_text("ABC") == "abc"

    def test_collapse_whitespace(self):
        assert _normalize_text("a   b\n\nc") == "a b c"


class TestTokenizeChinese:
    def test_chinese_chars(self):
        tokens = _tokenize_chinese("我是中文")
        assert tokens == ["我", "是", "中", "文"]

    def test_english(self):
        tokens = _tokenize_chinese("hello world")
        assert tokens == ["hello", "world"]

    def test_mixed(self):
        tokens = _tokenize_chinese("我是 test 中文")
        assert "我" in tokens
        assert "test" in tokens
        assert "中" in tokens


class TestSubstringMatch:
    def test_match(self):
        assert substring_match("The answer is 42 billion", "42 billion") is True

    def test_no_match(self):
        assert substring_match("The answer is unknown", "42 billion") is False

    def test_case_insensitive(self):
        assert substring_match("The ANSWER is 42 Billion", "42 billion") is True

    def test_empty_gold(self):
        assert substring_match("some text", "") is True


class TestComputeF1:
    def test_exact_match(self):
        assert compute_f1("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert compute_f1("hello", "world") == 0.0

    def test_partial_overlap(self):
        score = compute_f1("hello world foo", "hello world bar")
        assert 0.0 < score < 1.0

    def test_empty_gold(self):
        assert compute_f1("something", "") == 0.0

    def test_both_empty(self):
        assert compute_f1("", "") == 1.0

    def test_chinese(self):
        score = compute_f1("我是中文測試", "我是中文回答")
        # 共有: 我、是、中、文 = 4, pred=6, gold=6
        assert score > 0.0


# ── Scorer ────────────────────────────────────────────────────────────────────


class TestNIAHScorer:
    def test_get_name(self):
        scorer = NIAHScorer({})
        assert scorer.get_name() == "niah"

    def test_default_substring_mode(self):
        scorer = NIAHScorer({})
        assert scorer.scoring_mode == "substring"

    def test_substring_score_true(self):
        scorer = NIAHScorer({})
        assert scorer.score("The answer is 42 billion dollars", "42 billion") is True

    def test_substring_score_false(self):
        scorer = NIAHScorer({})
        assert scorer.score("I don't know", "42 billion") is False

    def test_exact_mode_true(self):
        scorer = NIAHScorer({"niah_scoring_mode": "exact"})
        assert scorer.score("42 billion", "42 billion") is True

    def test_exact_mode_false(self):
        scorer = NIAHScorer({"niah_scoring_mode": "exact"})
        assert scorer.score("42 billion dollars", "42 billion") is False

    def test_exact_mode_paragraph_spacing_true(self):
        scorer = NIAHScorer({"niah_scoring_mode": "exact"})
        assert scorer.score("段落 9", "段落9") is True

    def test_normalize_reference_label_paragraph_spacing(self):
        assert _normalize_reference_label("段落 27") == "段落27"
        assert _normalize_reference_label("paragraph 12") == "paragraph12"

    def test_f1_mode_above_threshold(self):
        scorer = NIAHScorer({"niah_scoring_mode": "f1", "niah_f1_threshold": 0.5})
        assert scorer.score("hello world foo", "hello world bar") is True

    def test_f1_mode_below_threshold(self):
        scorer = NIAHScorer({"niah_scoring_mode": "f1", "niah_f1_threshold": 0.99})
        assert scorer.score("hello world foo", "hello world bar") is False

    def test_score_none_predicted(self):
        scorer = NIAHScorer({})
        assert scorer.score(None, "42") is False

    def test_score_none_gold(self):
        scorer = NIAHScorer({})
        assert scorer.score("42", None) is False

    def test_normalize(self):
        scorer = NIAHScorer({})
        assert scorer.normalize("  hello  ") == "hello"
        assert scorer.normalize(None) == ""


# ── Generator: _read_haystack ─────────────────────────────────────────────────


class TestReadHaystack:
    def test_read_single_file(self, tmp_path):
        f = tmp_path / "haystack.txt"
        f.write_text("This is the haystack text.", encoding="utf-8")
        result = _read_haystack(str(f))
        assert result == "This is the haystack text."

    def test_read_directory(self, tmp_path):
        (tmp_path / "a.txt").write_text("File A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("File B", encoding="utf-8")
        result = _read_haystack(str(tmp_path))
        assert "File A" in result
        assert "File B" in result

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _read_haystack(str(tmp_path))

    def test_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            _read_haystack("/nonexistent/path/to/haystack")


# ── Generator: _insert_needle ─────────────────────────────────────────────────


class TestInsertNeedle:
    def test_needle_at_start(self):
        haystack = "A" * 1000
        result = _insert_needle(haystack, "NEEDLE", 500, 0.0)
        assert result.startswith("\nNEEDLE\n")

    def test_needle_at_end(self):
        haystack = "A" * 1000
        result = _insert_needle(haystack, "NEEDLE", 500, 1.0)
        assert result.endswith("\nNEEDLE\n")

    def test_needle_present(self):
        haystack = "A" * 1000
        result = _insert_needle(haystack, "NEEDLE", 500, 0.5)
        assert "NEEDLE" in result

    def test_short_haystack_repeats(self):
        haystack = "short"
        result = _insert_needle(haystack, "NEEDLE", 100, 0.5)
        assert "NEEDLE" in result
        assert len(result) > 100


# ── Generator: generate_niah_dataset ──────────────────────────────────────────


class TestGenerateNiahDataset:
    def test_generates_jsonl(self, tmp_path):
        # Create a simple haystack file
        haystack_file = tmp_path / "haystack.txt"
        haystack_file.write_text("A" * 500, encoding="utf-8")

        output_dir = tmp_path / "output"
        output_path = generate_niah_dataset(
            haystack_path=str(haystack_file),
            needle="The secret number is 42.",
            question="What is the secret number?",
            answer="42",
            context_lengths=[128, 256],
            needle_depths=[0, 50, 100],
            output_dir=str(output_dir),
        )

        assert os.path.exists(output_path)
        with open(output_path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        # 2 context_lengths x 3 depths = 6 rows
        assert len(rows) == 6

    def test_row_structure(self, tmp_path):
        haystack_file = tmp_path / "haystack.txt"
        haystack_file.write_text("B" * 500, encoding="utf-8")

        output_dir = tmp_path / "output"
        output_path = generate_niah_dataset(
            haystack_path=str(haystack_file),
            needle="Secret fact.",
            question="What?",
            answer="fact",
            context_lengths=[128],
            needle_depths=[50],
            output_dir=str(output_dir),
            language="zh",
        )

        with open(output_path, encoding="utf-8") as f:
            row = json.loads(f.readline())

        assert "id" in row
        assert "question" in row
        assert "answer" in row
        assert row["answer"] == "fact"
        assert row["context_length"] == 128
        assert row["needle_depth"] == 0.5
        assert row["language"] == "zh"
        assert row["source"] == "custom"

    def test_custom_prompt_template(self, tmp_path):
        haystack_file = tmp_path / "haystack.txt"
        haystack_file.write_text("C" * 500, encoding="utf-8")

        output_dir = tmp_path / "output"
        template = "Context: {context}\nQ: {question}"
        output_path = generate_niah_dataset(
            haystack_path=str(haystack_file),
            needle="Fact.",
            question="Q?",
            answer="A",
            context_lengths=[128],
            needle_depths=[50],
            output_dir=str(output_dir),
            prompt_template=template,
        )

        with open(output_path, encoding="utf-8") as f:
            row = json.loads(f.readline())

        assert row["question"].startswith("Context: ")
        assert "Q: Q?" in row["question"]


# ── Preset registration ──────────────────────────────────────────────────────


class TestNIAHPreset:
    def test_registered_in_presets(self):
        from twinkle_eval.metrics import PRESETS
        assert "niah" in PRESETS

    def test_create_metric_pair(self):
        from twinkle_eval.metrics import create_metric_pair
        extractor, scorer = create_metric_pair("niah")
        assert extractor.get_name() == "niah"
        assert scorer.get_name() == "niah"


# ── Example datasets ─────────────────────────────────────────────────────────


EXAMPLE_BASE = os.path.join(os.path.dirname(__file__), "..", "datasets", "example", "niah")


class TestExampleDatasets:
    @pytest.mark.skipif(
        not os.path.exists(os.path.join(EXAMPLE_BASE, "kamradt", "kamradt.jsonl")),
        reason="Kamradt example dataset not found",
    )
    def test_kamradt_loadable(self):
        path = os.path.join(EXAMPLE_BASE, "kamradt", "kamradt.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 10
        for row in rows:
            assert "id" in row
            assert "question" in row
            assert "answer" in row
            assert "context_length" in row
            assert "needle_depth" in row

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(EXAMPLE_BASE, "needlebench", "needlebench.jsonl")),
        reason="NeedleBench example dataset not found",
    )
    def test_needlebench_loadable(self):
        path = os.path.join(EXAMPLE_BASE, "needlebench", "needlebench.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 5
        for row in rows:
            assert "id" in row
            assert "question" in row
            assert "answer" in row

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(EXAMPLE_BASE, "longbench", "longbench_zh.jsonl")),
        reason="LongBench example dataset not found",
    )
    def test_longbench_loadable(self):
        path = os.path.join(EXAMPLE_BASE, "longbench", "longbench_zh.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 5
        for row in rows:
            assert "id" in row
            assert "question" in row
            assert "answer" in row

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(EXAMPLE_BASE, "kamradt", "kamradt.jsonl")),
        reason="Kamradt example dataset not found",
    )
    def test_kamradt_scorer_integration(self):
        """End-to-end: load a row, simulate a correct response, score it."""
        path = os.path.join(EXAMPLE_BASE, "kamradt", "kamradt.jsonl")
        with open(path, encoding="utf-8") as f:
            row = json.loads(f.readline())

        extractor = NIAHExtractor({})
        scorer = NIAHScorer({})

        # Simulate a model response that contains the gold answer
        simulated_response = f"Based on the document, {row['answer']}."
        extracted = extractor.extract(simulated_response)
        result = scorer.score(extracted, row["answer"])
        assert result is True

    @pytest.mark.skipif(
        not os.path.exists(os.path.join(EXAMPLE_BASE, "kamradt", "kamradt.jsonl")),
        reason="Kamradt example dataset not found",
    )
    def test_kamradt_scorer_wrong_answer(self):
        """End-to-end: a wrong response should score False."""
        path = os.path.join(EXAMPLE_BASE, "kamradt", "kamradt.jsonl")
        with open(path, encoding="utf-8") as f:
            row = json.loads(f.readline())

        extractor = NIAHExtractor({})
        scorer = NIAHScorer({})

        extracted = extractor.extract("I have no idea what you're talking about.")
        result = scorer.score(extracted, row["answer"])
        assert result is False
