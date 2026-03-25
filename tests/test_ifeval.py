"""Tests for IFEval evaluation."""

import json
import os

import pytest

from twinkle_eval.metrics.extractors.ifeval import IFEvalExtractor
from twinkle_eval.metrics.scorers.ifeval import (
    IFEvalScorer,
    _get_loose_variants,
    _remove_first_line,
    _remove_last_line,
    _remove_markdown,
    score_ifeval,
)


# ── Text transformation helpers ──────────────────────────────────────────────


class TestTextTransformations:
    def test_remove_markdown(self):
        assert _remove_markdown("Hello **world** and *test*") == "Hello world and test"

    def test_remove_markdown_no_change(self):
        assert _remove_markdown("No markdown here") == "No markdown here"

    def test_remove_first_line(self):
        assert _remove_first_line("First\nSecond\nThird") == "Second\nThird"

    def test_remove_first_line_single(self):
        assert _remove_first_line("Only one line") == ""

    def test_remove_last_line(self):
        assert _remove_last_line("First\nSecond\nThird") == "First\nSecond"

    def test_remove_last_line_single(self):
        assert _remove_last_line("Only one line") == ""

    def test_loose_variants_count(self):
        variants = _get_loose_variants("Some response")
        assert len(variants) == 8

    def test_loose_variants_includes_original(self):
        text = "Some response text"
        variants = _get_loose_variants(text)
        assert text in variants


# ── IFEvalScorer ──────────────────────────────────────────────────────────────


class TestIFEvalScorer:
    @pytest.fixture
    def scorer(self):
        return IFEvalScorer({})

    def test_get_name(self, scorer):
        assert scorer.get_name() == "ifeval"

    def test_normalize_passthrough(self, scorer):
        assert scorer.normalize("hello world") == "hello world"

    def test_score_no_comma_pass(self, scorer):
        gt = json.dumps({
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        })
        # Response with no comma
        assert scorer.score("Hello world this has no comma", gt) is True

    def test_score_no_comma_fail(self, scorer):
        gt = json.dumps({
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        })
        # Response with comma
        assert scorer.score("Hello, world", gt) is False

    def test_score_empty_response(self, scorer):
        gt = json.dumps({
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        })
        assert scorer.score("", gt) is False

    def test_score_invalid_gt(self, scorer):
        assert scorer.score("some response", "not json") is False

    def test_score_full_returns_four_metrics(self, scorer):
        result = scorer.score_full(
            "hello world no comma here",
            ["punctuation:no_comma"],
            [{}],
        )
        assert "prompt_strict" in result
        assert "prompt_loose" in result
        assert "instruction_strict" in result
        assert "instruction_loose" in result

    def test_score_full_strict_pass(self, scorer):
        result = scorer.score_full(
            "hello world",
            ["punctuation:no_comma"],
            [{}],
        )
        assert result["prompt_strict"] is True
        assert result["instruction_strict"] == [True]

    def test_score_full_strict_fail(self, scorer):
        result = scorer.score_full(
            "hello, world",
            ["punctuation:no_comma"],
            [{}],
        )
        assert result["prompt_strict"] is False
        assert result["instruction_strict"] == [False]

    def test_score_full_multiple_instructions(self, scorer):
        # No comma + at least 5 words
        result = scorer.score_full(
            "hello world foo bar baz",
            ["punctuation:no_comma", "length_constraints:number_words"],
            [{}, {"relation": "at least", "num_words": 5}],
        )
        assert result["instruction_strict"] == [True, True]
        assert result["prompt_strict"] is True

    def test_score_full_partial_pass(self, scorer):
        # No comma (pass) + at least 100 words (fail for short response)
        result = scorer.score_full(
            "hello world",
            ["punctuation:no_comma", "length_constraints:number_words"],
            [{}, {"relation": "at least", "num_words": 100}],
        )
        assert result["instruction_strict"][0] is True   # no comma
        assert result["instruction_strict"][1] is False  # word count
        assert result["prompt_strict"] is False

    def test_loose_tolerates_markdown(self, scorer):
        # Response has ** markdown but otherwise satisfies no_comma
        result = scorer.score_full(
            "**Hello** world",
            ["punctuation:no_comma"],
            [{}],
        )
        # Strict fails if ** triggers anything, but loose should pass after removal
        # Actually no_comma doesn't care about **, so both should pass
        assert result["prompt_loose"] is True


# ── IFEvalExtractor ───────────────────────────────────────────────────────────


class TestIFEvalExtractor:
    @pytest.fixture
    def extractor(self):
        return IFEvalExtractor({})

    def test_uses_ifeval_flag(self, extractor):
        assert extractor.uses_ifeval is True

    def test_extract_passthrough(self, extractor):
        assert extractor.extract("some response") == "some response"

    def test_extract_none(self, extractor):
        assert extractor.extract(None) is None

    def test_get_name(self, extractor):
        assert extractor.get_name() == "ifeval"


# ── score_ifeval function ─────────────────────────────────────────────────────


class TestScoreIfeval:
    def test_lowercase_instruction(self):
        result = score_ifeval(
            "all lowercase text here",
            ["change_case:english_lowercase"],
            [{}],
        )
        assert result["prompt_strict"] is True

    def test_lowercase_fails_uppercase(self):
        result = score_ifeval(
            "ALL UPPERCASE TEXT HERE",
            ["change_case:english_lowercase"],
            [{}],
        )
        assert result["prompt_strict"] is False

    def test_unknown_instruction_returns_false(self):
        result = score_ifeval(
            "some response",
            ["unknown:nonexistent_instruction"],
            [{}],
        )
        assert result["instruction_strict"] == [False]


# ── Example dataset ───────────────────────────────────────────────────────────


class TestExampleDataset:
    def test_example_dataset_exists(self):
        assert os.path.exists("datasets/example/ifeval/ifeval.jsonl")

    def test_example_dataset_format(self):
        with open("datasets/example/ifeval/ifeval.jsonl") as f:
            rows = [json.loads(l) for l in f if l.strip()]

        assert len(rows) >= 1
        for row in rows:
            assert "id" in row
            assert "question" in row
            assert "instruction_id_list" in row
            assert "kwargs" in row

            inst_ids = json.loads(row["instruction_id_list"])
            kwargs = json.loads(row["kwargs"])
            assert isinstance(inst_ids, list)
            assert isinstance(kwargs, list)
            assert len(inst_ids) == len(kwargs)
            assert len(inst_ids) >= 1

    def test_example_dataset_10_rows(self):
        with open("datasets/example/ifeval/ifeval.jsonl") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        assert len(rows) == 10

    def test_example_dataset_diverse_instructions(self):
        with open("datasets/example/ifeval/ifeval.jsonl") as f:
            rows = [json.loads(l) for l in f if l.strip()]

        all_ids = set()
        for row in rows:
            ids = json.loads(row["instruction_id_list"])
            all_ids.update(ids)

        # 應涵蓋多個不同的 instruction 類型
        assert len(all_ids) >= 5
