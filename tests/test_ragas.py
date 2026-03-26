"""Tests for RAGAS (Retrieval-Augmented Generation Assessment) evaluation."""

import json
import os

import pytest

from twinkle_eval.metrics.extractors.ragas import (
    RAGAS_METRICS,
    RAGASExtractor,
    _extract_json,
)
from twinkle_eval.metrics.scorers.ragas import RAGASScorer


# ── JSON extraction helpers ──────────────────────────────────────────────────


class TestExtractJson:
    def test_pure_json(self):
        text = '{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6}'
        result = _extract_json(text)
        assert result is not None
        assert result["faithfulness"] == 0.9

    def test_json_in_code_block(self):
        text = '```json\n{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6}\n```'
        result = _extract_json(text)
        assert result is not None
        assert result["answer_relevancy"] == 0.8

    def test_json_with_surrounding_text(self):
        text = 'Here is my evaluation:\n{"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5, "context_recall": 0.5}\nDone.'
        result = _extract_json(text)
        assert result is not None
        assert result["context_precision"] == 0.5

    def test_empty_string(self):
        assert _extract_json("") is None

    def test_none(self):
        assert _extract_json(None) is None

    def test_invalid_json(self):
        assert _extract_json("this is not json at all") is None

    def test_malformed_json(self):
        assert _extract_json("{faithfulness: 0.9}") is None


# ── Extractor ─────────────────────────────────────────────────────────────────


class TestRAGASExtractor:
    def setup_method(self):
        self.extractor = RAGASExtractor({})

    def test_get_name(self):
        assert self.extractor.get_name() == "ragas"

    def test_extract_valid(self):
        raw = '{"faithfulness": 0.9, "answer_relevancy": 0.85, "context_precision": 0.7, "context_recall": 0.8}'
        result = self.extractor.extract(raw)
        assert result is not None
        assert len(result) == 4
        assert result["faithfulness"] == 0.9
        assert result["answer_relevancy"] == 0.85

    def test_extract_none(self):
        assert self.extractor.extract(None) is None

    def test_extract_missing_metric(self):
        raw = '{"faithfulness": 0.9, "answer_relevancy": 0.8}'
        result = self.extractor.extract(raw)
        assert result is None  # Missing context_precision and context_recall

    def test_extract_clamps_values(self):
        raw = '{"faithfulness": 1.5, "answer_relevancy": -0.2, "context_precision": 0.5, "context_recall": 0.5}'
        result = self.extractor.extract(raw)
        assert result is not None
        assert result["faithfulness"] == 1.0
        assert result["answer_relevancy"] == 0.0

    def test_extract_from_code_block(self):
        raw = '```json\n{"faithfulness": 0.9, "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6}\n```'
        result = self.extractor.extract(raw)
        assert result is not None
        assert result["context_recall"] == 0.6

    def test_extract_invalid_type(self):
        raw = '{"faithfulness": "high", "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6}'
        result = self.extractor.extract(raw)
        assert result is None  # "high" can't be converted to float

    def test_extract_non_json(self):
        raw = "I cannot evaluate this response."
        result = self.extractor.extract(raw)
        assert result is None


# ── Scorer ────────────────────────────────────────────────────────────────────


class TestRAGASScorer:
    def test_get_name(self):
        scorer = RAGASScorer({})
        assert scorer.get_name() == "ragas"

    def test_default_threshold(self):
        scorer = RAGASScorer({})
        assert scorer.threshold == 0.5

    def test_custom_threshold(self):
        scorer = RAGASScorer({"ragas_threshold": 0.8})
        assert scorer.threshold == 0.8

    def test_score_above_threshold(self):
        scorer = RAGASScorer({})
        predicted = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        assert scorer.score(predicted, "any_gold") is True  # avg = 0.75

    def test_score_below_threshold(self):
        scorer = RAGASScorer({"ragas_threshold": 0.8})
        predicted = {
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        assert scorer.score(predicted, "any_gold") is False  # avg = 0.75 < 0.8

    def test_score_none_predicted(self):
        scorer = RAGASScorer({})
        assert scorer.score(None, "gold") is False

    def test_score_not_dict(self):
        scorer = RAGASScorer({})
        assert scorer.score("invalid", "gold") is False

    def test_score_missing_metric(self):
        scorer = RAGASScorer({})
        predicted = {"faithfulness": 0.9, "answer_relevancy": 0.8}
        assert scorer.score(predicted, "gold") is False

    def test_score_all_zeros(self):
        scorer = RAGASScorer({})
        predicted = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }
        assert scorer.score(predicted, "gold") is False  # avg = 0.0 < 0.5

    def test_score_all_ones(self):
        scorer = RAGASScorer({})
        predicted = {
            "faithfulness": 1.0,
            "answer_relevancy": 1.0,
            "context_precision": 1.0,
            "context_recall": 1.0,
        }
        assert scorer.score(predicted, "gold") is True

    def test_score_exact_threshold(self):
        scorer = RAGASScorer({"ragas_threshold": 0.5})
        predicted = {
            "faithfulness": 0.5,
            "answer_relevancy": 0.5,
            "context_precision": 0.5,
            "context_recall": 0.5,
        }
        assert scorer.score(predicted, "gold") is True  # avg = 0.5 >= 0.5

    def test_normalize_passthrough(self):
        scorer = RAGASScorer({})
        assert scorer.normalize("anything") == "anything"
        assert scorer.normalize(None) is None


# ── Metrics constants ─────────────────────────────────────────────────────────


class TestRAGASMetrics:
    def test_four_metrics(self):
        assert len(RAGAS_METRICS) == 4

    def test_metric_names(self):
        assert "faithfulness" in RAGAS_METRICS
        assert "answer_relevancy" in RAGAS_METRICS
        assert "context_precision" in RAGAS_METRICS
        assert "context_recall" in RAGAS_METRICS


# ── Preset registration ──────────────────────────────────────────────────────


class TestRAGASPreset:
    def test_registered_in_presets(self):
        from twinkle_eval.metrics import PRESETS
        assert "ragas" in PRESETS

    def test_create_metric_pair(self):
        from twinkle_eval.metrics import create_metric_pair
        extractor, scorer = create_metric_pair("ragas")
        assert extractor.get_name() == "ragas"
        assert scorer.get_name() == "ragas"


# ── Example dataset ──────────────────────────────────────────────────────────


EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "datasets", "example", "ragas", "wikieval.jsonl"
)


class TestExampleDataset:
    @pytest.mark.skipif(
        not os.path.exists(EXAMPLE_PATH),
        reason="RAGAS example dataset not found",
    )
    def test_loadable(self):
        with open(EXAMPLE_PATH, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert len(rows) == 10

    @pytest.mark.skipif(
        not os.path.exists(EXAMPLE_PATH),
        reason="RAGAS example dataset not found",
    )
    def test_required_fields(self):
        with open(EXAMPLE_PATH, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        for row in rows:
            assert "id" in row
            assert "question" in row
            assert "answer" in row
            assert "answer_type" in row

    @pytest.mark.skipif(
        not os.path.exists(EXAMPLE_PATH),
        reason="RAGAS example dataset not found",
    )
    def test_answer_types_distribution(self):
        with open(EXAMPLE_PATH, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        types = [r["answer_type"] for r in rows]
        assert types.count("good") == 5
        assert types.count("ungrounded") == 3
        assert types.count("poor") == 2

    @pytest.mark.skipif(
        not os.path.exists(EXAMPLE_PATH),
        reason="RAGAS example dataset not found",
    )
    def test_answer_metadata_parseable(self):
        with open(EXAMPLE_PATH, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        for row in rows:
            metadata = json.loads(row["answer"])
            assert "user_input" in metadata
            assert "response" in metadata
            assert "reference" in metadata
            assert "retrieved_contexts" in metadata
            assert "answer_type" in metadata

    @pytest.mark.skipif(
        not os.path.exists(EXAMPLE_PATH),
        reason="RAGAS example dataset not found",
    )
    def test_judge_prompt_contains_criteria(self):
        with open(EXAMPLE_PATH, encoding="utf-8") as f:
            row = json.loads(f.readline())
        question = row["question"]
        assert "Faithfulness" in question
        assert "Answer Relevancy" in question
        assert "Context Precision" in question
        assert "Context Recall" in question
