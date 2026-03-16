"""Tests for LogitEvaluationStrategy and score_continuation path."""
from unittest.mock import MagicMock, patch

import pytest

from twinkle_eval.evaluation_strategies import LogitEvaluationStrategy


# ── Strategy unit tests ──────────────────────────────────────────────────────

class TestLogitEvaluationStrategy:
    def setup_method(self):
        self.strategy = LogitEvaluationStrategy.__new__(LogitEvaluationStrategy)
        self.strategy.config = {}

    def test_uses_logprobs_flag(self):
        assert self.strategy.uses_logprobs is True

    def test_strategy_name(self):
        assert self.strategy.get_strategy_name() == "logit"

    def test_extract_answer_text_returns_none(self):
        """logit 策略不走文字解析路徑。"""
        assert self.strategy.extract_answer("A") is None

    def test_normalize_answer(self):
        assert self.strategy.normalize_answer("a") == "A"
        assert self.strategy.normalize_answer(" B ") == "B"

    def test_is_correct(self):
        assert self.strategy.is_correct("A", "A") is True
        assert self.strategy.is_correct("A", "B") is False


# ── score_continuation unit tests ────────────────────────────────────────────

class TestScoreContinuation:
    """測試 OpenAIModel.score_continuation 的邏輯。"""

    def _make_model(self):
        from twinkle_eval.models import OpenAIModel
        model = OpenAIModel.__new__(OpenAIModel)
        model.config = {
            "llm_api": {"api_key": "test", "base_url": "http://localhost", "disable_ssl_verify": False,
                        "max_retries": 1, "timeout": 10},
            "model": {"name": "test-model", "temperature": 0.0, "top_p": 1.0,
                      "max_tokens": 1, "frequency_penalty": 0.0, "presence_penalty": 0.0,
                      "extra_body": {}},
            "evaluation": {"evaluation_method": "logit"},
        }
        return model

    def test_score_continuation_returns_last_continuation_logprob(self):
        model = self._make_model()

        # Mock: context="Q\nAnswer:", continuation=" A"
        # tokens: ["Q", "\n", "Answer", ":", " A"]
        # logprobs: [None, -0.5, -0.3, -0.1, -0.8]  ← last is continuation
        mock_response = MagicMock()
        mock_response.choices[0].logprobs.token_logprobs = [None, -0.5, -0.3, -0.1, -0.8]
        mock_response.choices[0].logprobs.tokens = ["Q", "\n", "Answer", ":", " A"]

        mock_client = MagicMock()
        mock_client.completions.create.return_value = mock_response
        model.client = mock_client

        result = model.score_continuation("Q\nAnswer:", " A")
        assert result == pytest.approx(-0.8)

    def test_score_continuation_returns_neg_inf_on_error(self):
        model = self._make_model()
        mock_client = MagicMock()
        mock_client.completions.create.side_effect = Exception("API error")
        model.client = mock_client

        result = model.score_continuation("Q\nAnswer:", " A")
        assert result == float("-inf")

    def test_score_continuation_sums_multi_token_continuation(self):
        """multi-token continuation 應加總所有 continuation token 的 logprob。"""
        model = self._make_model()

        # context="Q:", continuation=" Paris" (2 tokens: " Par", "is")
        mock_response = MagicMock()
        mock_response.choices[0].logprobs.token_logprobs = [None, -0.2, -1.5, -0.9]
        mock_response.choices[0].logprobs.tokens = ["Q", ":", " Par", "is"]

        mock_client = MagicMock()
        mock_client.completions.create.return_value = mock_response
        model.client = mock_client

        result = model.score_continuation("Q:", " Paris")
        assert result == pytest.approx(-1.5 + -0.9)


# ── Integration: N parallel calls per question ───────────────────────────────

class TestLogitEvaluatorParallelCalls:
    """驗證 logit 模式下確實為每個選項各提交一個 future。"""

    def test_n_parallel_calls_per_question(self, tmp_path):
        import json
        from twinkle_eval.evaluators import Evaluator

        # 建立 1 題 ABCD 選擇題
        jsonl = tmp_path / "test.jsonl"
        jsonl.write_text(json.dumps({
            "question": "What is 1+1?",
            "A": "1", "B": "2", "C": "3", "D": "4",
            "answer": "B"
        }) + "\n")

        strategy = LogitEvaluationStrategy.__new__(LogitEvaluationStrategy)
        strategy.config = {}

        call_log = []

        mock_llm = MagicMock()
        mock_llm.score_continuation.side_effect = lambda ctx, cont: (
            call_log.append(cont) or (-0.5 if cont.strip() == "B" else -2.0)
        )

        evaluator = Evaluator(
            llm=mock_llm,
            evaluation_strategy=strategy,
            config={"llm_api": {"api_rate_limit": -1}, "evaluation": {"evaluation_method": "logit"}},
        )

        _, metrics, _ = evaluator.evaluate_file(str(jsonl), "20250101_0000")

        # 應對 A, B, C, D 各呼叫一次 score_continuation
        assert sorted(c.strip() for c in call_log) == ["A", "B", "C", "D"]
        # B 的 logprob 最高，應預測正確
        assert metrics["accuracy"] == pytest.approx(1.0)
