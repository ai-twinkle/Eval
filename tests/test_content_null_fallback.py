"""
tests/test_content_null_fallback.py

當 LLM 回應的 message.content 為 null 時（例如 ACE-1 模型在
skip_special_tokens=true 情況下），evaluators.py 必須 fallback 至
reasoning_content 進行答案提取，而非讓 extract_answer(None) 永遠回傳 None。

已知觸發情境：
  ACE-1 / Ace1-24B-NVFP4 透過 vLLM + LiteLLM，skip_special_tokens=true（預設）時：
  - message.content = null
  - message.reasoning_content = "推理過程...\n\nB"（答案在尾端）

防禦情境：
  content 與 reasoning_content 皆為 null 時，不得 raise 未預期的 AttributeError
  或 TypeError，應記錄 log_error 並將 predicted_answer 標記為 None（計為答錯）。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion(content, reasoning_content):
    """建立假的 ChatCompletion，content / reasoning_content 可獨立設定"""
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
    )
    usage = SimpleNamespace(
        completion_tokens=10,
        prompt_tokens=50,
        total_tokens=60,
    )
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _make_evaluator():
    from twinkle_eval.evaluators import Evaluator
    from twinkle_eval.evaluation_strategies import PatternMatchingStrategy

    mock_llm = MagicMock()
    config = {
        "llm_api": {"api_rate_limit": -1},
        "evaluation": {"shuffle_options": False},
    }
    return Evaluator(
        llm=mock_llm,
        evaluation_strategy=PatternMatchingStrategy(),
        config=config,
    )


# ---------------------------------------------------------------------------
# 測試案例
# ---------------------------------------------------------------------------

class TestContentNullFallback:
    """content=null 時應 fallback 至 reasoning_content"""

    def _run_single_question(self, evaluator, completion, tmp_path):
        """讓 evaluator 處理單題，回傳 predicted_answer"""
        import os
        from unittest.mock import patch

        evaluator.llm.call.return_value = completion

        jsonl_path = str(tmp_path / "eval_results_test_run0.jsonl")
        original_join = os.path.join

        def patched_join(*args):
            if len(args) == 2 and args[0] == "results" and "eval_results" in args[1]:
                return jsonl_path
            return original_join(*args)

        # 建立一個只有一題的臨時 JSONL 資料集
        dataset_path = str(tmp_path / "single.jsonl")
        import json
        with open(dataset_path, "w") as f:
            f.write(json.dumps({
                "question": "台灣的首都是哪裡？",
                "A": "台中", "B": "台北", "C": "高雄", "D": "台南",
                "answer": "B"
            }) + "\n")

        with patch("twinkle_eval.evaluators.os.makedirs"), \
             patch("twinkle_eval.evaluators.os.path.join", side_effect=patched_join):
            _, accuracy, _ = evaluator.evaluate_file(dataset_path, "test_run0")

        # 從 JSONL 讀取 predicted_answer
        with open(jsonl_path) as f:
            result = json.loads(f.readline())
        return result["predicted_answer"], result["is_correct"]

    def test_content_has_answer_normal_case(self, tmp_path):
        """正常情況：content 有答案 → 直接提取"""
        evaluator = _make_evaluator()
        completion = _make_completion(
            content="正確答案是 (B)",
            reasoning_content="推理過程..."
        )
        predicted, is_correct = self._run_single_question(evaluator, completion, tmp_path)
        assert predicted == "B"
        assert is_correct is True

    def test_content_null_falls_back_to_reasoning_content(self, tmp_path):
        """
        ACE-1 情境：content=null，reasoning_content 中有答案（符合 pattern）→
        應 fallback 至 reasoning_content 並成功提取
        """
        evaluator = _make_evaluator()
        completion = _make_completion(
            content=None,
            reasoning_content="根據台灣現行法律，台北市是首都。\n答案：B"
        )
        predicted, is_correct = self._run_single_question(evaluator, completion, tmp_path)
        assert predicted == "B", (
            f"content=null 時應 fallback 到 reasoning_content 提取答案，"
            f"但 predicted={predicted}"
        )
        assert is_correct is True

    def test_content_empty_string_falls_back(self, tmp_path):
        """content 為空字串時也應 fallback"""
        evaluator = _make_evaluator()
        completion = _make_completion(
            content="",
            reasoning_content="推理過程...\n答案：B"
        )
        predicted, is_correct = self._run_single_question(evaluator, completion, tmp_path)
        assert predicted == "B"
        assert is_correct is True

    def test_both_null_does_not_raise(self, tmp_path):
        """
        防禦情境：content 與 reasoning_content 皆為 null →
        不得 raise AttributeError / TypeError，
        predicted_answer 應為 None（計為答錯）
        """
        evaluator = _make_evaluator()
        completion = _make_completion(content=None, reasoning_content=None)

        # 不應拋出任何例外
        predicted, is_correct = self._run_single_question(evaluator, completion, tmp_path)
        assert predicted is None
        assert is_correct is False

    def test_both_null_no_crash_accuracy_zero(self, tmp_path):
        """content 與 reasoning_content 皆為 null 時，accuracy 應為 0.0，不是 exception"""
        import os, json
        from unittest.mock import patch

        evaluator = _make_evaluator()
        evaluator.llm.call.return_value = _make_completion(None, None)

        dataset_path = str(tmp_path / "single.jsonl")
        with open(dataset_path, "w") as f:
            f.write(json.dumps({
                "question": "台灣的首都是哪裡？",
                "A": "台中", "B": "台北", "C": "高雄", "D": "台南",
                "answer": "B"
            }) + "\n")

        jsonl_path = str(tmp_path / "eval_results_test2.jsonl")
        original_join = os.path.join

        def patched_join(*args):
            if len(args) == 2 and args[0] == "results" and "eval_results" in args[1]:
                return jsonl_path
            return original_join(*args)

        with patch("twinkle_eval.evaluators.os.makedirs"), \
             patch("twinkle_eval.evaluators.os.path.join", side_effect=patched_join):
            _, accuracy, _ = evaluator.evaluate_file(dataset_path, "test2")

        assert accuracy == 0.0
