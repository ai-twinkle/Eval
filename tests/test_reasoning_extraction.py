"""
tests/test_reasoning_extraction.py

統一推理輸出解析測試，涵蓋兩種常見情境：

情境 A（vLLM skip_special_tokens=true）：
  - content = null
  - vLLM < 0.13：只有 reasoning_content = "推理過程...\n答案：B"
  - vLLM 0.13.x：reasoning 與 reasoning_content 同時存在，但只有其中一個有值
  - vLLM >= 0.18：只有 reasoning = "推理過程...\n答案：B"
  → 優先使用 reasoning，否則 fallback 至 reasoning_content

情境 B（Ollama / inline think tag）：
  - content = "<think>推理過程...</think>答案：B"
  - reasoning = None
  - reasoning_content = None
  → 自動剝離 think block，從剩餘 content 提取答案

其他邊界案例：
  - 開頭 tag 被截斷（只有 </think>）
  - 多種 end tag（</think>、</reason>、</reasoning>）
  - strip 後 content 為空 → 優先 reasoning，否則 fallback reasoning_content
  - 三者皆 null → 不 crash，predicted=None
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion(content, reasoning_content=None, reasoning=None):
    message = SimpleNamespace(content=content)
    if reasoning_content is not None:
        message.reasoning_content = reasoning_content
    if reasoning is not None:
        message.reasoning = reasoning
    usage = SimpleNamespace(completion_tokens=10, prompt_tokens=50, total_tokens=60)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def _make_evaluator():
    from twinkle_eval.runners.evaluator import Evaluator
    from twinkle_eval.metrics.extractors.pattern import PatternExtractor
    from twinkle_eval.metrics.scorers.exact import ExactMatchScorer

    mock_llm = MagicMock()
    config = {
        "llm_api": {"api_rate_limit": -1},
        "evaluation": {"shuffle_options": False},
    }
    return Evaluator(llm=mock_llm, extractor=PatternExtractor(), scorer=ExactMatchScorer(), config=config)


def _run_single(evaluator, completion, tmp_path):
    """執行單題評測，回傳 (predicted_answer, is_correct)"""
    evaluator.llm.call.return_value = completion

    dataset_path = str(tmp_path / "q.jsonl")
    with open(dataset_path, "w") as f:
        f.write(json.dumps({
            "question": "台灣的首都？",
            "A": "台中", "B": "台北", "C": "高雄", "D": "台南",
            "answer": "B"
        }) + "\n")

    jsonl_path = str(tmp_path / "out.jsonl")
    original_join = os.path.join

    def patched_join(*args):
        if len(args) == 2 and args[0] == "results" and "eval_results" in args[1]:
            return jsonl_path
        return original_join(*args)

    with patch("twinkle_eval.runners.evaluator.os.makedirs"), \
         patch("twinkle_eval.runners.evaluator.os.path.join", side_effect=patched_join):
        evaluator.evaluate_file(dataset_path, "test_run0")

    with open(jsonl_path) as f:
        result = json.loads(f.readline())
    return result["predicted_answer"], result["is_correct"]


# ---------------------------------------------------------------------------
# 情境 A：content=null（新版 reasoning / 舊版 reasoning_content 相容）
# ---------------------------------------------------------------------------

class TestContentNullFallback:

    def test_null_content_uses_reasoning_content(self, tmp_path):
        evaluator = _make_evaluator()
        completion = _make_completion(None, reasoning_content="推理...\n答案：B")
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B"
        assert is_correct is True

    def test_empty_content_uses_reasoning_content(self, tmp_path):
        evaluator = _make_evaluator()
        completion = _make_completion("", reasoning_content="推理...\n答案：B")
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B"
        assert is_correct is True

    def test_null_content_uses_reasoning(self, tmp_path):
        evaluator = _make_evaluator()
        completion = _make_completion(None, reasoning="推理...\n答案：B")
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B"
        assert is_correct is True

    def test_null_content_uses_reasoning_when_both_fields_exist_but_old_is_none(self, tmp_path):
        evaluator = _make_evaluator()
        completion = _make_completion(
            None,
            reasoning="推理...\n答案：B",
            reasoning_content=None,
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B"
        assert is_correct is True

    def test_empty_reasoning_does_not_fallback_to_reasoning_content(self, tmp_path):
        evaluator = _make_evaluator()
        completion = _make_completion(
            None,
            reasoning="",
            reasoning_content="推理...\n答案：B",
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted is None
        assert is_correct is False


# ---------------------------------------------------------------------------
# 情境 B：inline think tag
# ---------------------------------------------------------------------------

class TestInlineThinkTag:

    def test_think_tag_stripped_answer_extracted(self, tmp_path):
        """<think>...</think> 後有答案 → 剝離 think block，提取答案"""
        evaluator = _make_evaluator()
        completion = _make_completion(
            content="<think>台灣現行法律規定台北為首都。</think>答案：B",
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B", f"應從 </think> 後提取答案，got: {predicted}"
        assert is_correct is True

    def test_truncated_start_tag_not_stripped(self, tmp_path):
        """開頭 <think> 被截斷，只剩 </think> → 格式不合格，原樣保留不剝離"""
        evaluator = _make_evaluator()
        # 只有結尾 tag，沒有開頭 tag → 不剝離，直接對整段 content 提取
        # 整段含 "答案：B" 仍可被 PatternMatchingStrategy 匹配
        completion = _make_completion(
            content="台灣現行法律規定台北為首都。</think>答案：B",
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        # 不剝離，但 PatternMatchingStrategy 仍可在原始 content 中找到 "答案：B"
        assert predicted == "B"
        assert is_correct is True

    def test_reason_tag_stripped(self, tmp_path):
        """</reason> tag 也應被處理"""
        evaluator = _make_evaluator()
        completion = _make_completion(
            content="<reason>推理過程</reason>答案：B",
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B"

    def test_reasoning_tag_stripped(self, tmp_path):
        """</reasoning> tag 也應被處理"""
        evaluator = _make_evaluator()
        completion = _make_completion(
            content="<reasoning>推理過程</reasoning>答案：B",
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B"

    def test_no_think_tag_unaffected(self, tmp_path):
        """沒有 think tag 的正常輸出不受影響"""
        evaluator = _make_evaluator()
        completion = _make_completion(content="答案：B")
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B"
        assert is_correct is True

    def test_think_tag_only_uses_reasoning(self, tmp_path):
        """think block 佔滿整個 content，剝離後為空 → 應優先使用 reasoning"""
        evaluator = _make_evaluator()
        completion = _make_completion(
            content="<think>推理...</think>",
            reasoning="答案：B",
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B", f"剝離後 content 為空應優先使用 reasoning，got: {predicted}"
        assert is_correct is True

    def test_think_tag_only_uses_reasoning_when_both_fields_exist_but_old_is_none(self, tmp_path):
        """vLLM 0.13.x：think block 剝空後，兩個屬性都存在但只有 reasoning 有值"""
        evaluator = _make_evaluator()
        completion = _make_completion(
            content="<think>推理...</think>",
            reasoning="答案：B",
            reasoning_content=None,
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B"
        assert is_correct is True

    def test_think_tag_only_fallback_to_reasoning_content(self, tmp_path):
        """think block 佔滿整個 content，剝離後為空且 reasoning=None → fallback reasoning_content"""
        evaluator = _make_evaluator()
        completion = _make_completion(
            content="<think>推理...</think>",
            reasoning=None,
            reasoning_content="答案：B",
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted == "B", f"剝離後 content 為空應 fallback，got: {predicted}"
        assert is_correct is True

    def test_think_tag_only_no_reasoning_content_returns_none(self, tmp_path):
        """think block 佔滿 content 且 reasoning/reasoning_content 都沒有 → predicted=None，不 crash"""
        evaluator = _make_evaluator()
        completion = _make_completion(
            content="<think>推理...</think>",
            reasoning=None,
            reasoning_content=None,
        )
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted is None
        assert is_correct is False


# ---------------------------------------------------------------------------
# 防禦：三者皆 null
# ---------------------------------------------------------------------------

class TestBothNull:

    def test_both_null_no_crash(self, tmp_path):
        evaluator = _make_evaluator()
        completion = _make_completion(None, None)
        predicted, is_correct = _run_single(evaluator, completion, tmp_path)
        assert predicted is None
        assert is_correct is False
