"""
tests/test_pr17_overwrite_bug.py

PR #17 回歸測試：驗證 evaluators.py 中 JSONL 寫入模式的正確性。

Bug 描述（已修正）：
  evaluate_file() 原先使用 open(..., 'w') 寫入 JSONL。
  當同一資料集目錄下有多個檔案（如 part_a.jsonl, part_b.jsonl），
  且同一 run（相同 timestamp）依序評測它們時，
  第二個檔案的結果會覆蓋第一個，造成 JSONL 詳細紀錄不完整。

  修正：改為 open(..., 'a') 模式，確保多檔結果累積。

  注意：results_{timestamp}.json 中的 accuracy 統計不受影響，
  因為準確率是在記憶體中計算後才寫入 JSON。
  受影響的只有 eval_results_{timestamp}_runN.jsonl 每題詳細紀錄。
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_completion(answer_letter: str = "A") -> SimpleNamespace:
    """建立一個假的 OpenAI ChatCompletion 物件"""
    message = SimpleNamespace(
        content=f"答案是 ({answer_letter})",
        reasoning_content=None,
    )
    usage = SimpleNamespace(
        completion_tokens=10,
        prompt_tokens=50,
        total_tokens=60,
    )
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


PART_A = "/tmp/eval_two_files/part_a.jsonl"
PART_B = "/tmp/eval_two_files/part_b.jsonl"


@pytest.fixture
def tmp_results(tmp_path):
    """提供一個臨時 results 目錄，避免污染真實 results/"""
    return str(tmp_path)


# ---------------------------------------------------------------------------
# 確認測試資料存在
# ---------------------------------------------------------------------------

def test_fixture_files_exist():
    """確認 /tmp/eval_two_files/ 的兩個 150 題檔案存在"""
    assert os.path.exists(PART_A), f"缺少測試資料: {PART_A}"
    assert os.path.exists(PART_B), f"缺少測試資料: {PART_B}"
    with open(PART_A) as f:
        assert sum(1 for _ in f) == 150, "part_a.jsonl 應有 150 行"
    with open(PART_B) as f:
        assert sum(1 for _ in f) == 150, "part_b.jsonl 應有 150 行"


# ---------------------------------------------------------------------------
# 核心行為驗證（純邏輯，不需 LLM）
# ---------------------------------------------------------------------------

class TestWriteModeSemantics:
    """
    直接模擬 evaluate_file() 的寫入行為，
    驗證 'a' 模式在多檔評測時能正確累積結果。
    """

    def _write_fake_results(self, path: str, n_questions: int, mode: str, label: str):
        """模擬 evaluate_file() 將 n_questions 筆結果寫入 JSONL"""
        with open(path, mode, encoding="utf-8") as f:
            for i in range(n_questions):
                record = {
                    "question_id": i,
                    "source_file": label,
                    "is_correct": True,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_a_mode_preserves_all_data(self, tmp_results):
        """'a' 模式：兩次寫入累積，150 + 150 = 300 行"""
        path = os.path.join(tmp_results, "eval_results_test_run0.jsonl")

        self._write_fake_results(path, 150, "a", "part_a")
        self._write_fake_results(path, 150, "a", "part_b")

        with open(path) as f:
            lines = f.readlines()

        assert len(lines) == 300, f"'a' 模式應有 300 行，實際 {len(lines)} 行"
        records = [json.loads(l) for l in lines]
        sources = {r["source_file"] for r in records}
        assert sources == {"part_a", "part_b"}, f"'a' 模式應包含兩個檔案的資料，但找到: {sources}"

    def test_w_mode_would_lose_data(self, tmp_results):
        """對照組：'w' 模式（舊行為）會導致資料遺失，只剩最後一個檔案的 150 行"""
        path = os.path.join(tmp_results, "eval_results_test_run0.jsonl")

        self._write_fake_results(path, 150, "w", "part_a")
        self._write_fake_results(path, 150, "w", "part_b")

        with open(path) as f:
            lines = f.readlines()

        # 'w' 模式：只剩 part_b
        assert len(lines) == 150
        records = [json.loads(l) for l in lines]
        assert all(r["source_file"] == "part_b" for r in records)


# ---------------------------------------------------------------------------
# 端對端驗證（使用 mock LLM）
# ---------------------------------------------------------------------------

class TestEndToEndWithMockLLM:
    """
    端對端驗證：使用 mock LLM（不呼叫真實 API）評測兩個 150 題檔案，
    確認 JSONL 輸出累積 300 行。
    """

    @pytest.mark.skipif(
        not os.path.exists(PART_A) or not os.path.exists(PART_B),
        reason="/tmp/eval_two_files/ 不存在，跳過端對端測試"
    )
    def test_two_file_eval_preserves_all_data(self, tmp_results):
        """
        'a' 模式（修正後）：評測兩個 150 題檔案後，
        JSONL 應有 300 行，兩個檔案的資料都完整保留。
        """
        from twinkle_eval.evaluators import Evaluator
        from twinkle_eval.evaluation_strategies import PatternMatchingStrategy

        mock_llm = MagicMock()
        mock_llm.call.return_value = _make_fake_completion("A")

        config = {
            "llm_api": {"api_rate_limit": -1},
            "evaluation": {"shuffle_options": False},
        }
        evaluator = Evaluator(
            llm=mock_llm,
            evaluation_strategy=PatternMatchingStrategy(),
            config=config,
        )

        timestamp = "e2e_append_run0"
        jsonl_path = os.path.join(tmp_results, f"eval_results_{timestamp}.jsonl")

        original_join = os.path.join

        def patched_join(*args):
            if len(args) == 2 and args[0] == "results" and "eval_results" in args[1]:
                return jsonl_path
            return original_join(*args)

        with patch("twinkle_eval.evaluators.os.makedirs"), \
             patch("twinkle_eval.evaluators.os.path.join", side_effect=patched_join):

            evaluator.evaluate_file(PART_A, timestamp)
            evaluator.evaluate_file(PART_B, timestamp)

        assert os.path.exists(jsonl_path), "JSONL 輸出檔案不存在"

        with open(jsonl_path) as f:
            line_count = sum(1 for _ in f)

        assert line_count == 300, (
            f"'a' 模式下 JSONL 應有 300 行（part_a 150 + part_b 150），"
            f"實際 {line_count} 行"
        )
