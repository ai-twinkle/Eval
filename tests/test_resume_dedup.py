"""--resume 題目層級去重（evaluator.completed_ids）的單元測試。

驗證：
1. completed_ids 中的題目會被跳過，不重複呼叫 API、不重複寫入 JSONL
2. JSONL 逐題結果包含 file 欄位（resume 比對依據）
"""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from twinkle_eval.metrics import create_metric_pair
from twinkle_eval.runners.evaluator import Evaluator


def _make_config() -> dict:
    return {
        "llm_api": {"api_rate_limit": -1},
        "model": {"name": "test-model"},
        "evaluation": {"evaluation_method": "box"},
    }


def _fake_completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning=None, reasoning_content=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


@pytest.fixture
def dataset_file(tmp_path):
    path = tmp_path / "mini.jsonl"
    rows = [
        {"question": f"Q{i}", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A"}
        for i in range(3)
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(path)


def _run(dataset_file, tmp_path, monkeypatch, completed_ids):
    monkeypatch.chdir(tmp_path)
    llm = MagicMock()
    llm.call.return_value = _fake_completion("\\boxed{A}")
    extractor, scorer = create_metric_pair("box", {})

    evaluator = Evaluator(
        llm=llm,
        extractor=extractor,
        scorer=scorer,
        config=_make_config(),
        eval_method="box",
    )
    _, metrics, results_path = evaluator.evaluate_file(
        dataset_file, "20260101_000000_run0", completed_ids=completed_ids
    )
    with open(results_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return llm, metrics, rows


class TestResumeDedup:
    def test_completed_dict_skips_and_backfills(self, dataset_file, tmp_path, monkeypatch):
        llm, metrics, rows = _run(
            dataset_file, tmp_path, monkeypatch, completed_ids={"0": True, "2": False}
        )

        # 只有 question_id=1 應被實際評測與寫入
        assert llm.call.call_count == 1
        assert [r["question_id"] for r in rows] == [1]
        # 但統計要涵蓋整個資料集（已完成的 0/2 回填進 accuracy）
        assert metrics["total_count"] == 3
        assert metrics["accuracy"] == pytest.approx(2 / 3)

    def test_completed_set_skips_without_backfill(self, dataset_file, tmp_path, monkeypatch):
        # set 形式僅跳過、不回填統計
        llm, metrics, rows = _run(dataset_file, tmp_path, monkeypatch, completed_ids={"0", "2"})

        assert llm.call.call_count == 1
        assert metrics["total_count"] == 1
        assert [r["question_id"] for r in rows] == [1]

    def test_no_completed_ids_runs_all(self, dataset_file, tmp_path, monkeypatch):
        llm, metrics, rows = _run(dataset_file, tmp_path, monkeypatch, completed_ids=None)

        assert llm.call.call_count == 3
        assert metrics["total_count"] == 3
        assert sorted(r["question_id"] for r in rows) == [0, 1, 2]

    def test_rows_contain_file_field(self, dataset_file, tmp_path, monkeypatch):
        _, _, rows = _run(dataset_file, tmp_path, monkeypatch, completed_ids=None)

        # file 欄位是 --resume 比對「已完成題目」的依據，必須存在
        assert all(r["file"] == dataset_file for r in rows)

    def test_all_completed_writes_nothing_but_keeps_stats(
        self, dataset_file, tmp_path, monkeypatch
    ):
        llm, metrics, rows = _run(
            dataset_file, tmp_path, monkeypatch, completed_ids={"0": True, "1": True, "2": False}
        )

        # 不再呼叫 API、不重寫 JSONL，但 summary 統計完整
        assert llm.call.call_count == 0
        assert rows == []
        assert metrics["total_count"] == 3
        assert metrics["accuracy"] == pytest.approx(2 / 3)
