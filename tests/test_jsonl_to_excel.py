"""--convert-to-excel（convert_jsonl_to_excel）的單元測試。"""

import json

import pytest

pytest.importorskip("openpyxl")

from twinkle_eval.main import convert_jsonl_to_excel


@pytest.fixture
def jsonl_file(tmp_path):
    path = tmp_path / "eval_results_20260101_000000_run0.jsonl"
    rows = [
        {
            "file": "datasets/example/mmlu/test.jsonl",
            "question_id": 0,
            "sample_id": 0,
            "question": "1 + 1 = ?",
            "correct_answer": "A",
            "predicted_answer": "A",
            "is_correct": True,
            "llm_output": "答案是 \\boxed{A}",
            "llm_reasoning_output": None,
            "usage_prompt_tokens": 10,
            "usage_completion_tokens": 5,
            "usage_total_tokens": 15,
        },
        {
            "file": "datasets/example/mmlu/test.jsonl",
            "question_id": 1,
            "sample_id": 0,
            "question": "2 + 2 = ?",
            "correct_answer": "B",
            "predicted_answer": "C",
            "is_correct": False,
            "llm_output": "\\boxed{C}",
            "llm_reasoning_output": "推理過程",
            # 額外欄位（如 logit 路徑的 logprob_scores）應被序列化保留
            "logprob_scores": {"A": -1.5, "B": None},
            "usage_prompt_tokens": None,
            "usage_completion_tokens": None,
            "usage_total_tokens": None,
        },
    ]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return path


class TestConvertJsonlToExcel:
    def test_basic_conversion(self, jsonl_file):
        import pandas as pd

        assert convert_jsonl_to_excel(str(jsonl_file)) == 0

        output = jsonl_file.with_suffix(".xlsx")
        assert output.exists()

        df = pd.read_excel(output)
        assert len(df) == 2
        # 常用欄位在前
        assert list(df.columns[:3]) == ["file", "question_id", "sample_id"]
        assert bool(df.iloc[0]["is_correct"]) is True
        assert df.iloc[1]["predicted_answer"] == "C"
        # dict 欄位以 JSON 字串保留
        assert json.loads(df.iloc[1]["logprob_scores"]) == {"A": -1.5, "B": None}

    def test_missing_file_returns_1(self, tmp_path):
        assert convert_jsonl_to_excel(str(tmp_path / "nope.jsonl")) == 1

    def test_empty_file_returns_1(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        assert convert_jsonl_to_excel(str(empty)) == 1

    def test_oversized_and_illegal_chars_sanitized(self, tmp_path):
        import pandas as pd

        path = tmp_path / "big.jsonl"
        row = {
            "file": "x.jsonl",
            "question_id": 0,
            "sample_id": 0,
            "question": "q",
            "correct_answer": "A",
            "predicted_answer": "A",
            "is_correct": True,
            # 超過 Excel 32767 字元上限 + openpyxl 不允許的控制字元
            "llm_output": ("長" * 40000) + "\x00\x08",
        }
        path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")

        assert convert_jsonl_to_excel(str(path)) == 0
        df = pd.read_excel(path.with_suffix(".xlsx"))
        cell = df.iloc[0]["llm_output"]
        assert len(cell) <= 32767
        assert "\x00" not in cell
