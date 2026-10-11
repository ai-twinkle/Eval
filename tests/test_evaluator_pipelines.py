"""Evaluator 層級的 mock-LLM 測試。

以假的 LLM 驅動 evaluate_file()，涵蓋：
- logprobs / vision / ifeval 三條評測路徑的端到端行為
- API 失敗的計分方式（部分失敗計為答錯、全部失敗拋例外、logprobs 單一選項失敗）
- --resume 回填（samples_per_question > 1、IFEval 附加指標）
- 結果檔名不受 RANK 環境變數 / distributed 設定影響
- scorer introspection 延遲到 IFEval 路徑才進行
- shuffle_options 只作用於選擇題路徑
"""

import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from twinkle_eval.core.exceptions import EvaluationError
from twinkle_eval.main import _resume_backfill_fields
from twinkle_eval.metrics import create_metric_pair
from twinkle_eval.runners.evaluator import Evaluator

# 1x1 PNG
_PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
)


def _config(**evaluation: Any) -> Dict[str, Any]:
    return {
        "llm_api": {"api_rate_limit": -1},
        "model": {"name": "test-model"},
        "evaluation": {"evaluation_method": "box", **evaluation},
    }


def _completion(*contents: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=c, reasoning=None, reasoning_content=None),
                finish_reason="stop",
            )
            for c in contents
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12),
    )


def _write_jsonl(path: Any, rows: List[Dict[str, Any]]) -> str:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return str(path)


def _mcq_rows(n: int) -> List[Dict[str, Any]]:
    return [
        {"question": f"Q{i}", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A"}
        for i in range(n)
    ]


def _evaluator(llm: Any, method: str, config: Optional[Dict[str, Any]] = None, **kw: Any):
    extractor, scorer = create_metric_pair(method, {})
    return Evaluator(
        llm=llm,
        extractor=extractor,
        scorer=scorer,
        config=config or _config(),
        eval_method=method,
        **kw,
    )


def _read_rows(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ── API 失敗的計分 ───────────────────────────────────────────────────────────


class TestFailureAccounting:
    def test_partial_failure_counts_as_incorrect(self, tmp_path):
        """部分題目失敗時，失敗題計入分母（答錯），不得把分數灌水。"""
        data = _write_jsonl(tmp_path / "d.jsonl", _mcq_rows(4))
        llm = MagicMock()

        def call(question_text, *args, **kwargs):
            if question_text.startswith(("Q0", "Q1")):
                raise TimeoutError("boom")
            return _completion("\\boxed{A}")

        llm.call.side_effect = call
        _, metrics, path = _evaluator(llm, "box").evaluate_file(data, "ts_run0")

        assert metrics["accuracy"] == pytest.approx(0.5)  # 不是 100%
        assert metrics["failed_count"] == 2
        assert metrics["total_count"] == 4
        # 失敗題不寫入 JSONL，--resume 時才會重新評測
        assert sorted(r["question_id"] for r in _read_rows(path)) == [2, 3]

    def test_partial_failure_with_multiple_samples(self, tmp_path):
        data = _write_jsonl(tmp_path / "d.jsonl", _mcq_rows(2))
        llm = MagicMock()

        def call(question_text, *args, **kwargs):
            if question_text.startswith("Q0"):
                raise TimeoutError("boom")
            return _completion("\\boxed{A}", "\\boxed{A}", "\\boxed{A}")

        llm.call.side_effect = call
        _, metrics, _ = _evaluator(llm, "box", samples_per_question=3).evaluate_file(
            data, "ts_run0"
        )

        # 失敗題的 3 個 sample 全數計入分母
        assert metrics["total_count"] == 6
        assert metrics["failed_count"] == 3
        assert metrics["accuracy"] == pytest.approx(0.5)

    def test_all_failed_raises(self, tmp_path):
        """全部失敗不得回報 accuracy=0，否則會被當成 0% 計入 repeat_runs 平均。"""
        data = _write_jsonl(tmp_path / "d.jsonl", _mcq_rows(2))
        llm = MagicMock()
        llm.call.side_effect = TimeoutError("down")

        with pytest.raises(EvaluationError):
            _evaluator(llm, "box").evaluate_file(data, "ts_run0")


# ── logprobs 路徑 ────────────────────────────────────────────────────────────


class TestLogprobsPipeline:
    def test_picks_highest_likelihood_option(self, tmp_path):
        data = _write_jsonl(
            tmp_path / "d.jsonl",
            [{"question": "Q0", "A": "a", "B": "b", "C": "c", "answer": "B", "subject": "x"}],
        )
        llm = MagicMock()
        scores = {" A": -3.0, " B": -0.5, " C": -2.0}
        llm.score_continuation.side_effect = lambda ctx, cont: scores[cont]

        _, metrics, path = _evaluator(llm, "logit").evaluate_file(data, "ts_run0")

        assert metrics["accuracy"] == 1.0
        row = _read_rows(path)[0]
        assert row["predicted_answer"] == "B"
        assert row["logprob_scores"] == {"A": -3.0, "B": -0.5, "C": -2.0}
        # metadata 欄位不得進入 prompt
        context = llm.score_continuation.call_args_list[0].args[0]
        assert "subject" not in context
        assert context.endswith("\nAnswer:")

    _ROWS = [
        {"question": "Q0", "A": "a", "B": "b", "answer": "B"},
        {"question": "Q1", "A": "a", "B": "b", "answer": "A"},
    ]

    @pytest.mark.parametrize(
        "failure",
        [float("-inf"), TimeoutError("boom")],
        ids=["returns-neg-inf", "raises"],
    )
    def test_single_option_failure_fails_question(self, tmp_path, failure):
        """任一選項失敗即整題失敗，不得只在倖存選項中挑答案。

        OpenAIModel.score_continuation 出錯時回傳 -inf 而非拋例外，兩種都要視為失敗。
        """
        data = _write_jsonl(tmp_path / "d.jsonl", self._ROWS)
        llm = MagicMock()

        def score(ctx, cont):
            if ctx.startswith("Q0") and cont == " A":
                if isinstance(failure, Exception):
                    raise failure
                return failure
            return {" A": -0.1, " B": -5.0}[cont]

        llm.score_continuation.side_effect = score
        _, metrics, path = _evaluator(llm, "logit").evaluate_file(data, "ts_run0")

        assert metrics["failed_count"] == 1
        assert metrics["total_count"] == 2
        assert metrics["accuracy"] == pytest.approx(0.5)
        # 失敗題不寫入 JSONL（--resume 會重跑）
        assert [r["question_id"] for r in _read_rows(path)] == [1]

    def test_unsupported_backend_surfaces_not_implemented(self, tmp_path):
        """後端不支援 score_continuation 是設定錯誤，不得被當成 API 失敗吞掉。"""
        data = _write_jsonl(tmp_path / "d.jsonl", self._ROWS)
        llm = MagicMock()
        llm.score_continuation.side_effect = NotImplementedError("no logit support")

        with pytest.raises(NotImplementedError):
            _evaluator(llm, "logit").evaluate_file(data, "ts_run0")

    def test_dead_endpoint_raises(self, tmp_path):
        """所有選項都是 -inf（端點掛掉或不支援 echo）時不得產生分數。"""
        data = _write_jsonl(tmp_path / "d.jsonl", self._ROWS)
        llm = MagicMock()
        llm.score_continuation.return_value = float("-inf")

        with pytest.raises(EvaluationError):
            _evaluator(llm, "logit").evaluate_file(data, "ts_run0")


# ── vision 路徑 ──────────────────────────────────────────────────────────────


class TestVisionPipeline:
    def test_sends_image_options_and_system_prompt(self, tmp_path):
        image = tmp_path / "img.png"
        image.write_bytes(_PNG_BYTES)
        data = _write_jsonl(
            tmp_path / "d.jsonl",
            [
                {
                    "question": "What is shown?",
                    "A": "cat",
                    "B": "dog",
                    "answer": "B",
                    "image_path": str(image),
                    "category": "animals",
                }
            ],
        )
        llm = MagicMock()
        llm.call.return_value = _completion("Answer: B")
        config = _config(evaluation_method="vision_mcq", system_prompt={"zh": "看圖作答"})

        _, metrics, path = _evaluator(llm, "vision_mcq", config).evaluate_file(data, "ts_run0")

        assert metrics["accuracy"] == 1.0
        messages = llm.call.call_args.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "看圖作答"}
        image_part, text_part = messages[1]["content"]
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        assert "A: cat" in text_part["text"] and "B: dog" in text_part["text"]
        assert "animals" not in text_part["text"]
        assert _read_rows(path)[0]["image_path"] == str(image)

    def test_shuffle_keeps_image_field(self, tmp_path):
        """shuffle_options 不得丟棄 image_path（否則 0 題被評測）。"""
        image = tmp_path / "img.png"
        image.write_bytes(_PNG_BYTES)
        data = _write_jsonl(
            tmp_path / "d.jsonl",
            [
                {
                    "question": "Q",
                    "A": "a",
                    "B": "b",
                    "C": "c",
                    "answer": "A",
                    "image_path": str(image),
                }
                for _ in range(3)
            ],
        )
        llm = MagicMock()
        llm.call.return_value = _completion("Answer: A")

        _, metrics, _ = _evaluator(
            llm, "vision_mcq", _config(evaluation_method="vision_mcq"), shuffle_options=True
        ).evaluate_file(data, "ts_run0")

        assert metrics["total_count"] == 3
        assert llm.call.call_count == 3


# ── IFEval 路徑 ──────────────────────────────────────────────────────────────


_IFEVAL_ROWS = [
    {
        "question": "Write without commas.",
        "instruction_id_list": json.dumps(["punctuation:no_comma"]),
        "kwargs": json.dumps([{}]),
    },
    {
        "question": "Write in all lowercase.",
        "instruction_id_list": json.dumps(["change_case:english_lowercase"]),
        "kwargs": json.dumps([{}]),
    },
]


class TestIFEvalPipeline:
    def _llm(self) -> MagicMock:
        llm = MagicMock()
        responses = {
            "Write without commas.": "no commas here",
            "Write in all lowercase.": "This Is Not Lowercase",
        }
        llm.call.side_effect = lambda question_text, *a, **k: _completion(responses[question_text])
        return llm

    def test_metrics(self, tmp_path):
        data = _write_jsonl(tmp_path / "d.jsonl", _IFEVAL_ROWS)
        llm = self._llm()

        _, metrics, path = _evaluator(llm, "ifeval").evaluate_file(data, "ts_run0")

        assert metrics["prompt_strict"] == pytest.approx(0.5)
        assert metrics["instruction_strict"] == pytest.approx(0.5)
        rows = sorted(_read_rows(path), key=lambda r: r["question_id"])
        assert [r["prompt_strict"] for r in rows] == [True, False]
        # IFEval 不送 system prompt
        assert llm.call.call_args.args[3] is False

    def test_failure_counts_in_instruction_metrics(self, tmp_path):
        """失敗題的 instruction 也要計入分母，instruction-level 指標不得被灌水。"""
        data = _write_jsonl(tmp_path / "d.jsonl", _IFEVAL_ROWS)
        llm = MagicMock()

        def call(question_text, *a, **k):
            if question_text == "Write in all lowercase.":
                raise TimeoutError("boom")
            return _completion("no commas here")

        llm.call.side_effect = call
        _, metrics, _ = _evaluator(llm, "ifeval").evaluate_file(data, "ts_run0")

        assert metrics["failed_count"] == 1
        assert metrics["prompt_strict"] == pytest.approx(0.5)
        assert metrics["prompt_loose"] == pytest.approx(0.5)
        assert metrics["instruction_strict"] == pytest.approx(0.5)
        assert metrics["instruction_loose"] == pytest.approx(0.5)

    def test_resume_backfill_matches_full_run(self, tmp_path):
        """resume 後的 prompt_loose / instruction_* 須與一次跑完相同。"""
        data = _write_jsonl(tmp_path / "d.jsonl", _IFEVAL_ROWS)
        _, full, path = _evaluator(self._llm(), "ifeval").evaluate_file(data, "full_run0")
        first = next(r for r in _read_rows(path) if r["question_id"] == 0)

        llm = self._llm()
        _, resumed, _ = _evaluator(llm, "ifeval").evaluate_file(
            data, "resume_run0", completed_ids={"0": [_resume_backfill_fields(first)]}
        )

        assert llm.call.call_count == 1
        for key in ("accuracy", "prompt_loose", "instruction_strict", "instruction_loose"):
            assert resumed[key] == pytest.approx(full[key]), key
        assert resumed["total_count"] == full["total_count"]

    def test_scorer_introspection_is_lazy(self):
        """非 IFEval 路徑在建構期不得對 scorer.score_full 做 introspection。"""
        extractor, scorer = create_metric_pair("box", {})
        scorer.score_full = max  # builtin：inspect.signature 會拋 ValueError

        Evaluator(llm=MagicMock(), extractor=extractor, scorer=scorer, config=_config())


# ── ASR 路徑 ────────────────────────────────────────────────────────────────


class TestASRPipeline:
    def test_failure_counts_as_full_error_rate(self, tmp_path):
        """失敗題視為空轉錄（WER / CER = 1.0），平均值不得只算成功的題目。"""
        pytest.importorskip("jiwer")
        data = _write_jsonl(
            tmp_path / "d.jsonl",
            [
                {"audio_path": "https://example.com/a.wav", "answer": "hello world"},
                {"audio_path": "https://example.com/b.wav", "answer": "good morning"},
            ],
        )
        llm = MagicMock()

        def call(*args, **kwargs):
            messages = args[7] if len(args) > 7 else kwargs.get("messages")
            if "b.wav" in json.dumps(messages):
                raise TimeoutError("boom")
            return _completion("hello world")

        llm.call.side_effect = call
        _, metrics, _ = _evaluator(llm, "asr").evaluate_file(data, "ts_run0")

        assert metrics["failed_count"] == 1
        assert metrics["avg_wer"] == pytest.approx(0.5)
        assert metrics["avg_cer"] == pytest.approx(0.5)


# ── --resume 回填 ────────────────────────────────────────────────────────────


class TestResumeBackfill:
    def test_multiple_samples_backfill_matches_full_run(self, tmp_path):
        """samples_per_question > 1 時，回填須涵蓋每個 sample，accuracy 與 pass@k 才會正確。"""
        data = _write_jsonl(tmp_path / "d.jsonl", _mcq_rows(3))
        answers = {
            "Q0": ("\\boxed{A}", "\\boxed{B}"),
            "Q1": ("\\boxed{B}", "\\boxed{C}"),
            "Q2": ("\\boxed{A}", "\\boxed{A}"),
        }

        def make_llm() -> MagicMock:
            llm = MagicMock()
            llm.call.side_effect = lambda q, *a, **k: _completion(*answers[q.split("\n")[0]])
            return llm

        kw = {"samples_per_question": 2, "pass_k": 2}
        _, full, path = _evaluator(make_llm(), "box", **kw).evaluate_file(data, "full_run0")

        completed: Dict[str, List[Dict[str, Any]]] = {}
        for row in _read_rows(path):
            if row["question_id"] in (0, 1):
                completed.setdefault(str(row["question_id"]), []).append(
                    _resume_backfill_fields(row)
                )

        llm = make_llm()
        _, resumed, _ = _evaluator(llm, "box", **kw).evaluate_file(
            data, "resume_run0", completed_ids=completed
        )

        assert llm.call.call_count == 1
        assert resumed["total_count"] == full["total_count"] == 6
        assert resumed["accuracy"] == pytest.approx(full["accuracy"])
        assert resumed["pass_at_k"] == pytest.approx(full["pass_at_k"])

    def test_backfill_keeps_unparsed_count(self, tmp_path):
        data = _write_jsonl(tmp_path / "d.jsonl", _mcq_rows(2))
        llm = MagicMock()
        llm.call.return_value = _completion("\\boxed{A}")

        _, metrics, _ = _evaluator(llm, "box").evaluate_file(
            data,
            "ts_run0",
            completed_ids={"0": [_resume_backfill_fields({"is_correct": False})]},
        )

        assert metrics["unparsed_count"] == 1
        assert metrics["total_count"] == 2


class TestResumeLoader:
    def test_handle_resume_keeps_every_sample(self, tmp_path, monkeypatch):
        from twinkle_eval import main as main_module

        os.makedirs("results")
        rows = [
            {
                "file": "d.jsonl",
                "question_id": 0,
                "sample_id": 0,
                "is_correct": True,
                "predicted_answer": "A",
            },
            {
                "file": "d.jsonl",
                "question_id": 0,
                "sample_id": 1,
                "is_correct": False,
                "predicted_answer": None,
            },
            {"question_id": 1, "is_correct": True, "predicted_answer": "A"},  # 舊格式：無 file
        ]
        _write_jsonl(tmp_path / "results" / "eval_results_20260101_000000_run0.jsonl", rows)

        captured: Dict[str, Any] = {}
        runner = MagicMock()
        runner.run_evaluation.side_effect = lambda fmts, completed_records: captured.update(
            completed_records
        )
        monkeypatch.setattr(main_module, "TwinkleEvalRunner", lambda path: runner)

        assert main_module._handle_resume("config.yaml", "20260101_000000", ["json"]) == 0
        assert captured == {
            "eval_results_20260101_000000_run0.jsonl": {
                "d.jsonl|0": [
                    {"is_correct": True, "unparsed": False},
                    {"is_correct": False, "unparsed": True},
                ]
            }
        }


# ── 結果檔名與 shuffle 範圍 ──────────────────────────────────────────────────


class TestOutputAndShuffleScope:
    def test_rank_env_does_not_change_filename(self, tmp_path, monkeypatch):
        """RANK 環境變數帶入的 distributed.rank 不得改變 JSONL 檔名（否則 --resume 對不上）。"""
        monkeypatch.setenv("RANK", "1")
        monkeypatch.delenv("SLURM_NODEID", raising=False)
        data = _write_jsonl(tmp_path / "d.jsonl", _mcq_rows(1))
        llm = MagicMock()
        llm.call.return_value = _completion("\\boxed{A}")
        config = _config()
        config["distributed"] = {"rank": 1, "world_size": 2}

        _, _, path = _evaluator(llm, "box", config).evaluate_file(data, "ts_run0")

        assert os.path.basename(path) == "eval_results_ts_run0.jsonl"

    def test_shuffle_not_applied_outside_mcq_pipelines(self, tmp_path):
        data = _write_jsonl(tmp_path / "d.jsonl", _IFEVAL_ROWS[:1])
        llm = MagicMock()
        llm.call.return_value = _completion("no commas")
        evaluator = _evaluator(llm, "ifeval", shuffle_options=True)
        evaluator.shuffle_question_options = MagicMock(side_effect=AssertionError)

        evaluator.evaluate_file(data, "ts_run0")

        evaluator.shuffle_question_options.assert_not_called()
