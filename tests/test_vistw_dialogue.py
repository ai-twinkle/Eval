"""VisTW-Dialogue 兩階段評測的測試（#151）。

階段 1（vistw_dialogue）走既有 vision 路徑生成自由回答；
階段 2（vistw_judge）走既有文字路徑，由 judge 給 0–10 分。
兩階段都不需要 evaluator 新增路徑。
"""

import json
from pathlib import Path

import pytest

from twinkle_eval.metrics import PRESETS, create_metric_pair
from twinkle_eval.metrics.scorers.vistw_judge import VisTWJudgeScorer

EXAMPLE = Path(__file__).resolve().parent.parent / "datasets" / "example" / "vistw_dialogue"
DATASET = EXAMPLE / "test.jsonl"


class TestPresets:
    def test_both_stages_registered(self):
        assert "vistw_dialogue" in PRESETS
        assert "vistw_judge" in PRESETS

    def test_stage1_uses_vision_stage2_does_not(self):
        gen_ex, _ = create_metric_pair("vistw_dialogue", {})
        judge_ex, _ = create_metric_pair("vistw_judge", {})
        assert getattr(gen_ex, "uses_vision", False) is True
        assert getattr(judge_ex, "uses_vision", False) is False

    def test_no_new_evaluator_flags_introduced(self):
        """設計前提：兩階段都走既有路徑，不得引入新的 uses_* flag。"""
        known = {
            "uses_logprobs",
            "uses_tool_calls",
            "uses_prompt_injection",
            "uses_ifeval",
            "uses_audio",
            "uses_vision",
        }
        for name in ("vistw_dialogue", "vistw_judge"):
            ex, _ = create_metric_pair(name, {})
            flags = {a for a in dir(ex) if a.startswith("uses_") and getattr(ex, a)}
            assert flags <= known, f"{name} 引入了新 flag: {flags - known}"


class TestJudgeScoreParsing:
    @pytest.fixture
    def scorer(self):
        return VisTWJudgeScorer()

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("評語……\n[評分]: 8", 8.0),
            ("[評分]：10", 10.0),
            ("[評分]: 0", 0.0),
            ("[評分] : 7", 7.0),
            ("[評分]: 9/10", 9.0),
        ],
    )
    def test_parses_valid_scores(self, scorer, response, expected):
        assert scorer.parse_score(response) == expected

    @pytest.mark.parametrize(
        "response",
        ["完全沒有分數格式", "[評分]: 15", "[評分]: 99", "", "分數是 8 分但沒用規定格式"],
    )
    def test_unparseable_returns_none_never_a_default(self, response):
        """judge-based 評分最危險的失效是格式不符時默默給中間值。

        解析失敗**必須**回傳 None，讓該題計入 unparsed，而不是給 0 或 5。
        """
        assert VisTWJudgeScorer().parse_score(response) is None

    def test_score_full_reports_parse_failure(self, scorer):
        assert scorer.score_full("沒有分數", "")["judge_parsed"] is False
        assert scorer.score_full("沒有分數", "")["judge_score"] is None

    def test_pass_threshold_configurable(self):
        strict = VisTWJudgeScorer({"vistw_judge_pass_threshold": 9.0})
        assert strict.score("[評分]: 8", "") is False
        assert strict.score("[評分]: 9", "") is True


class TestStage1Scorer:
    def test_accuracy_means_response_produced_not_correctness(self):
        """階段 1 的 accuracy 是回應產生率，不與 ground_truth 比對。"""
        _, scorer = create_metric_pair("vistw_dialogue", {})
        assert scorer.score("任何非空回答", "完全不同的參考答案") is True
        assert scorer.score("", "參考答案") is False
        assert scorer.score("   ", "參考答案") is False


@pytest.mark.skipif(not DATASET.exists(), reason="example 資料集尚未建立")
class TestExampleDataset:
    def test_fields(self):
        with open(DATASET, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) >= 10
        for r in rows:
            assert {"id", "image_path", "question", "ground_truth"} <= set(r)
            assert (EXAMPLE.parent.parent.parent / r["image_path"]).exists()
            assert "vistw_dialogue" in r["image_path"]

    def test_ids_unique(self):
        with open(DATASET, encoding="utf-8") as f:
            ids = [json.loads(line)["id"] for line in f if line.strip()]
        assert len(set(ids)) == len(ids)


class TestJudgePromptBuilder:
    def test_prompt_requires_the_parsed_format(self):
        """腳本產生的提示詞必須要求 [評分]: N，否則 scorer 永遠解析失敗。"""
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        from build_vistw_judge_dataset import JUDGE_PROMPT

        assert "[評分]: N" in JUDGE_PROMPT
        filled = JUDGE_PROMPT.format(question="Q", response="R", ground_truth="G")
        assert "Q" in filled and "R" in filled and "G" in filled


class TestScoreFullReachesTextPath:
    """回歸：score_full() 先前只有 uses_ifeval / uses_audio 兩條路徑會呼叫。

    VisTW-Dialogue 真正的指標是 0–10 平均分，若文字路徑不呼叫 score_full()，
    那個指標永遠不會出現在 metrics 或 JSONL 裡。
    """

    def test_avg_judge_score_in_metrics(self, tmp_path, monkeypatch):
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from twinkle_eval.runners.evaluator import Evaluator

        ds = tmp_path / "judge.jsonl"
        ds.write_text(
            "\n".join(
                json.dumps({"id": str(i), "question": "judge prompt", "answer": ""})
                for i in range(4)
            ),
            encoding="utf-8",
        )

        scores = iter(["[評分]: 8", "[評分]: 6", "[評分]: 10", "[評分]: 4"])
        llm = MagicMock()
        llm.call.side_effect = lambda *a, **k: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=next(scores), reasoning=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(completion_tokens=1, prompt_tokens=1, total_tokens=2),
        )

        ex, sc = create_metric_pair("vistw_judge", {})
        ev = Evaluator(
            llm=llm,
            extractor=ex,
            scorer=sc,
            config={"llm_api": {"api_rate_limit": -1}, "evaluation": {}},
            eval_method="vistw_judge",
        )
        monkeypatch.chdir(tmp_path)
        _, metrics, path = ev.evaluate_file(str(ds), "t")

        assert metrics["avg_judge_score"] == 7.0
        rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        assert sorted(r["judge_score"] for r in rows) == [4.0, 6.0, 8.0, 10.0]
