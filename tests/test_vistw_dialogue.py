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
            ("【評分】: 7", 7.0),  # 全形括號：中文模型常用
            ("[評分]: **7**", 7.0),  # judge 模仿 prompt 的粗體風格
            # 多次出現時取最後一個，與官方 split('[評分]: ')[-1] 一致。
            # judge 常先引用評分指南的範例分數，取第一個會抓到那個。
            ("範例：[評分]: 10。本回答有錯。\n[評分]: 3", 3.0),
        ],
    )
    def test_parses_valid_scores(self, scorer, response, expected):
        assert scorer.parse_score(response) == expected

    @pytest.mark.parametrize(
        "response",
        [
            "完全沒有分數格式",
            "[評分]: 15",
            "[評分]: 99",
            "",
            "分數是 8 分但沒用規定格式",
            "[評分]: 100",  # 三位數：不得被截成 10
            "[評分]: 007",  # 前導零：不得被截成 0
            "[評分]: 8.5",  # 小數：不得被截成 8
        ],
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
            assert {"id", "image_path", "question", "answer"} <= set(r)
            # 參考答案只放 answer——evaluator 會排除它。
            # 若另存成 ground_truth 等欄位，會被當成一般欄位印進 prompt，
            # 等於把正解直接餵給模型。
            assert "ground_truth" not in r
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


class TestScoreFullIsNotYetWired:
    """記錄事實：score_full() 的欄位到不了 JSONL。

    先前這裡用 inspect.getsource 數 score_full 出現次數——那是原始碼字串絆線，
    任何無關的 PR 在 evaluate_file 動一下就會在這個檔案爆掉，而且換個實作名稱
    就照樣通過。改成斷言實際寫出的 JSONL 欄位。
    """

    def test_judge_fields_absent_from_jsonl(self, tmp_path, monkeypatch):
        import json
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from twinkle_eval.runners.evaluator import Evaluator

        ds = tmp_path / "judge.jsonl"
        ds.write_text(json.dumps({"question": "judge prompt", "answer": ""}), encoding="utf-8")
        llm = MagicMock()
        llm.call.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="[評分]: 8", reasoning=None),
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

        row = json.loads(open(path, encoding="utf-8").readline())
        assert "judge_score" not in row, "若已接通，請更新 docs 與本測試"
        assert "judge_parsed" not in row
        assert "avg_judge_score" not in metrics
        # 分數仍存在於原始回應中，docs 教的解析方式依此
        assert sc.parse_score(row["predicted_answer"]) == 8.0
