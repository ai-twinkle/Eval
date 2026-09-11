"""選項重排（shuffle_options）與 TwinkleEvalRunner 單一實作的測試。

對應 issue：
- #140 shuffle_question_options 硬編碼 A/B/C/D
- #141 shuffle_question_options 丟棄非選項欄位
- #142 TwinkleEvalRunner 兩份實作分歧
"""

import random

import pytest

from twinkle_eval.datasets import index_to_label
from twinkle_eval.runners.evaluator import Evaluator, detect_option_keys


def make_evaluator(**kwargs) -> Evaluator:
    """建立僅用於測試 shuffle 的 Evaluator（不接觸 LLM/資料集）。"""
    config = {"llm_api": {"api_rate_limit": -1}, "evaluation": {"evaluation_method": "box"}}
    return Evaluator(
        llm=None, extractor=None, scorer=None, config=config, shuffle_options=True, **kwargs
    )


class TestIndexToLabel:
    def test_single_letters(self):
        assert [index_to_label(i) for i in range(4)] == ["A", "B", "C", "D"]

    def test_beyond_z(self):
        assert index_to_label(25) == "Z"
        assert index_to_label(26) == "AA"
        assert index_to_label(27) == "AB"


class TestDetectOptionKeys:
    def test_four_options(self):
        q = {"question": "q", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A"}
        assert detect_option_keys(q) == ["A", "B", "C", "D"]

    def test_ten_options_mmlu_pro_style(self):
        """MMLU-Pro / SuperGPQA 有 9–10 個選項，必須全部偵測到（#140）。"""
        q = {"question": "q", "answer": "J"}
        q.update({index_to_label(i): f"opt{i}" for i in range(10)})
        assert detect_option_keys(q) == list("ABCDEFGHIJ")

    def test_gap_does_not_truncate_options(self):
        """標籤不連續時仍須保留缺口之後的選項——截斷會讓正解從題目消失。"""
        q = {"question": "q", "A": "a", "B": "b", "D": "d", "answer": "A"}
        assert detect_option_keys(q) == ["A", "B", "D"]

    def test_excludes_uppercase_metadata_key(self):
        """ID 這類大寫短欄位不得被誤判為選項。"""
        q = {"question": "q", "A": "a", "B": "b", "ID": "x-1", "answer": "A"}
        assert detect_option_keys(q) == ["A", "B"]

    def test_no_options(self):
        assert detect_option_keys({"question": "q", "answer": "yes"}) == []

    def test_non_string_keys_are_safe(self):
        assert detect_option_keys({0: "zero", "A": "a", "B": "b"}) == ["A", "B"]


class TestShuffleQuestionOptions:
    def test_preserves_all_ten_options(self):
        """回歸 #140：超過 4 個選項時，E–J 不得被丟棄。"""
        ev = make_evaluator()
        q = {"question": "q", "answer": "J"}
        q.update({index_to_label(i): f"opt{i}" for i in range(10)})

        for seed in range(20):
            random.seed(seed)
            out = ev.shuffle_question_options(q)
            assert detect_option_keys(out) == list("ABCDEFGHIJ")
            assert sorted(out[k] for k in detect_option_keys(out)) == sorted(
                f"opt{i}" for i in range(10)
            )

    def test_answer_tracks_correct_text_across_shuffles(self):
        ev = make_evaluator()
        q = {"question": "q", "answer": "J"}
        q.update({index_to_label(i): f"opt{i}" for i in range(10)})

        for seed in range(50):
            random.seed(seed)
            out = ev.shuffle_question_options(q)
            assert out[out["answer"]] == "opt9"

    def test_preserves_non_option_fields(self):
        """回歸 #141：image_path 等欄位在重排後必須保留。"""
        ev = make_evaluator()
        q = {
            "question": "q",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "answer": "B",
            "image_path": "datasets/example/vision_mcq/images/1.jpg",
            "id": 42,
            "category": "science",
        }
        for seed in range(20):
            random.seed(seed)
            out = ev.shuffle_question_options(q)
            assert out["image_path"] == "datasets/example/vision_mcq/images/1.jpg"
            assert out["id"] == 42
            assert out["category"] == "science"
            assert out[out["answer"]] == "b"

    def test_duplicate_option_texts_map_to_original_key(self):
        """兩個選項文字相同時，正解須依原始鍵而非文字比對決定。"""
        ev = make_evaluator()
        q = {"question": "q", "A": "same", "B": "same", "C": "x", "D": "y", "answer": "C"}
        for seed in range(30):
            random.seed(seed)
            out = ev.shuffle_question_options(q)
            assert out[out["answer"]] == "x"

    def test_actually_permutes(self):
        """重排必須真的改變順序（而非每次都回傳原序）。"""
        ev = make_evaluator()
        q = {"question": "q", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A"}
        seen = set()
        for seed in range(30):
            random.seed(seed)
            out = ev.shuffle_question_options(q)
            seen.add(tuple(out[k] for k in "ABCD"))
        assert len(seen) > 1

    def test_answer_normalized_before_lookup(self):
        ev = make_evaluator()
        q = {"question": "q", "A": "a", "B": "b", "C": "c", "D": "d", "answer": " b "}
        random.seed(0)
        out = ev.shuffle_question_options(q)
        assert out[out["answer"]] == "b"

    @pytest.mark.parametrize(
        "q",
        [
            {"question": "q", "A": "a", "B": "b", "answer": "Z"},  # answer 不在選項中
            {"question": "q", "A": "a", "B": "b"},  # 缺 answer
            {"question": "q", "answer": "yes"},  # 無選項
            {"question": "q", "A": "a", "answer": "A"},  # 僅一個選項
        ],
    )
    def test_returns_unchanged_when_answer_unlocatable(self, q):
        """無法定位正解時原樣回傳，不得重排（重排後回填不了正解等同毀損題目）。"""
        ev = make_evaluator()
        random.seed(0)
        assert ev.shuffle_question_options(q) is q

    def test_does_not_mutate_input(self):
        ev = make_evaluator()
        q = {"question": "q", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A"}
        snapshot = dict(q)
        random.seed(1)
        ev.shuffle_question_options(q)
        assert q == snapshot


class TestRunnerSingleImplementation:
    """回歸 #142：TwinkleEvalRunner 只能有一份實作。"""

    def test_all_import_paths_resolve_to_same_class(self):
        import twinkle_eval
        from twinkle_eval.main import TwinkleEvalRunner as from_main
        from twinkle_eval.runners import TwinkleEvalRunner as from_runners
        from twinkle_eval.runners.standard import TwinkleEvalRunner as from_standard

        assert twinkle_eval.TwinkleEvalRunner is from_main is from_runners is from_standard

    def test_implementation_lives_in_runners_standard(self):
        """實作應位於 runners/standard.py（CLAUDE.md §4：main.py 不實作評測邏輯）。"""
        from twinkle_eval.runners.standard import TwinkleEvalRunner

        assert TwinkleEvalRunner.__module__ == "twinkle_eval.runners.standard"

    def test_resume_support_retained(self):
        """搬移後必須保留 --resume 用的 completed_records 參數。"""
        import inspect

        from twinkle_eval.runners.standard import TwinkleEvalRunner

        assert "completed_records" in inspect.signature(TwinkleEvalRunner.run_evaluation).parameters
        assert (
            "completed_records" in inspect.signature(TwinkleEvalRunner._evaluate_dataset).parameters
        )


class TestVisionShuffleEndToEnd:
    """端到端回歸 #141：vision_mcq + shuffle_options 必須真的評到題目。"""

    def _run(self, tmp_path, monkeypatch, dataset_rows, shuffle, tag="run0"):
        """以假 LLM 跑完整 evaluate_file，回傳 (metrics, 詳細結果列表)。"""
        import json
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from twinkle_eval.metrics.extractors.vision_mcq import VisionMCQExtractor
        from twinkle_eval.metrics.scorers.exact import ExactMatchScorer

        dataset_path = tmp_path / f"vision_{tag}.jsonl"
        dataset_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in dataset_rows),
            encoding="utf-8",
        )

        message = SimpleNamespace(content="答案是 \\boxed{A}", reasoning=None)
        usage = SimpleNamespace(completion_tokens=1, prompt_tokens=1, total_tokens=2)
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage
        )

        llm = MagicMock()
        llm.call.return_value = completion

        evaluator = Evaluator(
            llm=llm,
            extractor=VisionMCQExtractor(),
            scorer=ExactMatchScorer(),
            config={"llm_api": {"api_rate_limit": -1}, "evaluation": {}},
            eval_method="vision_mcq",
            shuffle_options=shuffle,
        )

        # evaluate_file 會寫進相對路徑 results/，切到 tmp_path 讓它自然建立，
        # 不必 patch 全域的 os.path.join / os.makedirs（那會影響整個 process）
        monkeypatch.chdir(tmp_path)
        _, metrics, results_path = evaluator.evaluate_file(str(dataset_path), f"test_{tag}")

        with open(results_path, encoding="utf-8") as f:
            details = [json.loads(line) for line in f if line.strip()]
        return metrics, details, llm

    @pytest.fixture
    def rows(self, tmp_path):
        """自建最小 PNG，避免依賴 repo 內的圖片資產與當前工作目錄。"""
        import base64

        # 1x1 PNG；_encode_image_to_data_uri 以 magic bytes 判型，不需要真實圖片內容
        png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
            "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        )
        img = tmp_path / "tiny.png"
        img.write_bytes(png)

        return [
            {
                "id": f"q{i}",
                "image_path": str(img),
                "question": f"question {i}",
                "A": "opt-a",
                "B": "opt-b",
                "C": "opt-c",
                "D": "opt-d",
                "answer": "A",
                "category": "coarse perception",
            }
            for i in range(3)
        ]

    def test_shuffle_enabled_still_evaluates_all_questions(self, tmp_path, monkeypatch, rows):
        """開啟 shuffle 時圖片欄位不得遺失，否則所有題目會被 skip。"""
        random.seed(0)
        metrics, details, _ = self._run(tmp_path, monkeypatch, rows, shuffle=True)
        assert metrics["total_count"] == len(rows), "題目在 shuffle 後被 skip（#141 回歸）"
        assert len(details) == len(rows)
        assert all(d["image_path"].endswith("tiny.png") for d in details)

    def test_shuffle_disabled_matches_shuffle_enabled_coverage(self, tmp_path, monkeypatch, rows):
        metrics_off, _, _ = self._run(tmp_path, monkeypatch, rows, shuffle=False, tag="off")
        random.seed(0)
        metrics_on, _, _ = self._run(tmp_path, monkeypatch, rows, shuffle=True, tag="on")
        assert metrics_off["total_count"] == metrics_on["total_count"] == len(rows)


class TestBuildQuestionText:
    """回歸 #143：metadata 欄位不得被當成選項送進 prompt。"""

    def test_excludes_metadata_fields(self):
        from twinkle_eval.runners.evaluator import build_question_text

        q = {
            "id": "gpqa_0",
            "question": "Why?",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "answer": "B",
            "domain": "Biology",
        }
        text = build_question_text(q)
        assert text == "Why?\nA: a\nB: b\nC: c\nD: d"
        assert "gpqa_0" not in text, "id 洩漏進 prompt"
        assert "Biology" not in text, "domain 洩漏進 prompt（等同學科提示）"

    def test_keeps_non_option_fields_when_no_options(self):
        """Text-to-SQL 這類題目需要 db_id / evidence 進 prompt，不得一刀切。"""
        from twinkle_eval.runners.evaluator import build_question_text

        q = {
            "question": "How many singers?",
            "db_id": "concert_singer",
            "evidence": "singer table",
            "answer": "SELECT count(*) FROM singer",
        }
        text = build_question_text(q)
        assert "db_id: concert_singer" in text
        assert "evidence: singer table" in text

    def test_exclude_param_applies_only_without_options(self):
        from twinkle_eval.runners.evaluator import build_question_text

        q = {"question": "Q", "image_path": "/x.png", "answer": "yes"}
        assert "/x.png" not in build_question_text(q, exclude=("image_path",))

    def test_shuffle_on_and_off_produce_same_option_set(self):
        """修 #143 後，shuffle 開關只影響順序，不影響 prompt 含哪些欄位。"""
        from twinkle_eval.runners.evaluator import build_question_text

        ev = make_evaluator()
        q = {
            "id": "x1",
            "question": "Q",
            "A": "a",
            "B": "b",
            "C": "c",
            "D": "d",
            "answer": "A",
            "domain": "Bio",
        }
        random.seed(7)
        off = build_question_text(q)
        on = build_question_text(ev.shuffle_question_options(q))

        def option_texts(text: str) -> list:
            return sorted(line.split(": ", 1)[1] for line in text.split("\n")[1:])

        assert option_texts(off) == option_texts(on) == ["a", "b", "c", "d"]
        for text in (off, on):
            assert "x1" not in text and "Bio" not in text


class TestDetectOptionKeysFallback:
    """非 A 起始的選項鍵需有回退，否則 logit 路徑會一題都不送出請求。"""

    def test_true_false_keys(self):
        assert detect_option_keys({"question": "q", "T": "yes", "F": "no", "answer": "T"}) == [
            "F",
            "T",
        ]

    def test_metadata_only_does_not_trigger_fallback(self):
        assert detect_option_keys({"question": "q", "ID": "x", "answer": "z"}) == []

    def test_single_letter_alone_is_not_options(self):
        assert detect_option_keys({"question": "q", "X": "1", "answer": "X"}) == []


class TestSemanticLabelsAreNotShuffled:
    """回歸：T/F、Y/N 這類標籤帶有語意，重排會讓標籤與內容錯位。"""

    def test_true_false_labels_never_shuffled(self):
        ev = make_evaluator()
        q = {"question": "地球是圓的嗎？", "T": "是", "F": "否", "answer": "T"}
        for seed in range(20):
            random.seed(seed)
            out = ev.shuffle_question_options(q)
            assert out["T"] == "是" and out["F"] == "否"
            assert out["answer"] == "T"

    def test_positional_labels_still_shuffled(self):
        ev = make_evaluator()
        q = {"question": "Q", "A": "a", "B": "b", "C": "c", "D": "d", "answer": "A"}
        seen = set()
        for seed in range(30):
            random.seed(seed)
            seen.add(tuple(ev.shuffle_question_options(q)[k] for k in "ABCD"))
        assert len(seen) > 1


class TestSingleOptionKeepsOtherFields:
    """回歸：只有一個選項鍵時不得把其餘欄位當成 metadata 丟棄。"""

    def test_single_option_key_is_not_treated_as_options(self):
        assert (
            detect_option_keys({"question": "Q", "A": "only", "context": "c", "answer": "A"}) == []
        )

    def test_context_survives(self):
        from twinkle_eval.runners.evaluator import build_question_text

        q = {"question": "Q?", "A": "only", "context": "very important context", "answer": "A"}
        assert "very important context" in build_question_text(q)


class TestDescribeDroppedFields:
    """被丟棄的欄位要能被觀察到，否則分數靜默下降無從察覺。"""

    def test_lists_metadata_fields(self):
        from twinkle_eval.runners.evaluator import describe_dropped_fields

        q = {"question": "Q", "A": "a", "B": "b", "answer": "A", "hint": "h", "id": "x"}
        assert sorted(describe_dropped_fields(q)) == ["hint", "id"]

    def test_empty_when_no_option_keys(self):
        from twinkle_eval.runners.evaluator import describe_dropped_fields

        assert describe_dropped_fields({"question": "Q", "db_id": "d", "answer": "SELECT 1"}) == []

    def test_exclude_param_removes_intentional_fields(self):
        from twinkle_eval.runners.evaluator import describe_dropped_fields

        q = {"question": "Q", "A": "a", "B": "b", "answer": "A", "image_path": "/x.png"}
        assert describe_dropped_fields(q, ("image_path",)) == []


class TestDroppedFieldsNoticeIsEmitted:
    """回歸：提示必須真的在評測時發出，且要看得見（logger 只寫檔案）。"""

    def test_notice_printed_during_evaluate_file(self, tmp_path, monkeypatch, capsys):
        import json
        from types import SimpleNamespace
        from unittest.mock import MagicMock

        from twinkle_eval.metrics.extractors.pattern import PatternExtractor
        from twinkle_eval.metrics.scorers.exact import ExactMatchScorer

        ds = tmp_path / "d.jsonl"
        ds.write_text(
            json.dumps(
                {
                    "question": "Q",
                    "A": "a",
                    "B": "b",
                    "answer": "A",
                    "hint": "important hint",
                    "id": "x",
                }
            ),
            encoding="utf-8",
        )

        message = SimpleNamespace(content="A", reasoning=None)
        usage = SimpleNamespace(completion_tokens=1, prompt_tokens=1, total_tokens=2)
        llm = MagicMock()
        llm.call.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=usage
        )

        evaluator = Evaluator(
            llm=llm,
            extractor=PatternExtractor(),
            scorer=ExactMatchScorer(),
            config={"llm_api": {"api_rate_limit": -1}, "evaluation": {}},
            eval_method="pattern",
        )
        monkeypatch.chdir(tmp_path)
        evaluator.evaluate_file(str(ds), "test_notice")

        out = capsys.readouterr().out
        assert "hint" in out, "被丟棄的欄位未提示到終端機"
        assert "不會進入 prompt" in out


class TestNonContiguousOptionKeys:
    """回歸：不連續的選項鍵（A,B,C,E）不得讓正解從 prompt 消失。

    由 PR #136 的審查發現——「由 A 起連續掃描」會在遇到缺口時停止，
    使缺口之後的選項（含正解）整個不出現在題目裡，題目變成無解。
    """

    def test_gap_in_labels_keeps_all_options(self):
        q = {"question": "Q", "A": "a", "B": "b", "C": "c", "E": "e", "answer": "E"}
        assert detect_option_keys(q) == ["A", "B", "C", "E"]

    def test_correct_answer_stays_visible_in_prompt(self):
        from twinkle_eval.runners.evaluator import build_question_text

        q = {"question": "Q", "A": "a", "B": "b", "C": "c", "E": "e", "answer": "E"}
        assert "E: e" in build_question_text(q)

    def test_two_letter_metadata_still_excluded(self):
        """修法不得讓 ID / NO 這類欄位重新被誤判為選項。"""
        q = {"question": "Q", "A": "a", "B": "b", "C": "c", "D": "d", "ID": "x", "answer": "A"}
        assert detect_option_keys(q) == ["A", "B", "C", "D"]

    def test_canonical_multi_letter_labels_kept(self):
        from twinkle_eval.runners.evaluator import index_to_label

        q = {"question": "Q", "answer": "A"}
        q.update({index_to_label(i): f"o{i}" for i in range(27)})
        assert detect_option_keys(q)[-1] == "AA"
