"""VisTW-MCQ 評測的測試（Milestone #23 Phase 1）。

VisTW-MCQ 沿用既有的 vision_mcq 評測方法，沒有新的 Extractor / Scorer，
因此測試聚焦在三件事：benchmark 註冊、example 資料集完整性，
以及既有的 VisionMCQExtractor 能否處理繁體中文回應。

對應 issue #150、#155。
"""

import json
from pathlib import Path

import pytest

from twinkle_eval.benchmarks import BENCHMARK_REGISTRY
from twinkle_eval.metrics import PRESETS
from twinkle_eval.metrics.extractors.vision_mcq import VisionMCQExtractor
from twinkle_eval.metrics.scorers.exact import ExactMatchScorer

EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "datasets" / "example" / "vistw_mcq"
DATASET = EXAMPLE_DIR / "test.jsonl"

#: VisTW-MCQ 的 21 個學科（論文與 HuggingFace config 名稱一致）
SUBJECTS = {
    "accounting",
    "arts",
    "biology",
    "chemistry",
    "chinese_literature",
    "dentistry",
    "electronic_circuits",
    "fundamentals_of_physical_therapy",
    "geography",
    "mathematics",
    "mechanics",
    "medical",
    "music",
    "natural_science",
    "navigation",
    "pharmaceutical_chemistry",
    "physics",
    "sociology",
    "statistics",
    "structural_engineering",
    "veterinary_medicine",
}


def load_records():
    with open(DATASET, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestBenchmarkRegistry:
    def test_registered(self):
        assert "vistw_mcq" in BENCHMARK_REGISTRY

    def test_entry_fields(self):
        entry = BENCHMARK_REGISTRY["vistw_mcq"]
        assert entry["source"] == "huggingface"
        assert entry["hf_id"] == "miulab/vistw-mcq"
        assert entry["split"] == "test"
        assert entry["license"] == "CC-BY-4.0"

    def test_uses_existing_vision_mcq_method(self):
        """VisTW-MCQ 不應引入新的評測方法。"""
        assert BENCHMARK_REGISTRY["vistw_mcq"]["eval_method"] == "vision_mcq"
        assert "vision_mcq" in PRESETS

    def test_no_new_preset_added(self):
        """Phase 1 不該新增 PRESETS 項目；Dialogue（#151）才需要。"""
        assert "vistw_mcq" not in PRESETS
        assert "vistw" not in PRESETS


@pytest.mark.skipif(not DATASET.exists(), reason="example 資料集尚未建立")
class TestExampleDataset:
    def test_dataset_file_exists(self):
        assert DATASET.exists()

    def test_record_count(self):
        """21 個學科各 1 題。"""
        assert len(load_records()) == 21

    def test_required_fields(self):
        required = {"id", "subject", "image_path", "question", "A", "B", "C", "D", "answer"}
        for r in load_records():
            assert required <= set(r), f"{r.get('id')} 缺少欄位: {required - set(r)}"
            for k in ("question", "A", "B", "C", "D"):
                assert str(r[k]).strip(), f"{r['id']} 的 {k} 是空的"

    def test_id_is_subject_prefixed(self):
        """id 必須是 {subject}_{qid}——這是圖片檔名不互相覆蓋的前提。"""
        for r in load_records():
            assert r["id"].startswith(r["subject"] + "_"), f"id 未加學科前綴: {r['id']}"

    def test_ids_are_unique(self):
        """qid 在跨學科之間不唯一，id 必須加上 subject 前綴，否則圖片檔名會互相覆蓋。"""
        ids = [r["id"] for r in load_records()]
        assert len(set(ids)) == len(ids)

    def test_all_subjects_covered(self):
        assert {r["subject"] for r in load_records()} == SUBJECTS

    def test_images_exist(self):
        repo_root = EXAMPLE_DIR.parent.parent.parent
        for r in load_records():
            img = repo_root / r["image_path"]
            assert "vistw_mcq" in r["image_path"], f"{r['id']} 的圖片不在 vistw_mcq 目錄下"
            assert img.exists(), f"{r['id']} 的圖片不存在: {r['image_path']}"
            assert img.stat().st_size > 0

    def test_answers_are_valid_option_keys(self):
        for r in load_records():
            assert r["answer"] in ("A", "B", "C", "D"), f"{r['id']}: answer={r['answer']!r}"

    def test_answer_distribution_is_not_degenerate(self):
        """答案不得集中在單一選項。

        若直接取每科第一題，21 題中有 17 題答案是 A——一個永遠回答 A 的模型
        就能拿 81%，這份 example 便失去 sanity check 的作用。
        """
        answers = [r["answer"] for r in load_records()]
        most_common = max(answers.count(k) for k in set(answers))
        assert most_common <= len(answers) * 0.5, f"答案過度集中: {most_common}/{len(answers)}"

    def test_no_upstream_metadata_leaked_into_records(self):
        """轉換時應剔除 source / stats 等上游 metadata。"""
        for r in load_records():
            assert "source" not in r
            assert "stats" not in r
            assert "qid" not in r

    def test_questions_are_chinese(self):
        """抽樣確認題幹確實是中文，而非英文資料集混入。"""
        records = load_records()
        has_cjk = sum(any("一" <= ch <= "鿿" for ch in r["question"]) for r in records)
        assert has_cjk >= len(records) * 0.8


class TestVisionMCQOnTraditionalChineseResponses:
    """既有的 VisionMCQExtractor 需能處理繁體中文回應。"""

    @pytest.fixture
    def extractor(self):
        return VisionMCQExtractor()

    @pytest.fixture
    def scorer(self):
        return ExactMatchScorer()

    @pytest.mark.parametrize(
        "response,expected",
        [
            ("答案是 \\boxed{B}", "B"),
            ("\\boxed{C}", "C"),
            ("正確答案為 (D)", "D"),
            ("故選 C。", "C"),
            ("B", "B"),
        ],
    )
    def test_extracts_from_chinese_response(self, extractor, response, expected):
        assert extractor.extract(response) == expected

    @pytest.mark.parametrize("response", ["我認為應該選 A", "選 A", "我選擇 B"])
    def test_known_gap_chinese_select_without_trailing_punctuation(self, extractor, response):
        """記錄已知缺口（#156），而非假裝它不存在。

        `選\\s*([A-Z])\\s*[項。.]` 要求結尾必須有「項」或句號，因此
        「我選擇 B」「選 A」這類常見繁中回應抓不到答案、該題被判錯。
        修好 #156 後這個測試會失敗——屆時請把這幾個案例移到上面的
        test_extracts_from_chinese_response 並刪除本測試。
        """
        assert extractor.extract(response) is None

    def test_boxed_takes_priority_over_echoed_options(self, extractor):
        """VLM 常先複述選項再給答案，必須取最後的結論而非第一個字母。"""
        response = "選項 A 不對，選項 B 也不對。\n綜合判斷，答案是 \\boxed{C}"
        assert extractor.extract(response) == "C"

    def test_unparseable_returns_none(self, extractor):
        assert extractor.extract("我無法判斷這張圖片的內容。") is None

    def test_scorer_matches(self, scorer):
        assert scorer.score(scorer.normalize(" b "), scorer.normalize("B")) is True
        assert scorer.score(scorer.normalize("A"), scorer.normalize("B")) is False
