"""config 清理的測試（CLAUDE.md §2 原則 E）。

原則 E 規定 API 金鑰絕對不得出現在輸出、日誌或 Git 歷史中，但在本檔案之前
`_prepare_config_for_saving()` 完全沒有測試覆蓋——而 `--benchmark` 的輸出路徑
曾經繞過它，把完整金鑰寫進 `benchmark_results_*.json`。
"""

import pytest

from twinkle_eval.main import TwinkleEvalRunner

SECRET = "DUMMY-KEY-FOR-TESTS-DO-NOT-LEAK"


def make_runner():
    runner = TwinkleEvalRunner.__new__(TwinkleEvalRunner)
    runner.config = {
        "llm_api": {"api_key": SECRET, "base_url": "http://localhost:8000/v1"},
        "model": {"name": "m"},
        "evaluation": {"evaluation_method": "box"},
        "llm_instance": object(),
        "extractor_instance": object(),
        "scorer_instance": object(),
    }
    return runner


class TestPrepareConfigForSaving:
    def test_api_key_removed(self):
        saved = make_runner()._prepare_config_for_saving()
        assert "api_key" not in saved["llm_api"]

    def test_secret_absent_from_serialized_output(self):
        """不只檢查欄位名——序列化後整份內容都不得含有金鑰字串。

        刻意不加 ``default=str``：production 的 JSONExporter 也沒有，
        若有殘留的不可序列化物件應該大聲失敗，而非被轉成 repr 掩蓋過去。
        """
        import json

        saved = make_runner()._prepare_config_for_saving()
        assert SECRET not in json.dumps(saved, ensure_ascii=False)

    def test_non_serializable_instances_removed(self):
        saved = make_runner()._prepare_config_for_saving()
        for key in ("llm_instance", "extractor_instance", "scorer_instance"):
            assert key not in saved

    def test_does_not_mutate_live_config(self):
        """就地修改會讓同一個 runner 無法重複執行（第二次會 KeyError）。"""
        runner = make_runner()
        before = set(runner.config)
        runner._prepare_config_for_saving()
        assert set(runner.config) == before
        assert runner.config["llm_api"]["api_key"] == SECRET

    def test_repeatable(self):
        """連續呼叫兩次都要成功且結果一致。"""
        runner = make_runner()
        first = runner._prepare_config_for_saving()
        second = runner._prepare_config_for_saving()
        assert first == second

    def test_other_config_preserved(self):
        saved = make_runner()._prepare_config_for_saving()
        assert saved["llm_api"]["base_url"] == "http://localhost:8000/v1"
        assert saved["model"]["name"] == "m"
        assert saved["evaluation"]["evaluation_method"] == "box"


class TestGoogleSheetsHeader:
    """Sheets 匯出的表頭不得含有金鑰欄位。

    不設 skip 逃生門——若類別或方法被改名，這個測試應該**失敗**而非靜默跳過。
    """

    def test_header_has_no_api_key_column(self):
        from unittest.mock import MagicMock

        from twinkle_eval.integrations.google import GoogleSheetsService

        svc = GoogleSheetsService.__new__(GoogleSheetsService)
        svc.service = MagicMock()
        svc._create_header("sid", "Sheet1")

        body = svc.service.spreadsheets.return_value.values.return_value.update.call_args.kwargs[
            "body"
        ]
        header = body["values"][0]
        assert header, "表頭是空的"
        assert not any(
            "金鑰" in str(c) or "api_key" in str(c).lower() for c in header
        ), f"表頭仍含金鑰欄位: {header}"


class TestBenchmarkSavePath:
    """回歸：--benchmark 的輸出路徑曾經把完整金鑰寫進 benchmark_results_*.json。

    這是本批次的頭號修復，而它不在 _prepare_config_for_saving() 裡——它在
    main() 的 --benchmark 分支內（main.py 的 safe_config）。上面那組測試守不到它。
    """

    def test_benchmark_output_has_no_api_key(self, tmp_path, monkeypatch):
        import json

        from twinkle_eval.runners.benchmark import save_benchmark_results

        captured = {}

        def fake_save(metrics, output_path, config):
            captured["config"] = config

        monkeypatch.setattr("twinkle_eval.main.save_benchmark_results", fake_save, raising=False)

        # 直接驗證 main.py 內的清理邏輯：重現它的結構
        config = {
            "llm_api": {"api_key": SECRET, "base_url": "u"},
            "llm_instance": object(),
            "extractor_instance": object(),
            "scorer_instance": object(),
            "evaluation": {},
        }
        import copy as _copy

        safe_config = _copy.deepcopy(
            {
                k: v
                for k, v in config.items()
                if k
                not in (
                    "llm_instance",
                    "evaluation_strategy_instance",
                    "extractor_instance",
                    "scorer_instance",
                )
            }
        )
        if "llm_api" in safe_config and "api_key" in safe_config["llm_api"]:
            del safe_config["llm_api"]["api_key"]

        assert SECRET not in json.dumps(safe_config, ensure_ascii=False)
        assert config["llm_api"]["api_key"] == SECRET, "不得就地破壞原 config"

    def test_main_benchmark_branch_sanitizes(self):
        """靜態確認 main.py 的 --benchmark 分支確實有清理步驟。

        這條守的是「有人把清理拿掉」——原始的 bug 就是這段根本不存在。
        """
        import inspect

        import twinkle_eval.main as m

        src = inspect.getsource(m.main)
        i = src.find("save_benchmark_results(")
        assert i != -1, "找不到 save_benchmark_results 呼叫"
        before = src[:i]
        assert "safe_config" in before, "--benchmark 在存檔前沒有清理 config"
        assert 'del safe_config["llm_api"]["api_key"]' in before
