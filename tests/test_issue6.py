"""
tests/test_issue6.py

Issue #6 回歸測試：
當 evaluation.dataset_paths 只有一個資料集時，評測失敗必須拋出明確例外，
而不是靜默地輸出空結果並以 exit 0 結束。
"""

from unittest.mock import MagicMock, patch

import pytest

from twinkle_eval.exceptions import EvaluationError


def _make_runner_with_config(dataset_paths: list):
    """建立一個已初始化的 TwinkleEvalRunner，不需要真實的 config.yaml"""
    from twinkle_eval.main import TwinkleEvalRunner

    runner = TwinkleEvalRunner.__new__(TwinkleEvalRunner)
    runner.config_path = "config.yaml"
    runner.results_dir = "results"
    runner.start_time = "20260316_0000"
    runner.start_datetime = None

    # 最小化 config，只放 run_evaluation() 所需的欄位
    runner.config = {
        "evaluation": {
            "dataset_paths": dataset_paths,
            "evaluation_method": "box",
            "repeat_runs": 1,
            "shuffle_options": False,
            "datasets_prompt_map": {},
        },
        "llm_instance": MagicMock(),
        "evaluation_strategy_instance": MagicMock(),
        "llm_api": {"api_rate_limit": -1},
    }
    return runner


class TestSingleDatasetFail:
    """Issue #6：單一 dataset_path 失敗時必須拋出明確錯誤"""

    def test_single_dataset_raises_when_evaluate_fails(self):
        """_evaluate_dataset() 拋出例外時，run_evaluation() 不得靜默結束，
        必須拋出 EvaluationError"""
        runner = _make_runner_with_config(["datasets/nonexistent/"])

        with patch.object(
            runner,
            "_evaluate_dataset",
            side_effect=FileNotFoundError("找不到評測檔案"),
        ):
            with patch.object(runner, "_prepare_config_for_saving", return_value={}):
                with pytest.raises(EvaluationError):
                    runner.run_evaluation()

    def test_single_dataset_raises_when_all_files_fail(self):
        """_evaluate_dataset() 回傳空 results（所有檔案皆失敗）時，
        run_evaluation() 必須拋出 EvaluationError"""
        runner = _make_runner_with_config(["datasets/empty/"])

        # _evaluate_dataset 正常回傳，但 results 是空的（所有檔案都失敗）
        with patch.object(
            runner,
            "_evaluate_dataset",
            return_value={"results": [], "average_accuracy": 0.0, "average_std": 0.0},
        ):
            with patch.object(runner, "_prepare_config_for_saving", return_value={}):
                with pytest.raises(EvaluationError):
                    runner.run_evaluation()


class TestAllDatasetsFail:
    """多個 dataset_paths 全部失敗時，同樣必須拋出明確錯誤"""

    def test_all_datasets_fail_raises_evaluation_error(self):
        """所有資料集評測均失敗時，run_evaluation() 必須拋出 EvaluationError"""
        runner = _make_runner_with_config(
            ["datasets/path_a/", "datasets/path_b/"]
        )

        with patch.object(
            runner,
            "_evaluate_dataset",
            side_effect=FileNotFoundError("找不到評測檔案"),
        ):
            with patch.object(runner, "_prepare_config_for_saving", return_value={}):
                with pytest.raises(EvaluationError):
                    runner.run_evaluation()


class TestPartialDatasetFail:
    """部分資料集失敗時，應繼續處理成功的資料集（不得整體中斷）"""

    def test_partial_failure_continues_and_succeeds(self):
        """部分資料集失敗時，run_evaluation() 應繼續評測其他資料集並正常完成"""
        runner = _make_runner_with_config(
            ["datasets/path_a/", "datasets/path_b/"]
        )

        good_result = {
            "results": [{"file": "a.csv", "accuracy_mean": 0.8, "accuracy_std": 0.0, "individual_runs": {}}],
            "average_accuracy": 0.8,
            "average_std": 0.0,
        }

        def side_effect(path, evaluator, **kwargs):
            if "path_a" in path:
                raise FileNotFoundError("找不到評測檔案")
            return good_result

        with patch.object(runner, "_evaluate_dataset", side_effect=side_effect):
            with patch.object(runner, "_prepare_config_for_saving", return_value={}):
                with patch("twinkle_eval.main.ResultsExporterFactory.export_results", return_value=["results/results.json"]):
                    with patch.object(runner, "_handle_google_services"):
                        # 部分失敗時，有成功的結果 → 不應拋出例外
                        result = runner.run_evaluation()
                        assert result  # 應回傳結果路徑
