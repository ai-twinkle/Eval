"""TwinkleEvalRunner — 標準評測執行器。

main.py 保持作為 CLI 入口，此處為核心執行邏輯（CLAUDE.md §4 模組職責邊界）。
"""

import copy
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np

from twinkle_eval.core.config import load_config
from twinkle_eval.core.exceptions import ConfigurationError, EvaluationError
from twinkle_eval.core.logger import log_error, log_info
from twinkle_eval.datasets import find_all_evaluation_files
from twinkle_eval.exporters import ResultsExporterFactory
from twinkle_eval.metrics import create_metric_pair
from twinkle_eval.runners.evaluator import Evaluator

__all__ = ["TwinkleEvalRunner"]


class TwinkleEvalRunner:
    """Twinkle Eval 主要執行器類別 - 負責控制整個評測流程"""

    def __init__(self, config_path: str = "config.yaml") -> None:
        """初始化 Twinkle Eval 執行器

        Args:
            config_path: 配置檔案路徑，預設為 config.yaml
        """
        self.config_path = config_path  # 配置檔案路徑
        self.config: Optional[Dict[str, Any]] = None  # 載入的配置字典
        self.start_time: Optional[str] = None  # 執行開始時間標記
        self.start_datetime: Optional[datetime] = None  # 執行開始的 datetime 物件
        self.results_dir = "results"  # 結果輸出目錄

    def initialize(self) -> None:
        """初始化評測執行器

        載入配置、設定時間標記、建立結果目錄

        Raises:
            Exception: 初始化過程中發生錯誤
        """
        try:
            self.config = load_config(self.config_path)  # 載入配置
            self.start_time = datetime.now().strftime("%Y%m%d_%H%M")  # 生成時間標記
            self.start_datetime = datetime.now()  # 記錄開始時間

            os.makedirs(self.results_dir, exist_ok=True)  # 建立結果目錄

            log_info(f"Twinkle Eval 初始化完成 - {self.start_time}")

        except Exception as e:
            log_error(f"初始化失敗: {e}")
            raise

    def _prepare_config_for_saving(self) -> Dict[str, Any]:
        """準備用於儲存的配置資料，移除敏感資訊

        在儲存配置到結果檔案前，需要移除 API 金鑰等敏感資訊
        和不可序列化的物件實例

        Returns:
            Dict[str, Any]: 清理後的配置字典
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        # 排除物件實例（不可序列化）後再複製，避免就地修改 self.config
        # 導致同一個 runner 無法重複執行
        config_without_instances = {
            k: v
            for k, v in self.config.items()
            if k
            not in (
                "llm_instance",
                "evaluation_strategy_instance",
                "extractor_instance",
                "scorer_instance",
            )
        }
        save_config = copy.deepcopy(config_without_instances)

        # 移除敏感資訊（API 金鑰）
        if "llm_api" in save_config and "api_key" in save_config["llm_api"]:
            del save_config["llm_api"]["api_key"]

        return save_config

    def _get_dataset_paths(self) -> List[str]:
        """從配置中取得資料集路徑清單

        支援單一路徑字串或路徑清單，統一轉換為清單格式

        Returns:
            List[str]: 資料集路徑清單
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        dataset_paths = self.config["evaluation"]["dataset_paths"]
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        return dataset_paths

    def _resolve_dataset_settings(self, dataset_path: str) -> Dict[str, Any]:
        """解析資料集的評測設定，套用 dataset_overrides（若有）。"""
        if self.config is None:
            raise ConfigurationError("配置未載入")

        eval_cfg = self.config["evaluation"]
        overrides = eval_cfg.get("dataset_overrides", {})
        dataset_abs = os.path.normpath(os.path.abspath(dataset_path))

        settings: Dict[str, Any] = {
            "evaluation_method": eval_cfg["evaluation_method"],
            "system_prompt_enabled": eval_cfg.get("system_prompt_enabled", True),
            "samples_per_question": eval_cfg.get("samples_per_question", 1),
            "pass_k": eval_cfg.get("pass_k", 1),
            "repeat_runs": eval_cfg.get("repeat_runs", 1),
            "shuffle_options": eval_cfg.get("shuffle_options", False),
            "model_overrides": {},
        }

        for prefix, cfg in overrides.items():
            if not isinstance(cfg, dict):
                continue
            try:
                prefix_abs = os.path.normpath(os.path.abspath(prefix))
                if not dataset_abs.startswith(prefix_abs):
                    continue
            except (OSError, ValueError):
                continue

            for key in (
                "evaluation_method",
                "system_prompt_enabled",
                "samples_per_question",
                "pass_k",
                "repeat_runs",
                "shuffle_options",
            ):
                if key in cfg:
                    settings[key] = cfg[key]
            for mk in (
                "temperature",
                "top_p",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
            ):
                if mk in cfg:
                    settings["model_overrides"][mk] = cfg[mk]

        return settings

    def _evaluate_dataset(
        self,
        dataset_path: str,
        evaluator: Evaluator,
        repeat_runs: int,
        pass_k: int,
        completed_records: Optional[Dict[str, Set[str]]] = None,
    ) -> Dict[str, Any]:
        """評測單一資料集

        對指定資料集中的所有檔案進行評測，支援多次執行並統計結果

        Args:
            dataset_path: 資料集路徑
            evaluator: 評測器實例
            repeat_runs: 重複執行次數
            pass_k: pass@k 的 k 值
            completed_records: resume 模式下已完成的紀錄

        Returns:
            Dict[str, Any]: 資料集評測結果，包含準確率統計和詳細結果
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        log_info(f"開始評測資料集: {dataset_path}")

        all_files = find_all_evaluation_files(dataset_path)  # 尋找所有評測檔案
        prompt_map = self.config["evaluation"].get("datasets_prompt_map", {})  # 資料集語言對應表
        dataset_lang = prompt_map.get(dataset_path, "zh")  # 當前資料集的語言，預設為中文

        results = []  # 儲存所有檔案的評測結果

        for idx, file_path in enumerate(all_files):
            file_accuracies = []  # 當前檔案的準確率結果
            file_pass_ats = []  # 當前檔案的 pass@k 結果
            file_results = []  # 當前檔案的詳細結果

            # 對當前檔案進行多次評測
            file_unparsed_counts: List[int] = []
            file_total_counts: List[int] = []
            for run in range(repeat_runs):
                # --resume：跳過已有完整結果的 run
                if completed_records is not None:
                    run_key = f"eval_results_{self.start_time}_run{run}.jsonl"
                    if run_key in completed_records:
                        from twinkle_eval.datasets.file import Dataset as _Dataset

                        ds_len = len(_Dataset(file_path))
                        completed_for_file = sum(
                            1
                            for rec in completed_records[run_key]
                            if rec.startswith(f"{file_path}|")
                        )
                        if completed_for_file >= ds_len:
                            log_info(f"⏭️  跳過已完成的檔案：{file_path} (run {run})")
                            continue

                try:
                    file_path_result, metrics, result_path = evaluator.evaluate_file(
                        file_path, f"{self.start_time}_run{run}", dataset_lang
                    )
                    file_accuracies.append(metrics["accuracy"])
                    file_pass_ats.append(metrics["pass_at_k"])
                    file_unparsed_counts.append(metrics.get("unparsed_count", 0))
                    file_total_counts.append(metrics.get("total_count", 0))
                    file_results.append((file_path_result, metrics, result_path))
                except Exception as e:
                    log_error(f"評測檔案 {file_path} 失敗: {e}")
                    continue

            # 為當前檔案計算統計數據
            if file_accuracies:
                mean_accuracy = float(np.mean(file_accuracies))  # 平均準確率
                std_accuracy = (
                    float(np.std(file_accuracies)) if len(file_accuracies) > 1 else 0.0
                )  # 標準差
                mean_pass_at_k = float(np.mean(file_pass_ats)) if file_pass_ats else 0.0
                total_unparsed = sum(file_unparsed_counts)
                total_evaluated = sum(file_total_counts)
                unparsed_rate = total_unparsed / total_evaluated if total_evaluated else 0.0

                results.append(
                    {
                        "file": file_path,
                        "accuracy_mean": mean_accuracy,
                        "accuracy_std": std_accuracy,
                        "pass_at_k_mean": mean_pass_at_k,
                        "pass_metric": f"pass@{pass_k}",
                        "unparsed_count": total_unparsed,
                        "unparsed_rate": round(unparsed_rate, 4),
                        "individual_runs": {
                            "accuracies": file_accuracies,
                            "pass_at_k": file_pass_ats,
                            "unparsed_counts": file_unparsed_counts,
                            "results": [r[2] for r in file_results],
                        },
                    }
                )

            # 進度指示器
            progress = (idx + 1) / len(all_files) * 100
            print(f"\r已執行 {progress:.1f}% ({idx + 1}/{len(all_files)}) ", end="")

        print()  # 進度完成後換行

        # 所有檔案均評測失敗時，拋出明確錯誤（而非以 accuracy=0 偽裝成正常結果）
        if not results:
            raise EvaluationError(
                f"資料集 {dataset_path} 中所有檔案評測均失敗，無法產生結果。\n"
                f"請確認評測設定（evaluation_method、system_prompt）以及 API 端點是否正常運作。"
            )

        # 計算資料集統計數據
        dataset_avg_accuracy = float(np.mean([r["accuracy_mean"] for r in results]))
        dataset_avg_std = float(np.mean([r["accuracy_std"] for r in results]))
        dataset_avg_pass_at_k = float(np.mean([r["pass_at_k_mean"] for r in results]))
        dataset_total_unparsed = sum(r["unparsed_count"] for r in results)
        dataset_avg_unparsed_rate = float(np.mean([r["unparsed_rate"] for r in results]))

        return {
            "results": results,
            "average_accuracy": dataset_avg_accuracy,
            "average_std": dataset_avg_std,
            "average_pass_at_k": dataset_avg_pass_at_k,
            "pass_metric": f"pass@{pass_k}",
            "total_unparsed_count": dataset_total_unparsed,
            "average_unparsed_rate": round(dataset_avg_unparsed_rate, 4),
        }

    def run_evaluation(
        self,
        export_formats: Optional[List[str]] = None,
        completed_records: Optional[Dict[str, Set[str]]] = None,
    ) -> str:
        """執行完整的評測流程

        這是主要的評測入口點，包含以下步驟：
        1. 建立評測器
        2. 對所有資料集進行評測
        3. 統計和輸出結果

        Args:
            export_formats: 輸出格式清單，預設為 ["json"]
            completed_records: resume 模式下已完成的紀錄，key 為 run 檔名，
                               value 為 "file|question_id" 的集合

        Returns:
            str: 主要結果檔案路徑
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        if export_formats is None:
            export_formats = ["json"]  # 預設輸出格式

        dataset_paths = self._get_dataset_paths()  # 取得資料集路徑
        dataset_results: Dict[str, Any] = {}  # 儲存所有資料集的結果

        llm_instance = self.config["llm_instance"]
        strategy_config = self.config["evaluation"].get("strategy_config", {})
        # 快取已建立的 (extractor, scorer) 配對，避免重複實例化
        default_method = self.config["evaluation"]["evaluation_method"]
        metric_cache: Dict[str, Any] = {
            default_method: create_metric_pair(default_method, strategy_config)
        }

        # 逐一評測每個資料集
        for dataset_path in dataset_paths:
            try:
                ds = self._resolve_dataset_settings(dataset_path)
                eval_method = ds["evaluation_method"]

                if eval_method not in metric_cache:
                    metric_cache[eval_method] = create_metric_pair(eval_method, strategy_config)

                extractor, scorer = metric_cache[eval_method]

                evaluator = Evaluator(
                    llm=llm_instance,
                    extractor=extractor,
                    scorer=scorer,
                    config=self.config,
                    eval_method=eval_method,
                    system_prompt_enabled=ds["system_prompt_enabled"],
                    samples_per_question=ds["samples_per_question"],
                    pass_k=ds["pass_k"],
                    shuffle_options=ds["shuffle_options"],
                    model_overrides=ds["model_overrides"],
                )

                dataset_result = self._evaluate_dataset(
                    dataset_path,
                    evaluator,
                    repeat_runs=ds["repeat_runs"],
                    pass_k=ds["pass_k"],
                    completed_records=completed_records,
                )
                if not dataset_result.get("results"):
                    log_error(f"資料集 {dataset_path} 評測完成但無有效結果，跳過")
                    continue
                dataset_result["evaluation_method"] = eval_method
                dataset_results[dataset_path] = dataset_result

                unparsed_info = ""
                if dataset_result.get("total_unparsed_count", 0) > 0:
                    unparsed_info = (
                        f"，無法解析: {dataset_result['total_unparsed_count']} "
                        f"({dataset_result['average_unparsed_rate']:.1%})"
                    )
                message = (
                    f"資料集 {dataset_path} 評測完成（模式: {eval_method}），"
                    f"平均正確率: {dataset_result['average_accuracy']:.2%} "
                    f"(±{dataset_result['average_std']:.2%}){unparsed_info}"
                )
                print(message)
                log_info(message)

            except ImportError as e:
                msg = f"\n❌ 資料集 {dataset_path} 評測失敗：缺少必要套件。\n   {e}\n"
                print(msg)
                log_error(msg.strip())
                continue
            except Exception as e:
                log_error(f"資料集 {dataset_path} 評測失敗: {e}")
                continue

        # 所有資料集均失敗時，拋出明確錯誤（而非靜默輸出空結果）
        if not dataset_results:
            failed_paths = ", ".join(dataset_paths)
            raise EvaluationError(
                f"所有資料集評測均失敗，未產生任何結果。\n"
                f"失敗路徑: {failed_paths}\n"
                f"請確認資料集路徑存在、格式正確，且評測設定完整。"
            )

        # 準備最終結果
        current_duration = (
            (datetime.now() - self.start_datetime).total_seconds() if self.start_datetime else 0
        )  # 計算執行時間
        final_results: Dict[str, Any] = {
            "timestamp": self.start_time,  # 執行時間標記
            "config": self._prepare_config_for_saving(),  # 清理後的配置
            "dataset_results": dataset_results,  # 所有資料集結果
            "duration_seconds": current_duration,  # 執行時間（秒）
        }

        # 以多種格式輸出結果
        base_output_path = os.path.join(self.results_dir, f"results_{self.start_time}")
        exported_files = ResultsExporterFactory.export_results(
            final_results, base_output_path, export_formats, self.config
        )

        # Google 服務整合
        self._handle_google_services(final_results, export_formats)

        log_info(f"評測完成，結果已匯出至: {', '.join(exported_files)}")
        return exported_files[0] if exported_files else ""

    def _handle_google_services(self, results: Dict[str, Any], export_formats: List[str]) -> None:
        """處理 Google 服務整合。"""
        if self.config is None:
            return

        google_services_config = self.config.get("google_services")
        if not google_services_config:
            return

        # 處理 Google Drive 檔案上傳（最新的 log 和 results）
        google_drive_config = google_services_config.get("google_drive", {})
        if google_drive_config.get("enabled", False):
            try:
                from twinkle_eval.integrations.google import GoogleDriveUploader

                uploader = GoogleDriveUploader(google_drive_config)
                upload_info = uploader.upload_latest_files(self.start_time, "logs", "results")

                if upload_info.get("uploaded_files"):
                    log_info(
                        f"成功建立資料夾: {upload_info['folder_name']} ({upload_info['folder_id']})"
                    )
                    log_info(f"成功上傳 {len(upload_info['uploaded_files'])} 個檔案到 Google Drive")

                    for file_info in upload_info["uploaded_files"]:
                        log_info(f"  - {file_info['type']}: {file_info['file_name']}")
            except Exception as e:
                log_error(f"Google Drive 檔案上傳失敗: {e}")

        # 處理 Google Sheets 結果匯出
        google_sheets_config = google_services_config.get("google_sheets", {})
        if google_sheets_config.get("enabled", False):
            try:
                # 檢查是否已經在 export_formats 中指定 google_sheets
                if "google_sheets" not in export_formats:
                    # 如果用戶沒有明確指定，我們自動執行 Google Sheets 匯出
                    sheets_exporter = ResultsExporterFactory.create_exporter(
                        "google_sheets", google_sheets_config
                    )
                    sheets_url = sheets_exporter.export(results, "google_sheets_export")
                    log_info(f"結果已自動匯出到 Google Sheets: {sheets_url}")
            except Exception as e:
                log_error(f"Google Sheets 結果匯出失敗: {e}")
