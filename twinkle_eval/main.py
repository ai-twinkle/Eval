import argparse
import copy
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from twinkle_eval.core.exceptions import ConfigurationError, EvaluationError

from .core.config import load_config
from .datasets import find_all_evaluation_files
from .runners.evaluator import Evaluator
from .exporters import ResultsExporterFactory
from .core.logger import log_error, log_info


def convert_json_to_html(json_file_path: str) -> int:
    """將 JSON 結果檔案轉換為 HTML 格式

    Args:
        json_file_path: JSON 結果檔案的路徑

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    import json

    try:
        # 檢查輸入檔案是否存在
        if not os.path.exists(json_file_path):
            print(f"❌ 檔案不存在: {json_file_path}")
            return 1

        # 載入 JSON 結果
        with open(json_file_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        # 建立 HTML 輸出器
        html_exporter = ResultsExporterFactory.create_exporter("html")

        # 產生輸出檔案路徑（與輸入檔案同目錄，但副檔名為 .html）
        output_path = os.path.splitext(json_file_path)[0] + ".html"

        # 執行轉換
        exported_file = html_exporter.export(results, output_path)

        print(f"✅ 成功轉換為 HTML: {exported_file}")
        return 0

    except json.JSONDecodeError as e:
        print(f"❌ JSON 檔案格式錯誤: {e}")
        return 1
    except Exception as e:
        print(f"❌ 轉換過程中發生錯誤: {e}")
        return 1


def get_available_templates() -> list[str]:
    """掃描 templates/ 目錄，回傳所有可用的 template 名稱（不含副檔名）。

    Returns:
        list[str]: 可用的 template 名稱列表，依字母排序
    """
    import glob

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")
    files = glob.glob(os.path.join(templates_dir, "*.yaml"))
    return sorted(os.path.splitext(os.path.basename(f))[0] for f in files)


def list_templates() -> int:
    """列出所有可用的設定檔範本。

    Returns:
        int: 程式退出代碼（0 表示成功）
    """
    available = get_available_templates()
    print("📋 可用的設定檔範本：")
    print()
    for name in available:
        print(f"  - {name}")
    print()
    print("💡 使用方式：")
    print("  twinkle-eval --init <name>    # 產生單一範本")
    print("  twinkle-eval --init all       # 產生全部範本")
    return 0


def create_default_config(template_name: str | None = None, output_dir: str = "configs") -> int:
    """在指定目錄下建立設定檔範本。

    Args:
        template_name: 要產生的範本名稱。None 或 "list" 列出所有可用範本，
                       "all" 產生全部，其他則產生指定的單一範本。
        output_dir: 輸出目錄，預設為 configs/

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    import shutil

    # 列出可用範本
    if template_name is None:
        return list_templates()

    available = get_available_templates()

    # 決定要產生哪些範本
    if template_name == "all":
        names = available
    elif template_name in available:
        names = [template_name]
    else:
        print(f"❌ 找不到範本：{template_name}")
        print()
        print("可用的範本：")
        for name in available:
            print(f"  - {name}")
        return 1

    templates_dir = os.path.join(os.path.dirname(__file__), "templates")

    try:
        os.makedirs(output_dir, exist_ok=True)

        created: list[str] = []
        skipped: list[str] = []

        for name in names:
            template_path = os.path.join(templates_dir, f"{name}.yaml")
            output_path = os.path.join(output_dir, f"{name}.yaml")

            if not os.path.exists(template_path):
                print(f"❌ 找不到範本檔案: {template_path}")
                return 1

            if os.path.exists(output_path):
                response = input(f"⚠️  '{output_path}' 已存在，是否覆蓋？(y/N): ")
                if response.lower() not in ["y", "yes", "是"]:
                    skipped.append(output_path)
                    continue

            shutil.copy2(template_path, output_path)
            created.append(output_path)

        print()
        for path in created:
            print(f"✅ 已建立：{path}")
        for path in skipped:
            print(f"⏭️  已跳過：{path}")

        if not created:
            return 0

        print()
        print("📝 接下來請編輯設定檔，填入：")
        print("  1. llm_api.base_url  — API 端點網址")
        print("  2. llm_api.api_key   — API 金鑰")
        print("  3. model.name        — 模型名稱")
        print("  4. evaluation.dataset_paths — 資料集路徑")
        print()
        print("💡 編輯完成後，執行評測：")
        for path in created:
            print(f"   twinkle-eval --config {path}")

        return 0

    except Exception as e:
        print(f"❌ 建立設定檔時發生錯誤: {e}")
        return 1


class TwinkleEvalRunner:
    """Twinkle Eval 主要執行器類別 - 負責控制整個評測流程"""

    def __init__(self, config_path: str = "config.yaml"):
        """初始化 Twinkle Eval 執行器

        Args:
            config_path: 配置檔案路徑，預設為 config.yaml
        """
        self.config_path = config_path  # 配置檔案路徑
        self.config = None  # 載入的配置字典
        self.start_time = None  # 執行開始時間標記
        self.start_datetime = None  # 執行開始的 datetime 物件
        self.results_dir = "results"  # 結果輸出目錄

    def initialize(self):
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

        # 移除物件實例（不可序列化）
        if "llm_instance" in self.config:
            del self.config["llm_instance"]

        save_config = copy.deepcopy(self.config)

        # 移除敏感資訊（API 金鑰）
        if "llm_api" in save_config and "api_key" in save_config["llm_api"]:
            del save_config["llm_api"]["api_key"]
        for key in ("evaluation_strategy_instance", "extractor_instance", "scorer_instance"):
            if key in save_config:
                del save_config[key]

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

            for key in ("evaluation_method", "system_prompt_enabled", "samples_per_question",
                        "pass_k", "repeat_runs", "shuffle_options"):
                if key in cfg:
                    settings[key] = cfg[key]
            for mk in ("temperature", "top_p", "max_tokens", "frequency_penalty", "presence_penalty"):
                if mk in cfg:
                    settings["model_overrides"][mk] = cfg[mk]

        return settings

    def _evaluate_dataset(
        self,
        dataset_path: str,
        evaluator: Evaluator,
        repeat_runs: int,
        pass_k: int,
        completed_records: Optional[Dict[str, set]] = None,
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
                        from .datasets.file import Dataset as _Dataset
                        ds_len = len(_Dataset(file_path))
                        completed_for_file = sum(
                            1 for rec in completed_records[run_key]
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
                mean_accuracy = np.mean(file_accuracies)  # 平均準確率
                std_accuracy = np.std(file_accuracies) if len(file_accuracies) > 1 else 0  # 標準差
                mean_pass_at_k = np.mean(file_pass_ats) if file_pass_ats else 0.0
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
        dataset_avg_accuracy = np.mean([r["accuracy_mean"] for r in results])
        dataset_avg_std = np.mean([r["accuracy_std"] for r in results])
        dataset_avg_pass_at_k = np.mean([r["pass_at_k_mean"] for r in results])
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
        completed_records: Optional[Dict[str, set]] = None,
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
        dataset_results = {}  # 儲存所有資料集的結果

        llm_instance = self.config["llm_instance"]
        strategy_config = self.config["evaluation"].get("strategy_config", {})
        # 快取已建立的 (extractor, scorer) 配對，避免重複實例化
        from .metrics import create_metric_pair
        default_method = self.config["evaluation"]["evaluation_method"]
        metric_cache = {default_method: create_metric_pair(default_method, strategy_config)}

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
                    dataset_path, evaluator,
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
        final_results = {
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

    def _handle_google_services(self, results: Dict[str, Any], export_formats: List[str]):
        """處理 Google 服務整合

        Args:
            results: 評測結果字典
            export_formats: 匯出格式列表
        """
        google_services_config = self.config.get("google_services")
        if not google_services_config:
            return

        # 處理 Google Drive 檔案上傳（最新的 log 和 results）
        google_drive_config = google_services_config.get("google_drive", {})
        if google_drive_config.get("enabled", False):
            try:
                from .integrations.google import GoogleDriveUploader

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


def create_cli_parser() -> argparse.ArgumentParser:
    """建立命令列介面解析器

    定義所有命令列參數和選項，支援多種評測和查詢功能

    Returns:
        argparse.ArgumentParser: 配置完成的命令列解析器
    """
    parser = argparse.ArgumentParser(
        description="🌟 Twinkle Eval - AI 模型評測工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  twinkle-eval                          # 使用預設配置執行
  twinkle-eval --config custom.yaml    # 使用自定義配置檔
  twinkle-eval --export json csv html google_sheets  # 輸出為多種格式
  twinkle-eval --list-llms             # 列出可用的 LLM 類型
  twinkle-eval --list-strategies       # 列出可用的評測策略

設定檔範本:
  twinkle-eval --init                  # 列出所有可用範本
  twinkle-eval --init multiple_choice  # 產生選擇題範本
  twinkle-eval --init all              # 產生全部範本

驗證與預覽:
  twinkle-eval --validate              # 驗證設定檔與資料集格式
  twinkle-eval --dry-run               # 預覽評測計畫（不呼叫 API）

接續中斷的評測:
  twinkle-eval --resume 20250401_1200  # 從指定時間戳記的中斷點繼續

結果格式轉換:
  twinkle-eval --convert-to-html results_20240101_1200.json  # 將 JSON 結果轉換為 HTML

效能基準測試:
  twinkle-eval --benchmark                           # 執行預設的基準測試
  twinkle-eval --benchmark --benchmark-requests 50  # 執行 50 個請求的測試
  twinkle-eval --benchmark --benchmark-concurrency 5 --benchmark-rate 2  # 5 並發，2 請求/秒

HuggingFace 資料集下載:
  twinkle-eval --download-dataset cais/mmlu          # 下載 MMLU 所有子集
  twinkle-eval --download-dataset cais/mmlu --dataset-subset anatomy  # 下載特定子集
  twinkle-eval --dataset-info cais/mmlu             # 查看資料集資訊
        """,
    )

    parser.add_argument(
        "--config", "-c", default="config.yaml", help="配置檔案路徑 (預設: config.yaml)"
    )

    parser.add_argument(
        "--export",
        "-e",
        nargs="+",
        default=["json"],
        choices=ResultsExporterFactory.get_available_types(),
        help="輸出格式 (預設: json)",
    )

    parser.add_argument("--list-llms", action="store_true", help="列出可用的 LLM 類型")

    parser.add_argument("--list-strategies", action="store_true", help="列出可用的評測策略")

    parser.add_argument("--list-exporters", action="store_true", help="列出可用的輸出格式")

    parser.add_argument("--version", action="store_true", help="顯示版本資訊")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="載入設定檔與資料集，顯示評測計畫但不呼叫 API",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="僅驗證設定檔格式與資料集格式是否正確",
    )

    parser.add_argument(
        "--resume",
        metavar="TIMESTAMP",
        help="從指定時間戳記的中斷點繼續評測（跳過已有結果的題目）",
    )

    parser.add_argument(
        "--init",
        nargs="?",
        const=None,
        default=False,
        metavar="TEMPLATE",
        help="產生設定檔範本到 configs/ 目錄。不帶參數列出所有可用範本，指定名稱產生單一範本，'all' 產生全部",
    )

    # HuggingFace 資料集下載相關命令
    parser.add_argument(
        "--download-dataset",
        metavar="DATASET_NAME",
        help="從 HuggingFace Hub 下載資料集 (例如: cais/mmlu)",
    )

    parser.add_argument(
        "--dataset-subset",
        metavar="SUBSET",
        help="指定資料集子集名稱 (與 --download-dataset 一起使用)",
    )

    parser.add_argument(
        "--dataset-split",
        metavar="SPLIT",
        default="test",
        help="指定資料集分割 (預設: test)",
    )

    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default="datasets",
        help="資料集下載輸出目錄 (預設: datasets)",
    )

    parser.add_argument(
        "--dataset-info",
        metavar="DATASET_NAME",
        help="獲取 HuggingFace 資料集資訊",
    )

    parser.add_argument(
        "--convert-to-html",
        metavar="JSON_FILE",
        help="將 JSON 結果檔案轉換為 HTML 格式",
    )

    parser.add_argument(
        "--finalize-results",
        metavar="TIMESTAMP",
        help=(
            "後處理指定時間戳記的評測結果：若找到分散式碎片則自動合併，"
            "若為單節點最終結果則直接上傳 (可搭配 --hf-repo-id)"
        ),
    )

    # HuggingFace 上傳參數
    parser.add_argument(
        "--hf-repo-id",
        help=(
            "Hugging Face dataset repo ID，用於上傳結果 "
            "(格式: namespace/repo-name，repo-name 必須以 -logs-and-scores 結尾)"
        ),
    )

    parser.add_argument(
        "--hf-variant",
        help="結果變體名稱（例如: low, medium, high），用於區分不同評測條件",
    )

    # NIAH 資料集生成命令
    parser.add_argument(
        "--generate-niah",
        action="store_true",
        help="生成自訂 NIAH (Needle in a Haystack) 測試集 JSONL",
    )

    parser.add_argument(
        "--haystack",
        metavar="PATH",
        help="Haystack 文本檔案或目錄路徑（與 --generate-niah 一起使用）",
    )

    parser.add_argument(
        "--needle",
        metavar="TEXT",
        help="要藏入 haystack 的事實/句子",
    )

    parser.add_argument(
        "--question",
        metavar="TEXT",
        help="對應 needle 的提問",
    )

    parser.add_argument(
        "--answer",
        metavar="TEXT",
        help="Ground truth 答案",
    )

    parser.add_argument(
        "--context-lengths",
        metavar="LENGTHS",
        default="1024,2048,4096,8192,16384,32768,65536,131072",
        help="Context 長度列表（以 token 為單位，逗號分隔，預設: 1024,...,131072）",
    )

    parser.add_argument(
        "--needle-depths",
        metavar="DEPTHS",
        default="0,10,20,30,40,50,60,70,80,90,100",
        help="Needle 插入深度列表（0-100 百分比，逗號分隔，預設: 0,10,...,100）",
    )

    parser.add_argument(
        "--niah-language",
        metavar="LANG",
        default="en",
        help="NIAH 語言代碼（預設: en）",
    )

    parser.add_argument(
        "--chars-per-token",
        type=int,
        default=4,
        help="每個 token 的平均字元數（英文≈4, 中文≈2，預設: 4）",
    )

    parser.add_argument(
        "--niah-prompt-template",
        metavar="TEMPLATE",
        help="自訂 prompt 模板，用 {context} 和 {question} 佔位符",
    )

    # Benchmark 相關命令
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="執行 LLM 效能基準測試",
    )

    parser.add_argument(
        "--benchmark-prompt",
        metavar="PROMPT",
        default="請用繁體中文回答：台灣的首都是哪裡？",
        help="基準測試使用的提示文字 (預設: 請用繁體中文回答：台灣的首都是哪裡？)",
    )

    parser.add_argument(
        "--benchmark-requests",
        type=int,
        default=100,
        help="基準測試的總請求數 (預設: 100)",
    )

    parser.add_argument(
        "--benchmark-concurrency",
        type=int,
        default=10,
        help="基準測試的並發請求數 (預設: 10)",
    )

    parser.add_argument(
        "--benchmark-rate",
        type=float,
        help="基準測試的請求速率 (請求/秒，不指定則全速發送)",
    )

    parser.add_argument(
        "--benchmark-duration",
        type=float,
        help="基準測試的最大執行時間 (秒，不指定則執行完所有請求)",
    )

    return parser


def _handle_validate(config_path: str) -> int:
    """處理 --validate：僅驗證設定檔與資料集格式。

    Args:
        config_path: 設定檔路徑

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    from .core.validators import ConfigValidator, DatasetValidator

    errors: list[str] = []

    # 1. 驗證設定檔存在與 YAML 語法
    try:
        ConfigValidator.validate_config_file(config_path)
        print(f"✅ 設定檔存在且可讀取：{config_path}")
    except Exception as e:
        print(f"❌ 設定檔錯誤：{e}")
        return 1

    try:
        ConfigValidator.validate_yaml_syntax(config_path)
        print("✅ YAML 語法正確")
    except Exception as e:
        print(f"❌ YAML 語法錯誤：{e}")
        return 1

    # 2. 驗證設定檔結構
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    try:
        ConfigValidator.validate_config_structure(config)
        print("✅ 設定檔結構正確")
    except Exception as e:
        errors.append(f"設定檔結構：{e}")

    # 3. 驗證資料集路徑與檔案
    eval_cfg = config.get("evaluation", {})
    dataset_paths = eval_cfg.get("dataset_paths", [])
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    for ds_path in dataset_paths:
        try:
            DatasetValidator.validate_dataset_path(ds_path)
            files = DatasetValidator.validate_dataset_files(ds_path)
            print(f"✅ 資料集 {ds_path}：找到 {len(files)} 個檔案")
        except Exception as e:
            errors.append(f"資料集 {ds_path}：{e}")

    # 4. 結果
    if errors:
        print()
        for err in errors:
            print(f"❌ {err}")
        return 1

    print()
    print("✅ 驗證全部通過")
    return 0


def _handle_dry_run(config_path: str) -> int:
    """處理 --dry-run：載入設定檔與資料集，顯示評測計畫但不呼叫 API。

    Args:
        config_path: 設定檔路徑

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    from .datasets.file import Dataset

    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"❌ 載入設定檔失敗：{e}")
        return 1

    model_name = config.get("model", {}).get("name", "未指定")
    eval_cfg = config.get("evaluation", {})
    eval_method = eval_cfg.get("evaluation_method", "未指定")
    repeat_runs = eval_cfg.get("repeat_runs", 1)
    shuffle = eval_cfg.get("shuffle_options", False)

    dataset_paths = eval_cfg.get("dataset_paths", [])
    if isinstance(dataset_paths, str):
        dataset_paths = [dataset_paths]

    print("📋 評測計畫預覽（Dry Run）")
    print("=" * 60)
    print(f"  模型：{model_name}")
    print(f"  評測方法：{eval_method}")
    print(f"  重複次數：{repeat_runs}")
    print(f"  選項隨機排列：{'是' if shuffle else '否'}")
    print()

    total_questions = 0
    total_files = 0

    for ds_path in dataset_paths:
        print(f"📁 資料集：{ds_path}")
        try:
            files = find_all_evaluation_files(ds_path)
            for file_path in files:
                ds = Dataset(file_path)
                count = len(ds)
                total_questions += count
                total_files += 1
                print(f"   📄 {os.path.basename(file_path)}：{count} 題")
        except Exception as e:
            print(f"   ❌ 無法讀取：{e}")

    print()
    print("-" * 60)
    print(f"  資料集檔案數：{total_files}")
    print(f"  總題數：{total_questions}")
    print(f"  總 API 呼叫數：{total_questions * repeat_runs}")
    print()
    print("💡 確認無誤後，移除 --dry-run 即可開始評測")
    return 0


def _handle_resume(config_path: str, timestamp: str, export_formats: list[str]) -> int:
    """處理 --resume：從指定時間戳記的中斷點繼續評測。

    讀取既有的 JSONL 結果檔，找出已完成的題目，跳過這些題目繼續評測。

    Args:
        config_path: 設定檔路徑
        timestamp: 中斷時的時間戳記（如 20250401_1200）
        export_formats: 輸出格式列表

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    import glob
    import json

    # 1. 找出既有的 JSONL 結果檔
    pattern = os.path.join("results", f"eval_results_{timestamp}_run*.jsonl")
    existing_files = sorted(glob.glob(pattern))

    if not existing_files:
        print(f"❌ 找不到時間戳記 {timestamp} 的結果檔案")
        print(f"   搜尋路徑：{pattern}")
        return 1

    # 2. 解析已完成的題目（per run）
    completed: dict[str, set[str]] = {}  # run_file -> set of (file|question_id)
    for result_file in existing_files:
        run_key = os.path.basename(result_file)
        completed[run_key] = set()
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    file_id = record.get("file", "")
                    q_id = str(record.get("question_id", ""))
                    completed[run_key].add(f"{file_id}|{q_id}")
        except Exception as e:
            print(f"⚠️  讀取 {result_file} 時發生錯誤：{e}")

    total_completed = sum(len(v) for v in completed.values())
    print(f"📋 找到 {len(existing_files)} 個結果檔案，共 {total_completed} 筆已完成紀錄")

    # 3. 正常載入 config 並執行評測，帶入 resume 資訊
    try:
        runner = TwinkleEvalRunner(config_path)
        runner.initialize()
        # 覆蓋時間戳記為原始的，確保結果寫入同一組檔案
        runner.start_time = timestamp
        runner.run_evaluation(export_formats, completed_records=completed)
    except Exception as e:
        log_error(f"Resume 執行失敗: {e}")
        return 1

    return 0


def main() -> int:
    """主程式入口點

    處理命令列參數並執行相應的功能，包括查詢功能和主要評測流程

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    parser = create_cli_parser()
    args = parser.parse_args()

    # 處理查詢命令
    if args.list_llms:
        from .models import LLMFactory

        print("可用的 LLM 類型:")
        for llm_type in LLMFactory.get_available_types():
            print(f"  - {llm_type}")
        return 0

    if args.list_strategies:
        from .metrics import get_available_methods

        print("可用的評測策略:")
        for strategy in get_available_methods():
            print(f"  - {strategy}")
        return 0

    if args.list_exporters:
        print("可用的輸出格式:")
        for exporter in ResultsExporterFactory.get_available_types():
            print(f"  - {exporter}")
        return 0

    if args.version:
        from . import get_info

        info = get_info()
        print(f"🌟 {info['name']} v{info['version']}")
        print(f"作者: {info['author']}")
        print(f"授權: {info['license']}")
        print(f"網址: {info['url']}")
        return 0

    if args.init is not False:
        return create_default_config(template_name=args.init)

    # HuggingFace 資料集相關命令
    if args.download_dataset:
        try:
            from .datasets import download_huggingface_dataset

            download_huggingface_dataset(
                dataset_name=args.download_dataset,
                subset=args.dataset_subset,
                split=args.dataset_split,
                output_dir=args.output_dir,
            )
            print(f"✅ 資料集下載完成，已快取到 HuggingFace 目錄")
            return 0
        except Exception as e:
            print(f"❌ 下載資料集失敗: {e}")
            return 1

    if args.dataset_info:
        try:
            from .datasets import list_huggingface_dataset_info

            info = list_huggingface_dataset_info(
                dataset_name=args.dataset_info, subset=args.dataset_subset
            )
            print(f"📊 資料集資訊: {info['dataset_name']}")
            print(f"可用配置: {', '.join(info['configs'])}")
            for config, splits in info["splits"].items():
                print(f"  {config}: {', '.join(splits)}")
            return 0
        except Exception as e:
            print(f"❌ 獲取資料集資訊失敗: {e}")
            return 1

    # JSON 轉 HTML 命令
    if args.convert_to_html:
        try:
            return convert_json_to_html(args.convert_to_html)
        except Exception as e:
            print(f"❌ 轉換失敗: {e}")
            return 1

    # 分散式結果合併與 HuggingFace 上傳
    if args.finalize_results:
        try:
            from .runners.finalize import finalize_results
            return finalize_results(
                args.finalize_results,
                getattr(args, "hf_repo_id", None),
                getattr(args, "hf_variant", None),
            )
        except Exception as e:
            print(f"❌ 合併結果失敗: {e}")
            return 1

    # NIAH 資料集生成命令
    if args.generate_niah:
        try:
            from .datasets.niah import generate_niah_dataset

            # 驗證必要參數
            missing = []
            for param in ("haystack", "needle", "question", "answer"):
                if not getattr(args, param, None):
                    missing.append(f"--{param}")
            if missing:
                print(f"❌ --generate-niah 需要以下參數: {', '.join(missing)}")
                return 1

            context_lengths = [int(x.strip()) for x in args.context_lengths.split(",")]
            needle_depths = [float(x.strip()) for x in args.needle_depths.split(",")]

            output_path = generate_niah_dataset(
                haystack_path=args.haystack,
                needle=args.needle,
                question=args.question,
                answer=args.answer,
                context_lengths=context_lengths,
                needle_depths=needle_depths,
                output_dir=args.output_dir,
                language=args.niah_language,
                chars_per_token=args.chars_per_token,
                prompt_template=args.niah_prompt_template,
            )

            print(f"✅ NIAH 測試集已生成: {output_path}")
            return 0

        except FileNotFoundError as e:
            print(f"❌ {e}")
            return 1
        except Exception as e:
            print(f"❌ 生成 NIAH 測試集失敗: {e}")
            log_error(f"NIAH 生成錯誤: {e}")
            return 1

    # Benchmark 命令
    if args.benchmark:
        try:
            from .runners.benchmark import BenchmarkRunner, print_benchmark_summary, save_benchmark_results
            from .core.config import load_config

            config = load_config(args.config)
            runner = BenchmarkRunner(config)

            print(f"🚀 開始執行 LLM 效能基準測試")
            print(f"   提示文字: {args.benchmark_prompt}")
            print(f"   請求數量: {args.benchmark_requests}")
            print(f"   並發數量: {args.benchmark_concurrency}")
            if args.benchmark_rate:
                print(f"   請求速率: {args.benchmark_rate} 請求/秒")
            if args.benchmark_duration:
                print(f"   最大時間: {args.benchmark_duration} 秒")
            print("-" * 60)

            metrics = runner.run_benchmark(
                prompt=args.benchmark_prompt,
                num_requests=args.benchmark_requests,
                concurrent_requests=args.benchmark_concurrency,
                request_rate=args.benchmark_rate,
                duration=args.benchmark_duration,
            )

            # 顯示結果摘要
            print_benchmark_summary(metrics)

            # 儲存結果
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_results_{timestamp}.json"
            if "llm_instance" in config:
                del config["llm_instance"]
            if "evaluation_strategy_instance" in config:
                del config["evaluation_strategy_instance"]
            save_benchmark_results(metrics, output_path, config)

            return 0

        except Exception as e:
            print(f"❌ 基準測試失敗: {e}")
            log_error(f"基準測試執行錯誤: {e}")
            return 1

    # --validate：僅驗證設定檔與資料集格式
    if args.validate:
        return _handle_validate(args.config)

    # --dry-run：載入設定檔與資料集，顯示評測計畫但不呼叫 API
    if args.dry_run:
        return _handle_dry_run(args.config)

    # --resume：從指定時間戳記的中斷點繼續評測
    if args.resume:
        return _handle_resume(args.config, args.resume, args.export)

    # 執行評測
    try:
        runner = TwinkleEvalRunner(args.config)
        runner.initialize()
        runner.run_evaluation(args.export)
    except Exception as e:
        log_error(f"執行失敗: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
