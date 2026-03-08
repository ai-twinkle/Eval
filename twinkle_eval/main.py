import argparse
import copy
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from twinkle_eval.exceptions import ConfigurationError

from .config import load_config
from .dataset import find_all_evaluation_files
from .evaluators import Evaluator
from .logger import log_error, log_info
from .results_exporters import ResultsExporterFactory


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


def create_default_config(output_path: str = "config.yaml") -> int:
    """創建預設配置檔案

    Args:
        output_path: 輸出檔案路徑，預設為 config.yaml

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    import shutil

    try:
        # 檢查檔案是否已存在
        if os.path.exists(output_path):
            response = input(f"⚠️  檔案 '{output_path}' 已存在，是否覆蓋？(y/N): ")
            if response.lower() not in ["y", "yes", "是"]:
                print("❌ 取消創建配置檔案")
                return 1

        # 找到範本檔案
        template_path = os.path.join(os.path.dirname(__file__), "config.template.yaml")

        if not os.path.exists(template_path):
            print(f"❌ 找不到配置範本檔案: {template_path}")
            return 1

        # 複製範本檔案
        shutil.copy2(template_path, output_path)

        print(f"✅ 配置檔案已創建: {output_path}")
        print()
        print("📝 接下來請編輯配置檔案，設定：")
        print("  1. LLM API 設定 (base_url, api_key)")
        print("  2. 模型名稱 (model.name)")
        print("  3. 資料集路徑 (evaluation.dataset_paths)")
        print()
        print("💡 編輯完成後，使用以下命令開始評測：")
        print(f"   twinkle-eval --config {output_path}")

        return 0

    except Exception as e:
        print(f"❌ 創建配置檔案時發生錯誤: {e}")
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
            self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成時間標記
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
        if "evaluation_strategy_instance" in save_config:
            del save_config["evaluation_strategy_instance"]

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

    def _evaluate_dataset(self, dataset_path: str, evaluator: Evaluator) -> Dict[str, Any]:
        """評測單一資料集

        對指定資料集中的所有檔案進行評測，支援多次執行並統計結果

        Args:
            dataset_path: 資料集路徑
            evaluator: 評測器實例

        Returns:
            Dict[str, Any]: 資料集評測結果，包含準確率統計和詳細結果
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        log_info(f"開始評測資料集: {dataset_path}")

        all_files = find_all_evaluation_files(dataset_path)  # 尋找所有評測檔案
        repeat_runs = self.config["evaluation"].get("repeat_runs", 1)  # 重複執行次數
        prompt_map = self.config["evaluation"].get("datasets_prompt_map", {})  # 資料集語言對應表
        dataset_lang = prompt_map.get(dataset_path, "zh")  # 當前資料集的語言，預設為中文

        results = []  # 儲存所有檔案的評測結果

        for idx, file_path in enumerate(all_files):
            file_accuracies = []  # 當前檔案的準確率結果
            file_results = []  # 當前檔案的詳細結果

            # 對當前檔案進行多次評測
            for run in range(repeat_runs):
                try:
                    file_path_result, accuracy, result_path = evaluator.evaluate_file(
                        file_path, f"{self.start_time}_run{run}", dataset_lang, run_index=run
                    )
                    file_accuracies.append(accuracy)
                    file_results.append((file_path_result, accuracy, result_path))
                except Exception as e:
                    log_error(f"評測檔案 {file_path} 失敗: {e}")
                    continue

            # 為當前檔案計算統計數據
            if file_accuracies:
                mean_accuracy = np.mean(file_accuracies)  # 平均準確率
                std_accuracy = np.std(file_accuracies) if len(file_accuracies) > 1 else 0  # 標準差

                results.append(
                    {
                        "file": file_path,
                        "accuracy_mean": mean_accuracy,
                        "accuracy_std": std_accuracy,
                        "individual_runs": {
                            "accuracies": file_accuracies,
                            "results": [r[2] for r in file_results],
                        },
                    }
                )

            # 進度指示器
            progress = (idx + 1) / len(all_files) * 100
            print(f"\r已執行 {progress:.1f}% ({idx + 1}/{len(all_files)}) ", end="")

        print()  # 進度完成後換行

        # 計算資料集統計數據
        dataset_avg_accuracy = (
            np.mean([r["accuracy_mean"] for r in results]) if results else 0
        )  # 資料集平均準確率
        dataset_avg_std = (
            np.mean([r["accuracy_std"] for r in results]) if results else 0
        )  # 資料集平均標準差

        return {
            "results": results,
            "average_accuracy": dataset_avg_accuracy,
            "average_std": dataset_avg_std,
        }

    def run_evaluation(self, export_formats: Optional[List[str]] = None) -> str:
        """執行完整的評測流程

        這是主要的評測入口點，包含以下步驟：
        1. 建立評測器
        2. 對所有資料集進行評測
        3. 統計和輸出結果

        Args:
            export_formats: 輸出格式清單，預設為 ["json"]

        Returns:
            str: 主要結果檔案路徑
        """
        if self.config is None:
            raise ConfigurationError("配置未載入")

        if export_formats is None:
            export_formats = ["json"]  # 預設輸出格式

        dataset_paths = self._get_dataset_paths()  # 取得資料集路徑
        dataset_results = {}  # 儲存所有資料集的結果

        # 建立評測器
        llm_instance = self.config["llm_instance"]
        evaluation_strategy_instance = self.config["evaluation_strategy_instance"]
        evaluator = Evaluator(llm_instance, evaluation_strategy_instance, self.config)

        # 逐一評測每個資料集
        for dataset_path in dataset_paths:
            try:
                dataset_result = self._evaluate_dataset(dataset_path, evaluator)
                dataset_results[dataset_path] = dataset_result

                message = (
                    f"資料集 {dataset_path} 評測完成，"
                    f"平均正確率: {dataset_result['average_accuracy']:.2%} "
                    f"(±{dataset_result['average_std']:.2%})"
                )
                print(message)
                log_info(message)

            except Exception as e:
                log_error(f"資料集 {dataset_path} 評測失敗: {e}")
                continue

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
                from .google_services import GoogleDriveUploader

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

    parser.add_argument("--init", action="store_true", help="創建預設配置檔案")

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

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # Upload command
    upload_parser = subparsers.add_parser("upload", help="上傳結果至 S3")
    upload_parser.add_argument("results_dir", help="要上傳的結果目錄路徑")
    upload_parser.add_argument("--bucket", required=True, help="S3 Bucket 名稱")
    upload_parser.add_argument("--key", help="S3 Key Prefix (資料夾路徑)")
    upload_parser.add_argument("--access-key", help="AWS Access Key ID")
    upload_parser.add_argument("--secret-key", help="AWS Secret Access Key")
    upload_parser.add_argument("--endpoint-url", help="S3 Endpoint URL")
    upload_parser.add_argument("--region", help="AWS Region")
    upload_parser.add_argument(
        "--no-verify-ssl", action="store_true", help="禁用 SSL 憑證驗證 (用於自簽憑證或開發環境)"
    )

    return parser


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
        from .evaluation_strategies import EvaluationStrategyFactory

        print("可用的評測策略:")
        for strategy in EvaluationStrategyFactory.get_available_types():
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

    if args.init:
        return create_default_config()

    # HuggingFace 資料集相關命令
    if args.download_dataset:
        try:
            from .dataset import download_huggingface_dataset

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
            from .dataset import list_huggingface_dataset_info

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

    # Upload command
    if hasattr(args, "command") and args.command == "upload":
        try:
            from .s3_uploader import S3Uploader

            uploader = S3Uploader(
                aws_access_key_id=args.access_key,
                aws_secret_access_key=args.secret_key,
                endpoint_url=args.endpoint_url,
                region_name=args.region,
                verify_ssl=not args.no_verify_ssl,
            )

            print(f"🚀 開始上傳至 S3")
            print(f"   本地目錄: {args.results_dir}")
            print(f"   Bucket: {args.bucket}")
            print(f"   Key Prefix: {args.key}")
            if args.endpoint_url:
                print(f"   Endpoint: {args.endpoint_url}")
            if args.no_verify_ssl:
                print(f"   ⚠️  SSL 驗證: 已停用")
            print("-" * 60)

            success = uploader.upload_directory(
                local_dir=args.results_dir, bucket=args.bucket, prefix=args.key
            )
            return 0 if success else 1
        except Exception as e:
            print(f"❌ 上傳失敗: {e}")
            log_error(f"S3 上傳錯誤: {e}")
            return 1

    # Benchmark 命令
    if args.benchmark:
        try:
            from .benchmark import BenchmarkRunner, print_benchmark_summary, save_benchmark_results
            from .config import load_config

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
