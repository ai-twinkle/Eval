import argparse
import copy
import os
import time

from .core.config import load_config
from .core.logger import log_error
from .datasets import find_all_evaluation_files
from .exporters import ResultsExporterFactory


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


def convert_jsonl_to_excel(jsonl_path: str) -> int:
    """將逐題結果 JSONL 檔案（eval_results_*.jsonl）轉換為 Excel 格式。

    Args:
        jsonl_path: JSONL 結果檔案的路徑

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """
    import json

    try:
        if not os.path.exists(jsonl_path):
            print(f"❌ 檔案不存在: {jsonl_path}")
            return 1

        try:
            import pandas as pd
            from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
        except ImportError:
            print("❌ 缺少 Excel 轉換所需套件，請執行: pip install twinkle-eval[excel]")
            return 1

        rows: list[dict] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

        if not rows:
            print(f"❌ 檔案沒有任何資料列: {jsonl_path}")
            return 1

        # Excel 單一儲存格上限 32767 字元；控制字元 openpyxl 會拒寫
        excel_cell_limit = 32767
        truncated = 0

        def _to_cell(value: object) -> object:
            nonlocal truncated
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            if isinstance(value, str):
                value = ILLEGAL_CHARACTERS_RE.sub("", value)
                if len(value) > excel_cell_limit:
                    value = value[: excel_cell_limit - 1] + "…"
                    truncated += 1
            return value

        rows = [{k: _to_cell(v) for k, v in row.items()} for row in rows]

        # 常用欄位排前面，其餘欄位（各 benchmark 的額外指標）依出現順序附加在後
        preferred = [
            "file",
            "question_id",
            "sample_id",
            "question",
            "correct_answer",
            "predicted_answer",
            "is_correct",
            "llm_output",
            "llm_reasoning_output",
            "usage_prompt_tokens",
            "usage_completion_tokens",
            "usage_total_tokens",
        ]
        all_keys: list[str] = []
        for row in rows:
            for k in row:
                if k not in all_keys:
                    all_keys.append(k)
        columns = [k for k in preferred if k in all_keys] + [
            k for k in all_keys if k not in preferred
        ]

        df = pd.DataFrame(rows, columns=columns)
        output_path = os.path.splitext(jsonl_path)[0] + ".xlsx"
        df.to_excel(output_path, index=False, engine="openpyxl")

        print(f"✅ 成功轉換為 Excel: {output_path}（{len(df)} 列）")
        if truncated:
            print(f"⚠️  有 {truncated} 個儲存格超過 Excel 上限（{excel_cell_limit} 字元）已截斷")
        return 0

    except json.JSONDecodeError as e:
        print(f"❌ JSONL 檔案格式錯誤: {e}")
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


# TwinkleEvalRunner 的唯一實作位於 runners/standard.py；
# 此處 re-export 以維持 `from twinkle_eval.main import TwinkleEvalRunner` 的相容性
from .runners.standard import TwinkleEvalRunner  # noqa: E402

__all__ = ["TwinkleEvalRunner", "create_cli_parser", "main"]


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
  twinkle-eval --convert-to-excel eval_results_20240101_120000_run0.jsonl  # 將逐題 JSONL 轉換為 Excel

效能基準測試:
  twinkle-eval --benchmark                           # 執行預設的基準測試
  twinkle-eval --benchmark --benchmark-requests 50  # 執行 50 個請求的測試
  twinkle-eval --benchmark --benchmark-concurrency 5 --benchmark-rate 2  # 5 並發，2 請求/秒

評測資料集下載:
  twinkle-eval --download-dataset list               # 列出所有可下載的 benchmark
  twinkle-eval --download-dataset mmlu gsm8k bbh     # 下載指定 benchmark
  twinkle-eval --download-dataset all                # 下載全部 benchmark
  twinkle-eval --download-dataset cais/mmlu          # 直接指定 HuggingFace ID
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
        help="驗證設定檔與資料集格式，並對 API 端點做一次試打確認可連線",
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
        nargs="+",
        metavar="NAME",
        help=(
            "下載評測資料集。支援 benchmark 短名稱（如 mmlu, gsm8k）、"
            "'all'（下載全部）、'list'（列出可用資料集）、或 HuggingFace ID（如 cais/mmlu）"
        ),
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
        "--convert-to-excel",
        metavar="JSONL_FILE",
        help="將逐題結果 JSONL 檔案轉換為 Excel 格式（需安裝 twinkle-eval[excel]）",
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

    # 4. 對 API 端點做一次試打，及早發現連線 / 金鑰 / 模型名稱錯誤
    if not errors:
        try:
            live_config = load_config(config_path)
            llm_type = live_config["llm_api"].get("type", "openai")
            if llm_type == "whisper":
                # whisper 後端的 call() 需要真實音檔，無法以文字試打
                print("⏭️  whisper 後端不支援文字試打，略過 API 連線測試")
            else:
                llm_instance = live_config["llm_instance"]
                # max_tokens 至少 16：Responses API 的 max_output_tokens 下限為 16
                response = llm_instance.call(
                    "ping",
                    system_prompt_enabled=False,
                    model_overrides={"max_tokens": 16},
                )
                model_name = getattr(response, "model", "") or live_config["model"]["name"]
                print(f"✅ API 端點試打成功（模型: {model_name}）")
        except Exception as e:
            errors.append(f"API 端點試打失敗：{e}")

    # 5. 結果
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

    # 2. 解析已完成的題目（per run）：file|question_id -> is_correct
    completed: dict[str, dict[str, bool]] = {}
    legacy_rows = 0  # 舊版結果檔沒有 file 欄位，無法比對來源檔案
    for result_file in existing_files:
        run_key = os.path.basename(result_file)
        completed[run_key] = {}
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    file_id = record.get("file", "")
                    if not file_id:
                        legacy_rows += 1
                        continue
                    q_id = str(record.get("question_id", ""))
                    completed[run_key][f"{file_id}|{q_id}"] = bool(record.get("is_correct", False))
        except Exception as e:
            print(f"⚠️  讀取 {result_file} 時發生錯誤：{e}")

    total_completed = sum(len(v) for v in completed.values())
    print(f"📋 找到 {len(existing_files)} 個結果檔案，共 {total_completed} 筆已完成紀錄")
    if legacy_rows:
        print(
            f"⚠️  有 {legacy_rows} 筆舊格式紀錄缺少 file 欄位，無法比對，"
            f"相關題目將重新評測（可能產生重複結果列）"
        )

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
        from .benchmarks import BENCHMARK_REGISTRY, download_benchmarks, list_benchmarks

        names = args.download_dataset

        # twinkle-eval --download-dataset list
        if names == ["list"]:
            list_benchmarks()
            return 0

        # 區分 registry 短名稱 vs HuggingFace ID
        registry_names = []
        hf_ids = []
        for name in names:
            if name == "all" or name in BENCHMARK_REGISTRY:
                registry_names.append(name)
            elif "/" in name:
                hf_ids.append(name)
            else:
                # 不在 registry 也不含 /，視為無效名稱
                print(f"❌ 找不到資料集：{name}")
                print("   使用 --download-dataset list 查看可用的 benchmark 名稱")
                print("   或使用 HuggingFace ID 格式（如 cais/mmlu）")
                return 1

        result = 0

        # 下載 registry 中的 benchmark
        if registry_names:
            result = download_benchmarks(registry_names, output_dir=args.output_dir)

        # 下載直接指定的 HuggingFace ID（保留向下相容）
        if hf_ids:
            from .datasets import download_huggingface_dataset

            for hf_id in hf_ids:
                try:
                    download_huggingface_dataset(
                        dataset_name=hf_id,
                        subset=args.dataset_subset,
                        split=args.dataset_split,
                        output_dir=args.output_dir,
                    )
                    print(f"✅ {hf_id} 下載完成")
                except Exception as e:
                    print(f"❌ 下載 {hf_id} 失敗: {e}")
                    result = 1

        return result

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

    # JSONL 轉 Excel 命令
    if args.convert_to_excel:
        try:
            return convert_jsonl_to_excel(args.convert_to_excel)
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
            from .core.config import load_config
            from .runners.benchmark import (
                BenchmarkRunner,
                print_benchmark_summary,
                save_benchmark_results,
            )

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

            # 儲存結果（排除不可序列化實例並移除 API 金鑰，避免敏感資訊寫入結果檔）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = f"benchmark_results_{timestamp}.json"
            safe_config = copy.deepcopy(
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
            save_benchmark_results(metrics, output_path, safe_config)

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
