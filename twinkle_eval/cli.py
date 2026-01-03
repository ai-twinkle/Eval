#!/usr/bin/env python3
"""
Twinkle Eval 命令列介面

提供 twinkle-eval 命令列工具的入口點，支援各種評測功能和配置選項。
"""

import os
import sys
from typing import List, Optional

# 確保能夠正確匯入模組
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from twinkle_eval.evaluation_strategies import EvaluationStrategyFactory
from twinkle_eval.logger import log_error
from twinkle_eval.main import create_cli_parser
from twinkle_eval.main import main as main_func
from twinkle_eval.models import LLMFactory
from twinkle_eval.results_exporters import ResultsExporterFactory


def main(args: Optional[List[str]] = None) -> int:
    """
    Twinkle Eval 命令列工具主入口點

    支援的命令範例：
    - twinkle-eval --config config.yaml
    - twinkle-eval --export json csv html
    - twinkle-eval --list-llms
    - twinkle-eval --list-strategies

    Args:
        args: 命令列參數列表，如果為 None 則使用 sys.argv

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示失敗）
    """

    # 設定命令列參數
    if args is not None:
        original_argv = sys.argv[:]
        sys.argv = ["twinkle-eval"] + args

    try:
        # 呼叫主程式函數
        return main_func()
    except KeyboardInterrupt:
        print("\n⚠️  使用者中斷執行")
        return 130  # Unix 慣例：128 + SIGINT(2)
    except Exception as e:
        log_error(f"執行時發生未預期的錯誤: {e}")
        return 1
    finally:
        # 恢復原始的 sys.argv
        if args is not None:
            sys.argv = original_argv


def print_version():
    """列印版本資訊"""
    from twinkle_eval import __author__, __version__

    print(f"🌟 Twinkle Eval v{__version__}")
    print(f"作者: {__author__}")
    print("GitHub: https://github.com/ai-twinkle/Eval")


def print_help():
    """列印詳細幫助資訊"""
    parser = create_cli_parser()
    parser.print_help()

    print("\n🚀 更多使用範例:")
    print("  # 使用預設配置執行評測")
    print("  twinkle-eval")
    print()
    print("  # 使用自定義配置檔案")
    print("  twinkle-eval --config my_config.yaml")
    print()
    print("  # 同時輸出多種格式")
    print("  twinkle-eval --export json csv html")
    print()
    print("  # 查看支援的功能")
    print("  twinkle-eval --list-llms")
    print("  twinkle-eval --list-strategies")
    print("  twinkle-eval --list-exporters")
    print()
    print("📖 詳細文件: https://github.com/ai-twinkle/Eval#readme")


def cli_list_llms():
    """列出支援的 LLM 類型"""
    print("🤖 支援的 LLM 類型:")
    for llm_type in LLMFactory.get_available_types():
        print(f"  - {llm_type}")


def cli_list_strategies():
    """列出支援的評測策略"""
    print("🎯 支援的評測策略:")
    for strategy in EvaluationStrategyFactory.get_available_types():
        print(f"  - {strategy}")


def cli_list_exporters():
    """列出支援的輸出格式"""
    print("📊 支援的輸出格式:")
    for exporter in ResultsExporterFactory.get_available_types():
        print(f"  - {exporter}")


if __name__ == "__main__":
    sys.exit(main())
