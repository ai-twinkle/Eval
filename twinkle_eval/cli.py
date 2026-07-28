#!/usr/bin/env python3
"""
Twinkle Eval 命令列介面

提供 twinkle-eval 命令列工具的入口點，支援各種評測功能和配置選項。
"""

import sys
from typing import List, Optional

from .core.logger import log_error
from .main import main as main_func


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


if __name__ == "__main__":
    sys.exit(main())
