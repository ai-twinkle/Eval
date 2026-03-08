import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# 取得啟動時間
start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 建立 logs 資料夾（如果不存在）
logs_dir = "logs"
os.makedirs(logs_dir, exist_ok=True)

# 設定帶有時間戳的 Log 檔名
log_filename = os.path.join(logs_dir, f"evaluation_{start_time}.log")

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)


def log_info(message):
    logging.info(message)


def log_error(message):
    logging.error(message)


def log_warning(message):
    logging.warning(message)


def upload_logs_to_drive(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """上傳 log 檔案到 Google Drive

    Args:
        config: Google Drive 配置

    Returns:
        List[Dict[str, str]]: 上傳檔案的資訊列表
    """
    if not config or not config.get("google_drive"):
        log_warning("Google Drive 配置未設定，跳過 log 檔案上傳")
        return []

    try:
        from .google_services import GoogleDriveUploader

        drive_config = config["google_drive"]
        uploader = GoogleDriveUploader(drive_config)

        # 上傳 logs 資料夾中的所有 log 檔案
        uploaded_files = uploader.upload_log_files(logs_dir)

        if uploaded_files:
            log_info(f"成功上傳 {len(uploaded_files)} 個 log 檔案到 Google Drive")
        else:
            log_warning("沒有找到可上傳的 log 檔案")

        return uploaded_files

    except ImportError:
        log_error("Google Drive 相關套件未安裝，無法上傳 log 檔案")
        return []
    except Exception as e:
        log_error(f"上傳 log 檔案到 Google Drive 時發生錯誤: {e}")
        return []


def get_current_log_file() -> str:
    """取得當前的 log 檔案路徑

    Returns:
        str: 當前 log 檔案的完整路徑
    """
    return log_filename
