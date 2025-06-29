import logging
import os
from datetime import datetime

# 取得啟動時間
start_time = datetime.now().strftime("%Y%m%d_%H%M")

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
