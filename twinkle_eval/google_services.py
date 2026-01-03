import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from twinkle_eval.exceptions import ConfigurationError
from twinkle_eval.logger import log_error, log_info
from twinkle_eval.results_exporters import ResultsExporter


class GoogleDriveUploader:
    """Google Drive 檔案上傳器"""

    # Google Drive API 權限範圍
    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self, config: Dict[str, Any]):
        """初始化 Google Drive 上傳器

        Args:
            config: Google Drive 配置字典
        """
        self.config = config
        self.service = None
        self._authenticate()

    def _authenticate(self):
        """進行 Google Drive API 驗證"""
        auth_method = self.config.get("auth_method", "service_account")

        if auth_method == "service_account":
            creds = self._authenticate_service_account()
        else:
            creds = self._authenticate_oauth()

        self.service = build("drive", "v3", credentials=creds)
        log_info("Google Drive API 驗證成功")

    def _authenticate_service_account(self):
        """使用 Service Account 進行驗證"""
        credentials_file = self.config.get("credentials_file")

        if not credentials_file:
            raise ConfigurationError("Google Drive Service Account credentials_file 未設定")

        if not os.path.exists(credentials_file):
            raise ConfigurationError(f"Service Account credentials 檔案不存在: {credentials_file}")

        try:
            creds = service_account.Credentials.from_service_account_file(
                credentials_file, scopes=self.SCOPES
            )
            log_info("使用 Service Account 驗證成功")
            return creds
        except Exception as e:
            raise ConfigurationError(f"Service Account 驗證失敗: {e}")

    def _authenticate_oauth(self):
        """使用 OAuth 流程進行驗證"""
        creds = None
        token_file = self.config.get("token_file", "token.json")
        credentials_file = self.config.get("credentials_file")

        if not credentials_file:
            raise ConfigurationError("Google Drive credentials_file 未設定")

        # 檢查是否有已儲存的憑證
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)

        # 如果沒有有效憑證，進行 OAuth 流程
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(credentials_file):
                    raise ConfigurationError(
                        f"Google Drive credentials 檔案不存在: {credentials_file}"
                    )

                flow = InstalledAppFlow.from_client_secrets_file(credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)

            # 儲存憑證以供下次使用
            with open(token_file, "w") as token:
                token.write(creds.to_json())

        return creds

    def upload_file(self, file_path: str, folder_id: Optional[str] = None) -> str:
        """上傳檔案到 Google Drive

        Args:
            file_path: 要上傳的檔案路徑
            folder_id: 目標資料夾 ID（可選）

        Returns:
            str: 上傳檔案的 Google Drive ID

        Raises:
            ConfigurationError: 當上傳失敗時拋出
        """
        try:
            if not os.path.exists(file_path):
                raise ConfigurationError(f"檔案不存在: {file_path}")

            file_name = os.path.basename(file_path)

            # 準備檔案元資料
            file_metadata = {"name": file_name}
            if folder_id:
                file_metadata["parents"] = [folder_id]

            # 準備檔案媒體
            media = MediaFileUpload(file_path, resumable=True)

            # 上傳檔案（支援 Shared Drive）
            file = (
                self.service.files()
                .create(
                    body=file_metadata,
                    media_body=media,
                    fields="id,webViewLink",
                    supportsAllDrives=True,
                )
                .execute()
            )

            file_id = file.get("id")
            web_view_link = file.get("webViewLink")

            log_info(f"檔案已成功上傳到 Google Drive: {file_name} (ID: {file_id})")
            log_info(f"檔案連結: {web_view_link}")

            return file_id

        except Exception as e:
            error_msg = f"Google Drive 上傳失敗: {e}"
            log_error(error_msg)
            raise ConfigurationError(error_msg) from e

    def upload_log_files(self, logs_directory: str = "logs") -> List[Dict[str, str]]:
        """批次上傳 log 檔案 - 只上傳最新的檔案

        Args:
            logs_directory: logs 資料夾路徑

        Returns:
            List[Dict[str, str]]: 上傳檔案的資訊列表
        """
        uploaded_files = []

        if not os.path.exists(logs_directory):
            log_error(f"Logs 資料夾不存在: {logs_directory}")
            return uploaded_files

        # 找出最新的 log 檔案
        log_files = []
        for file_name in os.listdir(logs_directory):
            if file_name.endswith(".log"):
                file_path = os.path.join(logs_directory, file_name)
                file_stat = os.stat(file_path)
                log_files.append(
                    {"name": file_name, "path": file_path, "mtime": file_stat.st_mtime}
                )

        if not log_files:
            log_info("沒有找到 log 檔案")
            return uploaded_files

        # 按修改時間排序，取最新的檔案
        latest_log = sorted(log_files, key=lambda x: x["mtime"], reverse=True)[0]

        folder_id = self.config.get("log_folder_id")

        try:
            log_info(f"上傳最新的 log 檔案: {latest_log['name']}")
            file_id = self.upload_file(latest_log["path"], folder_id)
            uploaded_files.append(
                {
                    "file_name": latest_log["name"],
                    "file_path": latest_log["path"],
                    "drive_id": file_id,
                }
            )
        except ConfigurationError as e:
            log_error(f"上傳 {latest_log['name']} 失敗: {e}")

        return uploaded_files

    def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> str:
        """在 Google Drive 中建立資料夾

        Args:
            folder_name: 資料夾名稱
            parent_folder_id: 父資料夾 ID（可選）

        Returns:
            str: 新建立的資料夾 ID

        Raises:
            ConfigurationError: 當建立資料夾失敗時拋出
        """
        try:
            # 準備資料夾元資料
            folder_metadata = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }

            if parent_folder_id:
                folder_metadata["parents"] = [parent_folder_id]

            # 建立資料夾（支援 Shared Drive）
            folder = (
                self.service.files()
                .create(body=folder_metadata, fields="id,name,webViewLink", supportsAllDrives=True)
                .execute()
            )

            folder_id = folder.get("id")
            web_view_link = folder.get("webViewLink")

            log_info(f"資料夾已成功建立: {folder_name} (ID: {folder_id})")
            log_info(f"資料夾連結: {web_view_link}")

            return folder_id

        except Exception as e:
            error_msg = f"Google Drive 建立資料夾失敗: {e}"
            log_error(error_msg)
            raise ConfigurationError(error_msg) from e

    def upload_latest_files(
        self,
        start_time: str,
        logs_directory: str = "logs",
        results_directory: str = "results",
    ) -> Dict[str, Any]:
        """上傳最新的 log、results 和 eval_results 檔案到新建立的資料夾

        Args:
            logs_directory: logs 資料夾路徑
            results_directory: results 資料夾路徑（eval_results_*.json 檔案也在此資料夾中）
            start_time: 評測開始時間標記，用於篩選對應的檔案

        Returns:
            Dict[str, Any]: 上傳結果資訊
        """
        upload_info = {
            "timestamp": datetime.now().isoformat(),
            "folder_id": None,
            "folder_name": None,
            "uploaded_files": [],
        }

        try:
            # 建立以時間戳命名的資料夾
            folder_name = f"Eval_{start_time}"

            parent_folder_id = self.config.get("log_folder_id")
            new_folder_id = self.create_folder(folder_name, parent_folder_id)

            upload_info["folder_id"] = new_folder_id
            upload_info["folder_name"] = folder_name

            # 上傳符合 key 的 log 檔案
            if os.path.exists(logs_directory):
                log_files = []
                for file_name in os.listdir(logs_directory):
                    if file_name.endswith(".log"):
                        # 只處理包含特定 key 的檔案
                        if start_time in file_name:
                            file_path = os.path.join(logs_directory, file_name)
                            file_stat = os.stat(file_path)
                            log_files.append(
                                {"name": file_name, "path": file_path, "mtime": file_stat.st_mtime}
                            )

                if log_files:
                    # 上傳符合 key 的 log 檔案
                    for log_file in log_files:
                        try:
                            log_info(f"上傳 log 檔案: {log_file['name']} 到新資料夾")
                            file_id = self.upload_file(log_file["path"], new_folder_id)
                            upload_info["uploaded_files"].append(
                                {
                                    "type": "log",
                                    "file_name": log_file["name"],
                                    "file_path": log_file["path"],
                                    "drive_id": file_id,
                                }
                            )
                        except ConfigurationError as e:
                            log_error(f"上傳 log 檔案失敗: {e}")

            # 上傳 results 資料夾中特定 key 的檔案
            if os.path.exists(results_directory):
                result_files = []
                eval_result_files = []

                for file_name in os.listdir(results_directory):
                    if file_name.endswith((".json", ".html", ".csv", ".xlsx", ".jsonl")):
                        # 只處理包含特定 key 的檔案
                        if start_time in file_name:
                            file_path = os.path.join(results_directory, file_name)
                            file_stat = os.stat(file_path)
                            file_info = {
                                "name": file_name,
                                "path": file_path,
                                "mtime": file_stat.st_mtime,
                            }

                            # 區分 eval_results_*.json 和其他 results 檔案
                            if file_name.startswith("eval_results_"):
                                eval_result_files.append(file_info)
                            else:
                                result_files.append(file_info)

                # 上傳符合 key 的一般 results 檔案
                if result_files:
                    for file_info in result_files:
                        try:
                            log_info(f"上傳 results 檔案: {file_info['name']} 到新資料夾")
                            file_id = self.upload_file(file_info["path"], new_folder_id)
                            upload_info["uploaded_files"].append(
                                {
                                    "type": "results",
                                    "file_name": file_info["name"],
                                    "file_path": file_info["path"],
                                    "drive_id": file_id,
                                }
                            )
                        except ConfigurationError as e:
                            log_error(f"上傳 results 檔案失敗: {e}")

                # 上傳符合 key 的 eval_results_*.jsonl 檔案
                if eval_result_files:
                    for file_info in eval_result_files:
                        try:
                            log_info(f"上傳 eval_results 檔案: {file_info['name']} 到新資料夾")
                            file_id = self.upload_file(file_info["path"], new_folder_id)
                            upload_info["uploaded_files"].append(
                                {
                                    "type": "eval_results",
                                    "file_name": file_info["name"],
                                    "file_path": file_info["path"],
                                    "drive_id": file_id,
                                }
                            )
                        except ConfigurationError as e:
                            log_error(f"上傳 eval_results 檔案失敗: {e}")

            log_info(
                f"成功上傳 {len(upload_info['uploaded_files'])} 個檔案到新資料夾: {folder_name}"
            )
            return upload_info

        except Exception as e:
            log_error(f"上傳最新檔案失敗: {e}")
            upload_info["error"] = str(e)
            return upload_info


class GoogleSheetsService:
    """Google Sheets 服務類別"""

    # Google Sheets API 權限範圍
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

    def __init__(self, config: Dict[str, Any]):
        """初始化 Google Sheets 服務

        Args:
            config: Google Sheets 配置字典
        """
        self.config = config
        self.service = None
        self._authenticate()

    def _authenticate(self):
        """進行 Google Sheets API 驗證"""
        auth_method = self.config.get("auth_method", "service_account")

        if auth_method == "service_account":
            creds = self._authenticate_service_account()
        else:
            creds = self._authenticate_oauth()

        self.service = build("sheets", "v4", credentials=creds)
        log_info("Google Sheets API 驗證成功")

    def _authenticate_service_account(self):
        """使用 Service Account 進行驗證"""
        credentials_file = self.config.get("credentials_file")

        if not credentials_file:
            raise ConfigurationError("Google Sheets Service Account credentials_file 未設定")

        if not os.path.exists(credentials_file):
            raise ConfigurationError(f"Service Account credentials 檔案不存在: {credentials_file}")

        try:
            creds = service_account.Credentials.from_service_account_file(
                credentials_file, scopes=self.SCOPES
            )
            log_info("使用 Service Account 驗證成功")
            return creds
        except Exception as e:
            raise ConfigurationError(f"Service Account 驗證失敗: {e}")

    def _authenticate_oauth(self):
        """使用 OAuth 流程進行驗證"""
        creds = None
        token_file = self.config.get("token_file", "sheets_token.json")
        credentials_file = self.config.get("credentials_file")

        if not credentials_file:
            raise ConfigurationError("Google Sheets credentials_file 未設定")

        # 檢查是否有已儲存的憑證
        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)

        # 如果沒有有效憑證，進行 OAuth 流程
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(credentials_file):
                    raise ConfigurationError(
                        f"Google Sheets credentials 檔案不存在: {credentials_file}"
                    )

                flow = InstalledAppFlow.from_client_secrets_file(credentials_file, self.SCOPES)
                creds = flow.run_local_server(port=0)

            # 儲存憑證以供下次使用
            with open(token_file, "w") as token:
                token.write(creds.to_json())

        return creds

    def append_results_to_sheet(
        self, spreadsheet_id: str, sheet_name: str, results: Dict[str, Any]
    ) -> bool:
        """將評測結果新增到指定的 Google Sheet

        Args:
            spreadsheet_id: Google Sheets 試算表 ID
            sheet_name: 工作表名稱
            results: 評測結果字典

        Returns:
            bool: 是否成功新增
        """
        try:
            # 確保表格有 header
            self._ensure_header_exists(spreadsheet_id, sheet_name)

            # 準備要寫入的資料
            rows = self._prepare_sheet_data(results)

            if not rows:
                log_info("沒有資料需要寫入 Google Sheets")
                return True

            # 指定範圍（從 A 欄開始）
            range_name = f"{sheet_name}!A:DD"  # 擴展到 DD 欄

            # 準備請求體
            body = {"values": rows}

            # 寫入資料
            result = (
                self.service.spreadsheets()
                .values()
                .append(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption="RAW",
                    body=body,
                )
                .execute()
            )

            updates = result.get("updates", {})
            updated_cells = updates.get("updatedCells", 0)

            log_info(
                f"已成功將 {len(rows)} 列資料寫入 Google Sheets，更新了 {updated_cells} 個儲存格"
            )

            return True

        except Exception as e:
            error_msg = f"Google Sheets 寫入失敗: {e}"
            log_error(error_msg)
            return False

    def _ensure_header_exists(self, spreadsheet_id: str, sheet_name: str):
        """確保表格有 header，如果沒有則建立

        Args:
            spreadsheet_id: Google Sheets 試算表 ID
            sheet_name: 工作表名稱
        """
        try:
            # 讀取第一列來檢查是否有 header
            range_name = f"{sheet_name}!A1:DD1"
            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_name)
                .execute()
            )

            values = result.get("values", [])

            # 如果第一列為空或不是完整的 header，則建立 header
            if not values or len(values[0]) < 10:  # 如果沒有資料或欄位數少於 10
                log_info("建立 Google Sheets Header...")
                self._create_header(spreadsheet_id, sheet_name)
            else:
                log_info("Google Sheets Header 已存在")

        except Exception as e:
            log_error(f"檢查 Header 失敗，嘗試建立新的 Header: {e}")
            self._create_header(spreadsheet_id, sheet_name)

    def _create_header(self, spreadsheet_id: str, sheet_name: str):
        """建立表格 header

        Args:
            spreadsheet_id: Google Sheets 試算表 ID
            sheet_name: 工作表名稱
        """
        # 完整的 header 定義
        header = [
            "時間戳記",  # A
            # LLM API 配置
            "API_基礎網址",  # B
            "API_金鑰",  # C
            "API_速率限制",  # D
            "最大重試次數",  # E
            "超時時間",  # F
            "SSL驗證設定",  # G
            # Model 配置
            "模型名稱",  # H
            "溫度參數",  # I
            "Top_P參數",  # J
            "最大Token數",  # K
            "頻率懲罰",  # L
            "存在懲罰",  # M
            # Environment 配置
            "GPU型號",  # N
            "GPU數量",  # O
            "GPU記憶體GB",  # P
            "CUDA版本",  # Q
            "驅動版本",  # R
            "TP大小",  # S
            "PP大小",  # T
            "框架",  # U
            "Python版本",  # V
            "PyTorch版本",  # W
            "節點數量",  # X
            # 資料集結果
            "資料集路徑",  # Y
            "平均準確率",  # Z
            "標準差",  # AA
            "檔案名稱",  # BB
            "準確率均值",  # CC
            "準確率標準差",  # DD
        ]

        try:
            # 清除第一列並寫入新的 header
            range_name = f"{sheet_name}!A1:DD1"
            body = {"values": [header]}

            self.service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id, range=range_name, valueInputOption="RAW", body=body
            ).execute()

            log_info(f"成功建立 Google Sheets Header，共 {len(header)} 個欄位")

        except Exception as e:
            log_error(f"建立 Header 失敗: {e}")
            raise

    def _prepare_sheet_data(self, results: Dict[str, Any]) -> List[List[str]]:
        """準備要寫入 Google Sheets 的資料

        Args:
            results: 評測結果字典

        Returns:
            List[List[str]]: 二維列表，每個子列表代表一列資料
        """
        rows = []

        timestamp = results.get("timestamp", "")
        config = results.get("config", {})
        environment = config.get("environment", {})
        llm_api = config.get("llm_api", {})
        model = config.get("model", {})

        # 擴展的基本資訊，包含完整配置
        base_info = [
            timestamp,  # A: 時間戳記
            # LLM API 配置
            llm_api.get("base_url", ""),  # B: API 基礎網址
            (
                llm_api.get("api_key", "")[:10] + "..." if llm_api.get("api_key") else ""
            ),  # C: API 金鑰（脫敏）
            str(llm_api.get("api_rate_limit", "")),  # D: API 速率限制
            str(llm_api.get("max_retries", "")),  # E: 最大重試次數
            str(llm_api.get("timeout", "")),  # F: 超時時間
            str(llm_api.get("disable_ssl_verify", "")),  # G: SSL 驗證設定
            # Model 配置
            model.get("name", ""),  # H: 模型名稱
            str(model.get("temperature", "")),  # I: 溫度參數
            str(model.get("top_p", "")),  # J: Top-p 參數
            str(model.get("max_tokens", "")),  # K: 最大 Token 數
            str(model.get("frequency_penalty", "")),  # L: 頻率懲罰
            str(model.get("presence_penalty", "")),  # M: 存在懲罰
            # Environment 配置
            environment.get("gpu_info", {}).get("model", ""),  # N: GPU 型號
            str(environment.get("gpu_info", {}).get("count", "")),  # O: GPU 數量
            str(environment.get("gpu_info", {}).get("memory_gb", "")),  # P: GPU 記憶體
            environment.get("gpu_info", {}).get("cuda_version", ""),  # Q: CUDA 版本
            environment.get("gpu_info", {}).get("driver_version", ""),  # R: 驅動版本
            str(environment.get("parallel_config", {}).get("tp_size", "")),  # S: TP 大小
            str(environment.get("parallel_config", {}).get("pp_size", "")),  # T: PP 大小
            environment.get("system_info", {}).get("framework", ""),  # U: 框架
            environment.get("system_info", {}).get("python_version", ""),  # V: Python 版本
            environment.get("system_info", {}).get("torch_version", ""),  # W: PyTorch 版本
            str(environment.get("system_info", {}).get("node_count", "")),  # X: 節點數量
        ]

        # 處理資料集結果
        for dataset_path, dataset_data in results.get("dataset_results", {}).items():
            dataset_base_info = base_info + [
                dataset_path,  # Y: 資料集路徑
                str(dataset_data.get("average_accuracy", 0)),  # Z: 平均準確率
                str(dataset_data.get("average_std", 0)),  # AA: 標準差
            ]

            # 如果沒有詳細結果，只新增摘要列
            if not dataset_data.get("results"):
                rows.append(
                    dataset_base_info + ["", "", ""]
                )  # BB, CC, DD: 檔案, 準確率均值, 準確率標準差
                continue

            # 新增每個檔案的結果
            for file_result in dataset_data.get("results", []):
                file_row = dataset_base_info + [
                    file_result.get("file", ""),  # BB: 檔案名稱
                    str(file_result.get("accuracy_mean", 0)),  # CC: 準確率均值
                    str(file_result.get("accuracy_std", 0)),  # DD: 準確率標準差
                ]
                rows.append(file_row)

        return rows


class GoogleSheetsExporter(ResultsExporter):
    """Google Sheets 結果匯出器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化 Google Sheets 匯出器

        Args:
            config: 配置字典，包含 Google Sheets 設定
        """
        super().__init__(config)

        if not self.config:
            raise ConfigurationError("GoogleSheetsExporter 需要配置參數")

        self.sheets_service = GoogleSheetsService(self.config)

    def get_file_extension(self) -> str:
        """取得檔案副檔名"""
        return ".gsheet"  # 這是個虛擬的副檔名，實際不會建立本地檔案

    def export(self, results: Dict[str, Any], output_path: str) -> str:
        """將結果匯出到 Google Sheets

        Args:
            results: 評測結果字典
            output_path: 輸出路徑（這裡主要用於記錄，實際寫入 Google Sheets）

        Returns:
            str: Google Sheets 的 URL
        """
        try:
            spreadsheet_id = self.config.get("spreadsheet_id")
            sheet_name = self.config.get("sheet_name", "Results")

            if not spreadsheet_id:
                raise ConfigurationError("Google Sheets spreadsheet_id 未設定")

            # 寫入資料到 Google Sheets
            success = self.sheets_service.append_results_to_sheet(
                spreadsheet_id, sheet_name, results
            )

            if success:
                sheets_url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
                log_info(f"結果已成功匯出到 Google Sheets: {sheets_url}")
                return sheets_url
            else:
                raise ConfigurationError("Google Sheets 寫入失敗")

        except Exception as e:
            error_msg = f"Google Sheets 匯出失敗: {e}"
            log_error(error_msg)
            raise ConfigurationError(error_msg) from e
