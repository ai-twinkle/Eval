"""Google Drive 與 Google Sheets 整合服務。"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

from twinkle_eval.core.abc import ResultsExporter
from twinkle_eval.exceptions import ConfigurationError
from twinkle_eval.logger import log_error, log_info


class GoogleDriveUploader:
    """Google Drive 檔案上傳器。"""

    SCOPES = ["https://www.googleapis.com/auth/drive.file"]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.service = None
        self._authenticate()

    def _authenticate(self) -> None:
        auth_method = self.config.get("auth_method", "service_account")

        if auth_method == "service_account":
            creds = self._authenticate_service_account()
        else:
            creds = self._authenticate_oauth()

        self.service = build("drive", "v3", credentials=creds)
        log_info("Google Drive API 驗證成功")

    def _authenticate_service_account(self):
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
        creds = None
        token_file = self.config.get("token_file", "token.json")
        credentials_file = self.config.get("credentials_file")

        if not credentials_file:
            raise ConfigurationError("Google Drive credentials_file 未設定")

        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)

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

            with open(token_file, "w") as token:
                token.write(creds.to_json())

        return creds

    def upload_file(self, file_path: str, folder_id: Optional[str] = None) -> str:
        """上傳檔案到 Google Drive，回傳 Drive 檔案 ID。"""
        try:
            if not os.path.exists(file_path):
                raise ConfigurationError(f"檔案不存在: {file_path}")

            file_name = os.path.basename(file_path)
            file_metadata: Dict[str, Any] = {"name": file_name}
            if folder_id:
                file_metadata["parents"] = [folder_id]

            media = MediaFileUpload(file_path, resumable=True)

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

    def create_folder(self, folder_name: str, parent_folder_id: Optional[str] = None) -> str:
        """在 Google Drive 中建立資料夾，回傳資料夾 ID。"""
        try:
            folder_metadata: Dict[str, Any] = {
                "name": folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }

            if parent_folder_id:
                folder_metadata["parents"] = [parent_folder_id]

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
        """上傳最新的 log、results 和 eval_results 檔案到新建立的資料夾。"""
        upload_info: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "folder_id": None,
            "folder_name": None,
            "uploaded_files": [],
        }

        try:
            folder_name = f"Eval_{start_time}"
            parent_folder_id = self.config.get("log_folder_id")
            new_folder_id = self.create_folder(folder_name, parent_folder_id)

            upload_info["folder_id"] = new_folder_id
            upload_info["folder_name"] = folder_name

            if os.path.exists(logs_directory):
                log_files = []
                for file_name in os.listdir(logs_directory):
                    if file_name.endswith(".log") and start_time in file_name:
                        file_path = os.path.join(logs_directory, file_name)
                        file_stat = os.stat(file_path)
                        log_files.append(
                            {"name": file_name, "path": file_path, "mtime": file_stat.st_mtime}
                        )

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

            if os.path.exists(results_directory):
                result_files = []
                eval_result_files = []

                for file_name in os.listdir(results_directory):
                    if file_name.endswith((".json", ".html", ".csv", ".xlsx", ".jsonl")):
                        if start_time in file_name:
                            file_path = os.path.join(results_directory, file_name)
                            file_stat = os.stat(file_path)
                            file_info = {
                                "name": file_name,
                                "path": file_path,
                                "mtime": file_stat.st_mtime,
                            }

                            if file_name.startswith("eval_results_"):
                                eval_result_files.append(file_info)
                            else:
                                result_files.append(file_info)

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
    """Google Sheets 服務類別。"""

    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.service = None
        self._authenticate()

    def _authenticate(self) -> None:
        auth_method = self.config.get("auth_method", "service_account")

        if auth_method == "service_account":
            creds = self._authenticate_service_account()
        else:
            creds = self._authenticate_oauth()

        self.service = build("sheets", "v4", credentials=creds)
        log_info("Google Sheets API 驗證成功")

    def _authenticate_service_account(self):
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
        creds = None
        token_file = self.config.get("token_file", "sheets_token.json")
        credentials_file = self.config.get("credentials_file")

        if not credentials_file:
            raise ConfigurationError("Google Sheets credentials_file 未設定")

        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, self.SCOPES)

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

            with open(token_file, "w") as token:
                token.write(creds.to_json())

        return creds

    def append_results_to_sheet(
        self, spreadsheet_id: str, sheet_name: str, results: Dict[str, Any]
    ) -> bool:
        """將評測結果新增到指定的 Google Sheet。"""
        try:
            self._ensure_header_exists(spreadsheet_id, sheet_name)
            rows = self._prepare_sheet_data(results)

            if not rows:
                log_info("沒有資料需要寫入 Google Sheets")
                return True

            range_name = f"{sheet_name}!A:DD"
            body = {"values": rows}

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

    def _ensure_header_exists(self, spreadsheet_id: str, sheet_name: str) -> None:
        try:
            range_name = f"{sheet_name}!A1:DD1"
            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_name)
                .execute()
            )

            values = result.get("values", [])

            if not values or len(values[0]) < 10:
                log_info("建立 Google Sheets Header...")
                self._create_header(spreadsheet_id, sheet_name)
            else:
                log_info("Google Sheets Header 已存在")

        except Exception as e:
            log_error(f"檢查 Header 失敗，嘗試建立新的 Header: {e}")
            self._create_header(spreadsheet_id, sheet_name)

    def _create_header(self, spreadsheet_id: str, sheet_name: str) -> None:
        header = [
            "時間戳記",
            "API_基礎網址",
            "API_金鑰",
            "API_速率限制",
            "最大重試次數",
            "超時時間",
            "SSL驗證設定",
            "模型名稱",
            "溫度參數",
            "Top_P參數",
            "最大Token數",
            "頻率懲罰",
            "存在懲罰",
            "GPU型號",
            "GPU數量",
            "GPU記憶體GB",
            "CUDA版本",
            "驅動版本",
            "TP大小",
            "PP大小",
            "框架",
            "Python版本",
            "PyTorch版本",
            "節點數量",
            "資料集路徑",
            "平均準確率",
            "標準差",
            "檔案名稱",
            "準確率均值",
            "準確率標準差",
        ]

        try:
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
        rows = []

        timestamp = results.get("timestamp", "")
        config = results.get("config", {})
        environment = config.get("environment", {})
        llm_api = config.get("llm_api", {})
        model = config.get("model", {})

        base_info = [
            timestamp,
            llm_api.get("base_url", ""),
            (
                llm_api.get("api_key", "")[:10] + "..." if llm_api.get("api_key") else ""
            ),
            str(llm_api.get("api_rate_limit", "")),
            str(llm_api.get("max_retries", "")),
            str(llm_api.get("timeout", "")),
            str(llm_api.get("disable_ssl_verify", "")),
            model.get("name", ""),
            str(model.get("temperature", "")),
            str(model.get("top_p", "")),
            str(model.get("max_tokens", "")),
            str(model.get("frequency_penalty", "")),
            str(model.get("presence_penalty", "")),
            environment.get("gpu_info", {}).get("model", ""),
            str(environment.get("gpu_info", {}).get("count", "")),
            str(environment.get("gpu_info", {}).get("memory_gb", "")),
            environment.get("gpu_info", {}).get("cuda_version", ""),
            environment.get("gpu_info", {}).get("driver_version", ""),
            str(environment.get("parallel_config", {}).get("tp_size", "")),
            str(environment.get("parallel_config", {}).get("pp_size", "")),
            environment.get("system_info", {}).get("framework", ""),
            environment.get("system_info", {}).get("python_version", ""),
            environment.get("system_info", {}).get("torch_version", ""),
            str(environment.get("system_info", {}).get("node_count", "")),
        ]

        for dataset_path, dataset_data in results.get("dataset_results", {}).items():
            dataset_base_info = base_info + [
                dataset_path,
                str(dataset_data.get("average_accuracy", 0)),
                str(dataset_data.get("average_std", 0)),
            ]

            if not dataset_data.get("results"):
                rows.append(dataset_base_info + ["", "", ""])
                continue

            for file_result in dataset_data.get("results", []):
                file_row = dataset_base_info + [
                    file_result.get("file", ""),
                    str(file_result.get("accuracy_mean", 0)),
                    str(file_result.get("accuracy_std", 0)),
                ]
                rows.append(file_row)

        return rows


class GoogleSheetsExporter(ResultsExporter):
    """Google Sheets 結果匯出器。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

        if not self.config:
            raise ConfigurationError("GoogleSheetsExporter 需要配置參數")

        self.sheets_service = GoogleSheetsService(self.config)

    def get_file_extension(self) -> str:
        return ".gsheet"

    def export(self, results: Dict[str, Any], output_path: str) -> str:
        """將結果匯出到 Google Sheets，回傳 Sheets URL。"""
        try:
            spreadsheet_id = self.config.get("spreadsheet_id")
            sheet_name = self.config.get("sheet_name", "Results")

            if not spreadsheet_id:
                raise ConfigurationError("Google Sheets spreadsheet_id 未設定")

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
