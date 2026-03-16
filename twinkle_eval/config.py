from typing import Any, Dict

import yaml

from .evaluation_strategies import EvaluationStrategyFactory
from .exceptions import ConfigurationError, ValidationError
from .logger import log_error, log_info
from .models import LLMFactory
from .validators import ConfigValidator, DatasetValidator


class ConfigurationManager:
    """配置管理器 - 負責載入和驗證評測系統的配置設定"""

    def __init__(self, config_path: str = "config.yaml"):
        """初始化配置管理器

        Args:
            config_path: 配置檔案路徑，預設為 config.yaml
        """
        self.config_path = config_path
        self.config = {}
        self.validator = ConfigValidator()

    def load_config(self) -> Dict[str, Any]:
        """載入並驗證配置檔案

        Returns:
            Dict[str, Any]: 已驗證的配置字典

        Raises:
            ConfigurationError: 配置載入或驗證失敗
            ValidationError: 配置格式或內容驗證失敗
        """
        try:
            # 驗證檔案是否存在且可讀取
            self.validator.validate_config_file(self.config_path)

            # 驗證 YAML 語法
            self.validator.validate_yaml_syntax(self.config_path)

            # 載入配置檔案
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            # 驗證配置結構
            self.validator.validate_config_structure(self.config)

            # 套用預設值並驗證
            self._apply_defaults()
            self._validate_dataset_paths()
            self._validate_google_services()
            self._instantiate_components()

            log_info("配置載入和驗證完成")
            return self.config

        except (ConfigurationError, ValidationError) as e:
            log_error(f"配置錯誤: {e}")
            raise
        except Exception as e:
            log_error(f"載入配置時發生未預期錯誤: {e}")
            raise ConfigurationError(f"配置載入失敗: {e}") from e

    def _apply_defaults(self):
        """為配置套用預設值

        當配置檔案中缺少某些選項時，自動填入合理的預設值
        """
        # 設定預設 LLM 類型
        if "type" not in self.config["llm_api"]:
            self.config["llm_api"]["type"] = "openai"

        # 設定預設 API 設定
        api_defaults = {
            "max_retries": 3,  # 最大重試次數
            "timeout": 600,  # 請求逾時時間（秒）
            "api_rate_limit": -1,  # API 速率限制（-1 表示無限制）
            "disable_ssl_verify": False,  # 是否停用 SSL 驗證
        }
        for key, value in api_defaults.items():
            if key not in self.config["llm_api"]:
                self.config["llm_api"][key] = value

        # 設定預設模型參數
        model_defaults = {
            "temperature": 0.0,  # 隨機性控制（0.0-1.0）
            "top_p": 0.9,  # 核心採樣參數（0.0-1.0）
            "max_tokens": 4096,  # 最大輸出 token 數
            "frequency_penalty": 0.0,  # 頻率懲罰（-2.0-2.0）
            "presence_penalty": 0.0,  # 存在懲罰（-2.0-2.0）
            "extra_body": {},  # 額外參數
        }
        for key, value in model_defaults.items():
            if key not in self.config["model"]:
                self.config["model"][key] = value

        # 設定預設評測設定
        eval_defaults = {
            "repeat_runs": 1,  # 重複執行次數
            "shuffle_options": False,  # 是否隨機打亂選項順序
            "datasets_prompt_map": {},  # 資料集語言對應表
            "strategy_config": {},  # 評測策略配置
            "dataset_overrides": {},  # 資料集客製化設定
            "samples_per_question": 1,  # 單題產生樣本數（用於 pass@k）
            "pass_k": 1,  # pass@k 的 k 值
            "system_prompt_enabled": True,  # 是否啟用 system prompt
        }
        for key, value in eval_defaults.items():
            if key not in self.config["evaluation"]:
                self.config["evaluation"][key] = value

        # 設定預設環境配置
        if "environment" not in self.config:
            self.config["environment"] = {}

        env_defaults = {
            "gpu_info": {
                "model": "Unknown",
                "count": 1,
                "memory_gb": 0,
                "cuda_version": "Unknown",
                "driver_version": "Unknown",
            },
            "parallel_config": {"tp_size": 1, "pp_size": 1},
            "system_info": {
                "framework": "Unknown",
                "python_version": "Unknown",
                "torch_version": "Unknown",
                "node_count": 1,
            },
        }
        for key, value in env_defaults.items():
            if key not in self.config["environment"]:
                self.config["environment"][key] = value

    def _validate_dataset_paths(self):
        """驗證資料集路徑是否存在且可存取

        檢查配置中指定的所有資料集路徑是否有效，並統計可用的資料集檔案數量

        Raises:
            ConfigurationError: 當資料集路徑無效時拋出
        """
        dataset_paths = self.config["evaluation"]["dataset_paths"]
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]

        for path in dataset_paths:
            try:
                DatasetValidator.validate_dataset_path(path)
                valid_files = DatasetValidator.validate_dataset_files(path)
                log_info(f"在 {path} 中找到 {len(valid_files)} 個有效的資料集檔案")
            except ValidationError as e:
                log_error(f"資料集驗證失敗 {path}: {e}")
                raise ConfigurationError(f"無效的資料集路徑 {path}: {e}") from e

    def _instantiate_components(self):
        """實例化 LLM 和評測策略元件

        根據配置建立 LLM 實例和評測策略實例，並將其加入配置中供後續使用

        Raises:
            ConfigurationError: 當元件實例化失敗時拋出
        """
        try:
            # 使用工廠模式建立 LLM 實例
            llm_type = self.config["llm_api"]["type"]
            self.config["llm_instance"] = LLMFactory.create_llm(llm_type, self.config)
            log_info(f"LLM 實例建立完成: {llm_type}")

        except Exception as e:
            available_types = ", ".join(LLMFactory.get_available_types())
            error_msg = f"不支援的 LLM API 類型: {llm_type}. 可用類型: {available_types}"
            log_error(error_msg)
            raise ConfigurationError(error_msg) from e

        try:
            # 使用工廠模式建立評測策略實例
            eval_method = self.config["evaluation"]["evaluation_method"]
            strategy_config = self.config["evaluation"].get("strategy_config", {})

            self.config["evaluation_strategy_instance"] = EvaluationStrategyFactory.create_strategy(
                eval_method, strategy_config
            )
            log_info(f"評測策略建立完成: {eval_method}")

        except Exception as e:
            available_types = ", ".join(EvaluationStrategyFactory.get_available_types())
            error_msg = f"不支援的評測方法: {eval_method}. 可用方法: {available_types}"
            log_error(error_msg)
            raise ConfigurationError(error_msg) from e

    def _validate_google_services(self):
        """驗證 Google 服務配置

        檢查 Google Drive 和 Google Sheets 的配置是否正確，
        包括驗證憑證檔案、必要參數等

        Raises:
            ConfigurationError: 當 Google 服務配置無效時拋出
        """
        google_services_config = self.config.get("google_services")
        if not google_services_config:
            return  # 如果沒有配置 Google 服務則跳過驗證

        # 驗證 Google Sheets 配置
        google_sheets_config = google_services_config.get("google_sheets", {})
        if google_sheets_config.get("enabled", False):
            self._validate_google_sheets_config(google_sheets_config)
            log_info("Google Sheets 配置驗證完成")

        # 驗證 Google Drive 配置
        google_drive_config = google_services_config.get("google_drive", {})
        if google_drive_config.get("enabled", False):
            try:
                self._validate_google_drive_config(google_drive_config)
                log_info("Google Drive 配置驗證完成")
            except ConfigurationError as e:
                # 如果是權限問題，建議使用 OAuth 方式
                if "不存在或 Service Account 無權限存取" in str(e):
                    auth_method = google_drive_config.get("auth_method", "service_account")
                    if auth_method == "service_account":
                        log_error(f"Service Account 驗證失敗: {e}")
                        log_info("建議解決方案:")
                        log_info("1. 將 Service Account Email 加入 Google Drive 資料夾共享")
                        log_info("2. 或改用 OAuth 驗證方式：設定 auth_method: 'oauth'")
                        # 不拋出錯誤，允許繼續執行，但會在實際上傳時失敗並提示
                    else:
                        raise
                else:
                    raise

    def _validate_google_sheets_config(self, config: Dict[str, Any]):
        """驗證 Google Sheets 特定配置

        Args:
            config: Google Sheets 配置字典

        Raises:
            ConfigurationError: 配置驗證失敗時拋出
        """
        # 檢查必要參數
        spreadsheet_id = config.get("spreadsheet_id")
        if not spreadsheet_id or not spreadsheet_id.strip():
            raise ConfigurationError("Google Sheets 配置錯誤: spreadsheet_id 為必填項目")

        # 驗證身份驗證配置
        self._validate_google_auth_config(config, "Google Sheets")

        try:
            # 嘗試建立 GoogleSheetsService 實例來驗證配置
            from .google_services import GoogleSheetsService

            sheets_service = GoogleSheetsService(config)

            # 嘗試驗證 spreadsheet 是否可以存取
            sheet_name = config.get("sheet_name", "Results")
            sheets_service.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()

            log_info(f"Google Sheets 連接測試成功 - 試算表 ID: {spreadsheet_id}")

        except Exception as e:
            raise ConfigurationError(f"Google Sheets 配置驗證失敗: {e}") from e

    def _validate_google_drive_config(self, config: Dict[str, Any]):
        """驗證 Google Drive 特定配置

        Args:
            config: Google Drive 配置字典

        Raises:
            ConfigurationError: 配置驗證失敗時拋出
        """
        # 驗證身份驗證配置
        self._validate_google_auth_config(config, "Google Drive")

        try:
            # 嘗試建立 GoogleDriveUploader 實例來驗證配置
            from .google_services import GoogleDriveUploader

            drive_uploader = GoogleDriveUploader(config)

            # 驗證 log_folder_id 是否存在且可存取（如果有設定的話）
            log_folder_id = config.get("log_folder_id")
            if log_folder_id and log_folder_id.strip():
                try:
                    # 嘗試取得資料夾資訊來驗證它是否存在且可存取
                    folder_info = (
                        drive_uploader.service.files()
                        .get(
                            fileId=log_folder_id, 
                            fields="id,name,mimeType",
                            supportsAllDrives=True
                        )
                        .execute()
                    )

                    # 檢查是否為資料夾
                    if folder_info.get("mimeType") != "application/vnd.google-apps.folder":
                        raise ConfigurationError(
                            f"Google Drive log_folder_id 指向的不是資料夾: {log_folder_id}"
                        )

                    log_info(
                        f"Google Drive 資料夾驗證成功 - {folder_info.get('name')} ({log_folder_id})"
                    )

                except Exception as folder_error:
                    if "File not found" in str(folder_error) or "notFound" in str(folder_error):
                        service_account_email = None
                        try:
                            import json

                            credentials_file = config.get("credentials_file")
                            with open(credentials_file, "r", encoding="utf-8") as f:
                                cred_data = json.load(f)
                                service_account_email = cred_data.get("client_email", "未知")
                        except:
                            service_account_email = "未知"

                        raise ConfigurationError(
                            f"Google Drive 資料夾不存在或 Service Account 無權限存取: {log_folder_id}\n"
                            f"Service Account: {service_account_email}\n"
                            f"請確認：\n"
                            f"1. 資料夾 ID 正確: {log_folder_id}\n"
                            f"2. 資料夾存在且未被刪除\n"
                            f"3. Service Account ({service_account_email}) 已被加入資料夾的共享權限"
                        ) from folder_error
                    else:
                        raise ConfigurationError(
                            f"Google Drive 資料夾驗證失敗: {folder_error}"
                        ) from folder_error

            log_info("Google Drive 配置驗證完成")

        except Exception as e:
            raise ConfigurationError(f"Google Drive 配置驗證失敗: {e}") from e

    def _validate_google_auth_config(self, config: Dict[str, Any], service_name: str):
        """驗證 Google 服務身份驗證配置

        Args:
            config: Google 服務配置字典
            service_name: 服務名稱（用於錯誤訊息）

        Raises:
            ConfigurationError: 身份驗證配置驗證失敗時拋出
        """
        import os

        auth_method = config.get("auth_method", "service_account")
        credentials_file = config.get("credentials_file")

        if not credentials_file or not credentials_file.strip():
            raise ConfigurationError(f"{service_name} 配置錯誤: credentials_file 為必填項目")

        if not os.path.exists(credentials_file):
            raise ConfigurationError(
                f"{service_name} 配置錯誤: 憑證檔案不存在 - {credentials_file}"
            )

        # 驗證憑證檔案格式
        if auth_method == "service_account":
            try:
                import json

                with open(credentials_file, "r", encoding="utf-8") as f:
                    cred_data = json.load(f)

                # 檢查 Service Account 必要欄位
                required_fields = [
                    "type",
                    "project_id",
                    "private_key_id",
                    "private_key",
                    "client_email",
                ]
                for field in required_fields:
                    if field not in cred_data:
                        raise ConfigurationError(
                            f"{service_name} Service Account 憑證檔案格式錯誤: 缺少必要欄位 '{field}'"
                        )

                if cred_data.get("type") != "service_account":
                    raise ConfigurationError(
                        f"{service_name} 憑證檔案格式錯誤: 類型應為 'service_account'"
                    )

            except json.JSONDecodeError as e:
                raise ConfigurationError(f"{service_name} 憑證檔案格式錯誤: {e}") from e
            except Exception as e:
                raise ConfigurationError(f"{service_name} 憑證檔案讀取失敗: {e}") from e


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """使用配置管理器載入配置

    這是一個便利函數，為外部模組提供簡單的配置載入介面

    Args:
        config_path: 配置檔案路徑，預設為 config.yaml

    Returns:
        Dict[str, Any]: 載入並驗證後的配置字典
    """
    manager = ConfigurationManager(config_path)
    return manager.load_config()
