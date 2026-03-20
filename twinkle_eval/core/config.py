import os
from typing import Any, Dict

import yaml

from twinkle_eval.core.exceptions import ConfigurationError, ValidationError
from twinkle_eval.core.logger import log_error, log_info
from twinkle_eval.metrics import create_metric_pair, get_available_methods
from twinkle_eval.models import LLMFactory
from twinkle_eval.core.validators import ConfigValidator, DatasetValidator


class ConfigurationManager:
    """配置管理器 - 負責載入和驗證評測系統的配置設定"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.validator = ConfigValidator()

    def load_config(self) -> Dict[str, Any]:
        """載入並驗證配置檔案。"""
        try:
            self.validator.validate_config_file(self.config_path)
            self.validator.validate_yaml_syntax(self.config_path)

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

            self.validator.validate_config_structure(self.config)

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

    def _apply_defaults(self) -> None:
        """為配置套用預設值。"""
        if "type" not in self.config["llm_api"]:
            self.config["llm_api"]["type"] = "openai"

        api_defaults = {
            "max_retries": 3,
            "timeout": 600,
            "api_rate_limit": -1,
            "disable_ssl_verify": False,
        }
        for key, value in api_defaults.items():
            if key not in self.config["llm_api"]:
                self.config["llm_api"][key] = value

        model_defaults = {
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 4096,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "extra_body": {},
        }
        for key, value in model_defaults.items():
            if key not in self.config["model"]:
                self.config["model"][key] = value

        eval_defaults = {
            "repeat_runs": 1,
            "shuffle_options": False,
            "datasets_prompt_map": {},
            "strategy_config": {},
            "dataset_overrides": {},
            "samples_per_question": 1,
            "pass_k": 1,
            "system_prompt_enabled": True,
        }
        for key, value in eval_defaults.items():
            if key not in self.config["evaluation"]:
                self.config["evaluation"][key] = value

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        dist_defaults = {"world_size": world_size, "rank": rank}
        if "distributed" not in self.config:
            self.config["distributed"] = dist_defaults
        else:
            for key, value in dist_defaults.items():
                if key not in self.config["distributed"]:
                    self.config["distributed"][key] = value

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

    def _validate_dataset_paths(self) -> None:
        """驗證資料集路徑是否存在且可存取。"""
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

    def _instantiate_components(self) -> None:
        """實例化 LLM 和評測 Extractor/Scorer 元件。"""
        try:
            llm_type = self.config["llm_api"]["type"]
            self.config["llm_instance"] = LLMFactory.create_llm(llm_type, self.config)
            log_info(f"LLM 實例建立完成: {llm_type}")

        except Exception as e:
            available_types = ", ".join(LLMFactory.get_available_types())
            error_msg = f"不支援的 LLM API 類型: {self.config['llm_api'].get('type', '?')}. 可用類型: {available_types}"
            log_error(error_msg)
            raise ConfigurationError(error_msg) from e

        try:
            eval_method = self.config["evaluation"]["evaluation_method"]
            strategy_config = self.config["evaluation"].get("strategy_config", {})

            extractor, scorer = create_metric_pair(eval_method, strategy_config)
            self.config["extractor_instance"] = extractor
            self.config["scorer_instance"] = scorer

            # 向下相容：保留 evaluation_strategy_instance 指向一個相容物件
            # （讓 main.py 仍可透過 config["evaluation_strategy_instance"] 取得）
            # 此處建立一個簡單的 shim 讓舊程式碼不報錯
            self.config["evaluation_strategy_instance"] = _CompatStrategyShim(extractor, scorer)

            log_info(f"評測策略建立完成: {eval_method}")

        except Exception as e:
            available_methods = ", ".join(get_available_methods())
            error_msg = f"不支援的評測方法: {self.config['evaluation'].get('evaluation_method', '?')}. 可用方法: {available_methods}"
            log_error(error_msg)
            raise ConfigurationError(error_msg) from e

    def _validate_google_services(self) -> None:
        """驗證 Google 服務配置。"""
        google_services_config = self.config.get("google_services")
        if not google_services_config:
            return

        google_sheets_config = google_services_config.get("google_sheets", {})
        if google_sheets_config.get("enabled", False):
            self._validate_google_sheets_config(google_sheets_config)
            log_info("Google Sheets 配置驗證完成")

        google_drive_config = google_services_config.get("google_drive", {})
        if google_drive_config.get("enabled", False):
            try:
                self._validate_google_drive_config(google_drive_config)
                log_info("Google Drive 配置驗證完成")
            except ConfigurationError as e:
                if "不存在或 Service Account 無權限存取" in str(e):
                    auth_method = google_drive_config.get("auth_method", "service_account")
                    if auth_method == "service_account":
                        log_error(f"Service Account 驗證失敗: {e}")
                        log_info("建議解決方案:")
                        log_info("1. 將 Service Account Email 加入 Google Drive 資料夾共享")
                        log_info("2. 或改用 OAuth 驗證方式：設定 auth_method: 'oauth'")
                    else:
                        raise
                else:
                    raise

    def _validate_google_sheets_config(self, config: Dict[str, Any]) -> None:
        spreadsheet_id = config.get("spreadsheet_id")
        if not spreadsheet_id or not spreadsheet_id.strip():
            raise ConfigurationError("Google Sheets 配置錯誤: spreadsheet_id 為必填項目")

        self._validate_google_auth_config(config, "Google Sheets")

        try:
            from twinkle_eval.integrations.google import GoogleSheetsService

            sheets_service = GoogleSheetsService(config)
            sheets_service.service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
            log_info(f"Google Sheets 連接測試成功 - 試算表 ID: {spreadsheet_id}")

        except Exception as e:
            raise ConfigurationError(f"Google Sheets 配置驗證失敗: {e}") from e

    def _validate_google_drive_config(self, config: Dict[str, Any]) -> None:
        self._validate_google_auth_config(config, "Google Drive")

        try:
            from twinkle_eval.integrations.google import GoogleDriveUploader

            drive_uploader = GoogleDriveUploader(config)

            log_folder_id = config.get("log_folder_id")
            if log_folder_id and log_folder_id.strip():
                try:
                    folder_info = (
                        drive_uploader.service.files()
                        .get(
                            fileId=log_folder_id,
                            fields="id,name,mimeType",
                            supportsAllDrives=True,
                        )
                        .execute()
                    )

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
                        except Exception:
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

    def _validate_google_auth_config(self, config: Dict[str, Any], service_name: str) -> None:
        auth_method = config.get("auth_method", "service_account")
        credentials_file = config.get("credentials_file")

        if not credentials_file or not credentials_file.strip():
            raise ConfigurationError(f"{service_name} 配置錯誤: credentials_file 為必填項目")

        if not os.path.exists(credentials_file):
            raise ConfigurationError(
                f"{service_name} 配置錯誤: 憑證檔案不存在 - {credentials_file}"
            )

        if auth_method == "service_account":
            try:
                import json

                with open(credentials_file, "r", encoding="utf-8") as f:
                    cred_data = json.load(f)

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


class _CompatStrategyShim:
    """向下相容 shim：讓舊程式碼仍可透過 evaluation_strategy_instance 使用。

    此 shim 將 EvaluationStrategy 的方法委派至新的 Extractor / Scorer 介面。
    """

    def __init__(self, extractor: Any, scorer: Any) -> None:
        self._extractor = extractor
        self._scorer = scorer
        self.uses_logprobs: bool = getattr(extractor, "uses_logprobs", False)

    def extract_answer(self, llm_output: str) -> Any:
        return self._extractor.extract(llm_output)

    def normalize_answer(self, answer: str) -> str:
        return self._scorer.normalize(answer)

    def is_correct(self, predicted: str, correct: str) -> bool:
        return self._scorer.score(predicted, correct)

    def get_strategy_name(self) -> str:
        return self._extractor.get_name()


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """便利函數：使用 ConfigurationManager 載入配置。"""
    manager = ConfigurationManager(config_path)
    return manager.load_config()
