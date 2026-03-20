"""自定義異常類別 - Twinkle Eval 系統中使用的所有異常類別

定義了系統中可能發生的各種錯誤類型，方便精確的錯誤處理和除錯
"""

from typing import Any, Optional


class TwinkleEvalError(Exception):
    """Twinkle Eval 系統的基本異常類別

    所有 Twinkle Eval 系統中的異常都從這個類別繼承，方便統一處理
    """

    def __init__(self, message: str, details: Optional[Any] = None):
        """初始化異常

        Args:
            message: 錯誤訊息
            details: 額外的錯誤詳細資訊（可選）
        """
        super().__init__(message)
        self.message = message
        self.details = details


class ConfigurationError(TwinkleEvalError):
    """配置錯誤 - 當配置檔案有問題或配置驗證失敗時拋出"""

    pass


class LLMError(TwinkleEvalError):
    """LLM 錯誤 - 當 LLM API 呼叫或操作失敗時拋出"""

    pass


class EvaluationError(TwinkleEvalError):
    """評測錯誤 - 當評測過程中發生錯誤時拋出"""

    pass


class DatasetError(TwinkleEvalError):
    """資料集錯誤 - 當資料集載入或處理失敗時拋出"""

    pass


class ExportError(TwinkleEvalError):
    """匯出錯誤 - 當結果匯出失敗時拋出"""

    pass


class ValidationError(TwinkleEvalError):
    """驗證錯誤 - 當資料或配置驗證失敗時拋出"""

    pass
