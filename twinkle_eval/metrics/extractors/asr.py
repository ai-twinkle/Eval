"""ASR（Automatic Speech Recognition）Extractor。

從語音辨識模型的輸出中提取轉錄文字。支援兩種 API 路徑：
- Whisper API（/v1/audio/transcriptions）：回應為結構化 JSON，直接取 text 欄位
- Chat Completions（多模態）：回應為一般 chat message，取 content 欄位

兩種路徑的轉錄文字最終都由 extract() 以 pass-through 方式回傳，
因為 LLM 輸出本身即為轉錄結果，無需進一步解析。
"""

from typing import Any, Dict, Optional

from twinkle_eval.core.abc import Extractor


class ASRExtractor(Extractor):
    """ASR 轉錄文字提取器。

    設定 uses_audio = True 讓 Evaluator 走音檔評測路徑。
    extract() 為 pass-through：LLM 輸出即為轉錄文字。
    """

    uses_audio: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def get_name(self) -> str:
        return "asr"

    def extract(self, llm_output: str) -> Optional[str]:
        """直接回傳 LLM 輸出文字作為轉錄結果。"""
        if not self.validate_output(llm_output):
            return None
        return llm_output.strip()
