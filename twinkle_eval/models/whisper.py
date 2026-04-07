"""Whisper 相容 API 的 LLM 實作。

支援 OpenAI Whisper API（/v1/audio/transcriptions）格式，
相容 OpenAI、Groq、faster-whisper-server 等服務。

使用方式：在 config.yaml 中設定 llm_api.type 為 "whisper"。
"""

import os
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from twinkle_eval.core.abc import LLM
from twinkle_eval.core.logger import log_error


class WhisperModel(LLM):
    """Whisper 相容格式的 ASR 模型實作。

    透過 /v1/audio/transcriptions 端點進行語音轉錄。
    call() 接收音檔路徑作為 question_text，回傳包裝為 ChatCompletion 的轉錄結果。
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.validate_config()
        self._initialize_client()

    def validate_config(self) -> bool:
        """驗證 Whisper API 所需的配置欄位。"""
        required_keys = ["api_key", "base_url"]
        for key in required_keys:
            if key not in self.config["llm_api"]:
                raise ValueError(f"缺少必要的配置欄位: llm_api.{key}")
        if "name" not in self.config.get("model", {}):
            raise ValueError("缺少必要的配置欄位: model.name")
        return True

    def _initialize_client(self) -> None:
        """初始化 OpenAI 客戶端。"""
        api_config = self.config["llm_api"]

        if api_config.get("disable_ssl_verify", False):
            httpx_client = httpx.Client(verify=False)
        else:
            httpx_client = httpx.Client()

        self.client = OpenAI(
            api_key=api_config["api_key"],
            base_url=api_config["base_url"],
            http_client=httpx_client,
            max_retries=api_config.get("max_retries", 3),
            timeout=api_config.get("timeout", 300),
        )

    def call(
        self,
        question_text: str,
        prompt_lang: str = "zh",
        eval_method: str = "",
        system_prompt_enabled: bool = True,
        num_samples: int = 1,
        model_overrides: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """呼叫 Whisper API 進行語音轉錄。

        Args:
            question_text: 音檔路徑（必須是可讀取的本機檔案路徑）。
            prompt_lang: 語言代碼（傳遞至 Whisper API 的 language 參數）。
            其他參數: 為符合 LLM ABC 介面而保留，Whisper 不使用。

        Returns:
            ChatCompletion: 將轉錄結果包裝為 ChatCompletion 格式。
        """
        audio_path = question_text  # 音檔路徑透過 question_text 傳入

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音檔不存在: {audio_path}")

        model_name = self.config["model"]["name"]
        overrides = model_overrides or {}
        temperature = overrides.get(
            "temperature", self.config["model"].get("temperature", 0.0)
        )

        # 呼叫 Whisper API
        with open(audio_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=model_name,
                file=audio_file,
                language=prompt_lang if prompt_lang != "zh" else "zh",
                temperature=temperature,
                response_format="verbose_json",
            )

        # 取得轉錄文字
        text = transcription.text if hasattr(transcription, "text") else str(transcription)

        # 包裝為 ChatCompletion 格式
        return ChatCompletion(
            id=f"whisper-{os.path.basename(audio_path)}",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=text,
                        role="assistant",
                    ),
                )
            ],
            created=0,
            model=model_name,
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
            ),
        )
