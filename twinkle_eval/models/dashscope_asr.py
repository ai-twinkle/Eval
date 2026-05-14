"""DashScope 原生 ASR 實作（Qwen3-ASR 系列）。

DashScope 的 Qwen3-ASR-Flash 不走 OpenAI-compatible /audio/transcriptions，
而是使用 multimodal-generation/generation 端點，將音檔以 base64 data URI 傳入。

使用方式：在 config.yaml 中設定 llm_api.type 為 "dashscope_asr"。
需設定環境變數：DASHSCOPE_API_KEY（或在 config 的 api_key 填入）。

限制：qwen3-asr-flash 每檔上限 10 MB / 5 分鐘。
"""

import base64
import os
import time
from typing import Any, Dict, List, Optional

import httpx
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from twinkle_eval.core.abc import LLM
from twinkle_eval.core.logger import log_error

_ENDPOINT = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

_MIME_MAP = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
    ".opus": "audio/opus",
    ".aac": "audio/aac",
}

_MAX_BYTES = 10 * 1024 * 1024  # 10 MB — qwen3-asr-flash hard limit


class DashScopeASRModel(LLM):
    """DashScope multimodal-generation 端點的 Qwen3-ASR 實作。

    call() 接收音檔路徑（透過 question_text），回傳包裝為 ChatCompletion 的轉錄文字。
    """
    _is_direct_asr = True  # marker: evaluator passes audio_path via question_text

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.validate_config()
        api_cfg = config["llm_api"]
        raw_key = api_cfg.get("api_key", "")
        # 支援 ${ENV_VAR} 替換
        if raw_key.startswith("${") and raw_key.endswith("}"):
            env_var = raw_key[2:-1]
            raw_key = os.environ.get(env_var, "")
        self._api_key = raw_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self._timeout = api_cfg.get("timeout", 180)
        self._max_retries = api_cfg.get("max_retries", 3)
        self._endpoint = api_cfg.get("base_url", _ENDPOINT).rstrip("/")
        # 若 base_url 是 compatible-mode 端點，換成 native 端點
        if "compatible-mode" in self._endpoint:
            self._endpoint = _ENDPOINT

    def validate_config(self) -> bool:
        if "name" not in self.config.get("model", {}):
            raise ValueError("缺少必要的配置欄位: model.name")
        return True

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
        """呼叫 DashScope Qwen3-ASR API 進行語音轉錄。

        Args:
            question_text: 音檔路徑。
            prompt_lang: 語言代碼（傳入 parameters.language）。
        """
        audio_path = question_text
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"音檔不存在: {audio_path}")

        file_size = os.path.getsize(audio_path)
        if file_size > _MAX_BYTES:
            raise ValueError(
                f"音檔 {audio_path} 大小 {file_size/1024/1024:.1f} MB 超過 "
                f"qwen3-asr-flash 上限 10 MB"
            )

        ext = os.path.splitext(audio_path)[1].lower()
        mime_type = _MIME_MAP.get(ext, "audio/wav")
        with open(audio_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{audio_b64}"

        model_name = self.config["model"]["name"]
        # nan 台語無對應 DashScope 語言代碼 → 不傳 language，讓模型自動偵測
        _LANG_REMAP: dict = {"nan": None}
        language_code = _LANG_REMAP.get(prompt_lang, prompt_lang)

        payload: Dict[str, Any] = {
            "model": model_name,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"audio": data_uri}],
                    }
                ]
            },
            "parameters": {
                "asr_options": {"enable_itn": False}
            },
        }
        if language_code:
            payload["parameters"]["language"] = language_code

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        text = self._post_with_retry(payload, headers)

        return ChatCompletion(
            id=f"dashscope-{os.path.basename(audio_path)}",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(content=text, role="assistant"),
                )
            ],
            created=int(time.time()),
            model=model_name,
            object="chat.completion",
            usage=CompletionUsage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
            ),
        )

    def _post_with_retry(self, payload: Dict[str, Any], headers: Dict[str, str]) -> str:
        last_err: Exception = RuntimeError("unknown error")
        for attempt in range(self._max_retries):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(self._endpoint, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["output"]["choices"][0]["message"]["content"][0]["text"]
                err_body = resp.text[:300]
                last_err = RuntimeError(
                    f"DashScope ASR HTTP {resp.status_code}: {err_body}"
                )
                if resp.status_code in (400, 401, 403):
                    raise last_err
                # 429 rate-limit: back off longer
                if resp.status_code == 429:
                    wait = 5 * (2 ** attempt)
                    log_error(f"DashScope ASR 429 rate-limit, retry {attempt + 1}/{self._max_retries} in {wait}s")
                    time.sleep(wait)
                    continue
            except httpx.TimeoutException as e:
                last_err = e
            except RuntimeError:
                raise
            except Exception as e:
                last_err = e

            if attempt < self._max_retries - 1:
                wait = 2 ** attempt
                log_error(f"DashScope ASR retry {attempt + 1}/{self._max_retries} in {wait}s: {last_err}")
                time.sleep(wait)

        raise last_err
