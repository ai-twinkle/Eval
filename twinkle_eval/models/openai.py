"""OpenAI 相容 API 的 LLM 實作。"""

from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion

from twinkle_eval.core.abc import LLM
from twinkle_eval.core.logger import log_error, log_warning


# 被 API 拒絕時可安全移除的選用取樣參數（新版推理模型多半不支援這些參數）
_DROPPABLE_PARAMS = {"temperature", "top_p", "frequency_penalty", "presence_penalty"}


def _rejected_parameter(error: Exception) -> tuple[Optional[str], Optional[str]]:
    """解析錯誤，回傳 (被拒絕的參數名稱, 錯誤代碼)；非參數拒絕錯誤則回傳 (None, None)。

    新版 OpenAI 推理模型（o 系列、gpt-5 系列等）會以 400 拒絕部分參數：
    - unsupported_parameter：參數本身不支援（如 max_tokens、top_p）
    - unsupported_value：參數值不支援（如 temperature 僅接受預設值）
    """
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("code") in (
            "unsupported_parameter",
            "unsupported_value",
        ):
            param = err.get("param")
            if isinstance(param, str) and param:
                return param, err.get("code")
    # 後備：部分相容端點不回傳結構化 body，僅能從訊息判斷 max_tokens 的情況
    msg = str(error)
    if "max_tokens" in msg and "max_completion_tokens" in msg:
        return "max_tokens", "unsupported_parameter"
    return None, None


class OpenAIModel(LLM):
    """OpenAI 相容格式的 LLM 實作。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.validate_config()
        self._initialize_client()
        # 首次遇到 API 回報 max_tokens 不支援後切為 True，之後直接送 max_completion_tokens
        self._use_max_completion_tokens = False
        # 已確認此模型不支援的選用參數，後續請求直接略過，不再每題撞一次 400
        self._unsupported_params: set = set()

    def validate_config(self) -> bool:
        """驗證 OpenAI 相容格式所需的配置欄位。"""
        required_keys = ["api_key", "base_url"]
        for key in required_keys:
            if key not in self.config["llm_api"]:
                raise ValueError(f"缺少必要的配置欄位: llm_api.{key}")
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
            max_retries=api_config["max_retries"],
            timeout=api_config["timeout"],
        )

    def _build_messages(
        self,
        question_text: str,
        prompt_lang: str,
        eval_method: str,
        system_prompt_enabled: bool,
    ) -> list:
        """依評測方法建立訊息列表。"""
        eval_config = self.config["evaluation"]
        method = eval_method or eval_config["evaluation_method"]

        # box 和 math 兩種方法都使用 system prompt
        uses_system_prompt = system_prompt_enabled and method in {"box", "math"}

        if uses_system_prompt:
            sys_prompt_cfg = eval_config.get("system_prompt", {})
            if isinstance(sys_prompt_cfg, dict):
                sys_prompt = sys_prompt_cfg.get(prompt_lang, sys_prompt_cfg.get("zh", ""))
            else:
                sys_prompt = sys_prompt_cfg

            return [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question_text},
            ]
        else:
            return [{"role": "user", "content": question_text}]

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
        """呼叫 OpenAI 相容 API 並回傳回應。

        Args:
            tools:    OpenAI tools 格式列表（FC 模式）。
            messages: 預先建構的 messages（BFCL 模式）。
                      若提供則略過 _build_messages()。
        """
        if messages is not None:
            built_messages = messages
        else:
            built_messages = self._build_messages(
                question_text, prompt_lang, eval_method, system_prompt_enabled
            )
        model_config = self.config["model"]
        overrides = model_overrides or {}

        max_tokens_param = (
            "max_completion_tokens" if self._use_max_completion_tokens else "max_tokens"
        )
        payload: Dict[str, Any] = {
            "model": model_config["name"],
            "temperature": overrides.get("temperature", model_config["temperature"]),
            "top_p": overrides.get("top_p", model_config["top_p"]),
            max_tokens_param: overrides.get("max_tokens", model_config["max_tokens"]),
            "messages": built_messages,
        }

        if num_samples > 1:
            payload["n"] = num_samples

        # 加入選用參數
        optional_params = ["frequency_penalty", "presence_penalty"]
        for param in optional_params:
            if param in overrides:
                payload[param] = overrides[param]
            elif param in model_config:
                payload[param] = model_config[param]

        if tools:
            payload["tools"] = tools

        if model_config["extra_body"]:
            payload["extra_body"] = model_config["extra_body"]

        # 移除先前已確認此模型不支援的參數
        for param in self._unsupported_params:
            payload.pop(param, None)

        # 每次成功的調整都會從 payload 移除或改名一個參數，迴圈必然終止
        while True:
            try:
                return self.client.chat.completions.create(**payload)
            except Exception as e:
                if self._adapt_payload_for_error(payload, e):
                    continue
                log_error(f"LLM API 錯誤: {e}")
                raise e

    def _adapt_payload_for_error(self, payload: Dict[str, Any], error: Exception) -> bool:
        """依 API 的參數拒絕錯誤調整 payload，回傳是否已調整（可重試）。

        - max_tokens 被拒 → 改名為 max_completion_tokens
        - 其他選用取樣參數被拒 → 直接移除（模型將使用其預設值）
        調整結果記錄在實例狀態，後續請求直接套用，不再重複撞錯。
        """
        param, code = _rejected_parameter(error)
        if param is None:
            return False
        model_name = self.config["model"]["name"]

        if param == "max_tokens" and "max_tokens" in payload:
            log_warning(
                f"模型 {model_name} 不支援 max_tokens，自動改用 max_completion_tokens 重試"
            )
            self._use_max_completion_tokens = True
            payload["max_completion_tokens"] = payload.pop("max_tokens")
            return True

        if param in _DROPPABLE_PARAMS and param in payload:
            log_warning(f"模型 {model_name} 不支援 {param}（{code}），自動移除該參數重試")
            self._unsupported_params.add(param)
            payload.pop(param)
            return True

        return False

    def score_continuation(self, context: str, continuation: str) -> float:
        """計算 log P(continuation | context)，用於 logit 評測策略。

        使用 /v1/completions 端點的 echo 模式：將 context + continuation 作為 prompt
        傳入，取得所有 token 的 logprob，再加總 continuation 部分的 log-likelihood。

        Args:
            context:      題目 context，通常以 "\\nAnswer:" 結尾。
            continuation: 要評分的選項文字，如 " A"（含 leading space，與 lm-harness 一致）。

        Returns:
            continuation 部分的 log-likelihood。若 API 不支援或發生錯誤，回傳 float("-inf")。
        """
        model_config = self.config["model"]
        full_prompt = context + continuation

        try:
            response = self.client.completions.create(
                model=model_config["name"],
                prompt=full_prompt,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )
            token_logprobs = response.choices[0].logprobs.token_logprobs
            tokens = response.choices[0].logprobs.tokens

            if not token_logprobs or not tokens:
                return float("-inf")

            context_char_len = len(context)
            cumulative = 0
            logprob_sum = 0.0
            found = False
            for token, lp in zip(tokens, token_logprobs):
                if cumulative >= context_char_len and lp is not None:
                    logprob_sum += lp
                    found = True
                cumulative += len(token)

            return logprob_sum if found else float("-inf")

        except Exception as e:
            log_error(f"score_continuation 失敗（模型: {model_config['name']}）: {e}")
            return float("-inf")
