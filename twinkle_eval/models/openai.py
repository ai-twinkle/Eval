"""OpenAI 相容 API 的 LLM 實作。"""

from typing import Any, Dict, Optional

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion

from twinkle_eval.core.abc import LLM
from twinkle_eval.core.logger import log_error


class OpenAIModel(LLM):
    """OpenAI 相容格式的 LLM 實作。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.validate_config()
        self._initialize_client()

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
    ) -> ChatCompletion:
        """呼叫 OpenAI 相容 API 並回傳回應。"""
        messages = self._build_messages(question_text, prompt_lang, eval_method, system_prompt_enabled)
        model_config = self.config["model"]
        overrides = model_overrides or {}

        payload: Dict[str, Any] = {
            "model": model_config["name"],
            "temperature": overrides.get("temperature", model_config["temperature"]),
            "top_p": overrides.get("top_p", model_config["top_p"]),
            "max_tokens": overrides.get("max_tokens", model_config["max_tokens"]),
            "messages": messages,
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

        if model_config["extra_body"]:
            payload["extra_body"] = model_config["extra_body"]

        try:
            response = self.client.chat.completions.create(**payload)
            return response
        except Exception as e:
            log_error(f"LLM API 錯誤: {e}")
            raise e

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
