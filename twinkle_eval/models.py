from abc import ABC, abstractmethod
from typing import Any, Dict, Type

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletion

from .logger import log_error


class LLM(ABC):
    """Abstract base class for all LLM implementations."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @abstractmethod
    def call(self, question_text: str, prompt_lang: str = "zh") -> ChatCompletion:
        """Call the LLM with a question and return the response."""
        pass

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate the configuration for this LLM."""
        pass


class OpenAIModel(LLM):
    """OpenAI-compatible LLM implementation."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.validate_config()
        self._initialize_client()

    def validate_config(self) -> bool:
        """Validate OpenAI-specific configuration."""
        required_keys = ["api_key", "base_url"]
        for key in required_keys:
            if key not in self.config["llm_api"]:
                raise ValueError(f"Missing required config key: llm_api.{key}")
        return True

    def _initialize_client(self):
        """Initialize the OpenAI client with proper configuration."""
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

    def _build_messages(self, question_text: str, prompt_lang: str) -> list:
        """Build message list based on evaluation method."""
        eval_config = self.config["evaluation"]

        if eval_config["evaluation_method"] == "box":
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

    def call(self, question_text: str, prompt_lang: str = "zh") -> ChatCompletion:
        """Call the OpenAI API with the given question."""
        messages = self._build_messages(question_text, prompt_lang)
        model_config = self.config["model"]

        payload = {
            "model": model_config["name"],
            "temperature": model_config["temperature"],
            "top_p": model_config["top_p"],
            "max_tokens": model_config["max_tokens"],
            "messages": messages,
        }

        # Add optional parameters if they exist
        optional_params = ["frequency_penalty", "presence_penalty"]
        for param in optional_params:
            if param in model_config:
                payload[param] = model_config[param]

        if model_config["extra_body"]:
            payload["extra_body"] = model_config["extra_body"]

        try:
            response = self.client.chat.completions.create(**payload)
            return response
        except Exception as e:
            log_error(f"LLM API 錯誤: {e}")
            raise e


class LLMFactory:
    """Factory class for creating LLM instances."""

    _registry: Dict[str, Type[LLM]] = {
        "openai": OpenAIModel,
    }

    @classmethod
    def register_llm(cls, name: str, llm_class: Type[LLM]):
        """Register a new LLM implementation."""
        cls._registry[name] = llm_class

    @classmethod
    def create_llm(cls, llm_type: str, config: Dict[str, Any]) -> LLM:
        """Create an LLM instance based on type."""
        if llm_type not in cls._registry:
            available_types = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported LLM type: {llm_type}. Available types: {available_types}"
            )

        llm_class = cls._registry[llm_type]
        return llm_class(config)

    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available LLM types."""
        return list(cls._registry.keys())
