"""OpenAI Responses API（/v1/responses）的 LLM 實作。

呼叫新版 Responses API，並將回應轉換為 ChatCompletion 相容格式，
讓 Evaluator 與各評測管線無需任何修改即可使用（依 CLAUDE.md 5.1 規範，
格式轉換在 call() 內部完成，不得修改 evaluators）。

支援：
- 文字與 multimodal（vision）輸入的自動格式轉換
- reasoning 模型的推理摘要（映射至 message.reasoning，
  與 vLLM reasoning 欄位的讀取路徑一致）
- tools / function calling（Chat Completions tools 格式自動轉為
  Responses API 的扁平格式，回傳的 function_call 轉回 tool_calls）
- num_samples > 1（Responses API 無 n 參數，內部以多次請求合併）

限制：
- frequency_penalty / presence_penalty 不受 Responses API 支援，會被忽略
- 不支援 ASR 音檔路徑（請改用 whisper 或 openai backend）
"""

import json
from typing import Any, Dict, List, Optional

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from twinkle_eval.core.logger import log_error
from twinkle_eval.models.openai import OpenAIModel


def _convert_content_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """將 Chat Completions 的 content part 轉為 Responses API 格式。"""
    part_type = part.get("type")
    if part_type == "text":
        return {"type": "input_text", "text": part.get("text", "")}
    if part_type == "image_url":
        image_url = part.get("image_url", {})
        url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
        converted: Dict[str, Any] = {"type": "input_image", "image_url": url}
        detail = image_url.get("detail") if isinstance(image_url, dict) else None
        if detail:
            converted["detail"] = detail
        return converted
    # 其他型別（如 audio）原樣傳遞，由 API 端回報錯誤
    return part


def convert_messages_to_responses_input(
    messages: List[Dict[str, Any]],
) -> tuple:
    """將 Chat Completions messages 轉為 Responses API 的 (instructions, input)。

    system / developer 訊息合併為 instructions，其餘訊息成為 input 列表，
    multimodal content parts 轉為 input_text / input_image 格式。
    """
    instructions_parts: List[str] = []
    input_items: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role in ("system", "developer"):
            if isinstance(content, str):
                instructions_parts.append(content)
            continue

        if isinstance(content, list):
            converted_content = [_convert_content_part(p) for p in content]
            input_items.append({"role": role, "content": converted_content})
        else:
            input_items.append({"role": role, "content": content})

    instructions = "\n\n".join(p for p in instructions_parts if p) or None
    return instructions, input_items


def convert_tools_to_responses_format(
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """將 Chat Completions tools 格式轉為 Responses API 的扁平 function 格式。

    Chat: {"type": "function", "function": {"name", "description", "parameters"}}
    Responses: {"type": "function", "name", "description", "parameters"}
    """
    converted = []
    for tool in tools:
        fn = tool.get("function", {})
        converted.append(
            {
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        )
    return converted


class OpenAIResponsesModel(OpenAIModel):
    """使用 OpenAI Responses API 的 LLM 實作。

    繼承 OpenAIModel 的客戶端初始化、配置驗證、訊息建構與
    score_continuation（logit 路徑仍走 /v1/completions）。
    """

    def _extract_output(self, response: Any) -> tuple:
        """從 Responses API 回應中抽取 (text, reasoning, tool_calls)。"""
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[ChatCompletionMessageToolCall] = []

        for item in getattr(response, "output", None) or []:
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in getattr(item, "content", None) or []:
                    if getattr(part, "type", None) == "output_text":
                        text_parts.append(getattr(part, "text", "") or "")
            elif item_type == "reasoning":
                for summary in getattr(item, "summary", None) or []:
                    summary_text = getattr(summary, "text", "") or ""
                    if summary_text:
                        reasoning_parts.append(summary_text)
            elif item_type == "function_call":
                arguments = getattr(item, "arguments", "") or "{}"
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments, ensure_ascii=False)
                tool_calls.append(
                    ChatCompletionMessageToolCall.construct(
                        id=getattr(item, "call_id", None) or getattr(item, "id", ""),
                        type="function",
                        function=Function.construct(
                            name=getattr(item, "name", ""),
                            arguments=arguments,
                        ),
                    )
                )

        text = "".join(text_parts) or None
        reasoning = "\n".join(reasoning_parts) or None
        return text, reasoning, tool_calls

    def _to_chat_completion(self, responses: List[Any]) -> ChatCompletion:
        """將一或多個 Responses API 回應合併為單一 ChatCompletion。"""
        choices: List[Choice] = []
        prompt_tokens = 0
        completion_tokens = 0
        has_usage = False

        for idx, response in enumerate(responses):
            text, reasoning, tool_calls = self._extract_output(response)
            message = ChatCompletionMessage.construct(
                role="assistant",
                content=text,
                # 對齊 vLLM reasoning 欄位：evaluator 的 _get_reasoning_text
                # 會優先讀取 message.reasoning
                reasoning=reasoning,
                tool_calls=tool_calls or None,
            )
            choices.append(
                Choice.construct(
                    index=idx,
                    message=message,
                    finish_reason="stop",
                )
            )

            usage = getattr(response, "usage", None)
            if usage is not None:
                has_usage = True
                prompt_tokens += getattr(usage, "input_tokens", 0) or 0
                completion_tokens += getattr(usage, "output_tokens", 0) or 0

        merged_usage = (
            CompletionUsage.construct(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            if has_usage
            else None
        )

        first = responses[0]
        return ChatCompletion.construct(
            id=getattr(first, "id", ""),
            choices=choices,
            created=int(getattr(first, "created_at", 0) or 0),
            model=getattr(first, "model", self.config["model"]["name"]),
            object="chat.completion",
            usage=merged_usage,
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
        """呼叫 Responses API 並回傳 ChatCompletion 相容格式的回應。"""
        if messages is not None:
            built_messages = messages
        else:
            built_messages = self._build_messages(
                question_text, prompt_lang, eval_method, system_prompt_enabled
            )
        model_config = self.config["model"]
        overrides = model_overrides or {}

        instructions, input_items = convert_messages_to_responses_input(built_messages)

        payload: Dict[str, Any] = {
            "model": model_config["name"],
            "input": input_items,
            "temperature": overrides.get("temperature", model_config["temperature"]),
            "top_p": overrides.get("top_p", model_config["top_p"]),
            "max_output_tokens": overrides.get("max_tokens", model_config["max_tokens"]),
        }

        if instructions:
            payload["instructions"] = instructions

        if tools:
            payload["tools"] = convert_tools_to_responses_format(tools)

        if model_config.get("extra_body"):
            payload["extra_body"] = model_config["extra_body"]

        try:
            # Responses API 沒有 n 參數，num_samples > 1 時以多次請求合併
            responses = [
                self.client.responses.create(**payload) for _ in range(max(1, num_samples))
            ]
            return self._to_chat_completion(responses)
        except Exception as e:
            log_error(f"Responses API 錯誤: {e}")
            raise e
