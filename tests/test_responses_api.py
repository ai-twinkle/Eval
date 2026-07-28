"""OpenAI Responses API 後端（models/responses.py）的單元測試。

全部為 pure unit test，不呼叫任何外部 API。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from twinkle_eval.models import LLMFactory, OpenAIResponsesModel
from twinkle_eval.models.responses import (
    convert_messages_to_responses_input,
    convert_tools_to_responses_format,
)


def _make_config() -> dict:
    return {
        "llm_api": {
            "base_url": "http://localhost:9999/v1",
            "api_key": "test-key",
            "max_retries": 1,
            "timeout": 5,
            "api_rate_limit": -1,
        },
        "model": {
            "name": "test-model",
            "temperature": 0.0,
            "top_p": 0.9,
            "max_tokens": 128,
            "extra_body": None,
        },
        "evaluation": {
            "evaluation_method": "box",
            "system_prompt": {"zh": "測試系統提示", "en": "test system prompt"},
        },
    }


def _make_model() -> OpenAIResponsesModel:
    model = OpenAIResponsesModel(_make_config())
    model.client = MagicMock()
    return model


def _fake_response(
    text: str = "答案是 \\boxed{A}",
    reasoning: str | None = None,
    tool_call: dict | None = None,
    input_tokens: int = 10,
    output_tokens: int = 20,
) -> SimpleNamespace:
    output = []
    if reasoning is not None:
        output.append(
            SimpleNamespace(
                type="reasoning",
                summary=[SimpleNamespace(text=reasoning)],
            )
        )
    if text is not None:
        output.append(
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text=text)],
            )
        )
    if tool_call is not None:
        output.append(
            SimpleNamespace(
                type="function_call",
                call_id=tool_call.get("call_id", "call_1"),
                name=tool_call["name"],
                arguments=tool_call.get("arguments", "{}"),
            )
        )
    return SimpleNamespace(
        id="resp_123",
        created_at=1700000000,
        model="test-model",
        output=output,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class TestFactoryRegistration:
    def test_openai_responses_registered(self):
        assert "openai_responses" in LLMFactory.get_available_types()

    def test_create_via_factory(self):
        llm = LLMFactory.create_llm("openai_responses", _make_config())
        assert isinstance(llm, OpenAIResponsesModel)


class TestMessageConversion:
    def test_system_message_becomes_instructions(self):
        instructions, input_items = convert_messages_to_responses_input(
            [
                {"role": "system", "content": "你是助理"},
                {"role": "user", "content": "問題"},
            ]
        )
        assert instructions == "你是助理"
        assert input_items == [{"role": "user", "content": "問題"}]

    def test_no_system_message(self):
        instructions, input_items = convert_messages_to_responses_input(
            [{"role": "user", "content": "問題"}]
        )
        assert instructions is None
        assert len(input_items) == 1

    def test_multimodal_content_converted(self):
        _, input_items = convert_messages_to_responses_input(
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,xx", "detail": "auto"},
                        },
                        {"type": "text", "text": "這是什麼？"},
                    ],
                }
            ]
        )
        parts = input_items[0]["content"]
        assert parts[0] == {
            "type": "input_image",
            "image_url": "data:image/png;base64,xx",
            "detail": "auto",
        }
        assert parts[1] == {"type": "input_text", "text": "這是什麼？"}


class TestToolConversion:
    def test_chat_tools_flattened(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "查天氣",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        converted = convert_tools_to_responses_format(tools)
        assert converted == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "查天氣",
                "parameters": {"type": "object", "properties": {}},
            }
        ]


class TestCall:
    def test_basic_call_returns_chat_completion(self):
        model = _make_model()
        model.client.responses.create.return_value = _fake_response()

        completion = model.call("測試問題")

        assert completion.object == "chat.completion"
        assert completion.choices[0].message.content == "答案是 \\boxed{A}"
        assert completion.usage.prompt_tokens == 10
        assert completion.usage.completion_tokens == 20
        assert completion.usage.total_tokens == 30

    def test_payload_uses_max_output_tokens(self):
        model = _make_model()
        model.client.responses.create.return_value = _fake_response()

        model.call("測試問題")

        payload = model.client.responses.create.call_args.kwargs
        assert payload["max_output_tokens"] == 128
        assert "max_tokens" not in payload
        assert payload["model"] == "test-model"

    def test_system_prompt_becomes_instructions(self):
        model = _make_model()
        model.client.responses.create.return_value = _fake_response()

        model.call("測試問題", eval_method="box", system_prompt_enabled=True)

        payload = model.client.responses.create.call_args.kwargs
        assert payload["instructions"] == "測試系統提示"

    def test_reasoning_mapped_to_message_reasoning(self):
        model = _make_model()
        model.client.responses.create.return_value = _fake_response(
            text="\\boxed{B}", reasoning="先思考選項"
        )

        completion = model.call("測試問題")
        message = completion.choices[0].message

        # evaluator._get_reasoning_text 讀取 message.reasoning
        assert message.reasoning == "先思考選項"
        assert message.content == "\\boxed{B}"

    def test_function_call_converted_to_tool_calls(self):
        model = _make_model()
        model.client.responses.create.return_value = _fake_response(
            text=None,
            tool_call={"name": "get_weather", "arguments": '{"city": "Taipei"}'},
        )

        completion = model.call("測試問題", tools=[{"type": "function", "function": {}}])
        tool_calls = completion.choices[0].message.tool_calls

        assert len(tool_calls) == 1
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[0].function.arguments == '{"city": "Taipei"}'

    def test_num_samples_merges_multiple_requests(self):
        model = _make_model()
        model.client.responses.create.side_effect = [
            _fake_response(text="第一次"),
            _fake_response(text="第二次"),
        ]

        completion = model.call("測試問題", num_samples=2)

        assert model.client.responses.create.call_count == 2
        assert [c.message.content for c in completion.choices] == ["第一次", "第二次"]
        # usage 為兩次請求加總
        assert completion.usage.prompt_tokens == 20
        assert completion.usage.total_tokens == 60

    def test_missing_usage_yields_none(self):
        model = _make_model()
        resp = _fake_response()
        resp.usage = None
        model.client.responses.create.return_value = resp

        completion = model.call("測試問題")
        assert completion.usage is None

    def test_api_error_propagates(self):
        model = _make_model()
        model.client.responses.create.side_effect = RuntimeError("connection failed")

        with pytest.raises(RuntimeError):
            model.call("測試問題")
