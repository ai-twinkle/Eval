"""BFCL 評測元件測試：ToolCallExtractor、BFCLPromptExtractor、BFCLScorer。"""

import json
import pytest

from twinkle_eval.metrics.extractors.tool_call import (
    ToolCallExtractor,
    convert_bfcl_functions_to_tools,
)
from twinkle_eval.metrics.extractors.bfcl_prompt import (
    BFCLPromptExtractor,
    build_bfcl_system_prompt,
    inject_bfcl_system_prompt,
    parse_bfcl_python_output,
)
from twinkle_eval.metrics.scorers.bfcl import BFCLScorer


# ── convert_bfcl_functions_to_tools ──────────────────────────────────────────

class TestConvertBfclFunctionsToTools:
    def test_basic_conversion(self):
        functions = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            }
        ]
        tools = convert_bfcl_functions_to_tools(functions)
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"

    def test_dict_to_object_conversion(self):
        functions = [
            {
                "name": "func",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "nested": {"type": "dict"},
                    },
                },
            }
        ]
        tools = convert_bfcl_functions_to_tools(functions)
        assert tools[0]["function"]["parameters"]["type"] == "object"
        assert tools[0]["function"]["parameters"]["properties"]["nested"]["type"] == "object"

    def test_dot_to_underscore_in_name(self):
        functions = [{"name": "module.sub.function", "parameters": {}}]
        tools = convert_bfcl_functions_to_tools(functions)
        assert tools[0]["function"]["name"] == "module_sub_function"

    def test_does_not_mutate_original(self):
        functions = [{"name": "f", "parameters": {"type": "dict"}}]
        original_type = functions[0]["parameters"]["type"]
        convert_bfcl_functions_to_tools(functions)
        assert functions[0]["parameters"]["type"] == original_type

    def test_array_items_conversion(self):
        functions = [
            {
                "name": "func",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "items": {"type": "dict"}},
                    },
                },
            }
        ]
        tools = convert_bfcl_functions_to_tools(functions)
        assert tools[0]["function"]["parameters"]["properties"]["items"]["items"]["type"] == "object"


# ── ToolCallExtractor ─────────────────────────────────────────────────────────

class TestToolCallExtractor:
    def setup_method(self):
        self.extractor = ToolCallExtractor()

    def test_uses_tool_calls_flag(self):
        assert self.extractor.uses_tool_calls is True

    def test_get_name(self):
        assert self.extractor.get_name() == "tool_call"

    def test_extract_valid_json(self):
        calls = [{"name": "get_weather", "arguments": {"location": "Tokyo"}}]
        result = self.extractor.extract(json.dumps(calls))
        assert json.loads(result) == calls

    def test_extract_none_on_empty(self):
        assert self.extractor.extract("") is None
        assert self.extractor.extract(None) is None

    def test_extract_none_on_invalid_json(self):
        assert self.extractor.extract("not json") is None

    def test_extract_none_on_empty_list(self):
        assert self.extractor.extract("[]") is None

    def test_extract_multiple_calls(self):
        calls = [
            {"name": "func1", "arguments": {"a": 1}},
            {"name": "func2", "arguments": {"b": "two"}},
        ]
        result = self.extractor.extract(json.dumps(calls))
        assert len(json.loads(result)) == 2


# ── parse_bfcl_python_output ──────────────────────────────────────────────────

class TestParseBfclPythonOutput:
    def test_single_call(self):
        result = parse_bfcl_python_output("[get_weather(location='Tokyo', unit='celsius')]")
        assert result is not None
        assert len(result) == 1
        assert result[0] == {"get_weather": {"location": "Tokyo", "unit": "celsius"}}

    def test_multiple_calls(self):
        result = parse_bfcl_python_output(
            "[get_weather(location='Tokyo'), get_time(timezone='JST')]"
        )
        assert result is not None
        assert len(result) == 2

    def test_without_brackets(self):
        result = parse_bfcl_python_output("get_weather(location='Tokyo')")
        assert result is not None
        assert result[0] == {"get_weather": {"location": "Tokyo"}}

    def test_numeric_args(self):
        result = parse_bfcl_python_output("[func(x=42, y=3.14)]")
        assert result is not None
        assert result[0] == {"func": {"x": 42, "y": 3.14}}

    def test_returns_none_on_garbage(self):
        result = parse_bfcl_python_output("this is not a function call")
        assert result is None

    def test_bool_args(self):
        result = parse_bfcl_python_output("[func(flag=True)]")
        assert result is not None
        assert result[0]["func"]["flag"] is True


# ── BFCLPromptExtractor ───────────────────────────────────────────────────────

class TestBFCLPromptExtractor:
    def setup_method(self):
        self.extractor = BFCLPromptExtractor()

    def test_uses_prompt_injection_flag(self):
        assert self.extractor.uses_prompt_injection is True

    def test_get_name(self):
        assert self.extractor.get_name() == "bfcl_prompt"

    def test_extract_single_call(self):
        output = "[get_weather(location='Tokyo', unit='celsius')]"
        result = self.extractor.extract(output)
        assert result is not None
        parsed = json.loads(result)
        assert parsed == [{"name": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}}]

    def test_extract_multiple_calls(self):
        output = "[func1(a=1), func2(b='hello')]"
        result = self.extractor.extract(output)
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "func1"
        assert parsed[1]["name"] == "func2"

    def test_extract_none_on_empty(self):
        assert self.extractor.extract("") is None
        assert self.extractor.extract(None) is None

    def test_extract_none_on_garbage(self):
        result = self.extractor.extract("I cannot answer this question.")
        assert result is None


# ── inject_bfcl_system_prompt ─────────────────────────────────────────────────

class TestInjectBfclSystemPrompt:
    FUNCTIONS = [{"name": "get_weather", "description": "Get weather", "parameters": {}}]

    def test_inject_no_existing_system(self):
        messages = [{"role": "user", "content": "What is the weather?"}]
        result = inject_bfcl_system_prompt(messages, self.FUNCTIONS)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "get_weather" in result[0]["content"]

    def test_inject_prepends_to_existing_system(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = inject_bfcl_system_prompt(messages, self.FUNCTIONS)
        assert result[0]["role"] == "system"
        assert "get_weather" in result[0]["content"]
        assert "You are helpful." in result[0]["content"]

    def test_does_not_mutate_original(self):
        messages = [{"role": "user", "content": "Hello"}]
        inject_bfcl_system_prompt(messages, self.FUNCTIONS)
        assert len(messages) == 1


# ── BFCLScorer ────────────────────────────────────────────────────────────────

class TestBFCLScorer:
    def setup_method(self):
        self.scorer = BFCLScorer()

    def _gold(self, category: str, ground_truth: list) -> str:
        return json.dumps({"category": category, "ground_truth": ground_truth})

    def _pred(self, calls: list) -> str:
        return json.dumps(calls)

    # basic

    def test_get_name(self):
        assert self.scorer.get_name() == "bfcl"

    def test_normalize_strips(self):
        assert self.scorer.normalize("  hello  ") == "hello"

    # simple ordered match

    def test_simple_correct(self):
        gold = self._gold("simple", [{"get_weather": {"location": ["Tokyo"]}}])
        pred = self._pred([{"name": "get_weather", "arguments": {"location": "Tokyo"}}])
        assert self.scorer.score(pred, gold) is True

    def test_simple_wrong_function(self):
        gold = self._gold("simple", [{"get_weather": {"location": ["Tokyo"]}}])
        pred = self._pred([{"name": "get_time", "arguments": {"location": "Tokyo"}}])
        assert self.scorer.score(pred, gold) is False

    def test_simple_wrong_value(self):
        gold = self._gold("simple", [{"get_weather": {"location": ["Tokyo"]}}])
        pred = self._pred([{"name": "get_weather", "arguments": {"location": "Osaka"}}])
        assert self.scorer.score(pred, gold) is False

    def test_simple_missing_required_param(self):
        gold = self._gold("simple", [{"get_weather": {"location": ["Tokyo"]}}])
        pred = self._pred([{"name": "get_weather", "arguments": {}}])
        assert self.scorer.score(pred, gold) is False

    # optional params

    def test_optional_param_may_be_absent(self):
        gold = self._gold("simple", [{"func": {"required_p": ["val"], "optional_p": ["", "celsius"]}}])
        pred = self._pred([{"name": "func", "arguments": {"required_p": "val"}}])
        assert self.scorer.score(pred, gold) is True

    def test_optional_param_wrong_value(self):
        gold = self._gold("simple", [{"func": {"required_p": ["val"], "optional_p": ["", "celsius"]}}])
        pred = self._pred([{"name": "func", "arguments": {"required_p": "val", "optional_p": "fahrenheit"}}])
        assert self.scorer.score(pred, gold) is False

    # multiple acceptable values

    def test_multiple_acceptable_values(self):
        gold = self._gold("simple", [{"func": {"city": ["Tokyo", "tokyo", "TOKYO"]}}])
        pred = self._pred([{"name": "func", "arguments": {"city": "tokyo"}}])
        assert self.scorer.score(pred, gold) is True

    # parallel unordered

    def test_parallel_correct_ordered(self):
        gold = self._gold("parallel", [
            {"func1": {"a": ["1"]}},
            {"func2": {"b": ["2"]}},
        ])
        pred = self._pred([
            {"name": "func1", "arguments": {"a": "1"}},
            {"name": "func2", "arguments": {"b": "2"}},
        ])
        assert self.scorer.score(pred, gold) is True

    def test_parallel_correct_unordered(self):
        gold = self._gold("parallel", [
            {"func1": {"a": ["1"]}},
            {"func2": {"b": ["2"]}},
        ])
        pred = self._pred([
            {"name": "func2", "arguments": {"b": "2"}},
            {"name": "func1", "arguments": {"a": "1"}},
        ])
        assert self.scorer.score(pred, gold) is True

    def test_parallel_wrong_count(self):
        gold = self._gold("parallel", [
            {"func1": {"a": ["1"]}},
            {"func2": {"b": ["2"]}},
        ])
        pred = self._pred([{"name": "func1", "arguments": {"a": "1"}}])
        assert self.scorer.score(pred, gold) is False

    # multiple function calls (ordered)

    def test_multiple_ordered_correct(self):
        gold = self._gold("multiple", [
            {"step1": {"x": ["hello"]}},
            {"step2": {"y": ["world"]}},
        ])
        pred = self._pred([
            {"name": "step1", "arguments": {"x": "hello"}},
            {"name": "step2", "arguments": {"y": "world"}},
        ])
        assert self.scorer.score(pred, gold) is True

    def test_multiple_wrong_order(self):
        gold = self._gold("multiple", [
            {"step1": {"x": ["hello"]}},
            {"step2": {"y": ["world"]}},
        ])
        pred = self._pred([
            {"name": "step2", "arguments": {"y": "world"}},
            {"name": "step1", "arguments": {"x": "hello"}},
        ])
        assert self.scorer.score(pred, gold) is False

    # edge cases

    def test_invalid_pred_json(self):
        gold = self._gold("simple", [{"func": {"a": ["1"]}}])
        assert self.scorer.score("not json", gold) is False

    def test_invalid_gold_json(self):
        pred = self._pred([{"name": "func", "arguments": {"a": "1"}}])
        assert self.scorer.score(pred, "not json") is False

    def test_empty_pred(self):
        gold = self._gold("simple", [{"func": {"a": ["1"]}}])
        assert self.scorer.score("[]", gold) is False

    def test_numeric_type_coercion(self):
        gold = self._gold("simple", [{"func": {"count": [5]}}])
        pred = self._pred([{"name": "func", "arguments": {"count": 5}}])
        assert self.scorer.score(pred, gold) is True

    def test_bool_not_confused_with_int(self):
        gold = self._gold("simple", [{"func": {"flag": [True]}}])
        pred = self._pred([{"name": "func", "arguments": {"flag": 1}}])
        # True != 1 in strict bool comparison
        assert self.scorer.score(pred, gold) is False

    def test_dot_to_underscore_in_pred(self):
        # Function name with _ (already normalized by ToolCallExtractor)
        gold = self._gold("simple", [{"module_func": {"a": ["val"]}}])
        pred = self._pred([{"name": "module_func", "arguments": {"a": "val"}}])
        assert self.scorer.score(pred, gold) is True

    def test_function_name_dot_vs_underscore(self):
        # BFCLScorer normalizes . → _ in both predicted and ground truth
        gold = self._gold("simple", [{"module.func": {"a": ["val"]}}])
        pred = self._pred([{"name": "module_func", "arguments": {"a": "val"}}])
        assert self.scorer.score(pred, gold) is True

    def test_string_normalization(self):
        # Punctuation and case should be stripped
        gold = self._gold("simple", [{"func": {"city": ["New York, USA"]}}])
        pred = self._pred([{"name": "func", "arguments": {"city": "new york usa"}}])
        assert self.scorer.score(pred, gold) is True
