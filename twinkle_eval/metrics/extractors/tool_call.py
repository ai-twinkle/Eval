"""FC 模式 Tool Call 抽取器。"""

import json
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Extractor


def convert_bfcl_functions_to_tools(functions: List[Dict]) -> List[Dict]:
    """將 BFCL function 定義轉換為 OpenAI tools 格式。

    主要差異：
    - BFCL 使用 "type": "dict"，OpenAI 要求 "type": "object"
    - BFCL function name 允許含 "."，OpenAI 不支援（轉成 "_"）
    """
    import copy

    tools = []
    for func in functions:
        func = copy.deepcopy(func)
        func["name"] = func["name"].replace(".", "_")
        if "parameters" in func:
            _convert_types_recursive(func["parameters"])
        tools.append({"type": "function", "function": func})
    return tools


def _convert_types_recursive(schema: Dict) -> None:
    """遞迴將 BFCL type 轉換為 OpenAI 相容 type。"""
    if schema.get("type") in ("dict",):
        schema["type"] = "object"
    for prop in schema.get("properties", {}).values():
        _convert_types_recursive(prop)
    if "items" in schema:
        _convert_types_recursive(schema["items"])


class ToolCallExtractor(Extractor):
    """FC 模式：從 ChatCompletion response 的 tool_calls 欄位抽取。

    Evaluator 在偵測到 uses_tool_calls=True 後，會將 response 的 tool_calls
    序列化為 JSON 字串傳入 extract()，而非傳入 message.content。

    extract() 輸入格式：
        '[{"name": "func", "arguments": {"param": "value"}}, ...]'

    extract() 輸出格式（同輸入，供 BFCLScorer 比對）：
        '[{"name": "func", "arguments": {"param": "value"}}, ...]'
    """

    uses_tool_calls: bool = True

    def get_name(self) -> str:
        return "tool_call"

    def extract(self, llm_output: str) -> Optional[str]:
        """解析 JSON 序列化的 tool_calls，回傳標準化格式。"""
        if not llm_output or not llm_output.strip():
            return None
        try:
            calls = json.loads(llm_output)
            if not calls:
                return None
            return json.dumps(calls, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return None
