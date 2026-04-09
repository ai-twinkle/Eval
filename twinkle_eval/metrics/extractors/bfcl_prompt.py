"""Prompting 模式 BFCL 抽取器（對齊官方 default system prompt 格式）。"""

import ast
import json
import re
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Extractor

# ── BFCL 官方 default system prompt 組成（ret_fmt=python, func_doc_fmt=json, style=classic）──

_PERSONA = "You are an expert in composing functions."

_TASK = (
    "You are given a question and a set of possible functions. "
    "Based on the question, you will need to make one or more function/tool calls "
    "to achieve the purpose. "
    "If none of the functions can be used, point it out. "
    "If the given question lacks the parameters required by the function, also point it out."
)

_TOOL_CALL_FORMAT = (
    "You should only return the function calls in your response.\n\n"
    "If you decide to invoke any of the function(s), you MUST put it in the format of "
    "[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]. "
    "You SHOULD NOT include any other text in the response."
)

_MULTITURN_BEHAVIOR = (
    "At each turn, you should try your best to complete the tasks requested by the user "
    "within the current turn. Continue to output functions to call until you have fulfilled "
    "the user's request to the best of your ability. Once you have no more functions to call, "
    "the system will consider the current turn complete and proceed to the next turn or task."
)


def build_bfcl_system_prompt(functions: List[Dict]) -> str:
    """依 BFCL 官方 default 格式建構 system prompt。

    對齊 BFCL DEFAULT_SYSTEM_PROMPT_FORMAT:
        ret_fmt=python & tool_call_tag=False & func_doc_fmt=json & prompt_fmt=plaintext & style=classic
    """
    func_doc = json.dumps(functions, indent=4, ensure_ascii=False)
    available_tools = (
        f"Here is a list of functions in JSON format that you can invoke.\n{func_doc}\n"
    )
    return (
        f"{_PERSONA}{_TASK}\n\n"
        f"{_TOOL_CALL_FORMAT}\n\n"
        f"{_MULTITURN_BEHAVIOR}\n\n"
        f"{available_tools}"
    )


def inject_bfcl_system_prompt(messages: List[Dict], functions: List[Dict]) -> List[Dict]:
    """將 function 定義以 BFCL system prompt 格式注入 messages。

    若 messages 第一條已是 system，則將 BFCL prompt 前置（對齊官方實作）；
    否則在最前面插入一條 system message。
    """
    system_prompt = build_bfcl_system_prompt(functions)
    messages = list(messages)
    if messages and messages[0]["role"] == "system":
        messages[0] = {
            **messages[0],
            "content": system_prompt + "\n\n" + messages[0]["content"],
        }
    else:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


# ── Output parser：BFCL Python 格式 → structured dict ──────────────────────────────


def _resolve_ast_call(call_node: ast.Call) -> Dict:
    """將 ast.Call 節點解析為 {func_name: {param: value}}。"""
    if isinstance(call_node.func, ast.Attribute):
        func_name = f"{ast.unparse(call_node.func.value)}.{call_node.func.attr}"
    elif isinstance(call_node.func, ast.Name):
        func_name = call_node.func.id
    else:
        func_name = ast.unparse(call_node.func)

    args: Dict[str, Any] = {}
    for kw in call_node.keywords:
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            args[kw.arg] = ast.unparse(kw.value)

    return {func_name: args}


def _try_parse_block(text: str) -> Optional[List[Dict]]:
    """嘗試將單一文字區塊解析為 BFCL Python call 列表。"""
    text = text.strip().strip("`\n ")
    if not text.startswith("["):
        text = "[" + text
    if not text.endswith("]"):
        text = text + "]"
    try:
        cleaned = text.strip().strip("'")
        parsed = ast.parse(cleaned, mode="eval")
        results = []
        body = parsed.body
        if isinstance(body, ast.Call):
            results.append(_resolve_ast_call(body))
        elif isinstance(body, ast.List):
            for elem in body.elts:
                if isinstance(elem, ast.Call):
                    results.append(_resolve_ast_call(elem))
        return results if results else None
    except (SyntaxError, AttributeError, ValueError, TypeError):
        return None


def parse_bfcl_python_output(text: str) -> Optional[List[Dict]]:
    """解析 BFCL prompting 模式的 Python 格式輸出。

    期望格式：[func_name1(param1=val1, param2=val2), func_name2(param=val)]
    對齊 BFCL 官方 default_decode_ast_prompting() 的邏輯。

    推理模型（reasoning model）可能會在 reasoning / reasoning_content 中把思考過程放在答案之前，
    例如：「Let me check... [func_name(param=val)]」
    此時直接解析全文會失敗，需要從後往前找最後一個 [identifier( 區塊。
    """
    text = text.strip()
    if not text:
        return None

    # 先嘗試直接解析（適用於乾淨輸出）
    result = _try_parse_block(text)
    if result is not None:
        return result

    # 找最後一個 [identifier( 區塊（適用於推理模型：思考 + 答案混在一起）
    matches = list(re.finditer(r"\[[A-Za-z_]", text))
    if matches:
        last_start = matches[-1].start()
        candidate = text[last_start:]
        result = _try_parse_block(candidate)
        if result is not None:
            return result

    return None


class BFCLPromptExtractor(Extractor):
    """Prompting 模式：從模型文字輸出解析 BFCL Python 格式 function call。

    Evaluator 在偵測到 uses_prompt_injection=True 後，會在呼叫 LLM 之前
    將 function 定義注入 system prompt（使用 inject_bfcl_system_prompt()），
    模型輸出的文字格式為：
        [func_name(param1=val1, param2=val2), ...]

    extract() 輸出格式（與 ToolCallExtractor 相同，供 BFCLScorer 比對）：
        '[{"name": "func", "arguments": {"param": "value"}}, ...]'
    """

    uses_prompt_injection: bool = True

    def get_name(self) -> str:
        return "bfcl_prompt"

    def extract(self, llm_output: str) -> Optional[str]:
        """解析 BFCL Python 格式輸出，轉成標準 tool call JSON 字串。"""
        if not llm_output or not llm_output.strip():
            return None

        calls = parse_bfcl_python_output(llm_output)
        if calls is None:
            return None

        # 轉換為統一格式：[{"name": "func", "arguments": {...}}]
        standard = [
            {"name": list(c.keys())[0], "arguments": list(c.values())[0]}
            for c in calls
        ]
        return json.dumps(standard, ensure_ascii=False)
