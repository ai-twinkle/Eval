"""BFCL ground truth 比對評分器。"""

import json
import re
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Scorer


# ── 值比對工具 ──────────────────────────────────────────────────────────────────


def _normalize_str(value: str) -> str:
    """正規化字串以便比較（對齊 BFCL ast_checker 邏輯）。"""
    return re.sub(r"[\s.,;:'\"!?]", "", str(value).lower().strip())


def _values_match(predicted_val: Any, acceptable_vals: List[Any]) -> bool:
    """判斷預測值是否符合任一可接受值。

    acceptable_vals 是 BFCL ground truth 中的值列表：
    - 空字串 "" 代表此參數可省略
    - 其餘為可接受的具體值（可有多個同義寫法）
    """
    non_empty = [v for v in acceptable_vals if v != ""]
    if not non_empty:
        # 全部都是 ""，代表此參數完全 optional，任何值都接受
        return True

    for acceptable in non_empty:
        if isinstance(acceptable, str) and isinstance(predicted_val, str):
            if _normalize_str(predicted_val) == _normalize_str(str(acceptable)):
                return True
        elif isinstance(acceptable, list) and isinstance(predicted_val, list):
            if len(acceptable) == len(predicted_val) and all(
                _values_match(p, [a]) for p, a in zip(predicted_val, acceptable)
            ):
                return True
        elif isinstance(acceptable, bool) or isinstance(predicted_val, bool):
            # bool 必須在 numeric 比較之前處理，避免 True==1 的誤判
            if acceptable is predicted_val:
                return True
        else:
            try:
                if type(predicted_val)(acceptable) == predicted_val:
                    return True
            except (TypeError, ValueError):
                pass
            if acceptable == predicted_val:
                return True
    return False


def _check_required_params(predicted_args: Dict, gt_params: Dict) -> bool:
    """驗證必填參數都有提供且值正確；選填參數（含 "" 的）若有提供則需正確。"""
    for param, acceptable_vals in gt_params.items():
        is_optional = "" in acceptable_vals
        non_empty = [v for v in acceptable_vals if v != ""]

        if is_optional:
            # 選填：若模型提供了值，該值必須正確
            if param in predicted_args and non_empty:
                if not _values_match(predicted_args[param], non_empty):
                    return False
        else:
            # 必填：必須存在且正確
            if param not in predicted_args:
                return False
            if not _values_match(predicted_args[param], acceptable_vals):
                return False
    return True


def _call_matches(predicted_call: Dict, gt_call: Dict) -> bool:
    """判斷一個預測 function call 是否符合 ground truth entry。

    Args:
        predicted_call: {"name": "func_name", "arguments": {"param": value}}
        gt_call: {"func_name": {"param": [acceptable_values...]}}
    """
    gt_func_name = list(gt_call.keys())[0]
    gt_params: Dict[str, List] = gt_call[gt_func_name]

    pred_name = predicted_call.get("name", "")
    pred_args = predicted_call.get("arguments", {})
    if isinstance(pred_args, str):
        try:
            pred_args = json.loads(pred_args)
        except (json.JSONDecodeError, TypeError):
            pred_args = {}

    # function name 比對（統一將 "." 替換為 "_"，對齊 OpenAI FC 模式轉換）
    normalized_pred = pred_name.replace(".", "_")
    normalized_gt = gt_func_name.replace(".", "_")
    if normalized_pred != normalized_gt:
        return False

    return _check_required_params(pred_args, gt_params)


def _score_ordered(predicted: List[Dict], ground_truth: List[Dict]) -> bool:
    """有序比對（simple / multiple / live_simple 等）。"""
    if len(predicted) != len(ground_truth):
        return False
    return all(_call_matches(p, g) for p, g in zip(predicted, ground_truth))


def _score_unordered(predicted: List[Dict], ground_truth: List[Dict]) -> bool:
    """無序比對（parallel / live_parallel 等）。"""
    if len(predicted) != len(ground_truth):
        return False
    used = [False] * len(ground_truth)
    for pred_call in predicted:
        matched = False
        for i, gt_call in enumerate(ground_truth):
            if not used[i] and _call_matches(pred_call, gt_call):
                used[i] = True
                matched = True
                break
        if not matched:
            return False
    return True


# ── BFCLScorer ──────────────────────────────────────────────────────────────────


class BFCLScorer(Scorer):
    """BFCL ground truth 比對評分器。

    FC 模式與 Prompting 模式共用此 scorer。

    gold 格式（JSON 字串）：
        {"category": "simple|parallel|multiple|live_simple|...",
         "ground_truth": [{"func_name": {"param": [acceptable_values...]}}, ...]}

    predicted 格式（JSON 字串，由 ToolCallExtractor 或 BFCLPromptExtractor 產生）：
        [{"name": "func_name", "arguments": {"param": value}}, ...]

    Category 決定比對策略：
    - "parallel" 在名稱中 → 無序比對
    - 其餘（simple / multiple / live_simple / live_multiple 等）→ 有序比對
    """

    def get_name(self) -> str:
        return "bfcl"

    def normalize(self, answer: str) -> str:
        """BFCL 答案已是 JSON 字串，直接 strip 後回傳。"""
        return answer.strip() if answer else ""

    def score(self, predicted: str, gold: str) -> bool:
        """比對預測 tool calls 與 BFCL ground truth。"""
        try:
            predicted_calls: List[Dict] = json.loads(predicted)
            gold_data: Dict = json.loads(gold)
        except (json.JSONDecodeError, TypeError, ValueError):
            return False

        ground_truth: List[Dict] = gold_data.get("ground_truth", [])
        category: str = gold_data.get("category", "simple")

        if not predicted_calls or not ground_truth:
            return False

        if "parallel" in category:
            return _score_unordered(predicted_calls, ground_truth)
        else:
            return _score_ordered(predicted_calls, ground_truth)
