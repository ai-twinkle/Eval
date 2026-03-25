"""IFBench Scorer。

封裝 AllenAI IFBench 的 strict/loose evaluation logic，
輸出四個指標：prompt-level 與 instruction-level 各有 strict/loose。

評分方式與 IFEval 相同：
- Strict：對原始 response 跑 checker
- Loose：對 8 種文字變體（去 markdown、去首行、去末行的 2³ 組合）跑 checker，
  任一通過即算通過
- Prompt-level：該題所有 instruction 都過才算這題正確
- Instruction-level：每條 instruction 各自算通過率
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Scorer


def _remove_markdown(text: str) -> str:
    """移除 markdown 粗體/斜體標記（* 與 **）。"""
    return re.sub(r"\*+", "", text)


def _remove_first_line(text: str) -> str:
    """移除第一行。"""
    lines = text.split("\n", 1)
    return lines[1] if len(lines) > 1 else ""


def _remove_last_line(text: str) -> str:
    """移除最後一行。"""
    lines = text.rsplit("\n", 1)
    return lines[0] if len(lines) > 1 else ""


def _get_loose_variants(response: str) -> List[str]:
    """產生 8 種 loose evaluation 文字變體（2³ 組合）。"""
    variants = []
    for remove_md in (False, True):
        for remove_first in (False, True):
            for remove_last in (False, True):
                r = response
                if remove_md:
                    r = _remove_markdown(r)
                if remove_first:
                    r = _remove_first_line(r)
                if remove_last:
                    r = _remove_last_line(r)
                variants.append(r)
    return variants


def _check_instruction(
    instruction_id: str,
    kwargs: dict,
    response: str,
    prompt: Optional[str] = None,
) -> bool:
    """對單一 response 跑單一 IFBench instruction checker。

    IFBench kwargs 包含 IFEval 繼承的空值欄位，必須過濾掉 None 值。
    某些 checker（如 RepeatChangeChecker）需要 prompt 參數。
    """
    try:
        from twinkle_eval.metrics.checkers.ifbench import INSTRUCTION_DICT
    except ImportError as e:
        raise ImportError(
            "IFBench checkers 載入失敗，請確認 emoji、syllapy 與 nltk 已安裝：\n"
            "  pip install twinkle-eval[ifbench]\n"
            "  python -c \"import nltk; nltk.download('punkt_tab'); "
            "nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger_eng')\""
        ) from e

    if instruction_id not in INSTRUCTION_DICT:
        return False

    checker_cls = INSTRUCTION_DICT[instruction_id]
    checker = checker_cls(instruction_id)

    # Filter out None values (IFEval-inherited kwargs that are always null)
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    checker.build_description(**filtered_kwargs)

    # Special handling: some checkers need prompt re-injection
    args = checker.get_instruction_args()
    if args and "prompt" in args and prompt:
        checker.build_description(prompt=prompt)

    try:
        if not response or not response.strip():
            return False
        return bool(checker.check_following(response))
    except Exception:
        return False


def score_ifbench(
    response: str,
    instruction_id_list: List[str],
    kwargs_list: List[dict],
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """對單一 response 計算所有 IFBench 指標。

    Args:
        response:             模型輸出的完整回答
        instruction_id_list:  該題的 instruction ID 列表
        kwargs_list:          對應每個 instruction 的參數字典列表
        prompt:               原始 prompt（某些 checker 如 RepeatChangeChecker 需要）

    Returns:
        {
            "prompt_strict": bool,
            "prompt_loose": bool,
            "instruction_strict": list[bool],
            "instruction_loose": list[bool],
        }
    """
    strict_results: List[bool] = []
    loose_results: List[bool] = []
    loose_variants = _get_loose_variants(response)

    for inst_id, kw in zip(instruction_id_list, kwargs_list):
        # Strict
        strict_ok = _check_instruction(inst_id, kw, response, prompt=prompt)
        strict_results.append(strict_ok)

        # Loose：任一 variant 通過即可
        loose_ok = any(
            _check_instruction(inst_id, kw, variant, prompt=prompt)
            for variant in loose_variants
        )
        loose_results.append(loose_ok)

    return {
        "prompt_strict": all(strict_results),
        "prompt_loose": all(loose_results),
        "instruction_strict": strict_results,
        "instruction_loose": loose_results,
    }


class IFBenchScorer(Scorer):
    """IFBench Scorer。

    `normalize()` 解析 response JSON string。
    `score()` 回傳 prompt-level strict 正確與否（作為主要 is_correct）。

    完整的四個指標透過 `score_full()` 取得。
    """

    def get_name(self) -> str:
        return "ifbench"

    def normalize(self, raw: Any) -> Any:
        """Pass-through：raw 就是 response string。"""
        return raw

    def score(self, predicted: Any, ground_truth: Any) -> bool:
        """回傳 prompt-level strict 正確與否。

        ground_truth 預期為 JSON string：
            {"instruction_id_list": [...], "kwargs": [...]}
        """
        if not predicted or not ground_truth:
            return False
        try:
            gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
            inst_ids = gt["instruction_id_list"]
            kwargs_list = gt["kwargs"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return False

        result = score_ifbench(predicted, inst_ids, kwargs_list)
        return result["prompt_strict"]

    def score_full(
        self,
        response: str,
        instruction_id_list: List[str],
        kwargs_list: List[dict],
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """回傳完整四個 IFBench 指標。"""
        return score_ifbench(response, instruction_id_list, kwargs_list, prompt=prompt)
