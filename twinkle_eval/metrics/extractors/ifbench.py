"""IFBench Extractor。

IFBench 評測不需要從 response 中提取特定格式，直接 pass-through 原始回答。
scorer 會拿完整的 response 去跑 58 種 OOD instruction checker。
"""

from typing import Any, Optional

from twinkle_eval.core.abc import Extractor


class IFBenchExtractor(Extractor):
    """IFBench Extractor — pass-through 原始 response。

    設定 uses_ifeval=True 讓 evaluator 進入 IFEval 專用路徑
    （IFBench 使用相同的 strict/loose 評分框架）。
    """

    uses_ifeval: bool = True

    def get_name(self) -> str:
        return "ifbench"

    def extract(self, raw: Optional[Any]) -> Optional[Any]:
        """直接回傳原始 response，不做任何提取。"""
        return raw
