"""數學題答案抽取器（從 \\boxed{} 中提取數學表達式）。"""

import re
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Extractor


class MathExtractor(Extractor):
    """從 \\boxed{} 中提取數學答案。

    使用括號計數法處理巢狀大括號，支援複雜數學表達式。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def get_name(self) -> str:
        return "math"

    @staticmethod
    def _extract_boxed_content(text: str) -> List[str]:
        """用括號計數提取所有 \\boxed{...} 的內容，支援巢狀大括號。"""
        results = []
        for match in re.finditer(r"\\{1,2}boxed?\{", text):
            start = match.end()
            depth = 1
            i = start
            while i < len(text) and depth > 0:
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                i += 1
            if depth == 0:
                results.append(text[start:i - 1])
        return results

    def extract(self, llm_output: str) -> Optional[str]:
        """從 \\boxed{} 中提取數學答案；若無則嘗試取最後一行非空文字。"""
        if not self.validate_output(llm_output):
            return None

        matches = self._extract_boxed_content(llm_output)
        if matches:
            return matches[-1].strip()

        # fallback：取最後一行非空內容
        lines = [line.strip() for line in llm_output.strip().splitlines() if line.strip()]
        return lines[-1] if lines else None
