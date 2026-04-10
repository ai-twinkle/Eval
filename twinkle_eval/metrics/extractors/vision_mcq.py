"""Vision MCQ（視覺多選題）Extractor。

從 VLM（Vision Language Model）的文字輸出中提取選擇題答案，
支援字母選項（A/B/C/D...）以及 Yes/No 二元判斷（用於 POPE 等幻覺偵測 benchmark）。

Vision 評測的圖片載入與 multimodal message 建構由 evaluator.py 處理；
本 Extractor 只負責從 LLM 回傳的文字中解析出答案。
"""

import re
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Extractor


class VisionMCQExtractor(Extractor):
    """Vision Multiple-Choice Extractor。

    設定 ``uses_vision = True`` 讓 Evaluator 走圖片評測路徑。
    extract() 同時支援字母答案與 Yes/No，以涵蓋 MMBench/MMStar/MMMU 等
    字母選項的 benchmark，以及 POPE 等 Yes/No 幻覺判斷 benchmark。

    與文字 MCQ 共用的 PatternExtractor 不同，本 Extractor 採用嚴格的
    VLM-friendly patterns，避免「I」「A」這類英文常用字被誤判為選項
    字母（PatternExtractor 的 ``[A-Z].`` 兜底會匹配「It does」抓到 I）。
    """

    uses_vision: bool = True

    #: Yes/No 二元答案的正則表達式（POPE 等 benchmark 使用）
    #: 注意：CJK 字符不能用 \b（word boundary 對 CJK 不適用），
    #: 半形/全形冒號都要支援。
    YESNO_PATTERNS: List[str] = [
        r"\b(?:answer\s*(?:is|:))\s*\*{0,2}(yes|no)\*{0,2}\b",
        r"答案[是為:：\s]+\*{0,2}(yes|no|是|否)\*{0,2}",
        r"^\s*\*{0,2}(yes|no)\b",
        r"\b(yes|no)\s*[.,。]?\s*$",
    ]

    #: 字母選項的正則表達式（嚴格 VLM-friendly，按優先序排列）。
    #: 每個 pattern 都要求明確的答案語境（"answer is"、bold markdown、行首等），
    #: 避免匹配英文文章中隨機的單字首字母。
    LETTER_PATTERNS: List[str] = [
        # 1. 明確的英文答案宣告：optional bold + 字母 + 分隔符
        r"(?:correct\s+)?answer\s+is\s*:?\s*\*{0,2}([A-Z])\*{0,2}\s*[:.\)\s]",
        r"(?:correct\s+)?(?:choice|option)\s+is\s*:?\s*\*{0,2}([A-Z])\*{0,2}\s*[:.\)\s]",
        r"(?:correct|right)\s+(?:choice|option|answer)\s*[:：]?\s*\*{0,2}([A-Z])\*{0,2}\s*[:.\)\s]",
        # 2. 明確的中文答案宣告
        r"答案[是為:：]\s*\*{0,2}([A-Z])\*{0,2}",
        r"正確(?:的)?(?:答案|選項|選擇)[是為:：]?\s*\*{0,2}([A-Z])\*{0,2}",
        r"選\s*\*{0,2}([A-Z])\*{0,2}\s*[項。.]",
        # 3. Markdown bold + 字母 + 分隔符（VLM 最常見格式：**A:** 或 **A)**）
        r"\*\*([A-Z])\*\*\s*[:.\)]",
        r"\*\*([A-Z])\s*[:.\)]",
        # 4. 行首字母 + 分隔符
        r"(?m)^\s*\(?([A-Z])\)?\s*[:.\)]\s+",
        # 5. 括號包字母（如 "(A)" 或 "(B)"）
        r"\(([A-Z])\)",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._yesno_patterns = [re.compile(p, re.IGNORECASE) for p in self.YESNO_PATTERNS]
        self._letter_patterns = [re.compile(p, re.IGNORECASE) for p in self.LETTER_PATTERNS]

    def get_name(self) -> str:
        return "vision_mcq"

    def extract(self, llm_output: str) -> Optional[str]:
        """從 VLM 輸出中提取選擇題答案。

        提取順序：
        1. 先嘗試 Yes/No（POPE 等 benchmark），因為「Yes」「No」字面也可能
           被字母 pattern 誤匹配為「Y」「N」
        2. 再嘗試嚴格的字母 pattern（要求明確答案語境或 markdown 強調）

        Returns:
            提取到的答案字串（"Yes"/"No" 或大寫字母），失敗時回傳 None。
        """
        if not self.validate_output(llm_output):
            return None

        # 1. 先試 Yes/No（更具體的格式）
        for pattern in self._yesno_patterns:
            match = pattern.search(llm_output)
            if match:
                token = match.group(1).strip().lower()
                if token in ("yes", "是"):
                    return "Yes"
                if token in ("no", "否"):
                    return "No"

        # 2. 再試字母選項（嚴格 VLM patterns）
        for pattern in self._letter_patterns:
            match = pattern.search(llm_output)
            if match:
                return match.group(1).strip().upper()

        return None
