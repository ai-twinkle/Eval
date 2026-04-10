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

    #: \boxed{} / \box{} 答案的正則表達式（最高優先序）。
    #: 推理型 VLM（reasoning model）通常被訓練成最後輸出 ``\boxed{答案}``，
    #: 這個格式來自數學 benchmark（GSM8K、MATH）並被推廣到所有 reasoning 場景。
    #: 由於 ``\boxed{X}`` 是模型刻意框出的答案，沒有歧義，應該最先嘗試。
    #:
    #: 同時支援單反斜線（``\boxed{A}``，raw LaTeX）與雙反斜線（``\\boxed{A}``，
    #: JSON-escaped）兩種輸出形式。
    BOXED_PATTERNS: List[str] = [
        # 字母答案：\boxed{A} / \boxed{**B**} / \box{C}
        r"\\{1,2}box(?:ed)?\s*\{\s*\*{0,2}([A-Z])\*{0,2}\s*\}",
        # Yes/No 答案：\boxed{Yes} / \boxed{是}
        r"\\{1,2}box(?:ed)?\s*\{\s*\*{0,2}(yes|no|是|否)\*{0,2}\s*\}",
    ]

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
    #:
    #: 共用 trailing lookahead ``(?=[\s:.,;!?)\]]|$)`` 確保字母後接標點/空白/EOS，
    #: 避免「A1」「AB」這類非答案 token 被誤抓。
    #:
    #: 重要：extract() 對每個 pattern 採用 ``findall + 取最後一個 match``，
    #: 因為 VLM 經常先回顯選項列表（A) cat / B) dog / ...）再給出最終答案，
    #: 取最後一個匹配可避免被選項列表的第一個字母誤導。
    LETTER_PATTERNS: List[str] = [
        # 1. 明確的英文答案宣告（"is" 或 ":" 任一即可，涵蓋 "Answer: A"、"Answer is A"）
        r"(?:correct\s+|final\s+)?answer(?:\s+is\s*[:：]?|\s*[:：])\s*\*{0,2}([A-Z])\*{0,2}(?=[\s:.,;!?)\]]|$)",
        r"(?:correct\s+|final\s+)?(?:choice|option)(?:\s+is\s*[:：]?|\s*[:：])\s*\*{0,2}([A-Z])\*{0,2}(?=[\s:.,;!?)\]]|$)",
        # 2. "Correct Answer: X" / "Correct Option: **B**" / "**Correct Answer: A**"
        r"(?:correct|right)\s+(?:choice|option|answer)\s*[:：]\s*\*{0,2}([A-Z])\*{0,2}(?=[\s:.,;!?)\]]|$)",
        # 3. 明確的中文答案宣告
        r"答案[是為:：]\s*\*{0,2}([A-Z])\*{0,2}",
        r"正確(?:的)?(?:答案|選項|選擇)[是為:：]?\s*\*{0,2}([A-Z])\*{0,2}",
        r"選\s*\*{0,2}([A-Z])\*{0,2}\s*[項。.]",
        # 4. Markdown bold + 字母 + 分隔符（VLM 最常見格式：**A:** 或 **A)**）
        r"\*\*([A-Z])\*\*\s*[:.\)]",
        r"\*\*([A-Z])\s*[:.\)]",
        # 5. 括號包字母（如 "(A)" 或 "(B)"）—— 強訊號，置於 line-start 之前
        r"\(([A-Z])\)",
        # 6. 末尾單獨字母（VLM 簡短回答：換行 + 單字母 + 可選句點 + EOS）
        #    例：「...The 3rd number is 9.\n\nB」。
        #    必須優先於 line-start letter，否則「A) cat / B) dog / C) bird / D」這種
        #    「選項回顯 + 答案在 echo list 之外」的情境會被誤抓成最後一個 echo 字母 C。
        r"(?:^|\n)\s*\*{0,2}([A-Z])\*{0,2}\s*[.。]?\s*$",
        # 7. 行首字母 + 分隔符（multi-line）—— 最弱的 fallback
        #    findall 會抓到所有匹配，extract() 會取最後一個。例如：
        #    「A) cat / B) dog / C) bird / C. answer」會正確抓到結尾的 C。
        r"(?m)^\s*\(?([A-Z])\)?\s*[:.\)]\s+",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._boxed_patterns = [re.compile(p, re.IGNORECASE) for p in self.BOXED_PATTERNS]
        self._yesno_patterns = [re.compile(p, re.IGNORECASE) for p in self.YESNO_PATTERNS]
        self._letter_patterns = [re.compile(p, re.IGNORECASE) for p in self.LETTER_PATTERNS]

    def get_name(self) -> str:
        return "vision_mcq"

    def extract(self, llm_output: str) -> Optional[str]:
        """從 VLM 輸出中提取選擇題答案。

        提取順序：
        1. 先嘗試 ``\\boxed{}`` / ``\\box{}`` —— 推理型 VLM 的標準輸出格式，
           沒有歧義，命中即回傳
        2. 再嘗試 Yes/No（POPE 等 benchmark），因為「Yes」「No」字面也可能
           被字母 pattern 誤匹配為「Y」「N」
        3. 最後嘗試嚴格的字母 pattern（要求明確答案語境或 markdown 強調）

        每個 pattern 採 ``findall + 取最後一個 match`` 策略：VLM 常先回顯
        選項列表（A) cat / B) dog / ...）再給最終答案，最後一個匹配通常
        才是真正的答案。例如 ``A) cat\\nB) dog\\nC) bird\\nAnswer: C`` 應
        抓到 ``C`` 而不是首行的 ``A``。

        Returns:
            提取到的答案字串（"Yes"/"No" 或大寫字母），失敗時回傳 None。
        """
        if not self.validate_output(llm_output):
            return None

        # 1. 最高優先序：\boxed{} / \box{}（推理型 VLM 的標準輸出）
        for pattern in self._boxed_patterns:
            matches = pattern.findall(llm_output)
            if matches:
                token = matches[-1].strip()
                if token.lower() in ("yes", "是"):
                    return "Yes"
                if token.lower() in ("no", "否"):
                    return "No"
                return token.upper()

        # 2. 再試 Yes/No（更具體的格式）
        for pattern in self._yesno_patterns:
            matches = pattern.findall(llm_output)
            if matches:
                token = matches[-1].strip().lower()
                if token in ("yes", "是"):
                    return "Yes"
                if token in ("no", "否"):
                    return "No"

        # 3. 最後試字母選項（嚴格 VLM patterns）
        for pattern in self._letter_patterns:
            matches = pattern.findall(llm_output)
            if matches:
                return matches[-1].strip().upper()

        return None
