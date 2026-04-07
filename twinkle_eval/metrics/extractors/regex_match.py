"""可設定正則表達式答案抽取器（提取完整答案字串）。"""

import re
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Extractor


class RegexMatchExtractor(Extractor):
    """使用可設定的正則表達式從 LLM 輸出中提取完整答案字串。

    與 PatternExtractor / CustomRegexExtractor 不同，本 Extractor 提取的是
    **完整答案字串**（而非單一選項字母），適用於 BBH 等混合格式 benchmark。

    預設包含多種常見的答案格式 pattern，依序嘗試匹配。

    Config 欄位（皆為可選）：
        answer_pattern (str | list[str]): 用於提取答案的正則表達式。
            捕捉組 1 的內容即為答案。可為單一字串或字串列表。
            預設: 內建多種常見 CoT 答案格式
    """

    DEFAULT_PATTERNS: List[str] = [
        # 標準 BBH CoT 格式
        r"[Tt]he answer is\s+(.+)",
        # "The correct answer/option is (X)" — 含選項或文字
        r"[Tt]he correct (?:answer|option) is:?\s*(.+)",
        # "The final answer is X"
        r"[Tt]he final answer is:?\s*(.+)",
        # "Final Answer:\nX" 或 "Final Answer: X"
        r"[Ff]inal [Aa]nswer:?\s*(.+)",
        # "Correct Answer:\n(X)"
        r"[Cc]orrect [Aa]nswer:?\s*(.+)",
        # "Therefore, the correct answer/option is:\n(X)"
        r"[Tt]herefore,?\s+the correct (?:answer|option) is:?\s*(.+)",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        raw = self._config.get("answer_pattern", self.DEFAULT_PATTERNS)
        if isinstance(raw, str):
            raw = [raw]
        self.patterns: List[str] = raw

    def get_name(self) -> str:
        return "regex_match"

    def extract(self, llm_output: str) -> Optional[str]:
        """使用正則表達式提取完整答案字串。

        策略：對整段文字做 re.DOTALL 搜尋（取最後一個匹配），
        再清理提取結果（取第一行非空內容、移除 markdown 粗體等）。
        """
        if not self.validate_output(llm_output):
            return None

        # 兩階段搜尋：先不跨行（取最後一個 match），再跨行（處理答案在下一行的情況）
        for flags in (re.IGNORECASE, re.IGNORECASE | re.DOTALL):
            for pattern in self.patterns:
                matches = list(re.finditer(pattern, llm_output, flags))
                if not matches:
                    continue

                raw = matches[-1].group(1).strip()
                answer = self._clean_answer(raw)
                if answer:
                    return answer

        # 最後嘗試：若最後幾行含有看起來像答案的內容
        lines = [line.strip() for line in llm_output.strip().splitlines() if line.strip()]
        for line in reversed(lines[-5:]):
            cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
            cleaned = re.sub(r"`(.+?)`", r"\1", cleaned)
            cleaned = re.sub(r"^[-*•]\s+", "", cleaned).strip()
            cleaned = re.sub(r"\.\s*$", "", cleaned)
            # 只接受看起來像簡短答案的行（MC、binary、數字、短文字）
            if re.match(
                r"^(\([A-Z]\)|Yes|No|True|False|[Vv]alid|[Ii]nvalid|-?\d+|[\]\[\)\(><}{]+)$",
                cleaned,
            ):
                return cleaned

        return None

    @staticmethod
    def _clean_answer(raw: str) -> Optional[str]:
        """清理提取到的原始答案字串。"""
        if not raw:
            return None

        # 若包含換行，取第一行非空內容
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return None
        answer = lines[0]

        # 移除 markdown 粗體 **X** → X
        answer = re.sub(r"\*\*(.+?)\*\*", r"\1", answer)

        # 移除 markdown 行內程式碼 `X` → X
        answer = re.sub(r"`(.+?)`", r"\1", answer)

        # 移除 \boxed{X} → X
        boxed = re.search(r"\\boxed\{(.+?)\}", answer)
        if boxed:
            answer = boxed.group(1)

        # 移除結尾句號
        answer = re.sub(r"\.\s*$", "", answer)

        # 移除首尾引號
        if len(answer) >= 2 and answer[0] == answer[-1] and answer[0] in "\"'":
            answer = answer[1:-1]

        # 移除列表前綴（如 "- No" → "No"）
        answer = re.sub(r"^[-*•]\s+", "", answer)

        # 若答案以 MC 格式開頭 (X)，只保留 (X) 部分
        mc_match = re.match(r"^(\([A-Z]\))\s", answer)
        if mc_match:
            answer = mc_match.group(1)

        return answer.strip() if answer.strip() else None
