"""VisTW-Dialogue 的 Extractor（階段 2：評分）。

judge 的提示詞烘焙在資料集的 ``question`` 欄位裡（與既有的 ragas 相同做法），
所以這裡也是 pass-through，由 Scorer 負責解析 judge 回應中的分數。
"""

from typing import Optional

from twinkle_eval.core.abc import Extractor


class VisTWJudgeExtractor(Extractor):
    """pass-through，走預設文字路徑。"""

    def get_name(self) -> str:
        return "vistw_judge"

    def extract(self, llm_output: Optional[str]) -> Optional[str]:
        if llm_output is None:
            return None
        text = llm_output.strip()
        return text or None
