"""VisTW-Dialogue 的 Extractor（階段 1：生成）。

VisTW-Dialogue 是開放式問答，沒有單一正解，所以這個 Extractor 是 pass-through——
把模型的自由回答原樣交出，由階段 2 的 judge 評分。

參考 VisTW（arXiv 2503.10427，NTU MiuLab，CC BY 4.0）。
"""

from typing import Optional

from twinkle_eval.core.abc import Extractor


class VisTWDialogueExtractor(Extractor):
    """pass-through，並標記走 vision 路徑。"""

    #: 走 evaluator 的 uses_vision 路徑（送圖片 + 文字）
    uses_vision: bool = True

    def get_name(self) -> str:
        return "vistw_dialogue"

    def extract(self, llm_output: Optional[str]) -> Optional[str]:
        """原樣回傳非空的回答；空回應回傳 None（計入 unparsed）。"""
        if llm_output is None:
            return None
        text = llm_output.strip()
        return text or None
