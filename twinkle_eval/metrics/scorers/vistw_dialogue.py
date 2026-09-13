"""VisTW-Dialogue 的 Scorer（階段 1：生成）。

生成階段沒有對錯可言——開放式問答的品質由階段 2 的 judge 評分。
這個 Scorer 只判斷「模型是否產出了非空回答」，因此該階段的 accuracy
應讀作**回應產生率**而非正確率。
"""

from typing import Any

from twinkle_eval.core.abc import Scorer


class VisTWDialogueScorer(Scorer):
    """以「是否產出非空回答」計分。"""

    def get_name(self) -> str:
        return "vistw_dialogue"

    def normalize(self, answer: Any) -> str:
        return "" if answer is None else str(answer).strip()

    def score(self, predicted: str, gold: str) -> bool:
        """回答非空即視為成功產生。

        不與 ``gold``（參考答案）比對——那是階段 2 judge 的工作。
        """
        return bool(predicted and predicted.strip())
