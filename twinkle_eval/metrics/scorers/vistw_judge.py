"""VisTW-Dialogue 的 Scorer（階段 2：評分）。

解析 judge 回應中的 0–10 分。評分規則與提示詞移植自 VisTW 官方實作
（https://github.com/TMMMU-Benchmark/evaluation，CC BY 4.0）：
judge 依 ``simplevals/prompts.py`` 的 HUMAN_GUIDELINE 給分，並以
``[評分]: N`` 的形式輸出。

官方對每個回答評分 5 次取平均；本專案以 ``repeat_runs: 5`` 達成，
並額外得到標準差，讓 judge 自身的變異可見。
"""

import re
from typing import Any, Dict, Optional

from twinkle_eval.core.abc import Scorer

#: judge 回應中分數的格式。官方用 ``response.split('[評分]: ')[-1]``，
#: 這裡用正則以容忍全形冒號與前後空白，但**不**放寬到「回應中任何數字」——
#: 那會在 judge 回應格式跑掉時默默抓到錯的數字。
#: 半形 [] 與全形【】都接受（中文模型常用後者）。
#: 允許數字前後有 **——JUDGE_PROMPT 本身滿是粗體標記，judge 很容易模仿該風格。
#: 分數後不得再接數字或小數點——否則 "100" 會被截成 10、"8.5" 被截成 8。
#: 官方用 ``split('[評分]: ')[-1]`` 取**最後一個**標記，這裡以 findall 取最後一個對齊，
#: 避免 judge 在評分指南裡先引用範例分數時抓到錯的那個。
_SCORE_PATTERN = re.compile(
    r"[\[【]\s*評分\s*[\]】]\s*[：:]\s*\*{0,2}(\d{1,2})\*{0,2}(?![\d.])(?:\s*/\s*10)?"
)

#: 及格線。score() 回傳 bool，用於 accuracy；真正的指標是 score_full 的平均分。
_DEFAULT_PASS_THRESHOLD = 6.0


class VisTWJudgeScorer(Scorer):
    """解析 judge 給的 0–10 分。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = self._config or {}
        self.pass_threshold: float = float(
            cfg.get("vistw_judge_pass_threshold", _DEFAULT_PASS_THRESHOLD)
        )

    def get_name(self) -> str:
        return "vistw_judge"

    def normalize(self, answer: Any) -> str:
        return "" if answer is None else str(answer).strip()

    def parse_score(self, judge_response: str) -> Optional[float]:
        """從 judge 回應解析分數；解析失敗或超出 0–10 回傳 None。

        **解析失敗一律回傳 None，絕不給預設分。** judge-based 評分最容易出現的
        失效就是格式不符時默默給一個中間值，使分數全面失真且無跡可循。
        """
        if not judge_response:
            return None
        matches = _SCORE_PATTERN.findall(judge_response)
        if not matches:
            return None
        value = float(matches[-1])
        if not 0.0 <= value <= 10.0:
            return None
        return value

    def score(self, predicted: str, gold: str) -> bool:
        """分數達及格線為 True。解析失敗為 False（並計入 unparsed）。"""
        value = self.parse_score(predicted)
        return value is not None and value >= self.pass_threshold

    def score_full(self, predicted: str, gold: str) -> Dict[str, Any]:
        """回傳完整評分資訊。

        ⚠️ **目前不會被呼叫。** evaluator 只在 ``uses_ifeval`` 與 ``uses_audio``
        兩條路徑呼叫 ``score_full()``，文字路徑沒有呼叫點；即使補上呼叫點，
        ``runners/standard.py`` 的 ``_evaluate_dataset()`` 也以白名單組結果，
        會把 metrics 裡的其他鍵丟掉，因此指標到不了 ``results_*.json``。

        這個方法先留著，等指標貫通的機制定案後即可生效（追蹤於 issue）。
        現階段 0–10 分請從 ``eval_results_*.jsonl`` 自行彙總，
        或看 ``accuracy``（及格率，門檻見 ``vistw_judge_pass_threshold``）。
        """
        value = self.parse_score(predicted)
        return {
            "judge_score": value,
            "judge_parsed": value is not None,
        }
