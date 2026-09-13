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
_SCORE_PATTERN = re.compile(r"\[\s*評分\s*\]\s*[：:]\s*(\d{1,2})(?:\s*/\s*10)?")

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
        m = _SCORE_PATTERN.search(judge_response)
        if not m:
            return None
        value = float(m.group(1))
        if not 0.0 <= value <= 10.0:
            return None
        return value

    def score(self, predicted: str, gold: str) -> bool:
        """分數達及格線為 True。解析失敗為 False（並計入 unparsed）。"""
        value = self.parse_score(predicted)
        return value is not None and value >= self.pass_threshold

    def score_full(self, predicted: str, gold: str) -> Dict[str, Any]:
        """回傳完整評分資訊，供 evaluator 併入 metrics 與 JSONL 明細。"""
        value = self.parse_score(predicted)
        return {
            "judge_score": value,
            "judge_parsed": value is not None,
            "judge_pass_threshold": self.pass_threshold,
        }
