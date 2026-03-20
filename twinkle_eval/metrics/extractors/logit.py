"""Logit（log-likelihood）評測路徑的 Extractor。"""

from typing import Any, Dict, Optional

from twinkle_eval.core.abc import Extractor


class LogitExtractor(Extractor):
    """Logit 評測策略的 Extractor。

    不解析生成文字，而是透過 score_continuation() 比較各選項的 log probability。
    Evaluator 偵測到 uses_logprobs=True 時會走 logit 路徑，不呼叫 extract()。
    """

    uses_logprobs: bool = True

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)

    def get_name(self) -> str:
        return "logit"

    def extract(self, llm_output: str) -> Optional[str]:
        # logit 路徑不使用文字抽取，此方法不會被 Evaluator 呼叫
        return None
