"""評測策略模組 - 向下相容的重新匯出層。

實際實作已遷移至 twinkle_eval.metrics 套件。
此檔案保留以確保現有程式碼的
`from twinkle_eval.evaluation_strategies import ...` 仍可正常運作。

新程式碼應改用：
    from twinkle_eval.metrics import create_metric_pair, get_available_methods
    from twinkle_eval.metrics.extractors.pattern import PatternExtractor
    from twinkle_eval.metrics.scorers.exact import ExactMatchScorer
"""

from .metrics.extractors.box import BoxExtractor as BoxExtractionStrategy
from .metrics.extractors.custom import CustomRegexExtractor as CustomRegexStrategy
from .metrics.extractors.logit import LogitExtractor as LogitEvaluationStrategy
from .metrics.extractors.math import MathExtractor as MathExtractionStrategy
from .metrics.extractors.pattern import PatternExtractor as PatternMatchingStrategy
from .metrics import get_available_methods

# EvaluationStrategy 向下相容 shim（委派至 Extractor/Scorer 介面）
from .core.abc import Extractor as EvaluationStrategy


class EvaluationStrategyFactory:
    """向下相容的工廠類別。新程式碼應改用 twinkle_eval.metrics.create_metric_pair。"""

    @classmethod
    def create_strategy(cls, strategy_type: str, config=None):
        """建立評測策略實例（向下相容介面）。"""
        from .metrics import create_metric_pair
        extractor, scorer = create_metric_pair(strategy_type, config or {})
        # 回傳一個 shim，讓舊程式碼可以繼續使用 extract_answer / normalize_answer / is_correct
        from .config import _CompatStrategyShim
        return _CompatStrategyShim(extractor, scorer)

    @classmethod
    def get_available_types(cls):
        """回傳所有可用的評測策略名稱。"""
        return get_available_methods()

    @classmethod
    def register_strategy(cls, name: str, strategy_class):
        """向評測策略登錄表新增自訂策略（向下相容介面）。"""
        # 無法直接對應到新架構，記錄警告
        import warnings
        warnings.warn(
            "EvaluationStrategyFactory.register_strategy 已過時。"
            "請改用 twinkle_eval.metrics.register_preset。",
            DeprecationWarning,
            stacklevel=2,
        )


__all__ = [
    "EvaluationStrategy",
    "PatternMatchingStrategy",
    "BoxExtractionStrategy",
    "CustomRegexStrategy",
    "MathExtractionStrategy",
    "LogitEvaluationStrategy",
    "EvaluationStrategyFactory",
]
