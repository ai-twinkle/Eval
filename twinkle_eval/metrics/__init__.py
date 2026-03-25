"""Metrics 模組：提供 Extractor、Scorer 以及評測方法 preset 系統。"""

from typing import Any, Dict, List, Optional, Tuple, Type

from twinkle_eval.core.abc import Extractor, Scorer

from .extractors.box import BoxExtractor
from .extractors.custom import CustomRegexExtractor
from .extractors.logit import LogitExtractor
from .extractors.math import MathExtractor
from .extractors.pattern import PatternExtractor
from .extractors.ifeval import IFEvalExtractor
from .scorers.exact import ExactMatchScorer
from .scorers.math import MathRulerScorer
from .scorers.ifeval import IFEvalScorer

# Preset：evaluation_method 字串 → (Extractor 類別, Scorer 類別)
PRESETS: Dict[str, Tuple[Type[Extractor], Type[Scorer]]] = {
    "pattern": (PatternExtractor, ExactMatchScorer),
    "box": (BoxExtractor, ExactMatchScorer),
    "logit": (LogitExtractor, ExactMatchScorer),
    "math": (MathExtractor, MathRulerScorer),
    "custom_regex": (CustomRegexExtractor, ExactMatchScorer),
    "ifeval": (IFEvalExtractor, IFEvalScorer),
}


def create_metric_pair(
    evaluation_method: str,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[Extractor, Scorer]:
    """依 evaluation_method 建立對應的 (Extractor, Scorer) 配對。

    Args:
        evaluation_method: 評測方法名稱，對應 PRESETS 中的 key。
        config: 傳遞給 Extractor / Scorer 建構子的配置字典。

    Returns:
        Tuple[Extractor, Scorer]: 建立完成的 extractor 和 scorer 實例。

    Raises:
        KeyError: 若 evaluation_method 不在 PRESETS 中。
    """
    if evaluation_method not in PRESETS:
        available = ", ".join(sorted(PRESETS))
        raise KeyError(
            f"evaluation_method '{evaluation_method}' 不存在。可用方法: {available}"
        )
    extractor_cls, scorer_cls = PRESETS[evaluation_method]
    cfg = config or {}
    return extractor_cls(cfg), scorer_cls(cfg)


def get_available_methods() -> List[str]:
    """回傳所有可用的評測方法名稱。"""
    return list(PRESETS.keys())


def register_preset(
    name: str,
    extractor_cls: Type[Extractor],
    scorer_cls: Type[Scorer],
) -> None:
    """向 PRESETS 登錄新的評測方法。

    Args:
        name: 評測方法名稱
        extractor_cls: Extractor 實作類別
        scorer_cls: Scorer 實作類別
    """
    PRESETS[name] = (extractor_cls, scorer_cls)


__all__ = [
    "PRESETS",
    "create_metric_pair",
    "get_available_methods",
    "register_preset",
    "Extractor",
    "Scorer",
    "PatternExtractor",
    "BoxExtractor",
    "LogitExtractor",
    "MathExtractor",
    "CustomRegexExtractor",
    "ExactMatchScorer",
    "MathRulerScorer",
    "IFEvalExtractor",
    "IFEvalScorer",
]
