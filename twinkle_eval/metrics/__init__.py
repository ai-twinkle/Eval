"""Metrics 模組：提供 Extractor、Scorer 以及評測方法 preset 系統。"""

from typing import Any, Dict, List, Optional, Tuple, Type

from twinkle_eval.core.abc import Extractor, Scorer

from .extractors.bfcl_prompt import BFCLPromptExtractor
from .extractors.box import BoxExtractor
from .extractors.custom import CustomRegexExtractor
from .extractors.logit import LogitExtractor
from .extractors.math import MathExtractor
from .extractors.pattern import PatternExtractor
from .extractors.tool_call import ToolCallExtractor
from .extractors.ifeval import IFEvalExtractor
from .extractors.ifbench import IFBenchExtractor
from .scorers.bfcl import BFCLScorer
from .scorers.exact import ExactMatchScorer
from .scorers.math import MathRulerScorer
from .scorers.ifeval import IFEvalScorer
from .scorers.ifbench import IFBenchScorer
from .extractors.niah import NIAHExtractor
from .scorers.niah import NIAHScorer
from .extractors.ragas import RAGASExtractor
from .scorers.ragas import RAGASScorer
from .extractors.text2sql import Text2SQLExtractor
from .extractors.regex_match import RegexMatchExtractor
from .scorers.text2sql import Text2SQLScorer
from .scorers.string_match import StringMatchScorer
from .extractors.asr import ASRExtractor
from .scorers.asr import ASRScorer

# Preset：evaluation_method 字串 → (Extractor 類別, Scorer 類別)
PRESETS: Dict[str, Tuple[Type[Extractor], Type[Scorer]]] = {
    "pattern": (PatternExtractor, ExactMatchScorer),
    "box": (BoxExtractor, ExactMatchScorer),
    "logit": (LogitExtractor, ExactMatchScorer),
    "math": (MathExtractor, MathRulerScorer),
    "custom_regex": (CustomRegexExtractor, ExactMatchScorer),
    "bfcl_fc": (ToolCallExtractor, BFCLScorer),
    "bfcl_prompt": (BFCLPromptExtractor, BFCLScorer),
    "ifeval": (IFEvalExtractor, IFEvalScorer),
    "ifbench": (IFBenchExtractor, IFBenchScorer),
    "niah": (NIAHExtractor, NIAHScorer),
    "ragas": (RAGASExtractor, RAGASScorer),
    "text2sql": (Text2SQLExtractor, Text2SQLScorer),
    "regex_match": (RegexMatchExtractor, StringMatchScorer),
    "asr": (ASRExtractor, ASRScorer),
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
    "ToolCallExtractor",
    "BFCLPromptExtractor",
    "ExactMatchScorer",
    "MathRulerScorer",
    "BFCLScorer",
    "IFEvalExtractor",
    "IFEvalScorer",
    "IFBenchExtractor",
    "IFBenchScorer",
    "NIAHExtractor",
    "NIAHScorer",
    "RAGASExtractor",
    "RAGASScorer",
    "Text2SQLExtractor",
    "Text2SQLScorer",
    "RegexMatchExtractor",
    "StringMatchScorer",
    "ASRExtractor",
    "ASRScorer",
]
