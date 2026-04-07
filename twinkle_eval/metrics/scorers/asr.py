"""ASR（Automatic Speech Recognition）Scorer。

計算語音辨識的 WER（Word Error Rate）與 CER（Character Error Rate）。
依語言自動選擇指標：中文/日文/韓文使用 CER，其他語言使用 WER。

依賴：jiwer（optional dependency，透過 pip install twinkle-eval[asr] 安裝）
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional

from twinkle_eval.core.abc import Scorer


# CJK Unicode 範圍（用於判斷是否為 CJK 字元）
_CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Unified Ideographs Extension A
    (0x20000, 0x2A6DF),  # CJK Unified Ideographs Extension B
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
]

# 使用 CER 的語言（ISO 639-1）
_CER_LANGUAGES = {"zh", "ja", "ko", "th"}


def _is_cjk_char(cp: int) -> bool:
    """判斷 Unicode code point 是否為 CJK 字元。"""
    return any(start <= cp <= end for start, end in _CJK_RANGES)


def _tokenize_mixed(text: str) -> List[str]:
    """混合中英文 tokenization：中文拆字元，英文保持 word。

    用於混合語言場景的 Mixed Error Rate 計算。
    """
    tokens: List[str] = []
    buf: List[str] = []

    for ch in text:
        if _is_cjk_char(ord(ch)):
            # flush English buffer
            if buf:
                word = "".join(buf).strip()
                if word:
                    tokens.append(word)
                buf = []
            tokens.append(ch)
        elif ch.isspace():
            if buf:
                word = "".join(buf).strip()
                if word:
                    tokens.append(word)
                buf = []
        else:
            buf.append(ch)

    if buf:
        word = "".join(buf).strip()
        if word:
            tokens.append(word)

    return tokens


class ASRScorer(Scorer):
    """ASR 評分器，計算 WER / CER。

    Config 選項：
        asr_language: str — 語言代碼（ISO 639-1），預設 "zh"。
            zh/ja/ko/th → CER，其他 → WER。
        asr_metric: str — 強制指定指標，"wer" / "cer" / "auto"（預設 "auto"）。
        normalize_unicode: bool — 是否做 NFKC 正規化（預設 True）。
        remove_punctuation: bool — 是否移除標點（預設 True）。
        to_lower: bool — 是否轉小寫（預設 True）。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._language: str = self._config.get("asr_language", "zh")
        self._metric_override: str = self._config.get("asr_metric", "auto")
        self._normalize_unicode: bool = self._config.get("normalize_unicode", True)
        self._remove_punctuation: bool = self._config.get("remove_punctuation", True)
        self._to_lower: bool = self._config.get("to_lower", True)

    def get_name(self) -> str:
        return "asr"

    @property
    def metric_name(self) -> str:
        """回傳目前使用的指標名稱。"""
        if self._metric_override in ("wer", "cer"):
            return self._metric_override
        return "cer" if self._language in _CER_LANGUAGES else "wer"

    def normalize(self, answer: str) -> str:
        """正規化轉錄文字。

        步驟：
        1. NFKC Unicode 正規化（全形轉半形等）
        2. 轉小寫
        3. 移除標點符號
        4. 壓縮連續空白
        """
        text = str(answer)

        if self._normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        if self._to_lower:
            text = text.lower()

        if self._remove_punctuation:
            # 移除所有 Unicode 標點類別字元（P 類）
            text = "".join(
                ch for ch in text
                if not unicodedata.category(ch).startswith("P")
            )

        # 壓縮空白
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def score(self, predicted: str, gold: str) -> bool:
        """判斷轉錄是否完全正確（正規化後完全比對）。

        注意：ASR 評測的主要指標是 WER/CER（連續值），不是 accuracy。
        此方法用於框架相容性（per-sample is_correct 欄位）。
        實際 WER/CER 值請使用 score_full()。
        """
        return predicted == gold

    def score_full(self, predicted: str, gold: str) -> Dict[str, Any]:
        """計算完整的 ASR 指標，包含 WER、CER 及 is_correct。

        Returns:
            Dict 包含：
                - is_correct: bool（完全比對）
                - wer: float（Word Error Rate）
                - cer: float（Character Error Rate）
                - metric: str（主要指標名稱 "wer" 或 "cer"）
                - metric_value: float（主要指標值）
                - predicted_normalized: str
                - gold_normalized: str
        """
        try:
            import jiwer
        except ImportError:
            raise ImportError(
                "ASR 評測需要 jiwer 套件。請安裝：pip install twinkle-eval[asr]"
            )

        pred_norm = self.normalize(predicted)
        gold_norm = self.normalize(gold)

        is_correct = pred_norm == gold_norm

        # 計算 WER
        if gold_norm.strip():
            wer = jiwer.wer(gold_norm, pred_norm)
        else:
            wer = 0.0 if not pred_norm.strip() else 1.0

        # 計算 CER
        if gold_norm.strip():
            cer = jiwer.cer(gold_norm, pred_norm)
        else:
            cer = 0.0 if not pred_norm.strip() else 1.0

        metric = self.metric_name
        metric_value = cer if metric == "cer" else wer

        return {
            "is_correct": is_correct,
            "wer": round(wer, 6),
            "cer": round(cer, 6),
            "metric": metric,
            "metric_value": round(metric_value, 6),
            "predicted_normalized": pred_norm,
            "gold_normalized": gold_norm,
        }
