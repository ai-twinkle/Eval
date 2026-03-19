"""核心抽象基底類別（ABCs）。

定義 LLM、Extractor、Scorer、ResultsExporter 四個核心介面。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from openai.types.chat import ChatCompletion


class LLM(ABC):
    """LLM 後端的抽象基底類別。"""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def call(
        self,
        question_text: str,
        prompt_lang: str = "zh",
        eval_method: str = "",
        system_prompt_enabled: bool = True,
        num_samples: int = 1,
        model_overrides: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletion:
        """呼叫 LLM 並回傳 ChatCompletion 格式的回應。"""
        ...

    @abstractmethod
    def validate_config(self) -> bool:
        """驗證此 LLM 後端所需的配置。"""
        ...

    def score_continuation(self, context: str, continuation: str) -> float:
        """計算 log P(continuation | context)，用於 logit 評測策略。

        透過 completions API 的 echo 模式取得 context+continuation 的 token logprobs，
        並回傳 continuation 部分的對數機率加總。
        子類別應覆寫此方法以提供正確實作；預設拋出 NotImplementedError。
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 尚未實作 score_continuation()，"
            "無法使用 logit 評測策略。"
        )


class Extractor(ABC):
    """從 LLM 輸出中抽取原始答案字串的抽象基底類別。"""

    #: 若為 True，表示此 Extractor 使用 logprobs 而非生成文字；Evaluator 會走 logit 路徑
    uses_logprobs: bool = False

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    @abstractmethod
    def extract(self, llm_output: str) -> Optional[str]:
        """從 LLM 輸出文字中抽取答案字串。

        Returns:
            抽取到的答案字串，若無法抽取則回傳 None。
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """回傳此 Extractor 的識別名稱。"""
        ...

    def validate_output(self, llm_output: Optional[str]) -> bool:
        """驗證 LLM 輸出格式是否有效。"""
        return isinstance(llm_output, str) and llm_output.strip() != ""


class Scorer(ABC):
    """將抽取出的答案與正解進行比對的抽象基底類別。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config = config or {}

    @abstractmethod
    def normalize(self, answer: str) -> str:
        """正規化答案以便比較。"""
        ...

    @abstractmethod
    def score(self, predicted: str, gold: str) -> bool:
        """判斷預測答案是否正確。"""
        ...

    @abstractmethod
    def get_name(self) -> str:
        """回傳此 Scorer 的識別名稱。"""
        ...


class ResultsExporter(ABC):
    """結果輸出器的抽象基底類別。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}

    @abstractmethod
    def export(self, results: Dict[str, Any], output_path: str) -> str:
        """將結果匯出至指定格式，回傳實際輸出路徑。"""
        ...

    @abstractmethod
    def get_file_extension(self) -> str:
        """回傳此輸出器使用的副檔名。"""
        ...
