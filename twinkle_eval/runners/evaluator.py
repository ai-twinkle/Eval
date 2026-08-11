import inspect
import json
import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from typing import Any, Dict, Iterator, List, Optional, Tuple

from tqdm import tqdm

from twinkle_eval.core.abc import Extractor, Scorer
from twinkle_eval.core.logger import log_error
from twinkle_eval.datasets import Dataset
from twinkle_eval.metrics.extractors.bfcl_prompt import inject_bfcl_system_prompt
from twinkle_eval.metrics.extractors.tool_call import convert_bfcl_functions_to_tools
from twinkle_eval.models import LLM


def _get_node_id() -> str:
    """取得當前節點識別碼，優先使用 SLURM_NODEID，否則回退至 node0。"""
    slurm_node = os.environ.get("SLURM_NODEID")
    return slurm_node if slurm_node is not None else "0"


def _excel_label(idx: int) -> str:
    """將 0-based 索引轉換為 Excel 風格的大寫字母標籤（A…Z、AA…AZ、BA…）。

    與 datasets 模組的 choices 展開邏輯一致。
    """
    letters = []
    while True:
        idx, rem = divmod(idx, 26)
        letters.append(string.ascii_uppercase[rem])
        if idx == 0:
            break
        idx -= 1
    return "".join(reversed(letters))


_THINK_TAG_PAIRS = [
    ("<think>", "</think>"),
    ("<reason>", "</reason>"),
    ("<reasoning>", "</reasoning>"),
]


def _strip_think_blocks(text: str) -> str:
    """剝離完整的推理 tag 對（需同時有開頭與結尾 tag），取結尾 tag 之後的內容。
    若 tag 不完整（如只有結尾 tag），視為格式不合格，原樣返回。
    """
    lower = text.lower()
    for start_tag, end_tag in _THINK_TAG_PAIRS:
        if start_tag in lower and end_tag in lower:
            idx = lower.rfind(end_tag)
            return text[idx + len(end_tag) :].strip()
    return text


def _get_reasoning_text(message: Any) -> Optional[str]:
    """優先讀取新版 reasoning，只有為 None 時才回退舊版 reasoning_content。"""
    reasoning = getattr(message, "reasoning", None)
    if reasoning is None:
        reasoning = getattr(message, "reasoning_content", None)
    return reasoning


#: 編碼圖片時的最大檔案大小（bytes），預設 50 MB。
#: 避免不小心把超大圖片塞進 base64 拖慢評測或撐爆 API 請求。
_MAX_IMAGE_BYTES = 50 * 1024 * 1024


def _detect_image_mime(data: bytes) -> str:
    """從圖片檔案的 magic bytes 偵測 MIME subtype。

    支援 JPEG / PNG / GIF / WEBP / BMP，無法辨識時回傳 "jpeg"（最寬鬆的兜底）。
    這個函式是處理「副檔名缺失或不正確」的情境，比 splitext 更可靠。
    """
    if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if len(data) >= 6 and data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if len(data) >= 2 and data[:2] == b"BM":
        return "bmp"
    return "jpeg"


def _encode_image_to_data_uri(
    image_path: str,
    max_image_size: Optional[int] = None,
) -> str:
    """將本地圖片檔案編碼為 base64 data URI。

    若 image_path 已是 http(s):// 開頭的 URL，直接回傳。
    若 max_image_size 指定且 Pillow 可用，會將圖片最長邊縮放至該大小。

    安全性與穩定性考量：
    - 拒絕超過 ``_MAX_IMAGE_BYTES`` 的檔案（避免把 GB 級檔案塞進 API request）
    - MIME type 從 magic bytes 偵測，而非僅依賴副檔名
    - 路徑經 ``os.path.realpath`` 解析，避免符號連結意外指向 base64 編碼後外洩

    Args:
        image_path:     本地檔案路徑或 HTTP/HTTPS URL
        max_image_size: 最長邊像素數；None 表不縮放

    Returns:
        可放入 OpenAI image_url.url 的字串（URL 或 data URI）。

    Raises:
        FileNotFoundError: 圖片檔案不存在
        ValueError:        檔案大小超過 ``_MAX_IMAGE_BYTES``
    """
    import base64

    if image_path.startswith(("http://", "https://")):
        return image_path

    # 解析符號連結並驗證檔案存在
    real_path = os.path.realpath(image_path)
    if not os.path.isfile(real_path):
        raise FileNotFoundError(f"圖片檔案不存在: {image_path}")

    file_size = os.path.getsize(real_path)
    if file_size > _MAX_IMAGE_BYTES:
        raise ValueError(
            f"圖片檔案過大 ({file_size / 1024 / 1024:.1f} MB > "
            f"{_MAX_IMAGE_BYTES / 1024 / 1024:.0f} MB): {image_path}。"
            f"請使用 strategy_config.max_image_size 縮放，或預先壓縮圖片。"
        )

    if max_image_size:
        try:
            import io as _io

            from PIL import Image  # type: ignore

            with Image.open(real_path) as img:
                img.thumbnail((max_image_size, max_image_size))
                buf = _io.BytesIO()
                # 使用偵測到的格式儲存（Pillow 認得的格式）
                save_format = (img.format or "JPEG").upper()
                if img.mode in ("RGBA", "LA", "P") and save_format == "JPEG":
                    img = img.convert("RGB")
                img.save(buf, format=save_format)
                payload = buf.getvalue()
                b64 = base64.b64encode(payload).decode("utf-8")
                mime_subtype = _detect_image_mime(payload)
                return f"data:image/{mime_subtype};base64,{b64}"
        except ImportError:
            log_error(
                "max_image_size 已設定但 Pillow 未安裝，跳過縮放。請執行 pip install twinkle-eval[vision]"
            )
        except Exception as e:
            log_error(f"圖片縮放失敗 ({image_path}): {e}，回退為原始檔案編碼")

    with open(real_path, "rb") as f:
        payload = f.read()
    mime_subtype = _detect_image_mime(payload)
    b64 = base64.b64encode(payload).decode("utf-8")
    return f"data:image/{mime_subtype};base64,{b64}"


def _build_vision_messages(
    image_url: str,
    question_text: str,
    image_detail: str = "auto",
) -> list:
    """建構 OpenAI multimodal messages（image_url + text）。"""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_url, "detail": image_detail},
                },
                {"type": "text", "text": question_text},
            ],
        }
    ]


class RateLimiter:
    def __init__(self, calls_per_second: float) -> None:
        self.no_limit = calls_per_second == -1
        self.interval = 1.0 / calls_per_second if not self.no_limit else 0
        self.last_call_time: float = 0

    def wait(self) -> None:
        if self.no_limit:
            return
        current_time = time.time()
        time_to_wait = self.interval - (current_time - self.last_call_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        self.last_call_time = time.time()


class _EvalAccumulator:
    """彙整單一檔案評測過程中的統計數據與逐題結果。

    取代原本分散在各評測路徑中重複的計數與紀錄樣板。
    """

    def __init__(self) -> None:
        self.total_correct = 0
        self.total_samples = 0
        self.total_unparsed = 0
        self.total_failed = 0
        self.question_stats: Dict[Any, Dict[str, int]] = {}
        self.detailed_results: List[Dict[str, Any]] = []

    def record(
        self,
        question_id: Any,
        is_correct: bool,
        unparsed: bool,
        detail: Optional[Dict[str, Any]],
    ) -> None:
        """記錄一筆樣本結果並更新統計。

        detail 為 None 表示僅計入統計、不寫入 JSONL
        （--resume 回填先前已完成題目時使用，避免重複寫入）。
        """
        self.question_stats.setdefault(question_id, {"correct": 0, "total": 0})
        if is_correct:
            self.question_stats[question_id]["correct"] += 1
            self.total_correct += 1
        if unparsed:
            self.total_unparsed += 1
        self.question_stats[question_id]["total"] += 1
        self.total_samples += 1
        if detail is not None:
            self.detailed_results.append(detail)


class Evaluator:
    #: 評測管線分派表：依 Extractor 宣告的 capability flag 依序比對，
    #: 全部不符則走預設的文字解析管線。新增評測類型時在此註冊，
    #: 不需修改 evaluate_file 本身。
    _PIPELINES: Tuple[Tuple[str, str], ...] = (
        ("uses_logprobs", "_run_logprobs_pipeline"),
        ("uses_tool_calls", "_run_tool_calls_pipeline"),
        ("uses_prompt_injection", "_run_prompt_injection_pipeline"),
        ("uses_ifeval", "_run_ifeval_pipeline"),
        ("uses_audio", "_run_audio_pipeline"),
        ("uses_vision", "_run_vision_pipeline"),
    )

    def __init__(
        self,
        llm: LLM,
        extractor: Extractor,
        scorer: Scorer,
        config: dict,
        eval_method: str = "",
        system_prompt_enabled: bool = True,
        samples_per_question: int = 1,
        pass_k: int = 1,
        shuffle_options: bool = False,
        model_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.llm = llm
        self.extractor = extractor
        self.scorer = scorer
        self.config = config
        self.eval_method = eval_method or config.get("evaluation", {}).get("evaluation_method", "")
        self.system_prompt_enabled = system_prompt_enabled
        self.rate_limiter = RateLimiter(calls_per_second=self.config["llm_api"]["api_rate_limit"])
        self.samples_per_question = max(1, int(samples_per_question))
        self.pass_k = max(1, int(pass_k))
        self.shuffle_options = bool(shuffle_options)
        self.model_overrides = model_overrides or {}
        # max_workers 未設定（null / 0 / 負值）時交由 ThreadPoolExecutor 使用預設值
        max_workers = self.config["llm_api"].get("max_workers")
        self._max_workers: Optional[int] = (
            int(max_workers) if max_workers and int(max_workers) > 0 else None
        )
        # scorer.score_full 的簽名在整次評測中不變，於此一次解析避免逐題呼叫 inspect
        self._score_full_accepts_prompt = hasattr(scorer, "score_full") and (
            "prompt" in inspect.signature(scorer.score_full).parameters
        )

    # ── 共用輔助 ────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_option_keys(question_data: dict) -> list:
        """動態偵測選項鍵（避免硬編碼 A/B/C/D）。

        只接受從 A 開始、依 Excel 風格標籤（A…Z、AA、AB…）連續出現的鍵，
        避免 ID、NO 之類的大寫 metadata 欄位被誤判為選項。
        """
        candidates = {
            k for k in question_data if isinstance(k, str) and k.isupper() and len(k) <= 2
        }
        option_keys: list = []
        idx = 0
        while True:
            label = _excel_label(idx)
            if label not in candidates:
                break
            option_keys.append(label)
            idx += 1
        return option_keys

    def shuffle_question_options(self, question_data: dict) -> dict:
        option_keys = self._detect_option_keys(question_data)

        if not option_keys:
            return question_data

        correct_ans = question_data.get("answer")
        if correct_ans not in option_keys:
            log_error("shuffle_options: 題目缺少 answer 欄位或 answer 不是選項鍵，跳過選項重排")
            return question_data

        shuffled_keys = option_keys[:]
        random.shuffle(shuffled_keys)

        # 保留所有其他欄位（id、image_path 等），僅重排選項內容並同步更新答案
        new_data = dict(question_data)
        for new_key, old_key in zip(option_keys, shuffled_keys):
            new_data[new_key] = question_data[old_key]
            if old_key == correct_ans:
                new_data["answer"] = new_key

        return new_data

    def _compose_question_text(self, q: dict, option_keys: list, exclude: tuple = ()) -> str:
        """組合題目文字。

        MCQ 題（有選項鍵）只帶入選項內容，避免 id、subject 等 metadata 欄位
        被當成選項渲染進 prompt；自由作答題（無選項鍵）保留除 question/answer
        以外的所有欄位（如 text2sql 的 db_id、evidence 需要進入 prompt）。
        """
        if option_keys:
            lines = [f"{k}: {q[k]}" for k in option_keys]
        else:
            excluded = {"question", "answer", *exclude}
            lines = [f"{k}: {v}" for k, v in q.items() if k not in excluded]
        return q["question"] + "\n" + "\n".join(lines)

    def _iter_questions(
        self,
        dataset: Dataset,
        completed_ids: Optional[Any],
    ) -> Iterator[Tuple[int, dict]]:
        """迭代題目，套用 shuffle 與 --resume 已完成題目跳過。"""
        for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
            if completed_ids is not None and str(idx) in completed_ids:
                continue
            if self.shuffle_options:
                q = self.shuffle_question_options(q)
            yield idx, q

    def _iter_completions(
        self,
        future_tasks: list,
        future_to_data: Dict[Any, Any],
        acc: _EvalAccumulator,
    ) -> Iterator[Tuple[Any, Any, Any]]:
        """迭代已完成的 futures，統一處理單題 API 失敗。

        單題在重試耗盡後仍失敗時跳過該題並計入 failed，
        避免整檔已完成的結果一併遺失。
        """
        for future in tqdm(as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"):
            try:
                completion = future.result()
            except Exception as e:
                log_error(f"問題 {future_to_data[future][2] + 1} API 呼叫失敗，跳過該題: {e}")
                acc.total_failed += 1
                continue
            yield completion, completion.usage, future_to_data[future]

    def _score_prediction(
        self, extraction_source: Optional[str], correct_answer: Any
    ) -> Tuple[Optional[str], Optional[str], bool]:
        """抽取 → 正規化 → 比對，回傳 (原始抽取值, 正規化答案, 是否正確)。"""
        predicted_raw = self.extractor.extract(extraction_source)
        predicted_answer = None if predicted_raw is None else self.scorer.normalize(predicted_raw)
        is_correct = (
            False
            if predicted_answer is None
            else self.scorer.score(predicted_answer, correct_answer)
        )
        return predicted_raw, predicted_answer, is_correct

    @staticmethod
    def _usage_fields(usage: Any) -> Dict[str, Any]:
        return {
            "usage_completion_tokens": usage.completion_tokens if usage else None,
            "usage_prompt_tokens": usage.prompt_tokens if usage else None,
            "usage_total_tokens": usage.total_tokens if usage else None,
        }

    def _select_pipeline(self) -> Any:
        """依 Extractor 的 capability flag 選擇評測管線。"""
        for flag, method_name in self._PIPELINES:
            if getattr(self.extractor, flag, False):
                return getattr(self, method_name)
        return self._run_text_pipeline

    # ── 主流程 ──────────────────────────────────────────────────────────────

    def evaluate_file(
        self,
        file_path: str,
        timestamp: str,
        prompt_lang: str = "zh",
        completed_ids: Optional[Any] = None,
    ) -> Tuple[str, Dict[str, Any], str]:
        """評測單一檔案。

        Args:
            file_path: 評測檔案路徑
            timestamp: 結果檔時間戳記（含 run 編號）
            prompt_lang: system prompt 語言
            completed_ids: --resume 模式下此檔案已完成的題目。
                           dict（question_id 字串 → is_correct）時會跳過這些題目
                           並將其結果回填進統計，讓 summary 涵蓋整個資料集；
                           set（僅 question_id 字串）時只跳過、不回填。
        """
        dataset = Dataset(file_path)
        acc = _EvalAccumulator()

        # --resume：回填先前已完成題目的結果（不重寫 JSONL），
        # 避免 resume 後的 accuracy 只以補跑的題目計算
        if isinstance(completed_ids, dict):
            for qid_str, was_correct in completed_ids.items():
                seed_qid: Any = int(qid_str) if str(qid_str).isdigit() else qid_str
                acc.record(seed_qid, bool(was_correct), unparsed=False, detail=None)

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            pipeline = self._select_pipeline()
            pipeline(dataset, executor, acc, prompt_lang, completed_ids, file_path)

        accuracy = acc.total_correct / acc.total_samples if acc.total_samples else 0

        # 計算 pass@k
        pass_at_k_values = []
        for key, stats in acc.question_stats.items():
            # 跳過內部統計 key（IFEval / ASR）
            if isinstance(key, str) and key.startswith("_"):
                continue
            c = stats["correct"]
            n = stats["total"]
            k = self.pass_k
            if n == 0 or k > n or c == 0:
                pass_at_k_values.append(0.0)
            else:
                pass_at_k_values.append(1.0 - comb(n - c, k) / comb(n, k))
        pass_at_k = sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0.0

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        node_id = _get_node_id()
        # rank 來自 config 的 distributed 區段（由 ConfigurationManager 從 RANK 環境變數帶入）
        rank = self.config.get("distributed", {}).get("rank", 0)
        if node_id != "0" or rank != 0:
            shard_suffix = f"_node{node_id}_rank{rank}"
        else:
            shard_suffix = ""
        results_path = os.path.join(results_dir, f"eval_results_{timestamp}{shard_suffix}.jsonl")

        with open(results_path, "a", encoding="utf-8") as f:
            for detail in acc.detailed_results:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

        unparsed_rate = acc.total_unparsed / acc.total_samples if acc.total_samples else 0.0
        print(
            f"✅ 評測完成，正確率: {accuracy:.1%} "
            f"({acc.total_correct}/{acc.total_samples})，結果已追加至 {results_path}"
        )
        if acc.total_unparsed > 0:
            print(f"⚠️  無法解析: {acc.total_unparsed}/{acc.total_samples} ({unparsed_rate:.1%})")
        if acc.total_failed > 0:
            # API 呼叫失敗被跳過的題目不計入 accuracy 分母，必須明確警告使用者
            print(f"⚠️  API 呼叫失敗被跳過: {acc.total_failed} 題（accuracy 僅以成功題目計算）")
        metrics: Dict[str, Any] = {
            "accuracy": accuracy,
            "pass_at_k": pass_at_k,
            "pass_metric": f"pass@{self.pass_k}",
            "pass_k": self.pass_k,
            "unparsed_count": acc.total_unparsed,
            "unparsed_rate": unparsed_rate,
            "failed_count": acc.total_failed,
            "total_count": acc.total_samples,
        }

        # ASR 額外指標
        if getattr(self.extractor, "uses_audio", False):
            asr_wer = acc.question_stats.get("_asr_wer", {})
            asr_cer = acc.question_stats.get("_asr_cer", {})
            avg_wer = asr_wer["sum"] / asr_wer["count"] if asr_wer.get("count") else None
            avg_cer = asr_cer["sum"] / asr_cer["count"] if asr_cer.get("count") else None
            if avg_wer is not None:
                metrics["avg_wer"] = round(avg_wer, 6)
            if avg_cer is not None:
                metrics["avg_cer"] = round(avg_cer, 6)
            if avg_wer is not None or avg_cer is not None:
                parts = []
                if avg_wer is not None:
                    parts.append(f"WER={avg_wer:.2%}")
                if avg_cer is not None:
                    parts.append(f"CER={avg_cer:.2%}")
                print(f"  ASR 指標: {' | '.join(parts)}")

        # IFEval 額外指標
        if getattr(self.extractor, "uses_ifeval", False):
            inst_strict = acc.question_stats.get("_ifeval_inst_strict", {})
            inst_loose = acc.question_stats.get("_ifeval_inst_loose", {})
            prompt_loose_count = sum(
                1 for d in acc.detailed_results if d.get("prompt_loose", False)
            )
            inst_strict_acc = (
                inst_strict["correct"] / inst_strict["total"] if inst_strict.get("total") else 0.0
            )
            inst_loose_acc = (
                inst_loose["correct"] / inst_loose["total"] if inst_loose.get("total") else 0.0
            )
            prompt_loose_acc = prompt_loose_count / acc.total_samples if acc.total_samples else 0.0
            metrics.update(
                {
                    "prompt_strict": accuracy,  # same as accuracy
                    "prompt_loose": prompt_loose_acc,
                    "instruction_strict": inst_strict_acc,
                    "instruction_loose": inst_loose_acc,
                }
            )
            print(
                f"  prompt strict={accuracy:.1%}  loose={prompt_loose_acc:.1%} | "
                f"instruction strict={inst_strict_acc:.1%}  loose={inst_loose_acc:.1%}"
            )

        return file_path, metrics, results_path

    # ── 各評測管線 ──────────────────────────────────────────────────────────

    def _run_logprobs_pipeline(
        self,
        dataset: Dataset,
        executor: ThreadPoolExecutor,
        acc: _EvalAccumulator,
        prompt_lang: str,
        completed_ids: Optional[Any],
        file_path: str,
    ) -> None:
        """logit 路徑：以各選項的 log-likelihood 決定答案。"""
        question_records = []

        for idx, q in self._iter_questions(dataset, completed_ids):
            option_keys = self._detect_option_keys(q)
            question_text = self._compose_question_text(q, option_keys)
            logit_context = question_text + "\nAnswer:"

            try:
                correct_answer = self.scorer.normalize(q["answer"])
            except (KeyError, AttributeError) as e:
                log_error(f"\n Error processing question {idx + 1}: {str(e)}")
                continue

            choice_futures: Dict[str, Any] = {}
            for choice_key in option_keys:
                self.rate_limiter.wait()
                choice_futures[choice_key] = executor.submit(
                    self.llm.score_continuation,
                    logit_context,
                    f" {choice_key}",
                )

            question_records.append(
                {
                    "idx": idx,
                    "question_text": question_text,
                    "correct_answer": correct_answer,
                    "choice_futures": choice_futures,
                }
            )

        for record in tqdm(question_records, desc="處理回應中"):
            question_id = record["idx"]
            question_text = record["question_text"]
            correct_answer = record["correct_answer"]

            scores: Dict[str, float] = {}
            for k, f in record["choice_futures"].items():
                try:
                    scores[k] = f.result()
                except Exception as e:
                    log_error(f"問題 {question_id + 1} 選項 {k} API 呼叫失敗: {e}")
                    scores[k] = float("-inf")

            if scores and any(v > float("-inf") for v in scores.values()):
                predicted_raw = max(scores, key=scores.get)
                predicted_answer: Optional[str] = self.scorer.normalize(predicted_raw)
            else:
                predicted_answer = None
                log_error(f"問題 {question_id + 1} 的所有選項均無法取得 log-likelihood")

            is_correct = (
                False
                if predicted_answer is None
                else self.scorer.score(predicted_answer, correct_answer)
            )

            acc.record(
                question_id,
                is_correct,
                unparsed=predicted_answer is None,
                detail={
                    "file": file_path,
                    "question_id": question_id,
                    "sample_id": 0,
                    "question": question_text,
                    "correct_answer": correct_answer,
                    "llm_output": None,
                    "llm_reasoning_output": None,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    # -inf 無法表示為合法 JSON（會輸出 -Infinity），改以 null 表示失敗選項
                    "logprob_scores": {
                        k: (None if v == float("-inf") else v) for k, v in scores.items()
                    },
                    **self._usage_fields(None),
                },
            )

    def _run_tool_calls_pipeline(
        self,
        dataset: Dataset,
        executor: ThreadPoolExecutor,
        acc: _EvalAccumulator,
        prompt_lang: str,
        completed_ids: Optional[Any],
        file_path: str,
    ) -> None:
        """BFCL FC 路徑：以原生 tool calls 作答。"""
        future_tasks: list = []
        future_to_data: Dict[Any, Any] = {}

        for idx, q in self._iter_questions(dataset, completed_ids):
            try:
                correct_answer = self.scorer.normalize(q["answer"])
                functions = json.loads(q.get("functions", "[]"))
                tools = convert_bfcl_functions_to_tools(functions)
                messages = json.loads(q["question"])
            except (KeyError, json.JSONDecodeError, AttributeError) as e:
                log_error(f"問題 {idx + 1} 資料格式錯誤: {e}")
                continue

            self.rate_limiter.wait()
            future = executor.submit(
                self.llm.call,
                "",
                prompt_lang,
                self.eval_method,
                False,
                self.samples_per_question,
                self.model_overrides,
                tools,
                messages,
            )
            future_tasks.append(future)
            future_to_data[future] = (q.get("question", ""), correct_answer, idx)

        for completion, usage, data in self._iter_completions(future_tasks, future_to_data, acc):
            question_text, correct_answer, question_id = data

            for sample_id, choice in enumerate(completion.choices[: self.samples_per_question]):
                message = choice.message
                tool_calls = getattr(message, "tool_calls", None)

                if tool_calls:
                    extraction_source = json.dumps(
                        [
                            {
                                "name": tc.function.name,
                                "arguments": json.loads(tc.function.arguments),
                            }
                            for tc in tool_calls
                        ],
                        ensure_ascii=False,
                    )
                else:
                    extraction_source = None
                    log_error(
                        f"問題 {question_id} 未回傳 tool_calls（finish_reason={choice.finish_reason}）"
                    )

                _, predicted_answer, is_correct = self._score_prediction(
                    extraction_source, correct_answer
                )

                acc.record(
                    question_id,
                    is_correct,
                    unparsed=predicted_answer is None,
                    detail={
                        "file": file_path,
                        "question_id": question_id,
                        "sample_id": sample_id,
                        "question": question_text,
                        "correct_answer": correct_answer,
                        "llm_output": json.dumps(
                            [tc.function.name for tc in tool_calls] if tool_calls else [],
                        ),
                        "llm_reasoning_output": None,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        **self._usage_fields(usage),
                    },
                )

    def _run_prompt_injection_pipeline(
        self,
        dataset: Dataset,
        executor: ThreadPoolExecutor,
        acc: _EvalAccumulator,
        prompt_lang: str,
        completed_ids: Optional[Any],
        file_path: str,
    ) -> None:
        """BFCL Prompting 路徑：以 system prompt 注入 function 定義。"""
        future_tasks: list = []
        future_to_data: Dict[Any, Any] = {}

        for idx, q in self._iter_questions(dataset, completed_ids):
            try:
                correct_answer = self.scorer.normalize(q["answer"])
                functions = json.loads(q.get("functions", "[]"))
                base_messages = json.loads(q["question"])
                messages = inject_bfcl_system_prompt(base_messages, functions)
            except (KeyError, json.JSONDecodeError, AttributeError) as e:
                log_error(f"問題 {idx + 1} 資料格式錯誤: {e}")
                continue

            self.rate_limiter.wait()
            future = executor.submit(
                self.llm.call,
                "",
                prompt_lang,
                self.eval_method,
                False,
                self.samples_per_question,
                self.model_overrides,
                None,
                messages,
            )
            future_tasks.append(future)
            future_to_data[future] = (q.get("question", ""), correct_answer, idx)

        for completion, usage, data in self._iter_completions(future_tasks, future_to_data, acc):
            question_text, correct_answer, question_id = data

            for sample_id, choice in enumerate(completion.choices[: self.samples_per_question]):
                message = choice.message
                content = message.content
                reasoning_content = _get_reasoning_text(message)
                if content:
                    content = _strip_think_blocks(content)
                extraction_source = content if content else reasoning_content
                if extraction_source is None:
                    log_error(f"問題 {question_id} 的 content 均為 null")

                _, predicted_answer, is_correct = self._score_prediction(
                    extraction_source, correct_answer
                )

                acc.record(
                    question_id,
                    is_correct,
                    unparsed=predicted_answer is None,
                    detail={
                        "file": file_path,
                        "question_id": question_id,
                        "sample_id": sample_id,
                        "question": question_text,
                        "correct_answer": correct_answer,
                        "llm_output": content,
                        "llm_reasoning_output": reasoning_content,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        **self._usage_fields(usage),
                    },
                )

    def _run_ifeval_pipeline(
        self,
        dataset: Dataset,
        executor: ThreadPoolExecutor,
        acc: _EvalAccumulator,
        prompt_lang: str,
        completed_ids: Optional[Any],
        file_path: str,
    ) -> None:
        """IFEval / IFBench 路徑：以 instruction checkers 評分。"""
        future_tasks: list = []
        future_to_data: Dict[Any, Any] = {}

        for idx, q in self._iter_questions(dataset, completed_ids):
            try:
                # 支援 IFEval（JSON string）與 IFBench（原生 list/dict）兩種格式
                raw_ids = q.get("instruction_id_list", "[]")
                raw_kwargs = q.get("kwargs", "[]")
                instruction_id_list = json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
                kwargs_list = json.loads(raw_kwargs) if isinstance(raw_kwargs, str) else raw_kwargs
                # IFEval uses "question", IFBench uses "prompt"
                question_text = q.get("question", "") or q.get("prompt", "")
            except (json.JSONDecodeError, AttributeError) as e:
                log_error(f"問題 {idx + 1} 資料格式錯誤: {e}")
                continue

            ground_truth = json.dumps(
                {
                    "instruction_id_list": instruction_id_list,
                    "kwargs": kwargs_list,
                },
                ensure_ascii=False,
            )

            self.rate_limiter.wait()
            future = executor.submit(
                self.llm.call,
                question_text,
                prompt_lang,
                self.eval_method,
                False,  # system_prompt_enabled=False for IFEval
                1,
                self.model_overrides,
            )
            future_tasks.append(future)
            future_to_data[future] = (
                question_text,
                ground_truth,
                idx,
                instruction_id_list,
                kwargs_list,
            )

        # 累積 instruction-level 統計（跨題目）
        all_inst_strict: list = []
        all_inst_loose: list = []

        for completion, usage, data in self._iter_completions(future_tasks, future_to_data, acc):
            question_text, ground_truth, question_id, inst_ids, kwargs_list = data

            message = completion.choices[0].message
            content = message.content
            reasoning_content = _get_reasoning_text(message)
            if content:
                content = _strip_think_blocks(content)
            response = content if content else (reasoning_content or "")

            # 計算四個指標
            if hasattr(self.scorer, "score_full"):
                # IFBench scorer 需要 prompt 參數（某些 checker 如 RepeatChangeChecker）
                if self._score_full_accepts_prompt:
                    ifeval_result = self.scorer.score_full(
                        response, inst_ids, kwargs_list, prompt=question_text
                    )
                else:
                    ifeval_result = self.scorer.score_full(response, inst_ids, kwargs_list)
            else:
                ifeval_result = {
                    "prompt_strict": False,
                    "prompt_loose": False,
                    "instruction_strict": [],
                    "instruction_loose": [],
                }

            prompt_strict = ifeval_result["prompt_strict"]
            prompt_loose = ifeval_result["prompt_loose"]
            inst_strict = ifeval_result["instruction_strict"]
            inst_loose = ifeval_result["instruction_loose"]

            all_inst_strict.extend(inst_strict)
            all_inst_loose.extend(inst_loose)

            # is_correct = prompt-level strict（主要指標）
            is_correct = prompt_strict

            acc.record(
                question_id,
                is_correct,
                unparsed=False,
                detail={
                    "file": file_path,
                    "question_id": question_id,
                    "sample_id": 0,
                    "question": question_text,
                    "correct_answer": ground_truth,
                    "llm_output": response,
                    "llm_reasoning_output": None,
                    "predicted_answer": response,
                    "is_correct": is_correct,
                    "prompt_strict": prompt_strict,
                    "prompt_loose": prompt_loose,
                    "instruction_strict": inst_strict,
                    "instruction_loose": inst_loose,
                    **self._usage_fields(usage),
                },
            )

        # 在 metrics 中補充 instruction-level 指標
        if all_inst_strict:
            acc.question_stats["_ifeval_inst_strict"] = {
                "correct": sum(all_inst_strict),
                "total": len(all_inst_strict),
            }
        if all_inst_loose:
            acc.question_stats["_ifeval_inst_loose"] = {
                "correct": sum(all_inst_loose),
                "total": len(all_inst_loose),
            }

    def _run_audio_pipeline(
        self,
        dataset: Dataset,
        executor: ThreadPoolExecutor,
        acc: _EvalAccumulator,
        prompt_lang: str,
        completed_ids: Optional[Any],
        file_path: str,
    ) -> None:
        """ASR 音檔路徑：語音轉錄並計算 WER/CER。"""
        future_tasks: list = []
        future_to_data: Dict[Any, Any] = {}

        for idx, q in self._iter_questions(dataset, completed_ids):
            try:
                correct_answer = q["answer"]
                # 音檔路徑：支援 audio_path 欄位或 question 欄位
                audio_path = q.get("audio_path") or q.get("question", "")
            except (KeyError, AttributeError) as e:
                log_error(f"問題 {idx + 1} 資料格式錯誤: {e}")
                continue

            self.rate_limiter.wait()

            if (
                hasattr(self.llm, "call")
                and getattr(type(self.llm), "__name__", "") == "WhisperModel"
            ):
                # Whisper API 路徑：直接傳音檔路徑
                future = executor.submit(
                    self.llm.call,
                    audio_path,
                    prompt_lang,
                    self.eval_method,
                    False,
                    1,
                    self.model_overrides,
                )
            else:
                # Chat Completions 多模態路徑：建構含音檔 URL 的 messages
                import base64

                audio_url = audio_path
                if os.path.isfile(audio_path):
                    with open(audio_path, "rb") as af:
                        b64 = base64.b64encode(af.read()).decode("utf-8")
                    ext = os.path.splitext(audio_path)[1].lstrip(".")
                    audio_url = f"data:audio/{ext};base64,{b64}"

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio_url", "audio_url": {"url": audio_url}},
                            {"type": "text", "text": "請將這段語音轉錄為文字，只輸出轉錄結果。"},
                        ],
                    }
                ]
                future = executor.submit(
                    self.llm.call,
                    "",
                    prompt_lang,
                    self.eval_method,
                    False,
                    1,
                    self.model_overrides,
                    None,
                    messages,
                )

            future_tasks.append(future)
            future_to_data[future] = (audio_path, correct_answer, idx)

        # 累積 ASR 指標
        all_wer: list = []
        all_cer: list = []
        jiwer_warned = False

        for completion, usage, data in self._iter_completions(future_tasks, future_to_data, acc):
            audio_path, correct_answer, question_id = data

            message = completion.choices[0].message
            content = message.content or ""

            predicted_raw = self.extractor.extract(content)
            predicted_answer = (
                None if predicted_raw is None else self.scorer.normalize(predicted_raw)
            )
            gold_normalized = self.scorer.normalize(correct_answer)

            # 計算完整 ASR 指標
            asr_detail: Dict[str, Any] = {}
            if hasattr(self.scorer, "score_full") and predicted_answer is not None:
                try:
                    asr_detail = self.scorer.score_full(predicted_raw, correct_answer)
                    all_wer.append(asr_detail.get("wer", 0.0))
                    all_cer.append(asr_detail.get("cer", 0.0))
                except ImportError as e:
                    # jiwer 未安裝時跳過 WER/CER 計算，僅記錄一次原因避免洗版
                    if not jiwer_warned:
                        log_error(f"WER/CER 計算失敗（jiwer 未安裝？）: {e}")
                        jiwer_warned = True

            is_correct = (
                False
                if predicted_answer is None
                else self.scorer.score(predicted_answer, gold_normalized)
            )

            result_entry: Dict[str, Any] = {
                "file": file_path,
                "question_id": question_id,
                "sample_id": 0,
                "question": audio_path,
                "correct_answer": correct_answer,
                "llm_output": content,
                "llm_reasoning_output": None,
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                **self._usage_fields(usage),
            }
            result_entry.update(asr_detail)
            acc.record(
                question_id,
                is_correct,
                unparsed=predicted_answer is None,
                detail=result_entry,
            )

        # 在 metrics 中補充 ASR 指標
        if all_wer:
            acc.question_stats["_asr_wer"] = {
                "sum": sum(all_wer),
                "count": len(all_wer),
            }
        if all_cer:
            acc.question_stats["_asr_cer"] = {
                "sum": sum(all_cer),
                "count": len(all_cer),
            }

    def _run_vision_pipeline(
        self,
        dataset: Dataset,
        executor: ThreadPoolExecutor,
        acc: _EvalAccumulator,
        prompt_lang: str,
        completed_ids: Optional[Any],
        file_path: str,
    ) -> None:
        """Vision 圖片路徑：multimodal MCQ 評測。"""
        # 從 extractor 設定讀取 strategy_config 內的 vision 參數
        vision_cfg = getattr(self.extractor, "_config", {}) or {}
        image_field = vision_cfg.get("image_field", "image_path")
        max_image_size = vision_cfg.get("max_image_size")
        image_detail = vision_cfg.get("image_detail", "auto")

        future_tasks: list = []
        future_to_data: Dict[Any, Any] = {}

        for idx, q in self._iter_questions(dataset, completed_ids):
            option_keys = self._detect_option_keys(q)

            image_path = q.get(image_field) or q.get("image_url") or q.get("image")
            if not image_path:
                log_error(f"問題 {idx + 1} 缺少圖片欄位 '{image_field}'，跳過")
                continue

            try:
                correct_answer = self.scorer.normalize(q["answer"])
            except (KeyError, AttributeError) as e:
                log_error(f"\n Error processing question {idx + 1}: {str(e)}")
                continue

            # 建構文字題目（與文字 MCQ 相同邏輯：question + 選項）
            question_text = self._compose_question_text(
                q, option_keys, exclude=(image_field, "image_url", "image", "id")
            )

            # 圖片編碼為 data URI 或直接使用 URL
            try:
                image_url = _encode_image_to_data_uri(image_path, max_image_size)
            except FileNotFoundError as e:
                log_error(f"問題 {idx + 1} 圖片載入失敗: {e}")
                continue

            messages = _build_vision_messages(image_url, question_text, image_detail)

            self.rate_limiter.wait()
            # Vision 路徑使用預先建構的 multimodal messages，
            # question_text / prompt_lang / system_prompt 等參數
            # 在 OpenAIModel.call 內會被略過（messages != None 走另一條分支），
            # 為了可讀性這裡只傳必要的 kwargs。
            future = executor.submit(
                self.llm.call,
                question_text="",
                prompt_lang=prompt_lang,
                eval_method=self.eval_method,
                system_prompt_enabled=self.system_prompt_enabled,
                num_samples=self.samples_per_question,
                model_overrides=self.model_overrides,
                messages=messages,
            )
            future_tasks.append(future)
            future_to_data[future] = (question_text, correct_answer, idx, image_path)

        for completion, usage, data in self._iter_completions(future_tasks, future_to_data, acc):
            question_text, correct_answer, question_id, image_path = data

            for sample_id, choice in enumerate(completion.choices[: self.samples_per_question]):
                message = choice.message
                content = message.content
                reasoning_content = _get_reasoning_text(message)

                if content:
                    content = _strip_think_blocks(content)
                if not content and reasoning_content:
                    content = _strip_think_blocks(reasoning_content)
                content = content or ""

                _, predicted_answer, is_correct = self._score_prediction(content, correct_answer)

                acc.record(
                    question_id,
                    is_correct,
                    unparsed=predicted_answer is None,
                    detail={
                        "file": file_path,
                        "question_id": question_id,
                        "sample_id": sample_id,
                        "question": question_text,
                        "image_path": image_path,
                        "correct_answer": correct_answer,
                        "llm_output": content,
                        "llm_reasoning_output": reasoning_content,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        **self._usage_fields(usage),
                    },
                )

    def _run_text_pipeline(
        self,
        dataset: Dataset,
        executor: ThreadPoolExecutor,
        acc: _EvalAccumulator,
        prompt_lang: str,
        completed_ids: Optional[Any],
        file_path: str,
    ) -> None:
        """預設文字解析路徑。"""
        future_tasks: list = []
        future_to_data: Dict[Any, Any] = {}

        for idx, q in self._iter_questions(dataset, completed_ids):
            option_keys = self._detect_option_keys(q)
            question_text = self._compose_question_text(q, option_keys)

            try:
                correct_answer = self.scorer.normalize(q["answer"])
            except (KeyError, AttributeError) as e:
                log_error(f"\n Error processing question {idx + 1}: {str(e)}")
                continue

            self.rate_limiter.wait()
            future = executor.submit(
                self.llm.call,
                question_text,
                prompt_lang,
                self.eval_method,
                self.system_prompt_enabled,
                self.samples_per_question,
                self.model_overrides,
            )
            future_tasks.append(future)
            future_to_data[future] = (question_text, correct_answer, idx)

        for completion, usage, data in self._iter_completions(future_tasks, future_to_data, acc):
            question_text, correct_answer, question_id = data

            for sample_id, choice in enumerate(completion.choices[: self.samples_per_question]):
                message = choice.message
                content = message.content
                reasoning_content = _get_reasoning_text(message)

                # 統一推理輸出解析：
                # A. inline think tag（如 Ollama）：content 含 <think>...</think>
                #    → 剝離 think block，只留結尾的答案部分
                # B. content=null（如 vLLM skip_special_tokens=true）：
                #    → 優先使用 reasoning，若為 None 再回退 reasoning_content
                if content:
                    content = _strip_think_blocks(content)

                extraction_source = content if content else reasoning_content
                if extraction_source is None:
                    log_error(
                        f"問題 {question_id} 的 content、reasoning、reasoning_content 均為 null，無法提取答案"
                    )

                _, predicted_answer, is_correct = self._score_prediction(
                    extraction_source, correct_answer
                )

                acc.record(
                    question_id,
                    is_correct,
                    unparsed=predicted_answer is None,
                    detail={
                        "file": file_path,
                        "question_id": question_id,
                        "sample_id": sample_id,
                        "question": question_text,
                        "correct_answer": correct_answer,
                        "llm_output": content,
                        "llm_reasoning_output": reasoning_content,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        **self._usage_fields(usage),
                    },
                )
