import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from twinkle_eval.core.abc import Extractor, Scorer
from twinkle_eval.core.logger import log_error, log_warning
from twinkle_eval.datasets import Dataset, index_to_label
from twinkle_eval.metrics.extractors.bfcl_prompt import (
    BFCLPromptExtractor,
    inject_bfcl_system_prompt,
)
from twinkle_eval.metrics.extractors.tool_call import (
    ToolCallExtractor,
    convert_bfcl_functions_to_tools,
)
from twinkle_eval.models import LLM


def _get_node_id() -> str:
    """取得當前節點識別碼，優先使用 SLURM_NODEID，否則回退至 node0。"""
    slurm_node = os.environ.get("SLURM_NODEID")
    return slurm_node if slurm_node is not None else "0"


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


def detect_option_keys(question_data: Dict[str, Any]) -> List[str]:
    """動態偵測題目字典中的選項鍵，依標籤順序回傳（A、B、…、Z、AA、AB、…）。

    以 ``index_to_label()`` 產生的標準標籤序列由 A 開始逐一比對，遇到第一個
    缺口即停止。這樣可支援任意數量的選項（如 MMLU-Pro 的 A–J），同時避免把
    ``ID`` 這類剛好是大寫短字串的中繼資料欄位誤判為選項。

    Args:
        question_data: 題目字典。

    Returns:
        依標籤順序排列的選項鍵列表；無選項時回傳空列表。
    """
    # 單一大寫字母一律視為候選選項鍵（涵蓋 A–D、不連續的 A/B/C/E、
    # 以及 T/F、Y/N 這類非 A 起始的標籤）。
    singles = {k for k in question_data if isinstance(k, str) and len(k) == 1 and k.isupper()}

    # 多字母標籤（AA、AB…）只在它延續標準序列時才算選項，
    # 藉此排除 ID、NO 這類剛好是兩個大寫字母的 metadata 欄位。
    ordered: List[str] = []
    idx = 0
    while True:
        label = index_to_label(idx)
        if label not in question_data:
            break
        ordered.append(label)
        idx += 1

    extras = sorted(singles - set(ordered))
    keys = ordered + extras
    return keys if len(keys) >= 2 else []


def describe_dropped_fields(
    question_data: Dict[str, Any],
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """回傳 ``build_question_text()`` 會略過的非選項欄位名稱。

    用於每個檔案發出一次提示，讓「某個作答必需的欄位（如 hint、context）被丟棄」
    這種靜默的分數下降變得可觀察。
    """
    keys = detect_option_keys(question_data)
    if not keys:
        return []
    skip = {"question", "answer", *keys, *(exclude or ())}
    return [k for k in question_data if k not in skip]


def build_question_text(
    question_data: Dict[str, Any],
    option_keys: Optional[List[str]] = None,
    exclude: Optional[Iterable[str]] = None,
) -> str:
    """組出送給模型的題目文字：題幹 + 選項。

    偵測得到選項鍵時**只列出選項**。這是為了避免 ``id`` / ``domain`` /
    ``discipline`` / ``category`` 這類 metadata 欄位被當成選項送進 prompt——
    那既是雜訊，其中的學科分類欄位更等同於免費提示，會高估分數（見 #143）。

    偵測不到選項鍵時（Text-to-SQL 等非選擇題）維持列出其餘所有欄位，
    因為 ``db_id`` / ``evidence`` 這類欄位本來就該進 prompt。

    Args:
        question_data: 題目字典。
        option_keys:   已算好的選項鍵；None 表示由本函式偵測。
        exclude:       無選項鍵時額外要排除的欄位（如圖片路徑欄位）。

    Returns:
        題幹加上選項（或其餘欄位）的完整題目文字。
    """
    keys = detect_option_keys(question_data) if option_keys is None else option_keys
    if keys:
        body = "\n".join(f"{k}: {question_data[k]}" for k in keys)
    else:
        skip = {"question", "answer", *(exclude or ())}
        body = "\n".join(f"{k}: {v}" for k, v in question_data.items() if k not in skip)
    return question_data["question"] + "\n" + body


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


class Evaluator:
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

    def shuffle_question_options(self, question_data: dict) -> dict:
        """隨機重排題目選項，消除模型對選項位置的偏好。

        選項鍵動態偵測（見 ``detect_option_keys``），支援任意數量的選項；
        非選項欄位（``image_path``、``id`` 等）原樣保留。

        若無法在選項鍵中定位正解（``answer`` 缺失、非字母標籤、或指向不存在的
        選項），則原樣回傳不做重排——重排後無法回填正解等同於毀損題目。
        """
        option_keys = detect_option_keys(question_data)
        if len(option_keys) < 2:
            return question_data

        # 只有 A 起始的位置性標籤才可重排。T/F、Y/N 這類標籤本身帶有語意，
        # 重排會讓標籤與內容錯位（T 指向「否」），使正確作答被判為錯。
        if option_keys != [index_to_label(i) for i in range(len(option_keys))]:
            return question_data

        correct_key = str(question_data.get("answer", "")).strip().upper()
        if correct_key not in option_keys:
            log_warning(
                f"跳過選項重排：answer={question_data.get('answer')!r} "
                f"不在偵測到的選項鍵 {option_keys} 中"
            )
            return question_data

        # 以「原始鍵」而非選項文字定位正解，避免兩個選項文字相同時比對到錯誤選項
        correct_index = option_keys.index(correct_key)
        texts = [question_data[k] for k in option_keys]

        order = list(range(len(option_keys)))
        random.shuffle(order)

        new_data = dict(question_data)
        for new_pos, old_index in enumerate(order):
            new_data[option_keys[new_pos]] = texts[old_index]
            if old_index == correct_index:
                new_data["answer"] = option_keys[new_pos]

        return new_data

    def evaluate_file(
        self, file_path: str, timestamp: str, prompt_lang: str = "zh"
    ) -> Tuple[str, Dict[str, Any], str]:
        dataset = Dataset(file_path)

        # 每個檔案提示一次：哪些非選項欄位不會進入 prompt。
        # 這些欄位多半是 metadata（id、domain），但若資料集把作答必需的內容
        # （hint、context）放在選項之外，就會被靜默丟棄而使分數無故下降。
        if dataset.data:
            vision_cfg = getattr(self.extractor, "_config", {}) or {}
            extra_exclude = (
                (vision_cfg.get("image_field", "image_path"), "image_url", "image")
                if getattr(self.extractor, "uses_vision", False)
                else ()
            )
            dropped = describe_dropped_fields(dataset.data[0], extra_exclude)
            if dropped:
                notice = (
                    f"ℹ️  {file_path}：以下欄位不會進入 prompt（只列出選項）："
                    f"{', '.join(dropped)}。"
                    "若其中含作答必需的內容（如 hint、context），請改寫進 question 欄位。"
                )
                # logger 只寫入 logs/ 檔案（basicConfig 未設 StreamHandler），
                # 而這個提示的目的正是讓靜默的分數下降被看見，所以同時印到終端機
                print(notice)
                log_warning(notice)

        total_correct_samples = 0
        total_samples = 0
        total_unparsed = 0
        detailed_results = []
        question_stats: Dict[int, Dict[str, int]] = {}

        with ThreadPoolExecutor() as executor:
            if self.extractor.uses_logprobs:
                # ── logit 路徑 ──────────────────────────────────────────────
                question_records = []

                for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
                    if self.shuffle_options:
                        q = self.shuffle_question_options(q)

                    option_keys = detect_option_keys(q)
                    question_text = build_question_text(q, option_keys)
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
                            "option_keys": option_keys,
                            "choice_futures": choice_futures,
                        }
                    )

                for record in tqdm(question_records, desc="處理回應中"):
                    question_id = record["idx"]
                    question_text = record["question_text"]
                    correct_answer = record["correct_answer"]
                    option_keys = record["option_keys"]

                    scores: Dict[str, float] = {
                        k: f.result() for k, f in record["choice_futures"].items()
                    }

                    if scores and any(v > float("-inf") for v in scores.values()):
                        predicted_raw = max(scores, key=scores.get)
                        predicted_answer: Optional[str] = self.scorer.normalize(predicted_raw)
                    else:
                        predicted_answer = None
                        log_error(f"問題 {question_id} 的所有選項均無法取得 log-likelihood")

                    is_correct = (
                        False
                        if predicted_answer is None
                        else self.scorer.score(predicted_answer, correct_answer)
                    )

                    question_stats.setdefault(question_id, {"correct": 0, "total": 0})
                    if is_correct:
                        question_stats[question_id]["correct"] += 1
                        total_correct_samples += 1
                    if predicted_answer is None:
                        total_unparsed += 1
                    question_stats[question_id]["total"] += 1
                    total_samples += 1

                    detailed_results.append(
                        {
                            "question_id": question_id,
                            "sample_id": 0,
                            "question": question_text,
                            "correct_answer": correct_answer,
                            "llm_output": None,
                            "llm_reasoning_output": None,
                            "predicted_answer": predicted_answer,
                            "is_correct": is_correct,
                            "logprob_scores": scores,
                            "usage_completion_tokens": None,
                            "usage_prompt_tokens": None,
                            "usage_total_tokens": None,
                        }
                    )

            elif getattr(self.extractor, "uses_tool_calls", False):
                # ── BFCL FC 路徑 ────────────────────────────────────────────
                future_tasks = []
                future_to_data: Dict[Any, Any] = {}

                for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
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

                for future in tqdm(
                    as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"
                ):
                    llm_chat_completion = future.result()
                    usage = llm_chat_completion.usage
                    question_text, correct_answer, question_id = future_to_data[future]

                    for sample_id, choice in enumerate(
                        llm_chat_completion.choices[: self.samples_per_question]
                    ):
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

                        predicted_raw = self.extractor.extract(extraction_source)
                        predicted_answer = (
                            None if predicted_raw is None else self.scorer.normalize(predicted_raw)
                        )
                        is_correct = (
                            False
                            if predicted_answer is None
                            else self.scorer.score(predicted_answer, correct_answer)
                        )

                        question_stats.setdefault(question_id, {"correct": 0, "total": 0})
                        if is_correct:
                            question_stats[question_id]["correct"] += 1
                            total_correct_samples += 1
                        if predicted_answer is None:
                            total_unparsed += 1
                        question_stats[question_id]["total"] += 1
                        total_samples += 1

                        detailed_results.append(
                            {
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
                                "usage_completion_tokens": (
                                    usage.completion_tokens if usage else None
                                ),
                                "usage_prompt_tokens": usage.prompt_tokens if usage else None,
                                "usage_total_tokens": usage.total_tokens if usage else None,
                            }
                        )

            elif getattr(self.extractor, "uses_prompt_injection", False):
                # ── BFCL Prompting 路徑 ─────────────────────────────────────
                future_tasks = []
                future_to_data: Dict[Any, Any] = {}

                for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
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

                for future in tqdm(
                    as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"
                ):
                    llm_chat_completion = future.result()
                    usage = llm_chat_completion.usage
                    question_text, correct_answer, question_id = future_to_data[future]

                    for sample_id, choice in enumerate(
                        llm_chat_completion.choices[: self.samples_per_question]
                    ):
                        message = choice.message
                        content = message.content
                        reasoning_content = _get_reasoning_text(message)
                        if content:
                            content = _strip_think_blocks(content)
                        extraction_source = content if content else reasoning_content
                        if extraction_source is None:
                            log_error(f"問題 {question_id} 的 content 均為 null")

                        predicted_raw = self.extractor.extract(extraction_source)
                        predicted_answer = (
                            None if predicted_raw is None else self.scorer.normalize(predicted_raw)
                        )
                        is_correct = (
                            False
                            if predicted_answer is None
                            else self.scorer.score(predicted_answer, correct_answer)
                        )

                        question_stats.setdefault(question_id, {"correct": 0, "total": 0})
                        if is_correct:
                            question_stats[question_id]["correct"] += 1
                            total_correct_samples += 1
                        if predicted_answer is None:
                            total_unparsed += 1
                        question_stats[question_id]["total"] += 1
                        total_samples += 1

                        detailed_results.append(
                            {
                                "question_id": question_id,
                                "sample_id": sample_id,
                                "question": question_text,
                                "correct_answer": correct_answer,
                                "llm_output": content,
                                "llm_reasoning_output": reasoning_content,
                                "predicted_answer": predicted_answer,
                                "is_correct": is_correct,
                                "usage_completion_tokens": (
                                    usage.completion_tokens if usage else None
                                ),
                                "usage_prompt_tokens": usage.prompt_tokens if usage else None,
                                "usage_total_tokens": usage.total_tokens if usage else None,
                            }
                        )

            elif getattr(self.extractor, "uses_ifeval", False):
                # ── IFEval / IFBench 路徑 ──────────────────────────────────
                future_tasks = []
                future_to_data: Dict[Any, Any] = {}

                for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
                    try:
                        # 支援 IFEval（JSON string）與 IFBench（原生 list/dict）兩種格式
                        raw_ids = q.get("instruction_id_list", "[]")
                        raw_kwargs = q.get("kwargs", "[]")
                        instruction_id_list = (
                            json.loads(raw_ids) if isinstance(raw_ids, str) else raw_ids
                        )
                        kwargs_list = (
                            json.loads(raw_kwargs) if isinstance(raw_kwargs, str) else raw_kwargs
                        )
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

                for future in tqdm(
                    as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"
                ):
                    llm_chat_completion = future.result()
                    usage = llm_chat_completion.usage
                    question_text, ground_truth, question_id, inst_ids, kwargs_list = (
                        future_to_data[future]
                    )

                    message = llm_chat_completion.choices[0].message
                    content = message.content
                    reasoning_content = _get_reasoning_text(message)
                    if content:
                        content = _strip_think_blocks(content)
                    response = content if content else (reasoning_content or "")

                    # 計算四個指標
                    if hasattr(self.scorer, "score_full"):
                        # IFBench scorer 需要 prompt 參數（某些 checker 如 RepeatChangeChecker）
                        import inspect

                        sig = inspect.signature(self.scorer.score_full)
                        if "prompt" in sig.parameters:
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

                    question_stats.setdefault(question_id, {"correct": 0, "total": 0})
                    if is_correct:
                        question_stats[question_id]["correct"] += 1
                        total_correct_samples += 1
                    question_stats[question_id]["total"] += 1
                    total_samples += 1

                    detailed_results.append(
                        {
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
                            "usage_completion_tokens": usage.completion_tokens if usage else None,
                            "usage_prompt_tokens": usage.prompt_tokens if usage else None,
                            "usage_total_tokens": usage.total_tokens if usage else None,
                        }
                    )

                # 在 metrics 中補充 instruction-level 指標
                if all_inst_strict:
                    question_stats["_ifeval_inst_strict"] = {
                        "correct": sum(all_inst_strict),
                        "total": len(all_inst_strict),
                    }
                if all_inst_loose:
                    question_stats["_ifeval_inst_loose"] = {
                        "correct": sum(all_inst_loose),
                        "total": len(all_inst_loose),
                    }

            elif getattr(self.extractor, "uses_audio", False):
                # ── ASR 音檔路徑 ───────────────────────────────────────────
                future_tasks = []
                future_to_data: Dict[Any, Any] = {}

                for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
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
                                    {
                                        "type": "text",
                                        "text": "請將這段語音轉錄為文字，只輸出轉錄結果。",
                                    },
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

                for future in tqdm(
                    as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"
                ):
                    llm_chat_completion = future.result()
                    usage = llm_chat_completion.usage
                    audio_path, correct_answer, question_id = future_to_data[future]

                    message = llm_chat_completion.choices[0].message
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
                        except ImportError:
                            pass  # jiwer 未安裝，跳過 WER/CER 計算

                    is_correct = (
                        False
                        if predicted_answer is None
                        else self.scorer.score(predicted_answer, gold_normalized)
                    )

                    question_stats.setdefault(question_id, {"correct": 0, "total": 0})
                    if is_correct:
                        question_stats[question_id]["correct"] += 1
                        total_correct_samples += 1
                    if predicted_answer is None:
                        total_unparsed += 1
                    question_stats[question_id]["total"] += 1
                    total_samples += 1

                    result_entry: Dict[str, Any] = {
                        "question_id": question_id,
                        "sample_id": 0,
                        "question": audio_path,
                        "correct_answer": correct_answer,
                        "llm_output": content,
                        "llm_reasoning_output": None,
                        "predicted_answer": predicted_answer,
                        "is_correct": is_correct,
                        "usage_completion_tokens": usage.completion_tokens if usage else None,
                        "usage_prompt_tokens": usage.prompt_tokens if usage else None,
                        "usage_total_tokens": usage.total_tokens if usage else None,
                    }
                    result_entry.update(asr_detail)
                    detailed_results.append(result_entry)

                # 在 metrics 中補充 ASR 指標
                if all_wer:
                    question_stats["_asr_wer"] = {
                        "sum": sum(all_wer),
                        "count": len(all_wer),
                    }
                if all_cer:
                    question_stats["_asr_cer"] = {
                        "sum": sum(all_cer),
                        "count": len(all_cer),
                    }

            elif getattr(self.extractor, "uses_vision", False):
                # ── Vision 圖片路徑 ─────────────────────────────────────────
                # 從 extractor 設定讀取 strategy_config 內的 vision 參數
                vision_cfg = getattr(self.extractor, "_config", {}) or {}
                image_field = vision_cfg.get("image_field", "image_path")
                max_image_size = vision_cfg.get("max_image_size")
                image_detail = vision_cfg.get("image_detail", "auto")

                future_tasks = []
                future_to_data: Dict[Any, Any] = {}

                for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
                    if self.shuffle_options:
                        q = self.shuffle_question_options(q)

                    option_keys = detect_option_keys(q)

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
                    question_text = build_question_text(
                        q,
                        option_keys,
                        exclude=(image_field, "image_url", "image", "id"),
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
                    future_to_data[future] = (
                        question_text,
                        correct_answer,
                        idx,
                        option_keys,
                        image_path,
                    )

                for future in tqdm(
                    as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"
                ):
                    llm_chat_completion = future.result()
                    usage = llm_chat_completion.usage
                    (
                        question_text,
                        correct_answer,
                        question_id,
                        option_keys,
                        image_path,
                    ) = future_to_data[future]

                    for sample_id, choice in enumerate(
                        llm_chat_completion.choices[: self.samples_per_question]
                    ):
                        message = choice.message
                        content = message.content
                        reasoning_content = _get_reasoning_text(message)

                        if content:
                            content = _strip_think_blocks(content)
                        if not content and reasoning_content:
                            content = _strip_think_blocks(reasoning_content)
                        content = content or ""

                        predicted_raw = self.extractor.extract(content)
                        predicted_answer = (
                            None if predicted_raw is None else self.scorer.normalize(predicted_raw)
                        )

                        is_correct = (
                            False
                            if predicted_answer is None
                            else self.scorer.score(predicted_answer, correct_answer)
                        )

                        question_stats.setdefault(question_id, {"correct": 0, "total": 0})
                        if is_correct:
                            question_stats[question_id]["correct"] += 1
                            total_correct_samples += 1
                        if predicted_answer is None:
                            total_unparsed += 1
                        question_stats[question_id]["total"] += 1
                        total_samples += 1

                        detailed_results.append(
                            {
                                "question_id": question_id,
                                "sample_id": sample_id,
                                "question": question_text,
                                "image_path": image_path,
                                "correct_answer": correct_answer,
                                "llm_output": content,
                                "llm_reasoning_output": reasoning_content,
                                "predicted_answer": predicted_answer,
                                "is_correct": is_correct,
                                "usage_completion_tokens": (
                                    usage.completion_tokens if usage else None
                                ),
                                "usage_prompt_tokens": usage.prompt_tokens if usage else None,
                                "usage_total_tokens": usage.total_tokens if usage else None,
                            }
                        )

            else:
                # ── 文字解析路徑 ────────────────────────────────────────────
                future_tasks = []
                future_to_data: Dict[Any, Any] = {}

                for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
                    if self.shuffle_options:
                        q = self.shuffle_question_options(q)

                    option_keys = detect_option_keys(q)
                    question_text = build_question_text(q, option_keys)

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
                    future_to_data[future] = (question_text, correct_answer, idx, option_keys)

                for future in tqdm(
                    as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"
                ):
                    llm_chat_completion = future.result()
                    usage = llm_chat_completion.usage
                    question_text, correct_answer, question_id, option_keys = future_to_data[future]

                    for sample_id, choice in enumerate(
                        llm_chat_completion.choices[: self.samples_per_question]
                    ):
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

                        predicted_raw = self.extractor.extract(extraction_source)
                        predicted_answer = (
                            None if predicted_raw is None else self.scorer.normalize(predicted_raw)
                        )

                        is_correct = (
                            False
                            if predicted_answer is None
                            else self.scorer.score(predicted_answer, correct_answer)
                        )

                        question_stats.setdefault(question_id, {"correct": 0, "total": 0})
                        if is_correct:
                            question_stats[question_id]["correct"] += 1
                            total_correct_samples += 1
                        if predicted_answer is None:
                            total_unparsed += 1
                        question_stats[question_id]["total"] += 1
                        total_samples += 1

                        detailed_results.append(
                            {
                                "question_id": question_id,
                                "sample_id": sample_id,
                                "question": question_text,
                                "correct_answer": correct_answer,
                                "llm_output": content,
                                "llm_reasoning_output": reasoning_content,
                                "predicted_answer": predicted_answer,
                                "is_correct": is_correct,
                                "usage_completion_tokens": usage.completion_tokens,
                                "usage_prompt_tokens": usage.prompt_tokens,
                                "usage_total_tokens": usage.total_tokens,
                            }
                        )

            accuracy = total_correct_samples / total_samples if total_samples else 0

            # 計算 pass@k
            pass_at_k_values = []
            for key, stats in question_stats.items():
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
        rank = self.model_overrides.get("_rank", 0)
        if node_id != "0" or rank != 0:
            shard_suffix = f"_node{node_id}_rank{rank}"
        else:
            shard_suffix = ""
        results_path = os.path.join(results_dir, f"eval_results_{timestamp}{shard_suffix}.jsonl")

        with open(results_path, "a", encoding="utf-8") as f:
            for detail in detailed_results:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

        unparsed_rate = total_unparsed / total_samples if total_samples else 0.0
        print(f"✅ 評測完成，結果已追加至 {results_path}")
        if total_unparsed > 0:
            print(f"⚠️  無法解析: {total_unparsed}/{total_samples} ({unparsed_rate:.1%})")
        metrics = {
            "accuracy": accuracy,
            "pass_at_k": pass_at_k,
            "pass_metric": f"pass@{self.pass_k}",
            "pass_k": self.pass_k,
            "unparsed_count": total_unparsed,
            "unparsed_rate": unparsed_rate,
            "total_count": total_samples,
        }

        # ASR 額外指標
        if getattr(self.extractor, "uses_audio", False):
            asr_wer = question_stats.get("_asr_wer", {})
            asr_cer = question_stats.get("_asr_cer", {})
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
            inst_strict = question_stats.get("_ifeval_inst_strict", {})
            inst_loose = question_stats.get("_ifeval_inst_loose", {})
            prompt_loose_count = sum(1 for d in detailed_results if d.get("prompt_loose", False))
            inst_strict_acc = (
                inst_strict["correct"] / inst_strict["total"] if inst_strict.get("total") else 0.0
            )
            inst_loose_acc = (
                inst_loose["correct"] / inst_loose["total"] if inst_loose.get("total") else 0.0
            )
            prompt_loose_acc = prompt_loose_count / total_samples if total_samples else 0.0
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
