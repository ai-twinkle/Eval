import json
import os
import random
import re
import time
from math import comb
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm

from .datasets import Dataset
from .core.abc import Extractor, Scorer
from .logger import log_error
from .models import LLM


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
            return text[idx + len(end_tag):].strip()
    return text


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
        # 動態偵測選項鍵（避免硬編碼 A/B/C/D）
        options = [(k, question_data[k]) for k in ["A", "B", "C", "D"] if k in question_data]

        if not options:
            return question_data

        correct_ans = question_data["answer"]
        correct_option_text = question_data.get(correct_ans)

        random.shuffle(options)

        new_data = {"question": question_data["question"]}

        for (old_key, text), (new_key, _) in zip(
            options, [("A", ""), ("B", ""), ("C", ""), ("D", "")]
        ):
            new_data[new_key] = text
            if text == correct_option_text:
                new_data["answer"] = new_key

        return new_data

    def evaluate_file(
        self, file_path: str, timestamp: str, prompt_lang: str = "zh"
    ) -> Tuple[str, Dict[str, Any], str]:
        dataset = Dataset(file_path)

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

                    option_keys = sorted(
                        [k for k in q if isinstance(k, str) and k.isupper() and len(k) <= 2]
                    )
                    question_text = (
                        q["question"]
                        + "\n"
                        + "\n".join(
                            [f"{k}: {v}" for k, v in q.items() if k not in ["question", "answer"]]
                        )
                    )
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

                    question_records.append({
                        "idx": idx,
                        "question_text": question_text,
                        "correct_answer": correct_answer,
                        "option_keys": option_keys,
                        "choice_futures": choice_futures,
                    })

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

                    detailed_results.append({
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
                    })

            else:
                # ── 文字解析路徑 ────────────────────────────────────────────
                future_tasks = []
                future_to_data: Dict[Any, Any] = {}

                for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
                    if self.shuffle_options:
                        q = self.shuffle_question_options(q)

                    option_keys = sorted(
                        [k for k in q if isinstance(k, str) and k.isupper() and len(k) <= 2]
                    )
                    question_text = (
                        q["question"]
                        + "\n"
                        + "\n".join(
                            [f"{k}: {v}" for k, v in q.items() if k not in ["question", "answer"]]
                        )
                    )

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
                        reasoning_content = getattr(message, "reasoning_content", None)

                        # 統一推理輸出解析：
                        # A. inline think tag（如 Ollama）：content 含 <think>...</think>
                        #    → 剝離 think block，只留結尾的答案部分
                        # B. content=null（如 vLLM skip_special_tokens=true）：
                        #    → fallback 至 reasoning_content
                        if content:
                            content = _strip_think_blocks(content)

                        extraction_source = content if content else reasoning_content
                        if extraction_source is None:
                            log_error(
                                f"問題 {question_id} 的 content 與 reasoning_content 均為 null，無法提取答案"
                            )

                        predicted_raw = self.extractor.extract(extraction_source)
                        predicted_answer = (
                            None
                            if predicted_raw is None
                            else self.scorer.normalize(predicted_raw)
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

                        detailed_results.append({
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
                        })

            accuracy = total_correct_samples / total_samples if total_samples else 0

            # 計算 pass@k
            pass_at_k_values = []
            for stats in question_stats.values():
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
        return file_path, metrics, results_path
