import json
import os
import random
import re
import socket
import string
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .dataset import Dataset
from .evaluation_strategies import EvaluationStrategy
from .logger import log_error
from .models import LLM


def _get_node_id() -> str:
    """取得當前節點識別碼，優先使用 SLURM_NODEID，否則回退至 node0。

    在多節點 SLURM 作業中，每個節點的 SLURM_NODEID 各不相同（0, 1, 2, …），
    確保各節點寫入不同的 JSONL 檔案，避免並發寫入衝突。
    """
    slurm_node = os.environ.get("SLURM_NODEID")
    if slurm_node is not None:
        return f"node{slurm_node}"
    return "node0"


class RateLimiter:
    def __init__(self, calls_per_second):
        self.no_limit = calls_per_second == -1
        self.interval = 1.0 / calls_per_second if not self.no_limit else 0
        self.last_call_time = 0

    def wait(self):
        if self.no_limit:
            return
        current_time = time.time()
        time_to_wait = self.interval - (current_time - self.last_call_time)
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        self.last_call_time = time.time()


class Evaluator:
    def __init__(self, llm: LLM, evaluation_strategy: EvaluationStrategy, config: dict):
        self.llm = llm
        self.evaluation_strategy = evaluation_strategy
        self.config = config
        self.rate_limiter = RateLimiter(calls_per_second=self.config["llm_api"]["api_rate_limit"])

    def shuffle_question_options(self, question_data):
        # 動態偵測所有大寫字母選項鍵（支援超過 4 個選項的資料集）
        # 偵測所有純大寫字母鍵作為選項（支援超過 4 個選項及雙字母鍵如 AA、AB）
        # 排序規則：先依長度（單字母 < 雙字母），再依字母順序，與 _choices_to_letter_keys 一致
        option_keys = sorted(
            [
                k
                for k in question_data
                if k and all(c in string.ascii_uppercase for c in k) and k not in ("answer",)
            ],
            key=lambda k: (len(k), k),
        )
        options = [(k, question_data[k]) for k in option_keys]

        if not options:
            return question_data

        correct_ans = question_data["answer"]
        correct_option_text = question_data.get(correct_ans)

        if correct_option_text is None:
            # answer 鍵不存在或對應值為 None，無法安全 shuffle，回傳原始資料
            log_error(
                f"shuffle_question_options: answer key '{correct_ans}' not found in options; skipping shuffle"
            )
            return question_data

        random.shuffle(options)

        # 以原始標籤清單作為新標籤（保持相同數量的選項）
        new_data = {
            k: v for k, v in question_data.items() if k not in option_keys and k != "answer"
        }
        for (_, text), new_key in zip(options, option_keys):
            new_data[new_key] = text
            if text == correct_option_text:
                new_data["answer"] = new_key

        return new_data

    def evaluate_file(self, file_path: str, timestamp: str, prompt_lang: str = "zh"):
        world_size = self.config.get("distributed", {}).get("world_size", 1)
        rank = self.config.get("distributed", {}).get("rank", 0)
        
        node_id = None
        if world_size > 1:
            slurm_node = os.environ.get("SLURM_NODEID")
            node_id = slurm_node if slurm_node is not None else "0"

        dataset = Dataset(file_path, node_id=node_id, rank=rank if world_size > 1 else None)
        shuffle_enabled = self.config["evaluation"].get("shuffle_options", False)

        start_idx = 0

        # 根據 WORLD_SIZE 和 RANK 進行資料切分 (Sharding)
        if world_size > 1:
            total_size = len(dataset.data)
            chunk_size = (total_size + world_size - 1) // world_size
            start_idx = rank * chunk_size
            end_idx = min(start_idx + chunk_size, total_size)
            if start_idx >= total_size:
                dataset.data = []
            else:
                dataset.data = dataset.data[start_idx:end_idx]

        total_correct = 0
        total_questions = 0
        detailed_results = []

        with ThreadPoolExecutor() as executor:
            future_tasks = []
            future_to_data = {}

            for idx, q in enumerate(tqdm(dataset, desc="處理題庫中"), start=start_idx):
                if shuffle_enabled:
                    q = self.shuffle_question_options(q)

                question_text = (
                    q["question"]
                    + "\n"
                    + "\n".join(
                        [f"{k}: {v}" for k, v in q.items() if k not in ["question", "answer"]]
                    )
                )

                try:
                    correct_answer = q["answer"].strip().upper()
                except (KeyError, AttributeError) as e:
                    log_error(f"\n Error processing question {idx + 1}: {str(e)}")
                    continue

                self.rate_limiter.wait()
                future = executor.submit(self.llm.call, question_text, prompt_lang)
                future_tasks.append(future)
                future_to_data[future] = (question_text, correct_answer, idx)

            for future in tqdm(
                as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"
            ):
                llm_chat_completion = future.result()

                message = llm_chat_completion.choices[0].message
                usage = llm_chat_completion.usage
                content = message.content
                reasoning_content = getattr(message, "reasoning_content", None)

                # 若設定 thinking_end_tag，則從回應中拆分思考過程與最終答案（相容開頭標籤被截斷的情況）。
                _thinking_start_tag = self.config["model"].get("thinking_start_tag")
                _thinking_end_tag = self.config["model"].get("thinking_end_tag")
                if (
                    reasoning_content is None
                    and _thinking_end_tag
                    and content
                    and _thinking_end_tag in content
                ):
                    _start = re.escape(_thinking_start_tag) if _thinking_start_tag else None
                    _end = re.escape(_thinking_end_tag)
                    _pattern = rf"{_start}(.*?){_end}|^(.*?){_end}" if _start else rf"^(.*?){_end}"
                    _think_match = re.search(_pattern, content, re.DOTALL)
                    if _think_match:
                        reasoning_content = next(
                            (g for g in _think_match.groups() if g is not None), ""
                        ).strip()
                        _strip_pattern = rf"({_start})?.*?{_end}" if _start else rf".*?{_end}"
                        content = re.sub(
                            _strip_pattern, "", content, count=1, flags=re.DOTALL
                        ).strip()

                question_text, correct_answer, question_id = future_to_data[future]
                predicted_answer = self.evaluation_strategy.extract_answer(content)

                is_correct = (
                    False
                    if predicted_answer is None
                    else predicted_answer.strip().upper() == correct_answer
                )
                if is_correct:
                    total_correct += 1
                total_questions += 1

                detailed_results.append(
                    {
                        "question_id": question_id,
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

            accuracy = total_correct / total_questions if total_questions else 0

        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        # 每個節點寫入獨立的 JSONL 檔案，避免多節點並發寫入同一檔案造成資料交錯。
        if world_size > 1:
            node_id = _get_node_id()
            results_path = os.path.join(results_dir, f"eval_results_{timestamp}_{node_id}_rank{rank}.jsonl")
        else:
            results_path = os.path.join(results_dir, f"eval_results_{timestamp}.jsonl")

        # append 模式：同一節點、同一 run 內的多個檔案依序累積，不覆蓋
        with open(results_path, "a", encoding="utf-8") as f:
            for detail in detailed_results:
                f.write(json.dumps(detail, ensure_ascii=False) + "\n")

        if world_size > 1:
            display_node_id = node_id.replace("node", "")
            print(f"[節點 {display_node_id} | Rank {rank}] ✅ 評測完成，結果已儲存至 {results_path}")
        else:
            print(f"✅ 評測完成，結果已儲存至 {results_path}")
            
        return file_path, accuracy, results_path
