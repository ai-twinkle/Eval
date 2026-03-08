import json
import os
import random
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from .dataset import Dataset
from .evaluation_strategies import EvaluationStrategy
from .logger import log_error
from .models import LLM


class RateLimiter:
    def __init__(self, calls_per_second):
        self.no_limit = calls_per_second == -1
        self.interval = 1.0 / calls_per_second if not self.no_limit else 0
        self.last_call_time = 0
        self.lock = threading.Lock()

    def wait(self):
        if self.no_limit:
            return
        
        with self.lock:
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

    def _rate_limited_call(self, question_text: str, prompt_lang: str):
        """Wrapper for LLM call with rate limiting inside the thread"""
        self.rate_limiter.wait()
        return self.llm.call(question_text, prompt_lang)

    def shuffle_question_options(self, question_data):
        options = []
        for key in ["A", "B", "C", "D"]:
            if key in question_data:
                options.append((key, question_data[key]))

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

    def evaluate_file(self, file_path: str, timestamp: str, prompt_lang: str = "zh", run_index: int = 0):
        dataset = Dataset(file_path)
        shuffle_enabled = self.config["evaluation"].get("shuffle_options", False)

        # Get metadata for JSONL output
        model_name = self.config["model"]["name"]
        model_version = self.config["model"].get("version", "N/A")
        model_endpoint = self.config["llm_api"]["base_url"]
        model_temperature = self.config["model"]["temperature"]
        model_top_p = self.config["model"]["top_p"]
        eval_run_id = self.config["evaluation"].get("run_id", "N/A")
        eval_creator = self.config["evaluation"]["creator"]
        eval_datetime = datetime.now().isoformat()

        total_correct = 0
        total_questions = 0
        detailed_results = []

        # Initialize results file path
        results_dir = os.path.join("results", f"eval_results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        file_name = os.path.basename(file_path)
        base_name, _ = os.path.splitext(file_name)
        results_path = os.path.join(results_dir, f"{base_name}.jsonl")
        
        # Ensure file exists and is empty
        with open(results_path, "w", encoding="utf-8") as f:
            pass

        with ThreadPoolExecutor() as executor:
            future_tasks = []
            future_to_data = {}

            for idx, q in enumerate(tqdm(dataset, desc="處理題庫中")):
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

                # self.rate_limiter.wait()  # Moved to inside the thread
                future = executor.submit(self._rate_limited_call, question_text, prompt_lang)
                future_tasks.append(future)
                future_to_data[future] = (question_text, correct_answer, idx)


            # Open file in append mode for incremental writing
            with open(results_path, "a", encoding="utf-8") as f_out:
                for future in tqdm(
                    as_completed(future_tasks), total=len(future_tasks), desc="處理回應中"
                ):
                    llm_chat_completion = future.result()

                    message = llm_chat_completion.choices[0].message
                    usage = llm_chat_completion.usage
                    content = message.content
                    reasoning_content = getattr(message, "reasoning_content", None)

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

                    result_detail = {
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
                        # Metadata fields
                        "model_name": model_name,
                        "model_version": model_version,
                        "model_endpoint": model_endpoint,
                        "temperature": model_temperature,
                        "top_p": model_top_p,
                        "eval_run_id": eval_run_id,
                        "creator": eval_creator,
                        "run_index": run_index,
                        "eval_datetime": eval_datetime,
                        "dataset_source": file_path,
                    }
                    
                    detailed_results.append(result_detail)
                    
                    # Write result immediately
                    f_out.write(json.dumps(result_detail, ensure_ascii=False) + "\n")
                    f_out.flush()

            accuracy = total_correct / total_questions if total_questions else 0

        print(f"✅ 評測完成，結果已儲存至 {results_path}")
        return file_path, accuracy, results_path
