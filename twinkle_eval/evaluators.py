import json
import os
import random
import time
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

    def evaluate_file(self, file_path: str, timestamp: str, prompt_lang: str = "zh"):
        dataset = Dataset(file_path)
        shuffle_enabled = self.config["evaluation"].get("shuffle_options", False)

        total_correct = 0
        total_questions = 0
        detailed_results = []

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
                reasoning_content = message.reasoning_content

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
                        "llm_resoning_output": reasoning_content,
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
        results_path = os.path.join(results_dir, f"eval_results_{timestamp}.json")

        result_data = {
            "timestamp": timestamp,
            "file": file_path,
            "accuracy": accuracy,
            "details": detailed_results,
        }

        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result_data, ensure_ascii=False) + "\n")

        print(f"✅ 評測完成，結果已儲存至 {results_path}")
        return file_path, accuracy, results_path
