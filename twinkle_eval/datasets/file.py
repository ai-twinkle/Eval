"""資料集載入和處理模組。

支援多種檔案格式的資料集載入，包括 JSON、JSONL、Parquet、Arrow、CSV 和 TSV
也支援從 HuggingFace Hub 下載資料集
"""

import json
import os
import string
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

from twinkle_eval.logger import log_error, log_info, log_warning


def _index_to_label(idx: int) -> str:
    """將 0-based 索引轉換為 Excel 風格的大寫字母標籤（A…Z、AA…AZ、BA…）。"""
    letters = []
    while True:
        idx, rem = divmod(idx, 26)
        letters.append(string.ascii_uppercase[rem])
        if idx == 0:
            break
        idx -= 1
    return "".join(reversed(letters))


def _normalize_record(record: dict) -> dict:
    """將 choices-list 格式正規化為具名字母鍵格式。

    支援兩種輸入格式：
    1. 整數索引 answer（MMLU HuggingFace 格式）：
       {"question": "...", "choices": [...], "answer": 1}
       → {"question": "...", "A": "opt0", "B": "opt1", ..., "answer": "B"}
    2. 字母 answer（choices 仍為 list 但 answer 已是字母）：
       {"question": "...", "choices": [...], "answer": "B"}
       → {"question": "...", "A": "opt0", ..., "answer": "B"}

    若 record 已是具名字母鍵格式則直接回傳（向下相容）。
    """
    choices = record.get("choices")
    answer = record.get("answer")

    if not (
        hasattr(choices, "__iter__")
        and not isinstance(choices, (str, bytes, dict))
        and len(choices) >= 2
    ):
        return record

    choices = list(choices)
    labels = [_index_to_label(i) for i in range(len(choices))]

    try:
        idx = int(answer)
    except (TypeError, ValueError):
        idx = None

    if idx is not None:
        if not (0 <= idx < len(choices)):
            return record
        answer_letter = labels[idx]
    else:
        answer_str = str(answer).strip().upper()
        if answer_str not in labels:
            return record
        answer_letter = answer_str

    normalized = {k: v for k, v in record.items() if k not in ("choices", "answer")}
    for label, text in zip(labels, choices):
        normalized[label] = text
    normalized["answer"] = answer_letter
    return normalized


class Dataset:
    """資料集類別 - 負責載入和管理單一資料集檔案

    支援多種檔案格式：
    - JSON: 單一 JSON 物件
    - JSONL: 每行一個 JSON 物件
    - Parquet: Apache Parquet 格式
    - Arrow: Apache Arrow 格式
    - CSV/TSV: 逗號或制表符分隔的文字檔

    自動將 MMLU HuggingFace 格式（choices + 整數 answer）正規化為 A/B/C/D 格式。
    """

    def __init__(self, file_path: str, node_id: Optional[str] = None, rank: Optional[int] = None):
        """初始化資料集

        Args:
            file_path: 資料集檔案路徑
            node_id: SLURM 節點 ID（分散式評測用，可選）
            rank: 分散式 rank（可選）
        """
        self.file_path = file_path
        self.node_id = node_id
        self.rank = rank
        self.data = self._load_data()

    def _load_data(self) -> list:
        ext = os.path.splitext(self.file_path)[-1].lower()
        print(f"正在讀取: {self.file_path}")
        try:
            if ext == ".json":
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif ext == ".jsonl":
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]
            elif ext in [".parquet", ".arrow"]:
                if ext == ".parquet":
                    df = pd.read_parquet(self.file_path)
                else:
                    table = pa.ipc.open_file(self.file_path).read_all()
                    df = table.to_pandas()
                if "question" not in df.columns:
                    raise ValueError(f"資料格式錯誤，檔案 `{self.file_path}` 缺少 `question` 欄位")
                if "answer" not in df.columns:
                    raise ValueError(f"資料格式錯誤，檔案 `{self.file_path}` 缺少 `answer` 欄位")
                df["answer"] = df["answer"].astype(str).str.strip().str.upper()
                data = df.to_dict(orient="records")
            elif ext in [".csv", ".tsv"]:
                sep = "," if ext == ".csv" else "\t"
                df = pd.read_csv(self.file_path, sep=sep)
                if "question" not in df.columns:
                    raise ValueError(f"資料格式錯誤，檔案 `{self.file_path}` 缺少 `question` 欄位")
                if "answer" not in df.columns:
                    raise ValueError(f"資料格式錯誤，檔案 `{self.file_path}` 缺少 `answer` 欄位")
                df["answer"] = df["answer"].astype(str).str.strip().str.upper()
                data = df.to_dict(orient="records")
            else:
                raise ValueError(f"不支援的檔案格式: {ext}")

            data = [_normalize_record(r) for r in data]
            if self.node_id is not None and self.rank is not None:
                log_info(
                    f"[節點 {self.node_id} | Rank {self.rank}] 成功讀取: {self.file_path}，共 {len(data)} 題"
                )
            else:
                log_info(f"成功讀取檔案: {self.file_path}，共 {len(data)} 題")
            return data
        except Exception as e:
            log_error(f"讀取資料錯誤: {e}")
            raise e

    def __iter__(self):
        return iter(self.data)

    def __len__(self) -> int:
        return len(self.data)


def find_all_evaluation_files(dataset_root: str) -> list:
    """在指定目錄中遞迴搜尋所有支援的評測檔案。

    支援的檔案格式包括：.json, .jsonl, .parquet, .arrow, .csv, .tsv

    Args:
        dataset_root: 資料集根目錄路徑

    Returns:
        list: 找到的所有評測檔案路徑列表

    Raises:
        FileNotFoundError: 當指定目錄中找不到任何支援的檔案時
    """
    supported_extensions = {".json", ".jsonl", ".parquet", ".arrow", ".csv", ".tsv"}
    all_files = []

    print(f"掃描目錄： {dataset_root}")
    for root, dirs, files in os.walk(dataset_root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for file in files:
            file_path = os.path.join(root, file)
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".lock":
                continue
            if ext in supported_extensions:
                all_files.append(file_path)
            else:
                print(f"⚠️ Warning: 跳過不支援的檔案 {file_path} (副檔名: {ext})")
    if not all_files:
        raise FileNotFoundError(f"在 {dataset_root} 下未找到可讀取的評測檔案")
    log_info(f"評測集資料夾： {dataset_root}")
    log_info(f"發現 {len(all_files)} 個評測檔案")
    return all_files


def download_huggingface_dataset(
    dataset_name: str,
    subset: Optional[str] = None,
    split: str = "test",
    output_dir: str = "datasets",
) -> str:
    """從 HuggingFace Hub 下載資料集。

    Args:
        dataset_name: HuggingFace 資料集名稱 (例如: "cais/mmlu")
        subset: 資料集子集名稱 (可選，如果為 None 則下載所有子集)
        split: 要下載的資料集分割 (預設: "test")
        output_dir: 輸出目錄 (預設: "datasets")

    Returns:
        str: 下載後的目錄路徑
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if subset is None:
        log_info(f"開始下載 HuggingFace 資料集的所有子集: {dataset_name}")
        try:
            configs = get_dataset_config_names(dataset_name)
            downloaded_count = 0

            with tqdm(configs, desc="下載子集", unit="subset") as pbar:
                for config in pbar:
                    try:
                        pbar.set_postfix({"目前": config})
                        log_info(f"下載子集: {config}")
                        _download_single_subset(dataset_name, config, split, output_dir)
                        downloaded_count += 1
                    except Exception as e:
                        log_warning(f"跳過子集 {config}: {e}")
                        continue

            if downloaded_count > 0:
                log_info(f"成功下載 {downloaded_count} 個子集")
                return output_dir
            else:
                raise Exception("沒有成功下載任何子集")

        except Exception as e:
            log_error(f"下載所有子集失敗: {e}")
            raise e
    else:
        log_info(f"開始下載 HuggingFace 資料集: {dataset_name}, 子集: {subset}")
        _download_single_subset(dataset_name, subset, split, output_dir)
        return output_dir


def _download_single_subset(
    dataset_name: str,
    subset: str,
    split: str,
    output_dir: Optional[str] = None,
) -> None:
    """下載單一子集的輔助函數，使用 HuggingFace 原始快取格式。"""
    try:
        hf_dataset = load_dataset(
            dataset_name,
            name=subset,
            split=split,
            trust_remote_code=False,
        )
        hf_dataset.to_parquet(f"{output_dir}/{dataset_name.replace('/', '__')}/{subset}.parquet")
    except Exception as e:
        log_error(f"下載子集 {subset} 失敗: {e}")
        raise e


def list_huggingface_dataset_info(dataset_name: str, subset: Optional[str] = None) -> Dict:
    """獲取 HuggingFace 資料集資訊。

    Args:
        dataset_name: HuggingFace 資料集名稱
        subset: 資料集子集名稱 (可選)

    Returns:
        dict: 資料集資訊，包含可用的分割、特徵等
    """
    try:
        configs = get_dataset_config_names(dataset_name)

        info: Dict = {
            "dataset_name": dataset_name,
            "configs": configs,
            "splits": {},
        }

        if subset:
            if subset in configs:
                splits = get_dataset_split_names(dataset_name, config_name=subset)
                info["splits"][subset] = splits
            else:
                log_warning(f"子集 '{subset}' 不存在於資料集 '{dataset_name}' 中")
        else:
            for config in configs[:5]:
                try:
                    splits = get_dataset_split_names(dataset_name, config_name=config)
                    info["splits"][config] = splits
                except Exception as e:
                    log_warning(f"無法獲取配置 '{config}' 的分割資訊: {e}")

        return info

    except Exception as e:
        log_error(f"獲取 HuggingFace 資料集資訊失敗: {e}")
        raise e
