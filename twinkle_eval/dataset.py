"""資料集載入和處理模組

支援多種檔案格式的資料集載入，包括 JSON、JSONL、Parquet、Arrow、CSV 和 TSV
也支援從 HuggingFace Hub 下載資料集
"""

import json
import os
import string as _string
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import pyarrow as pa
from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset
from tqdm import tqdm

from .logger import log_error, log_info, log_warning


def _choices_to_letter_keys(choices: list) -> list:
    """為 choices 清單產生對應的大寫字母標籤（A、B、C … Z、AA、AB …）。"""
    labels = []
    for i in range(len(choices)):
        if i < 26:
            labels.append(_string.ascii_uppercase[i])
        else:
            # 超過 26 個選項時使用雙字母，例如 AA、AB…
            labels.append(_string.ascii_uppercase[(i // 26) - 1] + _string.ascii_uppercase[i % 26])
    return labels


def _normalize_record(record: dict) -> dict:
    """將 choices-list 格式正規化為具名字母鍵格式。

    支援兩種輸入格式：

    1. 整數索引 answer（MMLU HuggingFace 格式）：
       {"question": "...", "choices": ["opt0", "opt1", ...], "answer": 1}
       → {"question": "...", "A": "opt0", "B": "opt1", ..., "answer": "B"}

    2. 字母 answer（choices 仍為 list 但 answer 已是字母）：
       {"question": "...", "choices": ["opt0", "opt1", ...], "answer": "B"}
       → {"question": "...", "A": "opt0", "B": "opt1", ..., "answer": "B"}

    若 record 已是具名字母鍵格式（A/B/C/… 分開的欄位）則直接回傳（向下相容）。
    支援超過 4 個選項的資料集。
    """
    choices = record.get("choices")
    answer = record.get("answer")

    # Accept list or any array-like sequence (e.g. numpy.ndarray from parquet)
    if not (
        hasattr(choices, "__iter__")
        and not isinstance(choices, (str, bytes, dict))
        and len(choices) >= 2
    ):
        return record

    # Ensure it's a plain Python list for consistent handling
    choices = list(choices)

    labels = _choices_to_letter_keys(choices)

    # 嘗試將 answer 解析成整數索引
    try:
        idx = int(answer)
    except (TypeError, ValueError):
        idx = None

    if idx is not None:
        # answer 是整數索引（例如 MMLU 的 0/1/2/3）
        if not (0 <= idx < len(choices)):
            return record  # 索引超出範圍，保持原始格式
        answer_letter = labels[idx]
    else:
        # answer 已是字母字串（例如 "B"）
        answer_str = str(answer).strip().upper()
        if answer_str not in labels:
            return record  # 無法對應到任何標籤，保持原始格式
        answer_letter = answer_str

    # 建立正規化後的 record
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

    def __init__(self, file_path: str):
        """初始化資料集

        Args:
            file_path: 資料集檔案路徑
        """
        self.file_path = file_path
        self.data = self._load_data()

    def _load_data(self):
        ext = os.path.splitext(self.file_path)[-1].lower()
        print(f"正在讀取: {self.file_path}")
        try:
            if ext == ".json":  # [{}...]
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            elif ext == ".jsonl":
                with open(self.file_path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]
            elif ext in [".parquet", ".arrow"]:
                if ext == ".parquet":
                    df = pd.read_parquet(self.file_path)
                else:  # .arrow
                    table = pa.ipc.open_file(self.file_path).read_all()
                    df = table.to_pandas()
                # 驗證必要欄位
                if "question" not in df.columns:
                    raise ValueError(f"資料格式錯誤，檔案 `{self.file_path}` 缺少 `question` 欄位")
                if "answer" not in df.columns:
                    raise ValueError(f"資料格式錯誤，檔案 `{self.file_path}` 缺少 `answer` 欄位")
                # 處理 answer 欄位
                df["answer"] = df["answer"].astype(str).str.strip().str.upper()
                data = df.to_dict(orient="records")
            elif ext in [".csv", ".tsv"]:
                sep = "," if ext == ".csv" else "\t"
                df = pd.read_csv(self.file_path, sep=sep)
                # 驗證必要欄位
                if "question" not in df.columns:
                    raise ValueError(f"資料格式錯誤，檔案 `{self.file_path}` 缺少 `question` 欄位")
                if "answer" not in df.columns:
                    raise ValueError(f"資料格式錯誤，檔案 `{self.file_path}` 缺少 `answer` 欄位")
                # 處理 answer 欄位
                df["answer"] = df["answer"].astype(str).str.strip().str.upper()
                data = df.to_dict(orient="records")
            else:
                raise ValueError(f"不支援的檔案格式: {ext}")

            log_info(f"成功讀取檔案: {self.file_path}，共 {len(data)} 題")
            data = [_normalize_record(r) for r in data]
            return data
        except Exception as e:
            log_error(f"讀取資料錯誤: {e}")
            raise e

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


def find_all_evaluation_files(dataset_root: str) -> list:
    """在指定目錄中遞迴搜尋所有支援的評測檔案

    支援的檔案格式包括：.json, .jsonl, .parquet, .arrow, .csv, .tsv
    會自動忽略以點開頭的隱藏目錄

    Args:
        dataset_root: 資料集根目錄路徑

    Returns:
        list: 找到的所有評測檔案路徑列表

    Raises:
        FileNotFoundError: 當指定目錄中找不到任何支援的檔案時
    """
    supported_extensions = {
        ".json",
        ".jsonl",
        ".parquet",
        ".arrow",
        ".csv",
        ".tsv",
    }  # 支援的檔案副檔名
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
    """從 HuggingFace Hub 下載資料集

    Args:
        dataset_name: HuggingFace 資料集名稱 (例如: "cais/mmlu")
        subset: 資料集子集名稱 (可選，如果為 None 則下載所有子集)
        split: 要下載的資料集分割 (預設: "test")
        output_dir: 輸出目錄 (預設: "datasets")

    Returns:
        str: 下載後的目錄路徑

    Raises:
        ImportError: 未安裝 datasets 套件
        Exception: 下載失敗
    """
    # 建立輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 如果沒有指定子集，下載所有子集
    if subset is None:
        log_info(f"開始下載 HuggingFace 資料集的所有子集: {dataset_name}")
        try:
            configs = get_dataset_config_names(dataset_name)
            downloaded_count = 0

            # 使用進度條顯示下載進度
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
        # 下載指定子集
        log_info(f"開始下載 HuggingFace 資料集: {dataset_name}, 子集: {subset}")
        _download_single_subset(dataset_name, subset, split, output_dir)
        return output_dir


def _download_single_subset(
    dataset_name: str,
    subset: str,
    split: str,
    output_dir: Optional[str] = None,
) -> None:
    """下載單一子集的輔助函數，使用 HuggingFace 原始快取格式"""
    try:
        # 直接下載資料集到 HuggingFace 快取目錄
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
    """獲取 HuggingFace 資料集資訊

    Args:
        dataset_name: HuggingFace 資料集名稱
        subset: 資料集子集名稱 (可選)

    Returns:
        dict: 資料集資訊，包含可用的分割、特徵等

    Raises:
        ImportError: 未安裝 datasets 套件
        Exception: 獲取資訊失敗
    """
    try:
        # 獲取資料集配置
        configs = get_dataset_config_names(dataset_name)

        info = {
            "dataset_name": dataset_name,
            "configs": configs,
            "splits": {},
        }

        # 如果指定了子集，只獲取該子集的資訊
        if subset:
            if subset in configs:
                splits = get_dataset_split_names(dataset_name, config_name=subset)
                info["splits"][subset] = splits
            else:
                log_warning(f"子集 '{subset}' 不存在於資料集 '{dataset_name}' 中")
        else:
            # 獲取所有配置的分割資訊
            for config in configs[:5]:  # 限制前5個配置以避免過多請求
                try:
                    splits = get_dataset_split_names(dataset_name, config_name=config)
                    info["splits"][config] = splits
                except Exception as e:
                    log_warning(f"無法獲取配置 '{config}' 的分割資訊: {e}")

        return info

    except Exception as e:
        log_error(f"獲取 HuggingFace 資料集資訊失敗: {e}")
        raise e
