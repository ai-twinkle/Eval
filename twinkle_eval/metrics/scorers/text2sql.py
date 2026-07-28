"""Text-to-SQL Scorer。

支援兩種評分模式：
- Exact Match (EM)：正規化 SQL 後比對字串
- Execution Accuracy (EX)：對 SQLite 資料庫執行 predicted 與 gold SQL，比較結果集

config 中可設定：
- ``text2sql_scoring_mode``：``"exec"``（預設）或 ``"em"``
- ``text2sql_db_base_path``：SQLite 資料庫所在的根目錄
  （EX 模式必填，資料庫路徑為 ``{db_base_path}/{db_id}/{db_id}.sqlite``）
- ``text2sql_timeout``：SQL 執行逾時秒數（預設 30）
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from twinkle_eval.core.abc import Scorer


def normalize_sql(sql: str) -> str:
    """正規化 SQL 以便字串比對。

    - 轉小寫
    - 移除多餘空白
    - 移除結尾分號
    - 移除別名中的 AS 關鍵字（``t1.col AS alias`` → ``t1.col alias``）不做，
      因為會破壞語義
    """
    sql = sql.strip().lower()
    sql = sql.rstrip(";").strip()
    sql = re.sub(r"\s+", " ", sql)
    return sql


def _parse_gold(gold: Any) -> Tuple[str, Optional[str]]:
    """解析 gold answer，回傳 (sql, db_id)。

    gold 可能是：
    - JSON 字串 ``{"sql": "...", "db_id": "..."}``
    - 純 SQL 字串
    """
    if gold is None:
        return "", None

    text = str(gold).strip()

    # 嘗試 JSON 解析
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed.get("sql", ""), parsed.get("db_id")
    except (json.JSONDecodeError, TypeError):
        pass

    return text, None


def execute_sql(db_path: str, sql: str, timeout: int = 30) -> Optional[List[Tuple[Any, ...]]]:
    """對 SQLite 資料庫執行 SQL，回傳結果集。

    Returns:
        結果列表（每列為一個 tuple），或 None（執行失敗時）。
    """
    if not os.path.isfile(db_path):
        return None
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
        conn.execute("PRAGMA query_only = ON")
        # 以 progress handler 實作查詢逾時：超過 deadline 即中止查詢
        # （sqlite3.connect 的 timeout 參數只處理鎖競爭，管不到長時間查詢）
        deadline = time.monotonic() + timeout
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 100_000)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        return results
    except sqlite3.Error:
        # SQL 執行失敗（語法錯誤、逾時中止等）視為無結果，由呼叫端評為不正確
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass


def result_sets_match(
    results_a: Optional[List[Tuple[Any, ...]]],
    results_b: Optional[List[Tuple[Any, ...]]],
) -> bool:
    """比較兩個結果集是否等價（忽略行順序）。"""
    if results_a is None or results_b is None:
        return False

    if len(results_a) != len(results_b):
        return False

    def normalize_row(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """正規化每個值以便比較。"""
        normalized = []
        for val in row:
            if isinstance(val, float):
                normalized.append(round(val, 6))
            elif isinstance(val, str):
                normalized.append(val.strip().lower())
            elif val is None:
                normalized.append(None)
            else:
                normalized.append(val)
        return tuple(normalized)

    set_a = sorted([normalize_row(r) for r in results_a], key=str)
    set_b = sorted([normalize_row(r) for r in results_b], key=str)
    return set_a == set_b


class Text2SQLScorer(Scorer):
    """Text-to-SQL Scorer — 支援 Exact Match 與 Execution Accuracy。

    config 可設定：
    - ``text2sql_scoring_mode``：``"exec"``（預設）或 ``"em"``
    - ``text2sql_db_base_path``：資料庫根目錄（EX 模式必填）
    - ``text2sql_timeout``：SQL 執行逾時秒數（預設 30）
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        cfg = config or {}
        self.scoring_mode: str = cfg.get("text2sql_scoring_mode", "exec")
        self.db_base_path: str = cfg.get("text2sql_db_base_path", "")
        self.timeout: int = cfg.get("text2sql_timeout", 30)

    def get_name(self) -> str:
        return "text2sql"

    def normalize(self, answer: Any) -> Any:
        """Gold answer 為 JSON metadata（含 db_id），不做正規化。"""
        return answer

    def score(self, predicted: Any, gold: Any) -> bool:
        """評分。

        Args:
            predicted: Text2SQLExtractor 回傳的 SQL 字串，或 None。
            gold: ground truth — JSON 字串 ``{"sql": "...", "db_id": "..."}``
                  或純 SQL 字串。

        Returns:
            True 若 predicted SQL 與 gold SQL 匹配。
        """
        if not predicted:
            return False

        gold_sql, db_id = _parse_gold(gold)
        if not gold_sql:
            return False

        pred_str = str(predicted)

        if self.scoring_mode == "em":
            return normalize_sql(pred_str) == normalize_sql(gold_sql)

        # Execution Accuracy mode
        if not db_id or not self.db_base_path:
            # Fallback to EM if no database info available
            return normalize_sql(pred_str) == normalize_sql(gold_sql)

        db_path = os.path.join(self.db_base_path, db_id, f"{db_id}.sqlite")
        if not os.path.isfile(db_path):
            # Fallback to EM
            return normalize_sql(pred_str) == normalize_sql(gold_sql)

        gold_results = execute_sql(db_path, gold_sql, self.timeout)
        pred_results = execute_sql(db_path, pred_str, self.timeout)

        if gold_results is None:
            # Gold SQL failed — fallback to EM
            return normalize_sql(pred_str) == normalize_sql(gold_sql)

        return result_sets_match(pred_results, gold_results)
