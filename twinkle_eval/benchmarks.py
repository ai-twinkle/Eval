"""Benchmark registry and unified dataset download utilities.

提供所有支援的評測資料集的集中管理，支援從 HuggingFace Hub 和 GitHub 下載。
"""

import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.logger import log_error, log_info, log_warning


# ---------------------------------------------------------------------------
# Benchmark Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ── 選擇題 ────────────────────────────────────────────────────────────
    "mmlu": {
        "source": "huggingface",
        "hf_id": "cais/mmlu",
        "split": "test",
        "description": "MMLU — 57 科目大規模多任務語言理解",
        "eval_method": "box",
        "license": "MIT",
    },
    "mmlu_pro": {
        "source": "huggingface",
        "hf_id": "TIGER-Lab/MMLU-Pro",
        "split": "test",
        "description": "MMLU-Pro — 更具鑑別力的多任務理解（10 選項）",
        "eval_method": "box",
        "license": "MIT",
    },
    "tmmluplus": {
        "source": "huggingface",
        "hf_id": "ikala/tmmluplus",
        "split": "test",
        "description": "TMMLU+ — 繁體中文多任務語言理解",
        "eval_method": "box",
        "license": "Apache-2.0",
    },
    "mmlu_redux": {
        "source": "huggingface",
        "hf_id": "edinburgh-dawg/mmlu-redux",
        "split": "test",
        "description": "MMLU-Redux — MMLU 修正版（3000 題人工重新標注）",
        "eval_method": "box",
        "license": "MIT",
    },
    "supergpqa": {
        "source": "huggingface",
        "hf_id": "m-a-p/SuperGPQA",
        "split": "test",
        "description": "SuperGPQA — 研究生等級跨領域問答",
        "eval_method": "box",
        "license": "Apache-2.0",
    },
    "gpqa": {
        "source": "huggingface",
        "hf_id": "Idavidrein/gpqa",
        "split": "train",
        "gated": True,
        "description": "GPQA — Google 研究等級科學問答（需申請存取權限）",
        "eval_method": "box",
        "license": "CC-BY-4.0",
    },
    "formosa_bench": {
        "source": "huggingface",
        "hf_id": "lianghsun/Formosa-bench",
        "split": "train",
        "description": "Formosa-bench — 台灣在地化多科目評測",
        "eval_method": "box",
        "license": "Apache-2.0",
    },
    # ── 數學 ──────────────────────────────────────────────────────────────
    "gsm8k": {
        "source": "huggingface",
        "hf_id": "openai/gsm8k",
        "split": "test",
        "subset": "main",
        "description": "GSM8K — 小學數學推理（8.5K 題）",
        "eval_method": "math",
        "license": "MIT",
    },
    "aime2025": {
        "source": "huggingface",
        "hf_id": "MathArena/aime_2025",
        "split": "train",
        "description": "AIME 2025 — 美國數學邀請賽（高難度）",
        "eval_method": "math",
        "license": "MIT",
    },
    # ── Regex Match ───────────────────────────────────────────────────────
    "bbh": {
        "source": "huggingface",
        "hf_id": "lukaemon/bbh",
        "split": "test",
        "description": "BIG-Bench Hard — 27 個高難度推理子任務",
        "eval_method": "regex_match",
        "license": "Apache-2.0",
    },
    # ── 指令遵循 ──────────────────────────────────────────────────────────
    "ifeval": {
        "source": "huggingface",
        "hf_id": "google/IFEval",
        "split": "train",
        "description": "IFEval — Google 25 類指令遵循評測",
        "eval_method": "ifeval",
        "license": "Apache-2.0",
    },
    "ifbench": {
        "source": "huggingface",
        "hf_id": "Yale-LILY/IFBench",
        "split": "test",
        "description": "IFBench — 58 類 OOD 指令遵循評測",
        "eval_method": "ifbench",
        "license": "MIT",
    },
    # ── 函式呼叫 ──────────────────────────────────────────────────────────
    "bfcl": {
        "source": "huggingface",
        "hf_id": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "split": "train",
        "description": "BFCL — Berkeley 函式呼叫排行榜",
        "eval_method": "bfcl_fc",
        "license": "Apache-2.0",
    },
    # ── 長文本 / NIAH ─────────────────────────────────────────────────────
    "needlebench": {
        "source": "huggingface",
        "hf_id": "opencompass/NeedleBench",
        "split": "test",
        "description": "NeedleBench — 多語言大海撈針測試",
        "eval_method": "niah",
        "license": "Apache-2.0",
    },
    "longbench": {
        "source": "github",
        "url": "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip",
        "description": "LongBench — 中文段落檢索長文本理解",
        "eval_method": "niah",
        "license": "MIT",
        "post_process": "longbench",
    },
    # ── RAG ────────────────────────────────────────────────────────────────
    "wikieval": {
        "source": "huggingface",
        "hf_id": "explodinggradients/WikiEval",
        "split": "train",
        "description": "WikiEval — RAG 品質評估（RAGAS 框架）",
        "eval_method": "ragas",
        "license": "Apache-2.0",
    },
    # ── ASR（語音辨識）────────────────────────────────────────────────────
    "librispeech": {
        "source": "huggingface",
        "hf_id": "openslr/librispeech_asr",
        "split": "test.clean",
        "description": "LibriSpeech test-clean — 英文朗讀語音辨識",
        "eval_method": "asr",
        "license": "CC-BY-4.0",
    },
    "aishell1": {
        "source": "huggingface",
        "hf_id": "carlot/AIShell",
        "split": "test",
        "description": "Aishell-1 — 中文普通話語音辨識（178 小時）",
        "eval_method": "asr",
        "license": "Apache-2.0",
    },
    "fleurs": {
        "source": "huggingface",
        "hf_id": "google/fleurs",
        "split": "test",
        "description": "Fleurs — 102 語言多語言語音辨識",
        "eval_method": "asr",
        "license": "CC-BY-4.0",
    },
    "common_voice": {
        "source": "huggingface",
        "hf_id": "mozilla-foundation/common_voice_17_0",
        "split": "test",
        "description": "Common Voice 17.0 — 群眾外包多語言語音辨識",
        "eval_method": "asr",
        "license": "CC0-1.0",
    },
    # ── Text-to-SQL ───────────────────────────────────────────────────────
    "spider": {
        "source": "huggingface",
        "hf_id": "xlangai/spider",
        "split": "validation",
        "description": "Spider 1.0 — 跨資料庫 Text-to-SQL",
        "eval_method": "text2sql",
        "license": "Apache-2.0",
    },
    "bird": {
        "source": "github",
        "url": "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip",
        "description": "BIRD — 大規模跨資料庫 Text-to-SQL（含 95 個資料庫）",
        "eval_method": "text2sql",
        "license": "CC-BY-SA-4.0",
        "post_process": "bird",
    },
    "spider2_lite": {
        "source": "github",
        "url": "https://github.com/xlang-ai/Spider2/archive/refs/heads/main.zip",
        "description": "Spider 2.0-lite — 85 題 SQLite Text-to-SQL",
        "eval_method": "text2sql",
        "license": "Apache-2.0",
        "post_process": "spider2_lite",
    },
}


def get_available_benchmarks() -> List[str]:
    """回傳所有可用 benchmark 名稱（依字母排序）。"""
    return sorted(BENCHMARK_REGISTRY.keys())


def list_benchmarks() -> None:
    """列出所有可用 benchmark 及其說明。"""
    print("📋 可下載的評測資料集：")
    print()

    # 依 eval_method 分群
    groups: Dict[str, list] = {}
    for name, info in BENCHMARK_REGISTRY.items():
        method = info.get("eval_method", "other")
        groups.setdefault(method, []).append((name, info))

    for method, items in groups.items():
        print(f"  [{method}]")
        for name, info in sorted(items):
            source_tag = "HF" if info["source"] == "huggingface" else "GitHub"
            gated_tag = " 🔒" if info.get("gated") else ""
            print(f"    {name:<20s} {info['description']} ({source_tag}){gated_tag}")
        print()

    print("💡 使用方式：")
    print("  twinkle-eval --download-dataset mmlu gsm8k    # 下載指定資料集")
    print("  twinkle-eval --download-dataset all            # 下載全部")


# ---------------------------------------------------------------------------
# Download orchestration
# ---------------------------------------------------------------------------


def download_benchmarks(
    names: List[str],
    output_dir: str = "datasets",
) -> int:
    """下載指定的 benchmark 資料集。

    Args:
        names: 要下載的 benchmark 名稱列表，或 ["all"] 下載全部
        output_dir: 輸出根目錄

    Returns:
        int: 程式退出代碼（0 表示成功，1 表示有錯誤）
    """
    if names == ["all"]:
        names = get_available_benchmarks()

    succeeded: list[str] = []
    skipped_gated: list[str] = []
    failed: list[tuple[str, str]] = []

    for name in names:
        info = BENCHMARK_REGISTRY.get(name)
        if info is None:
            failed.append((name, "不在 registry 中"))
            continue

        print(f"\n{'=' * 60}")
        print(f"📥 下載 {name}: {info['description']}")
        print(f"{'=' * 60}")

        try:
            if info["source"] == "huggingface":
                _download_hf_benchmark(name, info, output_dir, skipped_gated)
            elif info["source"] == "github":
                _download_github_benchmark(name, info, output_dir)
            succeeded.append(name)
        except _SkipGatedError:
            skipped_gated.append(name)
        except Exception as e:
            log_error(f"下載 {name} 失敗: {e}")
            failed.append((name, str(e)))

    # ── 摘要 ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("📋 下載結果摘要")
    print(f"{'=' * 60}")
    if succeeded:
        print(f"  ✅ 下載成功：{', '.join(succeeded)}")
    if skipped_gated:
        print(f"  ⏭️  因缺少 HuggingFace token 跳過：{', '.join(skipped_gated)}")
    if failed:
        for name, reason in failed:
            print(f"  ❌ {name}：{reason}")

    return 1 if failed else 0


class _SkipGatedError(Exception):
    """用於跳過 gated dataset 的內部例外。"""


# ---------------------------------------------------------------------------
# HuggingFace download
# ---------------------------------------------------------------------------


def _download_hf_benchmark(
    name: str,
    info: Dict[str, Any],
    output_dir: str,
    skipped_gated: list[str],
) -> None:
    """下載 HuggingFace 資料集到 datasets/{name}/。"""
    from datasets import get_dataset_config_names, load_dataset

    hf_id = info["hf_id"]
    split = info.get("split", "test")
    fixed_subset = info.get("subset")
    dest = os.path.join(output_dir, name)
    os.makedirs(dest, exist_ok=True)

    token: Optional[str] = None

    # Gated dataset: prompt for token
    if info.get("gated"):
        token = _prompt_hf_token(name)
        if token is None:
            raise _SkipGatedError()

    try:
        if fixed_subset:
            # 單一 subset（如 gsm8k/main）
            _save_hf_subset(hf_id, fixed_subset, split, dest, token)
        else:
            # 嘗試列出所有 subset
            try:
                configs = get_dataset_config_names(hf_id, token=token)
            except Exception:
                configs = ["default"]

            if len(configs) <= 1:
                _save_hf_subset(hf_id, configs[0] if configs else None, split, dest, token)
            else:
                log_info(f"  找到 {len(configs)} 個子集")
                from tqdm import tqdm

                for config in tqdm(configs, desc=f"下載 {name}", unit="subset"):
                    try:
                        _save_hf_subset(hf_id, config, split, dest, token)
                    except Exception as e:
                        log_warning(f"  跳過子集 {config}: {e}")

    except Exception as e:
        if info.get("gated") and "401" in str(e) or "403" in str(e):
            raise _SkipGatedError()
        raise


def _save_hf_subset(
    hf_id: str,
    subset: Optional[str],
    split: str,
    dest: str,
    token: Optional[str] = None,
) -> None:
    """下載並儲存單一 HuggingFace subset 為 parquet。"""
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": split, "trust_remote_code": False}
    if subset and subset != "default":
        kwargs["name"] = subset
    if token:
        kwargs["token"] = token

    ds = load_dataset(hf_id, **kwargs)
    filename = f"{subset}.parquet" if subset and subset != "default" else "data.parquet"
    ds.to_parquet(os.path.join(dest, filename))


def _prompt_hf_token(name: str) -> Optional[str]:
    """提示使用者輸入 HuggingFace token。回傳 None 表示跳過。"""
    print(f"\n⚠️  {name} 是受限資料集（gated dataset），需要 HuggingFace 存取權限")
    print("   請先至 HuggingFace 申請存取後，輸入你的 access token")
    try:
        token = input("   HuggingFace token（按 Enter 跳過）: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None

    if not token:
        print(f"   ⏭️  跳過 {name}")
        return None
    return token


# ---------------------------------------------------------------------------
# GitHub-based download
# ---------------------------------------------------------------------------


def _download_github_benchmark(
    name: str,
    info: Dict[str, Any],
    output_dir: str,
) -> None:
    """下載 GitHub-based 資料集。"""
    post_process = info.get("post_process", name)
    dest = os.path.join(output_dir, name)
    os.makedirs(dest, exist_ok=True)

    if post_process == "bird":
        _download_bird(info["url"], dest)
    elif post_process == "spider2_lite":
        _download_spider2_lite(info["url"], dest)
    elif post_process == "longbench":
        _download_longbench(info["url"], dest)
    else:
        raise ValueError(f"未知的後處理類型: {post_process}")


def _download_bird(url: str, dest: str) -> None:
    """下載 BIRD 資料集（dev set + databases）。"""
    import httpx

    log_info("下載 BIRD dev.zip ...")
    response = httpx.get(url, follow_redirects=True, timeout=300)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)

            # 找到 dev 目錄
            dev_dir = _find_subdir(tmp, "dev")
            if dev_dir is None:
                # 可能直接解壓到 tmp
                dev_dir = tmp

            # 1. 轉換 dev.json → JSONL
            dev_json = _find_file(dev_dir, "dev.json")
            if dev_json:
                _bird_json_to_jsonl(dev_json, os.path.join(dest, "bird_dev.jsonl"))

            # 2. 複製 tables.json
            tables = _find_file(dev_dir, "dev_tables.json")
            if tables is None:
                tables = _find_file(dev_dir, "tables.json")
            if tables:
                shutil.copy2(tables, os.path.join(dest, "tables.json"))

            # 3. 複製 databases
            db_src = _find_subdir(dev_dir, "dev_databases")
            if db_src is None:
                db_src = _find_subdir(dev_dir, "databases")
            if db_src:
                db_dest = os.path.join(dest, "databases")
                if os.path.exists(db_dest):
                    shutil.rmtree(db_dest)
                shutil.copytree(db_src, db_dest)

    _report_download(dest)


def _bird_json_to_jsonl(json_path: str, jsonl_path: str) -> None:
    """將 BIRD dev.json 轉換為本專案的 JSONL 格式。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            record = {
                "id": item.get("question_id", item.get("id", "")),
                "question": item.get("question", ""),
                "answer": json.dumps(
                    {"sql": item.get("SQL", ""), "db_id": item.get("db_id", "")},
                    ensure_ascii=False,
                ),
                "db_id": item.get("db_id", ""),
                "evidence": item.get("evidence", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    log_info(f"  轉換完成：{len(data)} 題 → {jsonl_path}")


def _download_spider2_lite(url: str, dest: str) -> None:
    """下載 Spider 2.0-lite（SQLite-only subset）。"""
    import httpx

    log_info("下載 Spider2 repo archive ...")
    response = httpx.get(url, follow_redirects=True, timeout=300)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        with tempfile.TemporaryDirectory() as tmp:
            zf.extractall(tmp)

            # 找到 spider2-lite 目錄
            lite_dir = _find_subdir_recursive(tmp, "spider2-lite")
            if lite_dir is None:
                raise FileNotFoundError("在 Spider2 repo 中找不到 spider2-lite 目錄")

            # 1. 找到 SQLite 題目 JSON
            examples_dir = _find_subdir(lite_dir, "baselines") or lite_dir
            spider2_json = _find_file(lite_dir, "spider2lite.json")
            if spider2_json is None:
                spider2_json = _find_file(lite_dir, "spider2-lite.json")

            if spider2_json:
                _spider2_json_to_jsonl(spider2_json, os.path.join(dest, "spider2_lite_dev.jsonl"))

            # 2. 複製 databases（只要 SQLite 的）
            db_src = _find_subdir_recursive(lite_dir, "local_sqlite")
            if db_src is None:
                db_src = _find_subdir_recursive(lite_dir, "databases")
            if db_src:
                db_dest = os.path.join(dest, "databases")
                if os.path.exists(db_dest):
                    shutil.rmtree(db_dest)
                shutil.copytree(db_src, db_dest)

    _report_download(dest)


def _spider2_json_to_jsonl(json_path: str, jsonl_path: str) -> None:
    """將 Spider 2.0-lite JSON 轉換為本專案的 JSONL 格式。"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in data:
            # 只保留 SQLite 題目
            if item.get("type", "").lower() not in ("sqlite", "local"):
                continue
            record = {
                "id": item.get("instance_id", item.get("id", "")),
                "question": item.get("instruction", item.get("question", "")),
                "answer": json.dumps(
                    {"sql": item.get("gold", item.get("sql", "")),
                     "db_id": item.get("db", item.get("db_id", ""))},
                    ensure_ascii=False,
                ),
                "db_id": item.get("db", item.get("db_id", "")),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    log_info(f"  轉換完成：{count} 題（SQLite only） → {jsonl_path}")


def _download_longbench(url: str, dest: str) -> None:
    """從 data.zip 下載 LongBench passage_retrieval_zh.jsonl。"""
    import httpx

    log_info("下載 LongBench data.zip（約 110 MB）...")
    response = httpx.get(url, follow_redirects=True, timeout=300)
    response.raise_for_status()

    target_name = "passage_retrieval_zh.jsonl"
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # 在 zip 中找目標檔案
        matched = [n for n in zf.namelist() if n.endswith(target_name)]
        if not matched:
            raise FileNotFoundError(f"在 data.zip 中找不到 {target_name}")

        output_path = os.path.join(dest, target_name)
        with zf.open(matched[0]) as src, open(output_path, "wb") as dst:
            dst.write(src.read())

    line_count = sum(1 for _ in open(output_path, "r", encoding="utf-8"))
    log_info(f"  完成：{line_count} 筆 → {output_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_subdir(root: str, name: str) -> Optional[str]:
    """在 root 下找第一層名稱匹配的子目錄。"""
    for entry in os.listdir(root):
        full = os.path.join(root, entry)
        if os.path.isdir(full) and entry.lower() == name.lower():
            return full
    return None


def _find_subdir_recursive(root: str, name: str) -> Optional[str]:
    """遞迴搜尋名稱匹配的子目錄。"""
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            if d.lower() == name.lower():
                return os.path.join(dirpath, d)
    return None


def _find_file(root: str, name: str) -> Optional[str]:
    """在目錄樹中找第一個名稱匹配的檔案。"""
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower() == name.lower():
                return os.path.join(dirpath, fn)
    return None


def _report_download(dest: str) -> None:
    """報告下載結果的檔案統計。"""
    total_files = sum(len(files) for _, _, files in os.walk(dest))
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(dest)
        for f in fns
    )
    size_mb = total_size / (1024 * 1024)
    log_info(f"  下載完成：{total_files} 個檔案，共 {size_mb:.1f} MB → {dest}")
