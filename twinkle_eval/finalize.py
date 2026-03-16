import glob
import json
import os
from collections import defaultdict
from typing import Optional

import numpy as np

from .results_exporters import ResultsExporterFactory


def finalize_results(timestamp: str, hf_repo_id: Optional[str] = None, hf_variant: Optional[str] = "default") -> int:
    """合併平行運算產生的碎片並重新計算評測指標，最後刪除碎片。
    若無碎片但存在單節點最終結果，則直接執行上傳。
    """

    results_dir = "results"
    json_shards = sorted(glob.glob(os.path.join(results_dir, f"results_{timestamp}_node*_rank*.json")))

    if not json_shards:
        # 單節點執行：直接上傳已存在的最終結果，無須合併
        single_node_result = os.path.join(results_dir, f"results_{timestamp}.json")
        if not os.path.exists(single_node_result):
            print(f"找不到時間戳記為 {timestamp} 的評測碎片或最終結果。")
            return 1

        print(f"未發現分散式碎片，以單節點模式直接上傳 {single_node_result}。")
        if hf_repo_id:
            try:
                from .hf_uploader import upload_results
                with open(single_node_result, "r", encoding="utf-8") as _f:
                    _result = json.load(_f)
                model_name = _result.get("config", {}).get("model", {}).get("name", "unknown_model")
                upload_results(
                    repo_id=hf_repo_id,
                    variant=hf_variant,
                    model_name=model_name,
                    results_dir=results_dir,
                    timestamp=timestamp,
                )
                print("上傳完成。")
            except Exception as e:
                print(f"上傳至 Hugging Face 失敗: {e}")
                return 1
        return 0

    print(f"找到 {len(json_shards)} 個 JSON 配置碎片。開始合併與統整數據...")

    with open(json_shards[0], "r", encoding="utf-8") as f:
        base_result = json.load(f)

    # ── 第一階段：收集元資料 ────────────────────────────────────────────────
    # run_idx -> 屬於該 run 的所有分片 JSONL 路徑列表
    run_jsonl_shards: dict[int, list[str]] = defaultdict(list)
    # ds_name -> file_path -> 該檔案包含的 run_idx 集合
    ds_file_runs: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    for shard_path in json_shards:
        with open(shard_path, "r", encoding="utf-8") as f:
            shard_data = json.load(f)

        for ds_name, ds_data in shard_data.get("dataset_results", {}).items():
            for file_res in ds_data.get("results", []):
                file_path = file_res["file"]
                jsonl_paths = file_res.get("individual_runs", {}).get("results", [])
                for run_idx, jsonl_path in enumerate(jsonl_paths):
                    ds_file_runs[ds_name][file_path].add(run_idx)
                    if jsonl_path not in run_jsonl_shards[run_idx]:
                        run_jsonl_shards[run_idx].append(jsonl_path)

    # ── 第二階段：串流合併所有分片 JSONL → 每個 run 產生一個合併後的 JSONL ─
    # run_idx -> 合併後的 JSONL 路徑
    run_merged_path: dict[int, str] = {}
    merged_jsonl_files: list[str] = []
    shard_jsonl_files: set[str] = set()  # 待清理的原始分片 JSONL

    try:
        for run_idx, shard_paths in sorted(run_jsonl_shards.items()):
            merged_name = os.path.join(results_dir, f"eval_results_{timestamp}_run{run_idx}.jsonl")
            run_merged_path[run_idx] = merged_name
            merged_jsonl_files.append(merged_name)

            # 從此 run 的每個分片蒐集所有記錄
            all_entries: list[dict] = []
            for j_path in shard_paths:
                if os.path.exists(j_path):
                    shard_jsonl_files.add(j_path)
                    with open(j_path, "r", encoding="utf-8") as jf:
                        for line in jf:
                            line = line.strip()
                            if line:
                                all_entries.append(json.loads(line))

            # 依 question_id 排序後一次寫入，避免反覆開檔
            all_entries.sort(key=lambda x: int(x.get("question_id", 0)))
            print(f"  run{run_idx}: 合併 {len(shard_paths)} 個碎片 → {len(all_entries)} 筆記錄")

            with open(merged_name, "w", encoding="utf-8") as jf:
                for entry in all_entries:
                    jf.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # ── 第三階段：掃描合併後的 JSONL，依來源檔案統計正確率 ──────────────
        # run_idx -> file_path -> 是否正確的布林串列
        run_file_correct: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))

        for run_idx, merged_path in run_merged_path.items():
            if not os.path.exists(merged_path):
                continue
            with open(merged_path, "r", encoding="utf-8") as jf:
                for line in jf:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    # 優先使用 source_file，其次嘗試 file 欄位
                    src_file = entry.get("source_file") or entry.get("file")
                    if src_file:
                        run_file_correct[run_idx][src_file].append(entry.get("is_correct", False))

        # ── 第四階段：組裝最終結果結構 ────────────────────────────────────────
        merged_dataset_results: dict = {}

        for ds_name, files_map in ds_file_runs.items():
            ds_results = []
            for file_path, run_indices in files_map.items():
                run_accuracies = []
                run_merged_jsonl_paths = []

                for run_idx in sorted(run_indices):
                    merged_path = run_merged_path.get(run_idx)
                    if merged_path:
                        run_merged_jsonl_paths.append(merged_path)

                    # 優先從第三階段的統計結果取得正確率
                    correct_list = run_file_correct.get(run_idx, {}).get(file_path)
                    if correct_list is not None:
                        acc = sum(correct_list) / len(correct_list) if correct_list else 0.0
                    else:
                        # 備援：直接從原始分片 JSON 讀取預先計算的正確率
                        acc = _acc_from_shards(json_shards, ds_name, file_path, run_idx)
                    run_accuracies.append(acc)

                mean_acc = float(np.mean(run_accuracies)) if run_accuracies else 0.0
                std_acc = float(np.std(run_accuracies)) if len(run_accuracies) > 1 else 0.0

                ds_results.append({
                    "file": file_path,
                    "accuracy_mean": mean_acc,
                    "accuracy_std": std_acc,
                    "individual_runs": {
                        "accuracies": run_accuracies,
                        "results": run_merged_jsonl_paths,
                    },
                })

            ds_avg_acc = float(np.mean([r["accuracy_mean"] for r in ds_results])) if ds_results else 0.0
            ds_avg_std = float(np.mean([r["accuracy_std"] for r in ds_results])) if ds_results else 0.0

            merged_dataset_results[ds_name] = {
                "results": ds_results,
                "average_accuracy": ds_avg_acc,
                "average_std": ds_avg_std,
            }

        final_results = {
            "timestamp": timestamp,
            "config": base_result["config"],
            "duration_seconds": base_result.get("duration_seconds", 0),
            "dataset_results": merged_dataset_results,
        }

        base_output_path = os.path.join(results_dir, f"results_{timestamp}")
        exported_files = ResultsExporterFactory.export_results(
            final_results, base_output_path, ["json"], base_result["config"]
        )
        print(f"✅ 合併完成，結果已匯出至: {', '.join(exported_files)}")

    finally:
        # ── 清理：即使程式中途被中斷或記憶體不足也保證執行 ────────────────────
        print("🧹 清理 Rank 分散式碎片...")
        for sp in json_shards:
            try:
                os.remove(sp)
            except OSError:
                pass
        for jp in shard_jsonl_files:
            try:
                os.remove(jp)
            except OSError:
                pass

    if hf_repo_id:
        try:
            from .hf_uploader import upload_results
            model_name = base_result["config"].get("model", {}).get("name", "unknown_model")
            upload_results(
                repo_id=hf_repo_id,
                variant=hf_variant,
                model_name=model_name,
                results_dir=results_dir,
                timestamp=timestamp,
            )
            print("✅ 成功上傳合併結果至 Hugging Face")
        except Exception as e:
            print(f"❌ 上傳至 Hugging Face 失敗: {e}")

    return 0


def _acc_from_shards(json_shards: list[str], ds_name: str, file_path: str, run_idx: int) -> float:
    """備援函式：直接從各分片 JSON 讀取已預先計算的正確率（當 JSONL 內無來源檔案欄位時使用）"""
    accs = []
    for sp in json_shards:
        with open(sp, "r", encoding="utf-8") as f:
            shard = json.load(f)
        for fr in shard.get("dataset_results", {}).get(ds_name, {}).get("results", []):
            if fr["file"] == file_path:
                run_accs = fr.get("individual_runs", {}).get("accuracies", [])
                if run_idx < len(run_accs):
                    accs.append(run_accs[run_idx])
    return float(np.mean(accs)) if accs else 0.0


# 向後相容別名
merge_distributed_results = finalize_results
