"""Hugging Face 資料集上傳服務

需要安裝 slurm extra：pip install twinkle-eval[slurm]
"""

import glob
import os
from typing import Optional

from .logger import log_error, log_info

try:
    from huggingface_hub import HfApi, HfFileSystem
    from huggingface_hub.utils import RepositoryNotFoundError
except ImportError:
    raise ImportError(
        "HuggingFace 上傳功能需要安裝額外套件，請執行：\n"
        "  pip install twinkle-eval[slurm]"
    )


def validate_repo_id(repo_id: str) -> None:
    """驗證 Hugging Face repo ID 格式，若不存在則自動創建"""
    # 檢查格式 <namespace>/<name>
    if "/" not in repo_id or len(repo_id.split("/")) != 2:
        raise ValueError(f"無效的 repo ID 格式: {repo_id}。必須為 <namespace>/<name>")
    
    # 檢查結尾
    if not repo_id.endswith("-logs-and-scores"):
        raise ValueError(f"Repo 名稱必須以 '-logs-and-scores' 結尾: {repo_id}")

    # 使用 HfApi 檢查存在性並確認為 dataset
    api = HfApi()
    try:
        # 若 repo 不是 dataset 或不存在，將在此拋出例外
        api.dataset_info(repo_id=repo_id)
        log_info(f"Dataset repo {repo_id} 已存在")
    except RepositoryNotFoundError:
        # Repo 不存在，自動創建以提升 UX
        log_info(f"Dataset repo {repo_id} 不存在，正在創建...")
        try:
            api.create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=False,
            )
            log_info(f"成功創建 Dataset repo: {repo_id}")
        except Exception as create_error:
            raise ValueError(f"創建 Dataset repo 失敗 {repo_id}: {create_error}")
    except Exception as e:
        # 例如: 認證錯誤
        raise ValueError(f"檢查 repo 時發生錯誤 {repo_id}: {e}")


def upload_results(
    repo_id: str,
    variant: Optional[str],
    model_name: str,
    results_dir: str,
    timestamp: str,
) -> None:
    """上傳評測結果至 Hugging Face dataset repo"""
    
    # 驗證 repo_id
    validate_repo_id(repo_id)
    
    # 識別要上傳的檔案
    
    files_to_upload = []
    
    # 主要結果檔案
    results_json = os.path.join(results_dir, f"results_{timestamp}.json")
    if os.path.exists(results_json):
        files_to_upload.append(results_json)
    
    # Run 檔案
    run_files = glob.glob(os.path.join(results_dir, f"eval_results_{timestamp}_run*.jsonl"))
    files_to_upload.extend(run_files)
    
    if not files_to_upload:
        log_info("本次執行未找到可上傳的檔案。")
        return

    # 決定上傳路徑
    # results/<model>/<variant>/
    variant_path = variant if variant else "default"
    # 確保 variant 安全：移除路徑分隔符、父目錄引用及前後空白
    variant_path = variant_path.strip().replace("/", "_").replace("\\", "_").replace("..", "_")
    #確保 model name 安全
    safe_model_name = model_name.replace("/", "__") 
    target_dir = f"results/{safe_model_name}/{variant_path}"
    
    api = HfApi()
    
    log_info(f"正在上傳 {len(files_to_upload)} 個檔案至 {repo_id}/{target_dir}...")
    
    try:
        fs = HfFileSystem()
        for file_path in files_to_upload:
            file_name = os.path.basename(file_path)
            path_in_repo = f"{target_dir}/{file_name}"
            
            # 檢查檔案是否存在以防止覆蓋
            # 需求: "既有 repo 檔案不得覆蓋"
            if fs.exists(f"datasets/{repo_id}/{path_in_repo}"):
                log_error(f"檔案 {path_in_repo} 已存在於 repo 中。跳過以防止覆蓋。")
                continue

            print(f"正在上傳 {file_name}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload evaluation results for {model_name} ({timestamp})"
            )
            
        dataset_url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{target_dir}"
        log_info(f"上傳完成。Dataset URL: {dataset_url}")
        print(f"✅ 上傳成功！查看網址: {dataset_url}")
        
    except Exception as e:
        log_error(f"上傳至 Hugging Face 失敗: {e}")
        raise
