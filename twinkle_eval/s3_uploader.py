import os
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from .logger import log_error, log_info


class S3Uploader:
    """S3 檔案上傳器"""

    def __init__(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        region_name: Optional[str] = None,
        verify_ssl: bool = True,
    ):
        """初始化 S3 上傳器

        Args:
            aws_access_key_id: AWS Access Key ID
            aws_secret_access_key: AWS Secret Access Key
            endpoint_url: S3 Endpoint URL
            region_name: AWS Region Name
            verify_ssl: 是否驗證 SSL 憑證
        """
        self.s3_client = self._create_s3_client(
            aws_access_key_id,
            aws_secret_access_key,
            endpoint_url,
            region_name,
            verify_ssl,
        )

    def _create_s3_client(
        self,
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        endpoint_url: Optional[str],
        region_name: Optional[str],
        verify_ssl: bool,
    ) -> Any:
        try:
            return boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
                endpoint_url=endpoint_url,
                verify=verify_ssl,
            )
        except Exception as e:
            log_error(f"建立 S3 client 失敗: {e}")
            raise

    def upload_directory(self, local_dir: str, bucket: str, prefix: str = "") -> bool:
        """上傳目錄中的所有檔案到 S3

        Args:
            local_dir: 本地目錄路徑
            bucket: S3 bucket 名稱
            prefix: S3 key 前綴 (資料夾結構)

        Returns:
            bool: 是否全部上傳成功
        """
        if not os.path.isdir(local_dir):
            log_error(f"目錄不存在: {local_dir}")
            return False

        success = True
        if prefix is None:
            prefix = ""

        # 將 local_dir 的最後一個資料夾名稱加入 prefix
        dir_name = os.path.basename(os.path.normpath(local_dir))
        prefix = os.path.join(prefix, dir_name)

        # 移除 prefix 開頭的斜線，避免 S3 路徑問題
        if prefix.startswith("/"):
            prefix = prefix[1:]

        # 確保 prefix 結尾有斜線 (如果有值的話)
        if prefix and not prefix.endswith("/"):
            prefix += "/"

        # 統一使用 forward slash
        prefix = prefix.replace("\\", "/")

        for root, _, files in os.walk(local_dir):
            for file in files:
                local_path = os.path.join(root, file)

                # 計算相對路徑，構建 S3 key
                relative_path = os.path.relpath(local_path, local_dir)
                # 使用 forward slash 即使在 Windows 上，S3 路徑標準
                s3_key = f"{prefix}{relative_path}".replace("\\", "/")

                try:
                    log_info(f"正在上傳: {local_path} -> s3://{bucket}/{s3_key}")
                    self.s3_client.upload_file(local_path, bucket, s3_key)
                except ClientError as e:
                    log_error(f"上傳失敗 {file}: {e}")
                    success = False
                except Exception as e:
                    log_error(f"發生錯誤 {file}: {e}")
                    success = False

        if success:
            log_info(f"✅ 目錄 {local_dir} 已成功上傳至 s3://{bucket}/{prefix}")
        else:
            log_error(f"❌ 上傳過程中發生部分錯誤")

        return success
