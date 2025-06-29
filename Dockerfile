# 使用 Python 3.11 官方基礎映像
FROM python:3.11-slim

# 設置工作目錄
WORKDIR /app

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安裝 uv（快速的 Python 套件管理器）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# 複製 uv 相關檔案
COPY pyproject.toml ./

# 設置 uv 環境變數
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 安裝依賴（使用 uv 會自動讀取 pyproject.toml 和 uv.lock）
RUN uv sync --frozen --no-dev

# 複製應用程式碼
COPY twinkle_eval/ ./twinkle_eval/
COPY setup.py MANIFEST.in ./

# 安裝專案本身
RUN uv pip install --no-deps .

# 複製配置檔案和資料集目錄結構
COPY config.yaml ./
RUN mkdir -p datasets results logs

# 設置環境變數
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 建立啟動腳本
RUN echo '#!/bin/bash\n\
echo "🌟 Twinkle Eval 容器已啟動"\n\
echo "使用 uv 作為套件管理器，提供更快的依賴解析與安裝"\n\
echo "正在檢查配置檔案..."\n\
\n\
# 檢查配置檔案是否存在\n\
if [ ! -f "config.yaml" ]; then\n\
    echo "❌ 未找到 config.yaml，正在建立預設配置檔案..."\n\
    uv run twinkle-eval --init\n\
    echo "✅ 已建立預設配置檔案，請編輯 config.yaml 後重新啟動容器"\n\
    exit 1\n\
fi\n\
\n\
# 檢查資料集目錄\n\
if [ ! "$(ls -A datasets)" ]; then\n\
    echo "❌ datasets 目錄為空，請放入評測資料集後重新啟動容器"\n\
    echo "支援格式: .csv, .json, .jsonl, .parquet"\n\
    exit 1\n\
fi\n\
\n\
echo "✅ 配置檢查通過，開始執行評測..."\n\
echo "=========================================="\n\
\n\
# 執行評測（使用 uv run 確保正確的環境）\n\
uv run twinkle-eval --config config.yaml --export json html\n\
\n\
echo "=========================================="\n\
echo "🎉 評測完成！結果已儲存至 results 目錄"\n\
echo "查看結果檔案:"\n\
ls -la results/\n\
' > /app/start.sh && chmod +x /app/start.sh

# 暴露可能需要的埠（如果有 API 服務）
EXPOSE 8000

# 設置預設執行命令
CMD ["/app/start.sh"]