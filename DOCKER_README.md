# 🐳 Twinkle Eval Docker 使用說明

本文件說明如何使用 Docker 容器化執行 Twinkle Eval 評測工具。

## 📋 目錄

- [功能特色](#功能特色)
- [前置要求](#前置要求)
- [快速開始](#快速開始)
- [使用方式](#使用方式)
- [配置說明](#配置說明)
- [常見問題](#常見問題)
- [進階用法](#進階用法)

## ✨ 功能特色

- **🚀 快速部署**: 一行命令即可啟動評測環境
- **📦 使用 uv**: 採用 uv 作為套件管理器，提供更快的依賴解析與安裝
- **🔧 自動化檢查**: 容器啟動時自動檢查配置和資料集
- **💾 結果持久化**: 評測結果自動儲存到主機目錄
- **🔄 彈性配置**: 支援 GPU、網路和環境變數配置
- **📝 完整日誌**: 詳細的執行日誌和錯誤處理

## 🛠 前置要求

- Docker Engine 20.10+ 
- Docker Compose 2.0+ (如使用 docker-compose)
- 至少 2GB 可用磁碟空間
- (可選) NVIDIA Docker 支援（如需使用 GPU）

## 🚀 快速開始
### 直接使用 Docker

```bash
# 建構映像
docker build -t twinkle-eval .

# 執行容器
docker run -it --rm \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/logs:/app/logs \
  twinkle-eval
```

## 📖 使用方式

### 基本使用流程

1. **準備資料集**: 將評測檔案放入 `datasets/` 目錄
2. **配置設定**: 編輯 `config.yaml` 設定 API 和模型參數
3. **啟動容器**: 使用上述命令啟動容器
4. **查看結果**: 評測完成後，結果會儲存在 `results/` 目錄

### 容器自動化流程

容器啟動時會依序執行以下步驟：

1. ✅ **環境檢查**: 檢查 uv 和 Python 環境
2. 🔍 **配置驗證**: 檢查 `config.yaml` 是否存在和有效
3. 📁 **資料集檢查**: 確認 `datasets/` 目錄包含評測檔案
4. 🏃 **執行評測**: 使用 `uv run` 執行 twinkle-eval
5. 💾 **儲存結果**: 將結果儲存到 `results/` 目錄
6. 📋 **顯示摘要**: 列出生成的結果檔案

## 🔧 配置說明

### 目錄結構

```
twinkle-eval/
├── config.yaml          # 主配置檔案
├── datasets/            # 評測資料集目錄
│   ├── dataset1.csv
│   ├── dataset2.json
│   └── ...
├── results/             # 評測結果輸出目錄
│   └── details/
├── logs/               # 日誌檔案目錄
├── Dockerfile          # Docker 映像定義
├── docker-compose.yml  # Docker Compose 配置
└── DOCKER_README.md    # 本說明文件
```

### 重要配置項目

編輯 `config.yaml` 時，請特別注意以下設定：

```yaml
llm_api:
  base_url: "your-api-endpoint"     # 替換為您的 API 端點
  api_key: "your-api-key"          # 替換為您的 API 金鑰
  api_rate_limit: 5                # 根據您的 API 限制調整

model:
  name: "your-model-name"          # 替換為您要評測的模型
  temperature: 0.0                 # 根據需求調整生成參數

evaluation:
  dataset_paths:                   # 指定資料集路徑
    - "datasets/"
  repeat_runs: 1                   # 設定重複執行次數
  shuffle_options: true            # 是否隨機排列選項
```

## ❓ 常見問題

### Q: 容器啟動後立即退出

**A**: 檢查以下項目：
- 確認 `config.yaml` 存在且格式正確
- 確認 `datasets/` 目錄包含有效的評測檔案
- 查看容器日誌：`docker-compose logs`

### Q: 找不到 uv.lock 檔案

**A**: 確保專案根目錄包含 `uv.lock` 檔案。如果沒有，請在專案目錄執行：
```bash
uv lock
```

### Q: API 連線失敗

**A**: 檢查以下設定：
- API 端點 URL 是否正確
- API 金鑰是否有效
- 網路連線是否正常
- 防火牆設定是否允許連接

### Q: 記憶體不足錯誤

**A**: 調整 Docker 記憶體限制：
```yaml
# 在 docker-compose.yml 中新增
services:
  twinkle-eval:
    deploy:
      resources:
        limits:
          memory: 4G
```

### Q: 權限問題（Linux/macOS）

**A**: 確保目錄權限正確：
```bash
sudo chown -R $USER:$USER datasets/ results/ logs/
chmod -R 755 datasets/ results/ logs/
```

## 🚀 進階用法

### GPU 支援

如果您的評測需要 GPU 支援，取消 `docker-compose.yml` 中的 GPU 配置註解：

```yaml
services:
  twinkle-eval:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### 自訂啟動腳本

您可以修改容器內的 `/app/start.sh` 來自訂啟動行為：

```bash
# 建立自訂啟動腳本
cat > custom_start.sh << 'EOF'
#!/bin/bash
echo "執行自訂評測流程..."
uv run twinkle-eval --config config.yaml --export json csv html
echo "評測完成，正在執行後處理..."
# 新增您的後處理邏輯
EOF

# 在 docker run 中使用自訂腳本
docker run -it --rm \
  -v $(pwd)/custom_start.sh:/app/start.sh \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/results:/app/results \
  twinkle-eval
```

### 批次評測多個配置

```bash
# 建立多個配置檔案
for config in config1.yaml config2.yaml config3.yaml; do
  docker run --rm \
    -v $(pwd)/$config:/app/config.yaml \
    -v $(pwd)/datasets:/app/datasets \
    -v $(pwd)/results:/app/results \
    twinkle-eval
done
```

### 環境變數配置

```bash
# 使用環境變數覆蓋配置
docker run --rm \
  -e TWINKLE_API_KEY="your-api-key" \
  -e TWINKLE_MODEL_NAME="your-model" \
  -v $(pwd)/datasets:/app/datasets \
  -v $(pwd)/results:/app/results \
  twinkle-eval
```

## 📚 相關連結

- [Twinkle Eval 主專案](https://github.com/ai-twinkle/Eval)
- [uv 官方文件](https://docs.astral.sh/uv/)
- [Docker 官方文件](https://docs.docker.com/)
- [Docker Compose 文件](https://docs.docker.com/compose/)

## 🆘 支援

如果您遇到問題，請：

1. 檢查本說明文件的常見問題章節
2. 查看專案的 [GitHub Issues](https://github.com/ai-twinkle/Eval/issues)
3. 提交新的 Issue 並提供詳細的錯誤資訊

---

**注意**: 使用 Docker 容器時，請確保遵守相關 API 服務的使用條款和限制。