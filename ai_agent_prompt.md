## 目標
你是一個負責執行 `Twinkle-Eval` 框架自動化評測流程的 AI 代理。你的任務是設定環境、下載資料集、執行評測，並將結果上傳至 S3 儲存服務。
要在 Twinkle-Eval 目錄下執行

## 上下文與參數
請使用以下特定參數執行任務：

- **目標模型 (Target Model)**
- **LLM API 端點 (Endpoint)**
- **LLM API 金鑰 (Key)**
- **評測資料集 (Dataset)**
- **S3 上傳設定**:
    - **儲存桶 (Bucket)**
    - **端點 (Endpoint)**
    - **Access Key**
    - **Secret Key**

## 執行步驟

請依序執行以下步驟。若任何步驟失敗，請立即停止並回報錯誤。

### 步驟 1：安裝套件
確保 `Twinkle-Eval` 套件已以編輯模式 (editable mode) 安裝。
**請使用 `uv` 進行安裝以加速過程**：
```bash
cd Twinkle-Eval
uv pip install --system -e .
```

### 步驟 2：設定環境
建立或覆蓋位於 `Twinkle-Eval/twinkle_eval/config.yaml` 的設定檔。

### 步驟 3：下載資料集
從 HuggingFace Hub 下載所需的資料集。
**指令**:
```bash
twinkle-eval --download-dataset <dataset_name>
```

### 步驟 4：執行評測
使用步驟 2 建立的設定檔執行評測。
**指令**:
```bash
twinkle-eval --config Twinkle-Eval/twinkle_eval/config.yaml
```

### 步驟 5：上傳結果
將 `eval_results_xxxx` 目錄中的內容上傳至 S3 儲存桶。
**指令**:
```bash
twinkle-eval upload eval_results_xxxx/ \
  --bucket <bucket_name> \
  --endpoint-url <endpoint_url> \
  --access-key <access_key> \
  --secret-key <secret_key> \
  --no-verify-ssl
```

## 成功標準
- 評測流程執行完畢且無錯誤。
- 評測結果成功上傳至指定的 S3 儲存桶。

## 常見問題
- 執行要在 twinkle_eval 目錄下
- 如果有問題應該要 cat logs/ 目錄下的日誌
- run id use summary + timestamp
