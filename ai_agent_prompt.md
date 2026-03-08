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

**重要**：請務必加入 `system_prompt` 設定，使用預設的中文提示詞（含 `\box{選項}` 格式），以確保模型能正確輸出答案格式。

```yaml
llm_api:
  base_url: "<endpoint>/v1"
  api_key: "<api_key>"
  disable_ssl_verify: false
  api_rate_limit: -1
  max_retries: 5
  timeout: 600

model:
  name: "<model_name>"
  version: "v1.0"
  temperature: 0.0
  top_p: 0.9
  max_tokens: 4096

evaluation:
  dataset_paths:
    - "datasets/<dataset_name>/"
  run_id: "exp-001"
  creator: "user"
  evaluation_method: "box"
  system_prompt:
    zh: |
      使用者將提供一個題目，並附上選項 A、B、C、D
      請仔細閱讀題目要求，根據題意選出最符合的選項，並將選項以以下格式輸出：
      \box{選項}
      請確保僅將選項包含在 { } 中，否則將不計算為有效答案。
      務必精確遵循輸出格式，避免任何多餘內容或錯誤格式。
  repeat_runs: 1
  shuffle_options: false
```

### 步驟 3：下載資料集
從 HuggingFace Hub 下載所需的資料集。
**注意**：某些資料集需要指定 `--dataset-split train`
**指令**:
```bash
twinkle-eval --download-dataset <dataset_name>
# 如有需要可先查詢資料集資訊
twinkle-eval --dataset-info <dataset_name>
```

### 步驟 4：執行評測
使用步驟 2 建立的設定檔執行評測。
**指令**:
```bash
twinkle-eval --config Twinkle-Eval/twinkle_eval/config.yaml
```

### 步驟 5：上傳結果
將 `results/eval_results_xxxx` 目錄中的內容上傳至 S3 儲存桶。
**指令**:
```bash
twinkle-eval upload results/eval_results_xxxx/ \
  --bucket <bucket_name> \
  --endpoint-url <endpoint_url> \
  --access-key <access_key> \
  --secret-key <secret_key> \
  --no-verify-ssl
```

## 成功標準
- 評測流程執行完畢且無錯誤。
- 評測結果成功上傳至指定的 S3 儲存桶。
- 使用 `github-mcp-create_issue_comment` MCP 工具在觸發此任務的 GitHub Issue 留言回報結果。

## 步驟 6：回報結果至 GitHub Issue

評測完成後，使用 `github-mcp-create_issue_comment` MCP 工具留言，參數如下：
- **owner**: `ap-mic-inc`
- **repo**: `rd-eval-dataset-tasks`
- **issue_number**: 請根據觸發的 Issue 編號填入

留言內容需包含：
- 評測狀態（成功 / 失敗）
- 目標模型與資料集
- 主要評測指標與分數
- S3 結果路徑
- 若失敗，附上錯誤訊息與 logs 內容

## 常見問題
- 執行要在 twinkle_eval 目錄下
- 如果有問題應該要 cat logs/ 目錄下的日誌
- run id use summary + timestamp
- 資料集路徑格式為 `datasets/<dataset_name>/`（HuggingFace 會將 `/` 轉為 `__`）
- 若上傳失敗，可使用 boto3 直接上傳
