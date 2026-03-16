# SLURM 與 vLLM 配置與執行指南

這份文件說明了如何在 SLURM 叢集環境下設定 vLLM 虛擬環境，並透過設定檔配置執行模型評測。

## 1. 初始環境設定 (vLLM 虛擬環境)

在第一次執行任何評測任務之前，必須先建立好專用的 vLLM 虛擬環境。這只需要在專案根目錄執行一次：

```bash
bash scripts/setup_vllm_venv.sh
```

此腳本會：
- 檢查是否安裝了 `uv` 工具（若無則透過 pip 安裝）。
- 在專案根目錄下建立名為 `vllm-venv` 的虛擬環境 (Python 3.12)。
- 安裝 vLLM 及相關套件 (PyTorch, Ray 等)，並自動驗證安裝。

## 2. 登入 Hugging Face (必要步驟)

為了讓程式能夠下載受保護的模型（如 Llama 3、Mistral 等）以及上傳評測結果，請確保在叢集的終端機中登入您的 Hugging Face 帳號。

1. 取得您的 Hugging Face Token (確保具有 `write` 權限以上傳結果)。
2. 執行以下指令登入：

```bash
hf auth login
```

這會將 Token 快取記錄於系統中，後續腳本將自動讀取使用。

## 3. 配置 SLURM 任務腳本

專案中提供了 `scripts/run_slurm_test.sh` (測試用) 與 `scripts/run_slurm_full.sh` (完整評測用)，執行前請務必根據您的需求修改檔案內的參數。

### A. SLURM #SBATCH 標頭參數設定
打開 `scripts/run_slurm_test.sh` 或 `scripts/run_slurm_full.sh`，修改檔案最上方的 `#SBATCH` 設定：

```bash
#SBATCH --job-name=my-eval-job       # 任務名稱 (識別用)
#SBATCH --partition=dev              # 佇列分區 (正式評測請用 normal 或 GPU 專用佇列)
#SBATCH --nodes=1                    # 使用節點數量 (多節點請修改)
#SBATCH --ntasks-per-node=1          # 每個節點的進程數
#SBATCH --gres=gpu:H200:8            # 使用 GPU 的規格與數量 (例如 H200 系列要求 8 張)
#SBATCH --account=YOUR_ACCOUNT       # 計畫代號或扣款帳號
```

### B. 使用者自定義參數 (腳本內部變數)
您也可以在腳本中找到以下區塊，根據這次要評測的模型調整變數：

- `MODEL_NAME`: Hugging Face 的模型 ID，例如 `"mistralai/Ministral-3-14B-Instruct-2512"`。
- `TP_SIZE` (Tensor Parallel Size): 張量平行數量，若模型太大單卡放不下，請設定為卡數 (例如 2, 4, 8) 分散模型權重。
- `PP_SIZE` (Pipeline Parallel Size): 管線平行數量，針對超大模型跨節點時使用，預設 1 即可。
- `HF_UPLOAD_REPO`: 評測結果上傳至 Hugging Face 的 Repo 名稱 (例: `your-username/my-eval-logs`)。
- `MAX_TOKENS` / `MAX_MODEL_LEN`: 生成輸出的 Tokens 上限以及模型支援的最大 Context 長度。

*提示：這些參數也可以在執行 `sbatch` 時透過後接參數覆蓋，例如：*
```bash
sbatch scripts/run_slurm_test.sh "mistralai/My-Model" 8 1 "my-org/my-repo"
```

### C. 評測配置檔參數 (YAML)
除了腳本中的變數，您也可以修改 `configs/config_slurm_test.yaml` 或 `configs/config_slurm_full.yaml`。雖然這些檔案中的配置（例如 `dataset_paths` 或 System Prompt）較不會頻繁更動，但如果您需要調整特定的評測細節、控制生成數量或其他 `twinkle-eval` 參數，請直接編輯這些 YAML 檔案。

## 4. 執行與監控 SLURM 任務

配置完成後，即可提交任務至排程系統：

```bash
# 提交測試評測任務
sbatch scripts/run_slurm_test.sh

# 提交完整評測任務
sbatch scripts/run_slurm_full.sh
```

### 常見管理指令：

- **查看任務排程狀態**：
  ```bash
  squeue -u <你的帳號>
  ```
- **取消任務**：
  ```bash
  scancel <JobID>
  ```
- **查看即時日誌**：
  任務送出後，會在專案下的 `logs/` 資料夾產生類似 `slurm_test_<JobID>.out` 及 `.err` 的日誌檔案。可使用 tail 即時查看輸出：
  ```bash
  tail -f logs/slurm_test_<JobID>.out
  ```
- **監測節點與 GPU 工作狀態**：
  在叢集中若想查看目前節點的 GPU 使用率與運行狀態，可以使用 `nvnodetop` 工具：
  ```bash
  uvx nvnodetop
  ```

## 5. 評測流程簡述

任務啟動後，執行腳本會自動進行以下流程：
1. 分析硬體與建立環境配置。
2. (若需) 下載 tw-legal-benchmark 資料集。
3. 自動以對應的 GPU 數量啟動本機/多節點的 vLLM OpenAI Server API (`vllm.entrypoints.openai.api_server`)。
4. 在背景使用 `twinkle-eval` 開始批次傳送 prompt 給 vLLM 解答。
5. 評等完成後，統一收集 json 結果，合併所有的 shards。
6. 自動化將成績及日誌推送至您指定的 Hugging Face Repo。
