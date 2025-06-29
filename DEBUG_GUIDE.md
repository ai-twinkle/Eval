# 🌟 Twinkle Eval Debug 指南

本指南將幫助您在 VSCode 中對 twinkle-eval 進行 debug。

## 📋 前置需求

1. **VSCode 擴充功能**
   - Python 擴充功能 (ms-python.python)
   - Python Debugger 擴充功能

2. **Python 環境**
   - 確保已安裝所有依賴套件
   - 建議使用虛擬環境

## 🚀 快速開始

### 1. 執行測試腳本

首先執行測試腳本來確認環境設定：

```bash
python debug_test.py
```

這個腳本會檢查：
- 基本模組匯入
- 配置檔案載入
- 工廠類別功能
- 資料集功能
- 主函數功能

### 2. 使用 VSCode Debug

#### 方法一：使用 Run and Debug 面板

1. 在 VSCode 中按 `Ctrl+Shift+D` (或 `Cmd+Shift+D` on Mac) 開啟 Run and Debug 面板
2. 從下拉選單中選擇適合的 debug 配置：
   - **Debug Twinkle Eval (基本評測)** - 使用預設配置進行評測
   - **Debug Twinkle Eval (自定義配置)** - 使用自定義配置檔案
   - **Debug Twinkle Eval (下載資料集)** - 下載 HuggingFace 資料集
   - **Debug Twinkle Eval (查詢功能)** - 列出可用的 LLM 類型和評測策略
   - **Debug Twinkle Eval (建立配置)** - 建立新的配置檔案
   - **Debug Twinkle Eval (模組測試)** - 以模組方式執行
   - **Debug 特定檔案** - 對當前開啟的檔案進行 debug

3. 按 `F5` 開始 debug

#### 方法二：使用快捷鍵

- `F5` - 開始 debug（使用預設配置）
- `Ctrl+F5` - 執行而不 debug

## 🔧 Debug 配置說明

### 基本評測配置
```json
{
    "name": "Debug Twinkle Eval (基本評測)",
    "args": ["--config", "config.yaml"]
}
```
- 使用 `config.yaml` 進行基本評測
- 適合測試完整的評測流程

### 自定義配置
```json
{
    "name": "Debug Twinkle Eval (自定義配置)",
    "args": ["--config", "${input:configPath}", "--export", "json", "csv"]
}
```
- 會提示您輸入配置檔案路徑
- 輸出多種格式的結果

### 下載資料集
```json
{
    "name": "Debug Twinkle Eval (下載資料集)",
    "args": ["--download-dataset", "cais/mmlu", "--dataset-subset", "anatomy"]
}
```
- 下載 MMLU 資料集的 anatomy 子集
- 適合測試資料集下載功能

## 🎯 常用 Debug 技巧

### 1. 設定中斷點

在程式碼中點擊行號左側或按 `F9` 來設定中斷點：

```python
# 在 main.py 的第 423 行設定中斷點
download_huggingface_dataset(
    dataset_name=args.download_dataset,  # ← 在這裡設定中斷點
    subset=args.dataset_subset,
    split=args.dataset_split,
    output_dir=args.output_dir,
)
```

### 2. 逐步執行

- `F10` - 逐步執行（Step Over）
- `F11` - 逐步進入（Step Into）
- `Shift+F11` - 逐步跳出（Step Out）
- `F5` - 繼續執行

### 3. 檢查變數

在 Debug 面板中：
- **Variables** - 查看當前作用域的變數
- **Watch** - 監控特定變數的值
- **Call Stack** - 查看函數呼叫堆疊

### 4. 條件中斷點

右鍵點擊中斷點，選擇 "Edit Breakpoint" 來設定條件：

```python
# 例如：只在特定條件下中斷
args.download_dataset == "cais/mmlu"
```

## 🐛 常見問題排解

### 1. 模組找不到

如果遇到 `ModuleNotFoundError`：

1. 確認 `PYTHONPATH` 設定正確
2. 檢查虛擬環境是否啟動
3. 確認所有依賴套件已安裝

### 2. 配置檔案問題

如果配置檔案載入失敗：

1. 檢查 `config.yaml` 格式是否正確
2. 確認檔案路徑是否正確
3. 使用 `--init` 重新建立配置檔案

### 3. 權限問題

如果遇到權限相關錯誤：

1. 確認對 `datasets/` 和 `results/` 目錄有寫入權限
2. 檢查 API 金鑰是否正確設定

## 📝 Debug 範例

### 範例 1：Debug 資料集下載

1. 選擇 "Debug Twinkle Eval (下載資料集)" 配置
2. 在 `datasets.py` 的 `download_huggingface_dataset` 函數中設定中斷點
3. 按 `F5` 開始 debug
4. 使用 `F10` 逐步執行，觀察變數值變化

### 範例 2：Debug 評測流程

1. 選擇 "Debug Twinkle Eval (基本評測)" 配置
2. 在 `main.py` 的 `run_evaluation` 方法中設定中斷點
3. 按 `F5` 開始 debug
4. 使用 `F11` 進入 `_evaluate_dataset` 方法

### 範例 3：Debug 配置載入

1. 選擇 "Debug Twinkle Eval (建立配置)" 配置
2. 在 `config.py` 的 `load_config` 函數中設定中斷點
3. 按 `F5` 開始 debug
4. 觀察配置檔案的載入過程

## 🔍 進階 Debug 技巧

### 1. 使用 Debug Console

在 debug 過程中，可以使用 Debug Console 執行 Python 程式碼：

```python
# 檢查當前變數
print(args)
print(config)

# 測試函數
from twinkle_eval.datasets import find_all_evaluation_files
files = find_all_evaluation_files("datasets")
print(f"找到 {len(files)} 個檔案")
```

### 2. 使用 Logging

在程式碼中加入 logging 來追蹤執行流程：

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def some_function():
    logger.debug("進入函數")
    # ... 程式碼 ...
    logger.debug("函數執行完成")
```

### 3. 使用 pdb

在程式碼中插入 `pdb` 來進行互動式 debug：

```python
import pdb

def some_function():
    # ... 程式碼 ...
    pdb.set_trace()  # 程式會在這裡暫停
    # ... 更多程式碼 ...
```

## 📚 相關資源

- [VSCode Python Debug 官方文件](https://code.visualstudio.com/docs/python/debugging)
- [Python Debugging 技巧](https://docs.python.org/3/library/pdb.html)
- [Twinkle Eval 專案文件](README.md)

---

💡 **提示**：如果遇到問題，請檢查 VSCode 的 Debug Console 輸出，通常會提供有用的錯誤資訊。 