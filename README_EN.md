![Twinkle Eval](assets/Twinkle_Eval.png)

# Twinkle Eval: Efficient and Accurate AI Evaluation Tool

English | [繁體中文](README.md)

[![Python](https://img.shields.io/badge/python-≥3.10-blue.svg?logo=python)](https://www.python.org)
![GitHub license](https://img.shields.io/github/license/ai-twinkle/Eval)
[![Website](https://img.shields.io/badge/Website-twinkleai.tw-blue?style=flat)](https://twinkleai.tw/)
[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-orange?logo=google-colab&style=for-the-badge)](https://colab.research.google.com/github/LiuYuWei/llm-colab-application/blob/main/Simon_LLM_Application_Twinkle_Eval_Tool_Google_Gemini_Model_Evaluation.ipynb)

This project is a Large Language Model (LLM) evaluation framework that uses concurrent and randomized testing methods to provide objective model performance analysis and stability assessment, supporting multiple common evaluation datasets.

## Table of Contents

- [Key Features](#key-features)
- [Performance Metrics](#performance-metrics)
- [Technical Highlights](#technical-highlights)
  - [Evaluation Methods](#evaluation-methods)
  - [Supported Formats and Common Datasets](#supported-formats-and-common-datasets)
  - [API Performance Settings](#api-performance-settings)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration File](#configuration-file)
  - [LLM API Configuration](#llm-api-configuration)
  - [Model Configuration](#model-configuration)
  - [Evaluation Configuration](#evaluation-configuration)
  - [Logging Configuration](#logging-configuration)
- [Output Results](#output-results)
- [Model Test Results Leaderboard](#model-test-results-leaderboard)
- [Contributors](#contributors)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Key Features

- **Automated Multi-file Evaluation**: Batch processing with unified result generation
- **Customizable Evaluation Parameters & Generation Control**: Configure temperature, top_p, and other generation parameters
- **Option Shuffling**: Prevents models from developing option order bias
- **Dual Evaluation Modes**: Supports both Pattern matching and Box scoring logic
- **Multi-run Analysis**: Set number of test runs to observe model performance stability
- **Accuracy and Stability Metrics**: Quantifies model answer accuracy and performance variance
- **LLM Inference and Statistical Logging**: For subsequent analysis of model performance across question types
- **OpenAI API Format Support**: Compatible with common GPT API input and output formats
- **Secure API Key Handling**: Prevents key exposure in code or logs
- **API Rate Limiting and Automatic Retry**: Reduces errors and improves API request success rate

## Performance Metrics

The chart below shows inference time comparison between Twinkle Eval and existing tool [iKala/ievals](https://github.com/iKala/ievals) on [ikala/tmmluplus](https://huggingface.co/datasets/ikala/tmmluplus) - **basic_medical_science** (954 questions) subtask across three models:

![TMMLU Evaluation Time Statistics](assets/tmmlu_eval_time_rounded_seconds.png)

- [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) (non-reasoning task): Twinkle Eval is **9.4x faster**
- [deepseek-ai/DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) (reasoning task): Twinkle Eval is **16.9x faster**
- [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) (non-reasoning task): Twinkle Eval is **14.5x faster**

These experimental results demonstrate that **Twinkle Eval significantly improves performance across different model sizes and task types, with up to a 17x speed advantage** while maintaining consistent accuracy. This dramatically reduces cycles and costs for LLM development workflows requiring extensive evaluation.

## Technical Highlights

### Evaluation Methods

- **Randomized Testing**: Based on [Changing Answer Order Can Decrease MMLU Accuracy](https://arxiv.org/html/2406.19470v1), implements **option shuffling** for more objective model capability assessment
- **Stability Analysis**: Supports multiple test runs with statistical analysis
- **Format Control**: Specify `\box{option}` or `\boxed{option}` format for strict output presentation management
- **Error Handling**: Automatic retry and timeout control mechanisms

### Supported Formats and Common Datasets

Any `.csv`, `.json`, `.jsonl`, or `.parquet` file conforming to the following format, with required fields (not limited to TMMLU+):

```csv
question,A,B,C,D,answer
```

Known evaluation datasets:

- [TMMLU+](https://huggingface.co/datasets/ikala/tmmluplus)
- [MMLU](https://github.com/hendrycks/test)
- [tw-legal-benchmark-v1](https://huggingface.co/datasets/lianghsun/tw-legal-benchmark-v1)
- [Formosa-bench](https://huggingface.co/datasets/lianghsun/Formosa-bench)

### API Performance Settings

- Request rate limiting: Unlimited or specified QPS (Queries Per Second) value
- Timeout configuration
- Optional SSL verification
- Error recovery mechanism

## Installation

### Method 1: Install via pip (Recommended)

```bash
# Install from PyPI (stable version)
pip install twinkle-eval

# Or install from GitHub (latest version)
pip install git+https://github.com/ai-twinkle/Eval.git
```

### Method 2: Install from Source

1. Clone the project locally
   ```bash
   git clone https://github.com/ai-twinkle/Eval.git
   cd Eval
   ```

2. Install package
   ```bash
   # Install production version
   pip install .

   # Or install development version (includes development tools)
   pip install -e ".[dev]"
   ```

## Usage

### Quick Start

1. After installation, create configuration file:
   ```bash
   # Use built-in command to create default configuration file
   twinkle-eval --init

   # Edit configuration file
   nano config.yaml
   ```

2. Prepare evaluation dataset:
   ```bash
   mkdir datasets
   # Place your dataset files in the datasets directory
   ```

3. Run evaluation:
   ```bash
   twinkle-eval --config config.yaml
   ```

### Command Line Options

After installation, you can use the `twinkle-eval` command:

```bash
# Create default configuration file
twinkle-eval --init

# Run evaluation with default configuration
twinkle-eval

# Use custom configuration file
twinkle-eval --config path/to/your/config.yaml

# Output results in multiple formats simultaneously
twinkle-eval --export json csv html

# List supported LLM types
twinkle-eval --list-llms

# List supported evaluation strategies
twinkle-eval --list-strategies

# List supported output formats
twinkle-eval --list-exporters

# Show version information
twinkle-eval --version

# Show complete help
twinkle-eval --help
```

### Python API Usage

You can also use Twinkle Eval directly in Python code:

```python
from twinkle_eval import TwinkleEvalRunner

# Create evaluation runner
runner = TwinkleEvalRunner("config.yaml")

# Initialize
runner.initialize()

# Run evaluation
results = runner.run_evaluation(export_formats=["json", "csv"])

print(f"Evaluation complete! Results saved to: {results}")
```

Evaluation results are saved in the `results` directory with timestamped filenames.

## Code Architecture

The refactored code adopts a modular design, mainly containing the following modules:

- **`cli.py`**: Command-line interface entry point
- **`main.py`**: Main program logic, handles evaluation flow control
- **`config.py`**: Configuration management, loads and validates configuration files
- **`models.py`**: LLM abstraction layer, supports multiple LLM APIs (currently supports OpenAI-compatible formats)
- **`dataset.py`**: Dataset loading and processing, supports JSON, JSONL, CSV, TSV, Parquet, Arrow formats
- **`evaluators.py`**: Core evaluation logic, includes parallel processing and progress tracking
- **`evaluation_strategies.py`**: Answer extraction strategies, includes Pattern, Box, and custom regex strategies
- **`results_exporters.py`**: Result export module, supports JSON, JSONL, CSV, HTML, Google Sheets formats
- **`validators.py`**: Validation tools, ensures correctness of configuration and datasets
- **`exceptions.py`**: Custom exception classes, provides precise error handling
- **`logger.py`**: Logging utilities, handles log recording and output
- **`benchmark.py`**: Performance benchmarking tool
- **`google_services.py`**: Google Drive and Google Sheets integration features

This modular design makes code easier to maintain and extend. Developers can easily:

- Add support for new LLM APIs (via Factory pattern)
- Implement new answer extraction strategies (via Strategy pattern)
- Add new output formats (via Exporter Factory)
- Integrate new cloud services (such as Google Drive, Google Sheets)

## Configuration File

The configuration file uses YAML format and contains the following main sections:

### LLM API Configuration

```yaml
llm_api:
  base_url: "http://your-openai-compatible-server/v1" # API server address
  api_key: "your-api-key" # API key
  disable_ssl_verify: false # Whether to disable SSL verification
  api_rate_limit: 2 # Requests per second limit (-1 for unlimited)
  max_retries: 5 # Number of retries on API call failure
  timeout: 600 # API call timeout (seconds)
```

### Model Configuration

```yaml
model:
  name: "model-name" # Model name
  temperature: 0.0 # Temperature parameter
  top_p: 0.9 # Top-p probability threshold
  max_tokens: 4096 # Maximum output tokens
  frequency_penalty: 0.0 # Frequency penalty
  presence_penalty: 0.0 # Presence penalty
```

### Evaluation Configuration

```yaml
evaluation:
  dataset_paths: # Dataset paths
    - "datasets/dataset1/"
    - "datasets/dataset2/"
  evaluation_method: "box" # Evaluation method (supports "pattern" or "box")
  system_prompt:        # System prompt, only used in box evaluation method
    zh: |
      使用者將提供一個題目，並附上選項 A、B、C、D
      請仔細閱讀題目要求，根據題意選出最符合的選項，並將選項以以下格式輸出：
      \box{選項}
      請確保僅將選項包含在 { } 中，否則將不計算為有效答案。
      務必精確遵循輸出格式，避免任何多餘內容或錯誤格式。
    en: |
      The user will provide a question along with options A, B, C, and D.
      Please read the question carefully and select the option that best fits the requirements.
      Output the selected option in the following format:
      \box{Option}
      Make sure to include only the option within the curly braces; otherwise, it will not be considered a valid answer.
      Strictly follow the output format and avoid any extra content or incorrect formatting.
  datasets_prompt_map:
    "datasets/mmlu/": "en" # Specify dataset to use English prompt
  repeat_runs: 5 # Number of repeated runs for a single dataset
  shuffle_options: true # Whether to randomly shuffle options
```

### Logging Configuration

```yaml
logging:
  level: "INFO" # Log level (options: DEBUG, INFO, WARNING, ERROR)
```

## Output Results

This project primarily outputs `results_{timestamp}.json` summary results, and optionally outputs `eval_results_{timestamp}.jsonl` detailed results (when using JSONL format export).

### `results_{timestamp}.json`

This file is mainly used to **consolidate summary information for the entire evaluation**, suitable for:

- Quickly viewing model performance across multiple datasets
- Comparing average accuracy of different models and configurations
- Referencing model parameters and API settings used
- Using timestamp as evaluation version control record basis

```json
{
  "timestamp": "20250314_1158", // Evaluation execution timestamp
  "results": [
    // Evaluation results for each test file
    {
      "file": "datasets/test/basic_medical_science_train.csv", // Test file path
      "accuracy": 0.4 // Model accuracy on this file
    },
    {
      "file": "datasets/test/culinary_skills_dev.csv",
      "accuracy": 0.4
    }
  ],
  "average_accuracy": 0.4, // Average accuracy across all datasets
  "config": {
    "llm_api": {
      "base_url": "http://localhost:8002/v1/", // Model API endpoint
      "api_key": "EMPTY" // API key (empty here)
    },
    "model": {
      "name": "checkpoint-108", // Model name used
      "temperature": 0, // Temperature parameter (affects randomness)
      "top_p": 0.9, // Top-p sampling parameter
      "max_tokens": 4096, // Maximum generation length
      "frequency_penalty": 0,
      "presence_penalty": 0
    },
    "evaluation": {
      "dataset_path": "datasets/test/", // Evaluation dataset directory
      "api_concurrency": 40, // Concurrent requests (affects inference speed)
      "evaluation_method": "box", // Evaluation mode is box
      "system_prompt": { // System prompt
        "zh": "...", // Chinese prompt
        "en": "..."  // English prompt
      },
      "datasets_prompt_map": {
        "datasets/mmlu/": "en"
      }
    }
  },
  "logging": {
    "level": "INFO" // Log level
  }
}
```

### `eval_results_{timestamp}.jsonl`

This file (JSONL format) is used to **record answer status for each question in a single test file**, suitable for:

- Analyzing incorrect answers and understanding model error tendencies
- Pairing with data visualization (such as confusion matrix, error rate heatmaps)

```json
{
  "timestamp": "20250314_1158",  // Evaluation execution timestamp
  "file": "datasets/test/basic_medical_science_train.csv",  // Test file path
  "accuracy": 0.4,  // Model's overall accuracy on this file

  "details": [  // Evaluation details for each question
    {
      "question_id": 0,  // Question number
      "question": "Which of the following is located only in the kidney cortex? A: Papillary duct ...",  // Question content and options
      "correct_answer": "C",  // Correct answer
      "predicted_answer": "C",  // Model predicted answer
      "is_correct": true  // Whether prediction is correct
    },
    {
      "question_id": 1,
      ...
    }
  ]
}
```

## Model Test Results Leaderboard

For the latest model evaluation results, visit the [TW Eval Leaderboard](https://apps.twinkleai.tw/tw-eval-leaderboard/?lang=en). The leaderboard is continuously updated with new evaluation scores.

## Contributors

[![Teds Lin](https://img.shields.io/badge/GitHub-Teds%20Lin-blue?logo=github)](https://github.com/teds-lin)
[![Liang Hsun Huang](https://img.shields.io/badge/GitHub-Huang%20Liang%20Hsun-blue?logo=github)](https://github.com/lianghsun)
[![Min Yi Chen](https://img.shields.io/badge/GitHub-Min%20Yi%20Chen-blue?logo=github)](https://github.com/cyc00518)
[![Dave Sung](https://img.shields.io/badge/GitHub-Dave%20Sung-blue?logo=github)](https://github.com/k1dav)
[![Thomas Liang](https://img.shields.io/badge/GitHub-Thomas%20Liang-blue?logo=github)](https://github.com/thliang01)

This project is developed through collaboration between [Twinkle AI](https://github.com/ai-twinkle) and [APMIC](https://www.apmic.ai/).

## License

The source code in this repository is open-sourced under the [MIT](https://github.com/ai-twinkle/Eval?tab=MIT-1-ov-file#readme) license.

## Citation

If you find this evaluation tool helpful, please cite it as follows:

```bibtex
@misc{twinkle_eval,
  author       = {Teds Lin, Liang Hsun Huang, Min Yi Chen, Dave Sung and Thomas Liang},
  title        = {Twinkle Eval: An Efficient and Accurate AI Evaluation Tool.},
  year         = {2025},
  url          = {https://github.com/ai-twinkle/Eval},
  note         = {GitHub repository}
}
```

## Acknowledgments

During the development of this project, we referenced design pattern concepts from the [iKala/ievals](https://github.com/iKala/ievals) project, which provided valuable inspiration for our design direction. We extend our sincere thanks.

We also thank [Simon Liu](https://simonliuyuwei-4ndgcf4.gamma.site/) for providing the Colab [demo example](https://colab.research.google.com/github/LiuYuWei/llm-colab-application/blob/main/Simon_LLM_Application_Twinkle_Eval_Tool_Google_Gemini_Model_Evaluation.ipynb), which helps us present the tool's usage and practical application scenarios more intuitively.
