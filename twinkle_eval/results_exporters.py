import csv
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

import pandas as pd


class ResultsExporter(ABC):
    """Abstract base class for results exporters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    def export(self, results: Dict[str, Any], output_path: str) -> str:
        """Export results to specified format."""
        pass

    @abstractmethod
    def get_file_extension(self) -> str:
        """Get the file extension for this exporter."""
        pass


class JSONExporter(ResultsExporter):
    """Export results to JSON format."""

    def get_file_extension(self) -> str:
        return ".json"

    def export(self, results: Dict[str, Any], output_path: str) -> str:
        """Export results to JSON file."""
        if not output_path.endswith(self.get_file_extension()):
            output_path += self.get_file_extension()

        # Ensure environment config is included
        enhanced_results = self._enhance_with_environment(results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enhanced_results, f, indent=4, ensure_ascii=False)

        return output_path

    def _enhance_with_environment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure environment config is included in results."""
        enhanced = results.copy()
        
        # Check if config and environment already exist
        if "config" not in enhanced:
            enhanced["config"] = {}
        
        if "environment" not in enhanced["config"]:
            enhanced["config"]["environment"] = {
                "gpu_info": {
                    "model": "N/A",
                    "count": "N/A",
                    "memory_gb": "N/A",
                    "cuda_version": "N/A",
                    "driver_version": "N/A"
                },
                "parallel_config": {
                    "tp_size": "N/A",
                    "pp_size": "N/A"
                },
                "system_info": {
                    "framework": "N/A",
                    "python_version": "N/A",
                    "torch_version": "N/A",
                    "node_count": "N/A"
                }
            }
        
        return enhanced


class CSVExporter(ResultsExporter):
    """Export results to CSV format."""

    def get_file_extension(self) -> str:
        return ".csv"

    def export(self, results: Dict[str, Any], output_path: str) -> str:
        """Export results to CSV file."""
        if not output_path.endswith(self.get_file_extension()):
            output_path += self.get_file_extension()

        # Flatten the results structure for CSV export
        flattened_data = self._flatten_results(results)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if flattened_data:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)

        return output_path

    def _flatten_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten nested results structure for CSV export."""
        flattened = []

        timestamp = results.get("timestamp", "")
        config = results.get("config", {})
        environment = config.get("environment", {})

        base_info = {
            "timestamp": timestamp,
            "config_model_name": config.get("model", {}).get("name", ""),
            "config_temperature": config.get("model", {}).get("temperature", ""),
            "gpu_model": environment.get("gpu_info", {}).get("model", ""),
            "gpu_count": environment.get("gpu_info", {}).get("count", ""),
            "gpu_memory_gb": environment.get("gpu_info", {}).get("memory_gb", ""),
            "cuda_version": environment.get("gpu_info", {}).get("cuda_version", ""),
            "tp_size": environment.get("parallel_config", {}).get("tp_size", ""),
            "pp_size": environment.get("parallel_config", {}).get("pp_size", ""),
            "framework": environment.get("system_info", {}).get("framework", ""),
        }

        for dataset_path, dataset_data in results.get("dataset_results", {}).items():
            dataset_info = base_info.copy()
            dataset_info.update(
                {
                    "dataset_path": dataset_path,
                    "dataset_average_accuracy": dataset_data.get("average_accuracy", 0),
                    "dataset_average_std": dataset_data.get("average_std", 0),
                }
            )

            for file_result in dataset_data.get("results", []):
                file_info = dataset_info.copy()
                file_info.update(
                    {
                        "file": file_result.get("file", ""),
                        "accuracy_mean": file_result.get("accuracy_mean", 0),
                        "accuracy_std": file_result.get("accuracy_std", 0),
                    }
                )
                flattened.append(file_info)

        return flattened


class ExcelExporter(ResultsExporter):
    """Export results to Excel format."""

    def get_file_extension(self) -> str:
        return ".xlsx"

    def export(self, results: Dict[str, Any], output_path: str) -> str:
        """Export results to Excel file."""
        if not output_path.endswith(self.get_file_extension()):
            output_path += self.get_file_extension()

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            # Summary sheet
            summary_data = self._create_summary_data(results)
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Detailed results for each dataset
            for dataset_path, dataset_data in results.get("dataset_results", {}).items():
                sheet_name = os.path.basename(dataset_path)[:30]  # Excel sheet name limit
                detailed_data = self._create_detailed_data(dataset_data)
                if detailed_data:
                    detailed_df = pd.DataFrame(detailed_data)
                    detailed_df.to_excel(writer, sheet_name=sheet_name, index=False)

        return output_path

    def _create_summary_data(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create summary data for Excel export."""
        summary = []

        config = results.get("config", {})
        environment = config.get("environment", {})

        for dataset_path, dataset_data in results.get("dataset_results", {}).items():
            summary.append(
                {
                    "Dataset": dataset_path,
                    "Average Accuracy": dataset_data.get("average_accuracy", 0),
                    "Standard Deviation": dataset_data.get("average_std", 0),
                    "Number of Files": len(dataset_data.get("results", [])),
                    "GPU Model": environment.get("gpu_info", {}).get("model", ""),
                    "GPU Count": environment.get("gpu_info", {}).get("count", ""),
                    "TP Size": environment.get("parallel_config", {}).get("tp_size", ""),
                    "PP Size": environment.get("parallel_config", {}).get("pp_size", ""),
                    "Framework": environment.get("system_info", {}).get("framework", ""),
                }
            )

        return summary

    def _create_detailed_data(self, dataset_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create detailed data for Excel export."""
        detailed = []

        for file_result in dataset_data.get("results", []):
            detailed.append(
                {
                    "File": file_result.get("file", ""),
                    "Accuracy Mean": file_result.get("accuracy_mean", 0),
                    "Accuracy Std": file_result.get("accuracy_std", 0),
                    "Individual Accuracies": str(
                        file_result.get("individual_runs", {}).get("accuracies", [])
                    ),
                }
            )

        return detailed


class HTMLExporter(ResultsExporter):
    """Export results to HTML format."""

    def get_file_extension(self) -> str:
        return ".html"

    def export(self, results: Dict[str, Any], output_path: str) -> str:
        """Export results to HTML file."""
        if not output_path.endswith(self.get_file_extension()):
            output_path += self.get_file_extension()

        html_content = self._generate_html(results)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

    def _generate_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML content from results."""
        # Ensure environment config is included
        enhanced_results = self._enhance_with_environment(results)
        
        # Load detailed results from referenced files
        enhanced_results = self._load_detailed_results(enhanced_results)
        
        return self._generate_summary_html(enhanced_results)

    def _enhance_with_environment(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure environment config is included in results."""
        enhanced = results.copy()
        
        # Check if config and environment already exist
        if "config" not in enhanced:
            enhanced["config"] = {}
        
        if "environment" not in enhanced["config"]:
            enhanced["config"]["environment"] = {
                "gpu_info": {
                    "model": "N/A",
                    "count": "N/A",
                    "memory_gb": "N/A",
                    "cuda_version": "N/A",
                    "driver_version": "N/A"
                },
                "parallel_config": {
                    "tp_size": "N/A",
                    "pp_size": "N/A"
                },
                "system_info": {
                    "framework": "N/A",
                    "python_version": "N/A",
                    "torch_version": "N/A",
                    "node_count": "N/A"
                }
            }
        
        return enhanced

    def _load_detailed_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Load detailed results from referenced files."""
        enhanced = results.copy()
        
        for dataset_path, dataset_data in enhanced.get("dataset_results", {}).items():
            for file_result in dataset_data.get("results", []):
                # Check if individual_runs contains result file references
                individual_runs = file_result.get("individual_runs", {})
                result_files = individual_runs.get("results", [])
                
                if result_files:
                    # Load the first result file for detailed data
                    result_file_path = result_files[0]
                    try:
                        with open(result_file_path, "r", encoding="utf-8") as f:
                            detailed_data = json.load(f)
                        
                        # Add detailed data to file_result
                        if "details" in detailed_data:
                            file_result["details"] = detailed_data["details"]
                            file_result["file"] = detailed_data.get("file", file_result.get("file", ""))
                            file_result["accuracy"] = detailed_data.get("accuracy", file_result.get("accuracy_mean", 0))
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        print(f"⚠️ Warning: Could not load detailed results from {result_file_path}: {e}")
        
        return enhanced

    def _generate_summary_html(self, results: Dict[str, Any]) -> str:
        """Generate HTML content for summary and detailed results."""
        timestamp = results.get("timestamp", "")
        config = results.get("config", {})
        environment = config.get("environment", {})
        gpu_info = environment.get("gpu_info", {})
        parallel_config = environment.get("parallel_config", {})
        system_info = environment.get("system_info", {})
        
        # Check if we have detailed results data
        has_detailed_data = False
        for dataset_path, dataset_data in results.get("dataset_results", {}).items():
            for file_result in dataset_data.get("results", []):
                if "details" in file_result:
                    has_detailed_data = True
                    break
            if has_detailed_data:
                break
        
        if has_detailed_data:
            # Handle detailed result display
            title = "Twinkle Eval 詳細結果"
            
            # Get all detailed results from all datasets
            all_details = []
            all_files = []
            total_accuracy = 0
            total_files = 0
            
            for dataset_path, dataset_data in results.get("dataset_results", {}).items():
                for file_result in dataset_data.get("results", []):
                    if "details" in file_result:
                        all_details.extend(file_result["details"])
                        all_files.append(file_result.get("file", ""))
                        total_accuracy += file_result.get("accuracy", file_result.get("accuracy_mean", 0))
                        total_files += 1
            
            # Calculate overall accuracy
            overall_accuracy = total_accuracy / total_files if total_files > 0 else 0
            file_paths = ", ".join(all_files) if all_files else ""
            details = all_details
            
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title} - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
        .header {{ background-color: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px; }}
        .environment {{ background-color: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; border: 1px solid #ddd; }}
        .env-header {{ cursor: pointer; display: flex; justify-content: space-between; align-items: center; padding: 10px; background-color: #e9ecef; border-radius: 5px; margin-bottom: 10px; }}
        .env-header:hover {{ background-color: #dee2e6; }}
        .env-content {{ display: none; }}
        .env-content.show {{ display: block; }}
        .env-section {{ margin: 10px 0; }}
        .collapse-icon {{ transition: transform 0.3s ease; }}
        .collapse-icon.rotated {{ transform: rotate(180deg); }}
        .file-info {{ background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        .accuracy {{ color: #2e7d32; font-weight: bold; font-size: 1.2em; }}
        .question-item {{ background-color: #ffffff; margin: 15px 0; padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); border-left: 5px solid #007bff; }}
        .question-item.correct {{ border-left-color: #28a745; }}
        .question-item.incorrect {{ border-left-color: #dc3545; }}
        .question-text {{ font-weight: bold; margin-bottom: 15px; line-height: 1.6; }}
        .answer-section {{ margin: 10px 0; }}
        .correct-answer {{ color: #28a745; font-weight: bold; }}
        .predicted-answer {{ color: #007bff; font-weight: bold; }}
        .incorrect-answer {{ color: #dc3545; font-weight: bold; }}
        .llm-output {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; white-space: pre-wrap; line-height: 1.4; }}
        .reasoning {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; white-space: pre-wrap; line-height: 1.4; }}
        .usage-info {{ font-size: 0.9em; color: #6c757d; margin-top: 10px; }}
        .status-badge {{ padding: 5px 10px; border-radius: 15px; color: white; font-size: 0.9em; font-weight: bold; }}
        .correct-badge {{ background-color: #28a745; }}
        .incorrect-badge {{ background-color: #dc3545; }}
        .summary-stats {{ display: flex; gap: 20px; margin: 20px 0; }}
        .stat-item {{ background-color: #ffffff; padding: 15px; border-radius: 10px; flex: 1; text-align: center; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .stat-label {{ color: #6c757d; margin-top: 5px; }}
        
        /* Tab styles */
        .tabs {{ margin: 20px 0; }}
        .tab-buttons {{ display: flex; background-color: #f8f9fa; border-radius: 10px 10px 0 0; border: 1px solid #ddd; border-bottom: none; }}
        .tab-button {{ flex: 1; padding: 15px; background: none; border: none; cursor: pointer; font-size: 16px; font-weight: bold; transition: all 0.3s ease; }}
        .tab-button.active {{ background-color: #ffffff; color: #007bff; }}
        .tab-button.correct.active {{ color: #28a745; }}
        .tab-button.incorrect.active {{ color: #dc3545; }}
        .tab-content {{ display: none; background-color: #ffffff; border: 1px solid #ddd; border-radius: 0 0 10px 10px; padding: 20px; }}
        .tab-content.active {{ display: block; }}
    </style>
    <script>
        function toggleEnvironment() {{
            const content = document.querySelector('.env-content');
            const icon = document.querySelector('.collapse-icon');
            content.classList.toggle('show');
            icon.classList.toggle('rotated');
        }}
        
        function showTab(tabName) {{
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            
            // Remove active class from all buttons
            document.querySelectorAll('.tab-button').forEach(button => {{
                button.classList.remove('active');
            }});
            
            // Show selected tab content
            document.getElementById(tabName + '-tab').classList.add('active');
            
            // Add active class to clicked button
            document.querySelector(`[onclick="showTab('${{tabName}}')"]`).classList.add('active');
        }}
        
        window.onload = function() {{
            // Show first tab by default
            showTab('correct');
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>🌟 {title}</h1>
        <p><strong>時間戳記：</strong> {timestamp}</p>
        <p><strong>檔案：</strong> {file_paths}</p>
        <p><strong>總體準確率：</strong> <span class="accuracy">{overall_accuracy:.2%}</span></p>
        <p><strong>Model:</strong> {config.get("model", {}).get("name", "N/A")}</p>
        <p><strong>Temperature:</strong> {config.get("model", {}).get("temperature", "N/A")}</p>
    </div>
    
    <div class="environment">
        <div class="env-header" onclick="toggleEnvironment()">
            <h2>🖥️ Environment Information</h2>
            <span class="collapse-icon">▼</span>
        </div>
        <div class="env-content">
            <div class="env-section">
                <h3>GPU Configuration</h3>
                <p><strong>Model:</strong> {gpu_info.get("model", "N/A")}</p>
                <p><strong>Count:</strong> {gpu_info.get("count", "N/A")}</p>
                <p><strong>Memory:</strong> {gpu_info.get("memory_gb", "N/A")} GB</p>
                <p><strong>CUDA Version:</strong> {gpu_info.get("cuda_version", "N/A")}</p>
                <p><strong>Driver Version:</strong> {gpu_info.get("driver_version", "N/A")}</p>
            </div>
            <div class="env-section">
                <h3>Parallel Configuration</h3>
                <p><strong>TP Size:</strong> {parallel_config.get("tp_size", "N/A")}</p>
                <p><strong>PP Size:</strong> {parallel_config.get("pp_size", "N/A")}</p>
            </div>
            <div class="env-section">
                <h3>System Information</h3>
                <p><strong>Framework:</strong> {system_info.get("framework", "N/A")}</p>
                <p><strong>Python Version:</strong> {system_info.get("python_version", "N/A")}</p>
                <p><strong>PyTorch Version:</strong> {system_info.get("torch_version", "N/A")}</p>
                <p><strong>Node Count:</strong> {system_info.get("node_count", "N/A")}</p>
            </div>
        </div>
    </div>
    
    <div class="summary-stats">
        <div class="stat-item">
            <div class="stat-number">{len(details)}</div>
            <div class="stat-label">總題數</div>
        </div>
        <div class="stat-item">
            <div class="stat-number" style="color: #28a745;">{sum(1 for d in details if d.get('is_correct', False))}</div>
            <div class="stat-label">正確題數</div>
        </div>
        <div class="stat-item">
            <div class="stat-number" style="color: #dc3545;">{sum(1 for d in details if not d.get('is_correct', False))}</div>
            <div class="stat-label">錯誤題數</div>
        </div>
        <div class="stat-item">
            <div class="stat-number" style="color: #007bff;">{sum(d.get('usage_total_tokens', 0) for d in details):,}</div>
            <div class="stat-label">總 Token 使用量</div>
        </div>
    </div>
"""
            
            # Add tabbed questions
            correct_details = [d for d in details if d.get('is_correct', False)]
            incorrect_details = [d for d in details if not d.get('is_correct', False)]
            
            html += f"""
    <div class="tabs">
        <div class="tab-buttons">
            <button class="tab-button correct" onclick="showTab('correct')">✓ 正確答題 ({len(correct_details)})</button>
            <button class="tab-button incorrect" onclick="showTab('incorrect')">✗ 錯誤答題 ({len(incorrect_details)})</button>
        </div>
        
        <div id="correct-tab" class="tab-content">
"""
            
            for i, detail in enumerate(correct_details, 1):
                question_id = detail.get("question_id", i)
                question = detail.get("question", "")
                correct_answer = detail.get("correct_answer", "")
                predicted_answer = detail.get("predicted_answer", "")
                llm_output = detail.get("llm_output", "")
                reasoning = detail.get("llm_resoning_output", "")
                usage_completion = detail.get("usage_completion_tokens", 0)
                usage_prompt = detail.get("usage_prompt_tokens", 0)
                usage_total = detail.get("usage_total_tokens", 0)

                html += f"""
            <div class="question-item correct">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h3>第 {question_id} 題</h3>
                    <span class="status-badge correct-badge">✓ 正確</span>
                </div>
                
                <div class="question-text">{question}</div>
                
                <div class="answer-section">
                    <p><strong>正確答案：</strong> <span class="correct-answer">{correct_answer}</span></p>
                    <p><strong>預測答案：</strong> <span class="correct-answer">{predicted_answer}</span></p>
                </div>
                
                <div class="llm-output">
                    <strong>LLM 輸出：</strong>
                    {llm_output}
                </div>
                
                {f'<div class="reasoning"><strong>推理過程：</strong>{reasoning}</div>' if reasoning else ''}
                
                <div class="usage-info">
                    <strong>Token 使用量：</strong> 
                    提示 {usage_prompt:,} | 完成 {usage_completion:,} | 總計 {usage_total:,}
                </div>
            </div>
"""
            
            html += """
        </div>
        
        <div id="incorrect-tab" class="tab-content">
"""
            
            for i, detail in enumerate(incorrect_details, 1):
                question_id = detail.get("question_id", i)
                question = detail.get("question", "")
                correct_answer = detail.get("correct_answer", "")
                predicted_answer = detail.get("predicted_answer", "")
                llm_output = detail.get("llm_output", "")
                reasoning = detail.get("llm_resoning_output", "")
                usage_completion = detail.get("usage_completion_tokens", 0)
                usage_prompt = detail.get("usage_prompt_tokens", 0)
                usage_total = detail.get("usage_total_tokens", 0)

                html += f"""
            <div class="question-item incorrect">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h3>第 {question_id} 題</h3>
                    <span class="status-badge incorrect-badge">✗ 錯誤</span>
                </div>
                
                <div class="question-text">{question}</div>
                
                <div class="answer-section">
                    <p><strong>正確答案：</strong> <span class="correct-answer">{correct_answer}</span></p>
                    <p><strong>預測答案：</strong> <span class="incorrect-answer">{predicted_answer}</span></p>
                </div>
                
                <div class="llm-output">
                    <strong>LLM 輸出：</strong>
                    {llm_output}
                </div>
                
                {f'<div class="reasoning"><strong>推理過程：</strong>{reasoning}</div>' if reasoning else ''}
                
                <div class="usage-info">
                    <strong>Token 使用量：</strong> 
                    提示 {usage_prompt:,} | 完成 {usage_completion:,} | 總計 {usage_total:,}
                </div>
            </div>
"""
            
            html += """
        </div>
    </div>
"""
        else:
            # Handle summary result display
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Twinkle Eval Results - {timestamp}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .environment {{ background-color: #f8f9fa; padding: 15px; margin: 20px 0; border-radius: 5px; border: 1px solid #ddd; }}
        .env-header {{ cursor: pointer; display: flex; justify-content: space-between; align-items: center; padding: 10px; background-color: #e9ecef; border-radius: 5px; margin-bottom: 10px; }}
        .env-header:hover {{ background-color: #dee2e6; }}
        .env-content {{ display: none; }}
        .env-content.show {{ display: block; }}
        .env-section {{ margin: 10px 0; }}
        .collapse-icon {{ transition: transform 0.3s ease; }}
        .collapse-icon.rotated {{ transform: rotate(180deg); }}
        .dataset {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .dataset-header {{ background-color: #e8f4f8; padding: 15px; }}
        .results-table {{ width: 100%; border-collapse: collapse; }}
        .results-table th, .results-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .results-table th {{ background-color: #f2f2f2; }}
        .accuracy {{ color: #2e7d32; font-weight: bold; }}
    </style>
    <script>
        function toggleEnvironment() {{
            const content = document.querySelector('.env-content');
            const icon = document.querySelector('.collapse-icon');
            content.classList.toggle('show');
            icon.classList.toggle('rotated');
        }}
    </script>
</head>
<body>
    <div class="header">
        <h1>🌟 Twinkle Eval Results</h1>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <p><strong>Model:</strong> {config.get("model", {}).get("name", "N/A")}</p>
        <p><strong>Temperature:</strong> {config.get("model", {}).get("temperature", "N/A")}</p>
    </div>
    
    <div class="environment">
        <div class="env-header" onclick="toggleEnvironment()">
            <h2>🖥️ Environment Information</h2>
            <span class="collapse-icon">▼</span>
        </div>
        <div class="env-content">
            <div class="env-section">
                <h3>GPU Configuration</h3>
                <p><strong>Model:</strong> {gpu_info.get("model", "N/A")}</p>
                <p><strong>Count:</strong> {gpu_info.get("count", "N/A")}</p>
                <p><strong>Memory:</strong> {gpu_info.get("memory_gb", "N/A")} GB</p>
                <p><strong>CUDA Version:</strong> {gpu_info.get("cuda_version", "N/A")}</p>
                <p><strong>Driver Version:</strong> {gpu_info.get("driver_version", "N/A")}</p>
            </div>
            <div class="env-section">
                <h3>Parallel Configuration</h3>
                <p><strong>TP Size:</strong> {parallel_config.get("tp_size", "N/A")}</p>
                <p><strong>PP Size:</strong> {parallel_config.get("pp_size", "N/A")}</p>
            </div>
            <div class="env-section">
                <h3>System Information</h3>
                <p><strong>Framework:</strong> {system_info.get("framework", "N/A")}</p>
                <p><strong>Python Version:</strong> {system_info.get("python_version", "N/A")}</p>
                <p><strong>PyTorch Version:</strong> {system_info.get("torch_version", "N/A")}</p>
                <p><strong>Node Count:</strong> {system_info.get("node_count", "N/A")}</p>
            </div>
        </div>
    </div>
"""

            # Add dataset results for summary
            for dataset_path, dataset_data in results.get("dataset_results", {}).items():
                html += f"""
    <div class="dataset">
        <div class="dataset-header">
            <h2>📊 Dataset: {dataset_path}</h2>
            <p><strong>Average Accuracy:</strong> <span class="accuracy">{dataset_data.get("average_accuracy", 0):.2%}</span></p>
            <p><strong>Standard Deviation:</strong> {dataset_data.get("average_std", 0):.4f}</p>
        </div>
        <table class="results-table">
            <tr>
                <th>File</th>
                <th>Accuracy Mean</th>
                <th>Accuracy Std</th>
            </tr>
"""

                for file_result in dataset_data.get("results", []):
                    html += f"""
            <tr>
                <td>{file_result.get("file", "")}</td>
                <td class="accuracy">{file_result.get("accuracy_mean", 0):.2%}</td>
                <td>{file_result.get("accuracy_std", 0):.4f}</td>
            </tr>
"""

                html += """
        </table>
    </div>
"""

        html += """
</body>
</html>
"""

        return html


class ResultsExporterFactory:
    """Factory class for creating results exporters."""

    _registry: Dict[str, Type[ResultsExporter]] = {
        "json": JSONExporter,
        "csv": CSVExporter,
        "excel": ExcelExporter,
        "html": HTMLExporter,
    }

    @classmethod
    def register_exporter(cls, name: str, exporter_class: Type[ResultsExporter]):
        """Register a new results exporter."""
        if not issubclass(exporter_class, ResultsExporter):
            raise ValueError(f"Exporter class must inherit from ResultsExporter base class")
        cls._registry[name] = exporter_class

    @classmethod
    def create_exporter(
        cls, exporter_type: str, config: Optional[Dict[str, Any]] = None
    ) -> ResultsExporter:
        """Create a results exporter instance based on type."""
        if exporter_type not in cls._registry:
            available_types = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unsupported exporter type: {exporter_type}. Available types: {available_types}"
            )

        exporter_class = cls._registry[exporter_type]
        return exporter_class(config)

    @classmethod
    def get_available_types(cls) -> List[str]:
        """Get list of available exporter types."""
        return list(cls._registry.keys())

    @classmethod
    def export_results(
        cls, results: Dict[str, Any], output_path: str, formats: List[str]
    ) -> List[str]:
        """Export results to multiple formats."""
        exported_files = []

        for format_type in formats:
            try:
                exporter = cls.create_exporter(format_type)
                base_path = os.path.splitext(output_path)[0]
                format_path = base_path + exporter.get_file_extension()
                exported_file = exporter.export(results, format_path)
                exported_files.append(exported_file)
            except Exception as e:
                print(f"⚠️ Warning: Failed to export to {format_type}: {e}")

        return exported_files
