"""Validation utilities for Twinkle Eval."""

import os
from typing import Any, Dict, List, Optional

import yaml

from .exceptions import ConfigurationError, ValidationError


class ConfigValidator:
    """Validator for configuration files and settings."""

    REQUIRED_SECTIONS = ("llm_api", "model", "evaluation")
    REQUIRED_LLM_API_FIELDS = ("api_key", "base_url")
    REQUIRED_MODEL_FIELDS = ("name",)
    REQUIRED_EVALUATION_FIELDS = ("dataset_paths", "evaluation_method")

    @classmethod
    def validate_config_file(cls, config_path: str) -> bool:
        """Validate that configuration file exists and is readable."""
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        if not os.path.isfile(config_path):
            raise ConfigurationError(f"Configuration path is not a file: {config_path}")

        if not os.access(config_path, os.R_OK):
            raise ConfigurationError(f"Configuration file is not readable: {config_path}")

        return True

    @classmethod
    def validate_yaml_syntax(cls, config_path: str) -> bool:
        """Validate YAML syntax of configuration file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                yaml.safe_load(f)
            return True
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML syntax in {config_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading configuration file {config_path}: {e}")

    @classmethod
    def validate_config_structure(cls, config: Dict[str, Any]) -> bool:
        """Validate the structure of configuration dictionary."""
        # Check required sections
        for section in cls.REQUIRED_SECTIONS:
            if section not in config:
                raise ValidationError(f"Missing required configuration section: {section}")

            if not isinstance(config[section], dict):
                raise ValidationError(f"Configuration section '{section}' must be a dictionary")

        # Validate LLM API section
        cls._validate_llm_api_config(config["llm_api"])

        # Validate model section
        cls._validate_model_config(config["model"])

        # Validate evaluation section
        cls._validate_evaluation_config(config["evaluation"])

        return True

    @classmethod
    def _validate_llm_api_config(cls, llm_config: Dict[str, Any]) -> bool:
        """Validate LLM API configuration."""
        for field in cls.REQUIRED_LLM_API_FIELDS:
            if field not in llm_config:
                raise ValidationError(f"Missing required LLM API field: {field}")

            if not isinstance(llm_config[field], str) or not llm_config[field].strip():
                raise ValidationError(f"LLM API field '{field}' must be a non-empty string")

        # Validate optional fields
        if "type" in llm_config and not isinstance(llm_config["type"], str):
            raise ValidationError("LLM API 'type' must be a string")

        if "max_retries" in llm_config:
            if not isinstance(llm_config["max_retries"], int) or llm_config["max_retries"] < 0:
                raise ValidationError("LLM API 'max_retries' must be a non-negative integer")

        if "timeout" in llm_config:
            if not isinstance(llm_config["timeout"], (int, float)) or llm_config["timeout"] <= 0:
                raise ValidationError("LLM API 'timeout' must be a positive number")

        if "api_rate_limit" in llm_config:
            if not isinstance(llm_config["api_rate_limit"], (int, float)):
                raise ValidationError("LLM API 'api_rate_limit' must be a number")

        return True

    @classmethod
    def _validate_model_config(cls, model_config: Dict[str, Any]) -> bool:
        """Validate model configuration."""
        for field in cls.REQUIRED_MODEL_FIELDS:
            if field not in model_config:
                raise ValidationError(f"Missing required model field: {field}")

            if not isinstance(model_config[field], str) or not model_config[field].strip():
                raise ValidationError(f"Model field '{field}' must be a non-empty string")

        # Validate optional numeric fields
        numeric_fields = [
            "temperature",
            "top_p",
            "max_tokens",
        ]
        for field in numeric_fields:
            if field in model_config:
                if not isinstance(model_config[field], (int, float)):
                    raise ValidationError(f"Model field '{field}' must be a number")

                # Specific validations
                if field == "temperature" and not (0 <= model_config[field] <= 1):
                    raise ValidationError("Model 'temperature' must be between 0 and 1")

                if field == "top_p" and not (0 <= model_config[field] <= 1):
                    raise ValidationError("Model 'top_p' must be between 0 and 1")

                if field == "max_tokens" and model_config[field] <= 0:
                    raise ValidationError("Model 'max_tokens' must be positive")

        return True

    @classmethod
    def _validate_evaluation_config(cls, eval_config: Dict[str, Any]) -> bool:
        """Validate evaluation configuration."""
        for field in cls.REQUIRED_EVALUATION_FIELDS:
            if field not in eval_config:
                raise ValidationError(f"Missing required evaluation field: {field}")

        # Validate dataset paths
        dataset_paths = eval_config["dataset_paths"]
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        elif not isinstance(dataset_paths, list):
            raise ValidationError("Evaluation 'dataset_paths' must be a string or list of strings")

        for path in dataset_paths:
            if not isinstance(path, str) or not path.strip():
                raise ValidationError("All dataset paths must be non-empty strings")

        # Validate evaluation method
        eval_method = eval_config["evaluation_method"]
        if not isinstance(eval_method, str) or not eval_method.strip():
            raise ValidationError("Evaluation 'evaluation_method' must be a non-empty string")

        # Validate optional fields
        if "repeat_runs" in eval_config:
            repeat_runs = eval_config["repeat_runs"]
            if not isinstance(repeat_runs, int) or repeat_runs <= 0:
                raise ValidationError("Evaluation 'repeat_runs' must be a positive integer")

        if "shuffle_options" in eval_config:
            if not isinstance(eval_config["shuffle_options"], bool):
                raise ValidationError("Evaluation 'shuffle_options' must be a boolean")

        if "datasets_prompt_map" in eval_config:
            prompt_map = eval_config["datasets_prompt_map"]
            if not isinstance(prompt_map, dict):
                raise ValidationError("Evaluation 'datasets_prompt_map' must be a dictionary")

            for key, value in prompt_map.items():
                if not isinstance(key, str) or not isinstance(value, str):
                    raise ValidationError("All entries in 'datasets_prompt_map' must be strings")

        return True


class DatasetValidator:
    """Validator for dataset files and directories."""

    SUPPORTED_EXTENSIONS = (".json", ".jsonl", ".parquet", ".csv", ".tsv")
    REQUIRED_COLUMNS = ("question", "answer")
    VALID_OPTION_COLUMNS = ("A", "B", "C", "D")

    @classmethod
    def validate_dataset_path(cls, dataset_path: str) -> bool:
        """Validate that dataset path exists and is accessible."""
        if not os.path.exists(dataset_path):
            raise ValidationError(f"Dataset path does not exist: {dataset_path}")

        if not os.path.isdir(dataset_path):
            raise ValidationError(f"Dataset path is not a directory: {dataset_path}")

        if not os.access(dataset_path, os.R_OK):
            raise ValidationError(f"Dataset directory is not readable: {dataset_path}")

        return True

    @classmethod
    def validate_dataset_files(cls, dataset_path: str) -> List[str]:
        """Validate and return list of valid dataset files."""
        valid_files = []

        for root, dirs, files in os.walk(dataset_path):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                file_path = os.path.join(root, file)
                ext = os.path.splitext(file)[-1].lower()

                if ext in cls.SUPPORTED_EXTENSIONS:
                    if cls._validate_file_access(file_path):
                        valid_files.append(file_path)

        if not valid_files:
            raise ValidationError(f"No valid dataset files found in: {dataset_path}")

        return valid_files

    @classmethod
    def _validate_file_access(cls, file_path: str) -> bool:
        """Validate file access permissions."""
        if not os.path.isfile(file_path):
            return False

        if not os.access(file_path, os.R_OK):
            return False

        return True

    @classmethod
    def validate_dataset_content(cls, data: List[Dict[str, Any]], file_path: str) -> bool:
        """Validate dataset content structure."""
        if not data:
            raise ValidationError(f"Dataset file is empty: {file_path}")

        for idx, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValidationError(f"Row {idx} in {file_path} is not a dictionary")

            # Check required columns
            for col in cls.REQUIRED_COLUMNS:
                if col not in row:
                    raise ValidationError(
                        f"Missing required column '{col}' in row {idx} of {file_path}"
                    )

                if not isinstance(row[col], str) or not row[col].strip():
                    raise ValidationError(
                        f"Column '{col}' in row {idx} of {file_path} must be a non-empty string"
                    )

            # Validate answer format
            answer = row["answer"].strip().upper()
            if answer not in cls.VALID_OPTION_COLUMNS:
                raise ValidationError(
                    f"Invalid answer '{answer}' in row {idx} of {file_path}. Must be A, B, C, or D"
                )

            # Check if corresponding option exists
            if answer not in row:
                raise ValidationError(
                    f"Answer '{answer}' has no corresponding option in row {idx} of {file_path}"
                )

        return True


class RuntimeValidator:
    """Validator for runtime conditions and states."""

    @classmethod
    def validate_llm_response(cls, response: Optional[str], context: str = "") -> bool:
        """Validate LLM response."""
        if response is None:
            raise ValidationError(
                f"LLM returned None response{' for ' + context if context else ''}"
            )

        if not response.strip():
            raise ValidationError(
                f"LLM returned empty response{' for ' + context if context else ''}"
            )

        return True

    @classmethod
    def validate_accuracy_calculation(cls, correct: int, total: int) -> bool:
        """Validate accuracy calculation inputs."""
        if not isinstance(correct, int) or correct < 0:
            raise ValidationError("Correct count must be a non-negative integer")

        if not isinstance(total, int) or total <= 0:
            raise ValidationError("Total count must be a positive integer")

        if correct > total:
            raise ValidationError("Correct count cannot exceed total count")

        return True

    @classmethod
    def validate_export_path(cls, export_path: str) -> bool:
        """Validate export path."""
        if not isinstance(export_path, str) or not export_path.strip():
            raise ValidationError("Export path must be a non-empty string")

        # Check if directory exists or can be created
        directory = os.path.dirname(export_path)
        if directory and not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                raise ValidationError(f"Cannot create export directory {directory}: {e}")

        # Check write permissions
        if directory and not os.access(directory, os.W_OK):
            raise ValidationError(f"No write permission for export directory: {directory}")

        return True
