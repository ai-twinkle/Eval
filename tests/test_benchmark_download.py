"""Tests for benchmark registry and download utilities."""

import pytest

from twinkle_eval.benchmarks import (
    BENCHMARK_REGISTRY,
    get_available_benchmarks,
    list_benchmarks,
)


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------


class TestBenchmarkRegistry:
    """驗證 registry 結構完整性。"""

    def test_registry_not_empty(self):
        assert len(BENCHMARK_REGISTRY) > 0

    def test_all_entries_have_required_fields(self):
        required = {"source", "description", "eval_method", "license"}
        for name, info in BENCHMARK_REGISTRY.items():
            missing = required - set(info.keys())
            assert not missing, f"{name} 缺少必填欄位: {missing}"

    def test_hf_entries_have_hf_id_and_split(self):
        for name, info in BENCHMARK_REGISTRY.items():
            if info["source"] == "huggingface":
                assert "hf_id" in info, f"{name} 缺少 hf_id"
                assert "split" in info, f"{name} 缺少 split"
                assert "/" in info["hf_id"], f"{name} 的 hf_id 格式不正確: {info['hf_id']}"

    def test_github_entries_have_url_and_post_process(self):
        for name, info in BENCHMARK_REGISTRY.items():
            if info["source"] == "github":
                assert "url" in info, f"{name} 缺少 url"
                assert "post_process" in info, f"{name} 缺少 post_process"
                assert info["url"].startswith("http"), f"{name} 的 url 格式不正確"

    def test_source_is_valid(self):
        valid_sources = {"huggingface", "github"}
        for name, info in BENCHMARK_REGISTRY.items():
            assert info["source"] in valid_sources, (
                f"{name} 的 source '{info['source']}' 不在 {valid_sources}"
            )

    def test_gated_flag_only_on_hf(self):
        for name, info in BENCHMARK_REGISTRY.items():
            if info.get("gated"):
                assert info["source"] == "huggingface", (
                    f"{name} 標記為 gated 但 source 不是 huggingface"
                )

    def test_known_benchmarks_exist(self):
        expected = [
            "mmlu", "mmlu_pro", "tmmluplus", "mmlu_redux", "supergpqa", "gpqa",
            "formosa_bench", "gsm8k", "aime2025", "bbh", "ifeval", "ifbench",
            "bfcl", "spider", "needlebench", "longbench", "wikieval", "bird",
            "spider2_lite",
        ]
        for name in expected:
            assert name in BENCHMARK_REGISTRY, f"缺少預期的 benchmark: {name}"

    def test_benchmark_count(self):
        assert len(BENCHMARK_REGISTRY) >= 19


# ---------------------------------------------------------------------------
# get_available_benchmarks
# ---------------------------------------------------------------------------


class TestGetAvailableBenchmarks:
    def test_returns_sorted_list(self):
        result = get_available_benchmarks()
        assert result == sorted(result)

    def test_returns_all_registry_keys(self):
        result = get_available_benchmarks()
        assert set(result) == set(BENCHMARK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# list_benchmarks
# ---------------------------------------------------------------------------


class TestListBenchmarks:
    def test_list_benchmarks_runs_without_error(self, capsys):
        list_benchmarks()
        captured = capsys.readouterr()
        assert "可下載的評測資料集" in captured.out
        assert "mmlu" in captured.out
        assert "HF" in captured.out

    def test_list_shows_github_tag(self, capsys):
        list_benchmarks()
        captured = capsys.readouterr()
        assert "GitHub" in captured.out

    def test_list_shows_gated_tag(self, capsys):
        list_benchmarks()
        captured = capsys.readouterr()
        assert "🔒" in captured.out


# ---------------------------------------------------------------------------
# Short name resolution (in CLI handler)
# ---------------------------------------------------------------------------


class TestShortNameResolution:
    """驗證短名稱與 HuggingFace ID 的區分邏輯。"""

    def test_registry_name_recognized(self):
        assert "mmlu" in BENCHMARK_REGISTRY
        assert "gsm8k" in BENCHMARK_REGISTRY
        assert "bird" in BENCHMARK_REGISTRY

    def test_hf_id_has_slash(self):
        """HuggingFace ID 包含 /，短名稱不含。"""
        for name in BENCHMARK_REGISTRY:
            assert "/" not in name, f"registry key '{name}' 不應包含 /"

    def test_all_is_not_a_benchmark_name(self):
        assert "all" not in BENCHMARK_REGISTRY

    def test_list_is_not_a_benchmark_name(self):
        assert "list" not in BENCHMARK_REGISTRY


# ---------------------------------------------------------------------------
# Gated dataset detection
# ---------------------------------------------------------------------------


class TestGatedDataset:
    def test_gpqa_is_gated(self):
        assert BENCHMARK_REGISTRY["gpqa"].get("gated") is True

    def test_non_gated_datasets(self):
        non_gated = ["mmlu", "gsm8k", "bbh", "ifeval", "bird"]
        for name in non_gated:
            assert not BENCHMARK_REGISTRY[name].get("gated"), (
                f"{name} 不應為 gated"
            )


# ---------------------------------------------------------------------------
# GitHub source detection
# ---------------------------------------------------------------------------


class TestGitHubSource:
    def test_github_sources(self):
        github_benchmarks = ["bird", "spider2_lite", "longbench"]
        for name in github_benchmarks:
            assert BENCHMARK_REGISTRY[name]["source"] == "github", (
                f"{name} 應為 github source"
            )

    def test_hf_sources(self):
        hf_benchmarks = ["mmlu", "gsm8k", "ifeval", "bfcl"]
        for name in hf_benchmarks:
            assert BENCHMARK_REGISTRY[name]["source"] == "huggingface", (
                f"{name} 應為 huggingface source"
            )


# ---------------------------------------------------------------------------
# Eval method mapping
# ---------------------------------------------------------------------------


class TestEvalMethodMapping:
    def test_mcq_benchmarks_use_box(self):
        box_benchmarks = ["mmlu", "mmlu_pro", "tmmluplus", "formosa_bench"]
        for name in box_benchmarks:
            assert BENCHMARK_REGISTRY[name]["eval_method"] == "box", (
                f"{name} 應使用 box eval method"
            )

    def test_math_benchmarks(self):
        assert BENCHMARK_REGISTRY["gsm8k"]["eval_method"] == "math"
        assert BENCHMARK_REGISTRY["aime2025"]["eval_method"] == "math"

    def test_text2sql_benchmarks(self):
        for name in ["spider", "bird", "spider2_lite"]:
            assert BENCHMARK_REGISTRY[name]["eval_method"] == "text2sql"

    def test_bbh_uses_regex_match(self):
        assert BENCHMARK_REGISTRY["bbh"]["eval_method"] == "regex_match"
