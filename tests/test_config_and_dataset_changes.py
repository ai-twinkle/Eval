"""answer 正規化、split 下載 fallback、config 驗證、--validate 試打與 Responses 後端的單元測試。"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from twinkle_eval.core.exceptions import ValidationError
from twinkle_eval.core.validators import ConfigValidator
from twinkle_eval.datasets.file import Dataset, _download_single_subset

# ── answer 正規化：所有檔案格式一致 ──────────────────────────────────────────

_ROWS = [
    {"question": "Q0", "A": "a", "B": "b", "answer": " b "},  # 選項字母 → 正規化
    {"question": "Q1", "A": "a", "B": "b", "answer": "No"},  # 不是選項鍵 → 原樣
    {"question": "Q2", "answer": "\\frac{1}{2}"},  # 自由作答 → 原樣
]
_EXPECTED = ["B", "No", "\\frac{1}{2}"]


@pytest.mark.parametrize("fmt", ["jsonl", "json", "csv", "parquet"])
def test_answer_normalization_is_format_independent(tmp_path, fmt):
    import pandas as pd

    path = tmp_path / f"d.{fmt}"
    if fmt == "jsonl":
        path.write_text("\n".join(json.dumps(r) for r in _ROWS), encoding="utf-8")
    elif fmt == "json":
        path.write_text(json.dumps(_ROWS), encoding="utf-8")
    elif fmt == "csv":
        pd.DataFrame(_ROWS).to_csv(path, index=False)
    else:
        pytest.importorskip("pyarrow")
        pd.DataFrame(_ROWS).to_parquet(path, index=False)

    assert [r["answer"] for r in Dataset(str(path))] == _EXPECTED


# ── split 下載 fallback ──────────────────────────────────────────────────────


class _FakeHFDataset:
    def __init__(self, written):
        self._written = written

    def to_parquet(self, path):
        self._written.append(path)


def _patch_hf(monkeypatch, splits):
    import datasets

    written = []
    monkeypatch.setattr(datasets, "get_dataset_split_names", lambda *a, **k: splits)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: _FakeHFDataset(written))
    return written


def test_download_requested_split_when_present(monkeypatch):
    written = _patch_hf(monkeypatch, ["train", "test"])
    _download_single_subset("org/ds", "sub", "test", "out")
    assert written == ["out/org__ds/sub.parquet"]


def test_download_falls_back_to_all_splits(monkeypatch, capsys):
    written = _patch_hf(monkeypatch, ["exam_a", "exam_b"])
    _download_single_subset("org/ds", "sub", "test", "out")
    # 各 split 分別存檔，不互相覆蓋
    assert written == ["out/org__ds/sub__exam_a.parquet", "out/org__ds/sub__exam_b.parquet"]
    assert "改為下載所有 splits" in capsys.readouterr().out


def test_download_fallback_default_subset_uses_split_name(monkeypatch):
    written = _patch_hf(monkeypatch, ["exam_a"])
    _download_single_subset("org/ds", "default", "test", "out")
    assert written == ["out/org__ds/exam_a.parquet"]


# ── config 驗證 ──────────────────────────────────────────────────────────────


def _config(**overrides):
    config = {
        "llm_api": {"base_url": "http://x/v1", "api_key": "k"},
        "model": {"name": "m"},
        "evaluation": {"dataset_paths": ["d/"], "evaluation_method": "box"},
    }
    for section, values in overrides.items():
        config[section].update(values)
    return config


@pytest.mark.parametrize("value", [None, 1, 32])
def test_max_workers_accepts_positive_int_or_null(value):
    assert ConfigValidator.validate_config_structure(_config(llm_api={"max_workers": value}))


@pytest.mark.parametrize("value", [0, -1, 2.5, "8", True])
def test_max_workers_rejects_invalid(value):
    with pytest.raises(ValidationError, match="max_workers"):
        ConfigValidator.validate_config_structure(_config(llm_api={"max_workers": value}))


def test_temperature_and_top_p_may_be_null():
    config = _config(model={"temperature": None, "top_p": None})
    assert ConfigValidator.validate_config_structure(config)


# ── --validate 試打 ──────────────────────────────────────────────────────────


def _live_config(extractor):
    llm = MagicMock()
    original_client = llm.client
    llm.call.return_value = SimpleNamespace(model="served-model")
    return {"llm_instance": llm, "extractor_instance": extractor, "model": {"name": "m"}}, (
        original_client
    )


def test_ping_uses_short_timeout_and_restores_client():
    from twinkle_eval.main import _ping_llm

    config, original_client = _live_config(SimpleNamespace())
    assert _ping_llm(config) is None

    original_client.with_options.assert_called_once_with(timeout=30, max_retries=0)
    config["llm_instance"].call.assert_called_once()
    assert config["llm_instance"].client is original_client


def test_ping_reports_errors():
    from twinkle_eval.main import _ping_llm

    config, _ = _live_config(SimpleNamespace())
    config["llm_instance"].call.side_effect = RuntimeError("401 invalid api key")
    assert "401" in _ping_llm(config)


def test_ping_logit_method_uses_completions_logprobs():
    from twinkle_eval.main import _ping_llm

    config, _ = _live_config(SimpleNamespace(uses_logprobs=True))
    llm = config["llm_instance"]
    llm.score_continuation.return_value = float("-inf")

    assert "logprobs" in _ping_llm(config)
    llm.call.assert_not_called()

    llm.score_continuation.return_value = -1.5
    assert _ping_llm(config) is None


def test_ping_skips_audio_methods():
    from twinkle_eval.main import _ping_llm

    config, _ = _live_config(SimpleNamespace(uses_audio=True))
    assert _ping_llm(config) is None
    config["llm_instance"].call.assert_not_called()


# ── Responses 後端 ───────────────────────────────────────────────────────────


def _responses_model(**model_cfg):
    from twinkle_eval.models.responses import OpenAIResponsesModel

    config = {
        "llm_api": {"base_url": "http://x/v1", "api_key": "k", "max_retries": 0, "timeout": 5},
        "model": {"name": "m", "temperature": 0.0, "top_p": 0.9, "max_tokens": 64, **model_cfg},
        "evaluation": {"evaluation_method": "box"},
    }
    model = OpenAIResponsesModel(config)
    model.client = MagicMock()
    model.client.responses.create.return_value = SimpleNamespace(
        id="r1",
        created_at=0,
        model="m",
        status="completed",
        incomplete_details=None,
        output=[],
        output_text="hi",
        usage=None,
    )
    return model


def test_responses_omits_null_sampling_params():
    model = _responses_model(temperature=None, top_p=None)
    model.call("hello", system_prompt_enabled=False)
    payload = model.client.responses.create.call_args.kwargs
    assert "temperature" not in payload and "top_p" not in payload


@pytest.mark.parametrize(
    "status,reason,tool_calls,expected",
    [
        ("completed", None, None, "stop"),
        ("incomplete", "max_output_tokens", None, "length"),
        ("incomplete", "content_filter", None, "content_filter"),
        ("completed", None, [object()], "tool_calls"),
    ],
)
def test_responses_finish_reason(status, reason, tool_calls, expected):
    from twinkle_eval.models.responses import OpenAIResponsesModel

    response = SimpleNamespace(
        status=status,
        incomplete_details=SimpleNamespace(reason=reason) if reason else None,
    )
    assert OpenAIResponsesModel._finish_reason(response, tool_calls) == expected


# ── 時間標記與 dataset_overrides ─────────────────────────────────────────────


def test_run_timestamp_from_env(monkeypatch, capsys):
    from twinkle_eval.runners.standard import TwinkleEvalRunner

    monkeypatch.setenv("TWINKLE_EVAL_RUN_TIMESTAMP", "20260101_120000")
    assert TwinkleEvalRunner._resolve_run_timestamp() == "20260101_120000"
    assert "TWINKLE_EVAL_RUN_TIMESTAMP" in capsys.readouterr().out


def test_run_timestamp_rejects_malformed_env(monkeypatch):
    from twinkle_eval.core.exceptions import ConfigurationError
    from twinkle_eval.runners.standard import TwinkleEvalRunner

    monkeypatch.setenv("TWINKLE_EVAL_RUN_TIMESTAMP", "yesterday")
    with pytest.raises(ConfigurationError):
        TwinkleEvalRunner._resolve_run_timestamp()


def test_run_timestamp_defaults_to_seconds(monkeypatch):
    import re

    from twinkle_eval.runners.standard import TwinkleEvalRunner

    monkeypatch.delenv("TWINKLE_EVAL_RUN_TIMESTAMP", raising=False)
    assert re.fullmatch(r"\d{8}_\d{6}", TwinkleEvalRunner._resolve_run_timestamp())


def test_dataset_override_sampling_params_may_be_null():
    config = _config(evaluation={"dataset_overrides": {"d/": {"temperature": None, "top_p": None}}})
    assert ConfigValidator.validate_config_structure(config)


def test_google_log_upload_includes_current_log(tmp_path, monkeypatch):
    """runner 與 logger 的時間戳不同秒時，本次程序的 log 檔仍須被上傳。"""
    pytest.importorskip("googleapiclient")
    from twinkle_eval.core import logger as logger_module
    from twinkle_eval.integrations.google import GoogleDriveUploader

    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "evaluation_20260101_120001.log").write_text("current")
    (logs / "evaluation_20250101_000000.log").write_text("old")
    monkeypatch.setattr(logger_module, "log_filename", "logs/evaluation_20260101_120001.log")

    uploader = GoogleDriveUploader.__new__(GoogleDriveUploader)
    uploader.config = {}
    uploader.create_folder = MagicMock(return_value="folder")
    uploader.upload_file = MagicMock(return_value="file-id")

    results = tmp_path / "results"
    results.mkdir()
    uploader.upload_latest_files("20260101_120000", str(logs), str(results))

    uploaded = [call.args[0] for call in uploader.upload_file.call_args_list]
    assert uploaded == [str(logs / "evaluation_20260101_120001.log")]
