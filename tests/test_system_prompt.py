"""system_prompt 送達路徑的測試（#144）。

修正前 `_build_messages()` 以 `method in {"box", "math"}` 的白名單決定是否送出
system prompt，而 vision 路徑根本不經過該函式。結果是除了 box / math 之外的
所有評測方法——包含官方範本就設了 system_prompt 的 regex_match——設了也不會
進 request。
"""

import pathlib

import pytest

from twinkle_eval.core.prompts import resolve_system_prompt
from twinkle_eval.models.openai import OpenAIModel
from twinkle_eval.runners.evaluator import _build_vision_messages


class TestResolveSystemPrompt:
    @pytest.mark.parametrize(
        "cfg,lang,enabled,expected",
        [
            ({"system_prompt": {"zh": "中文", "en": "english"}}, "zh", True, "中文"),
            ({"system_prompt": {"zh": "中文", "en": "english"}}, "en", True, "english"),
            ({"system_prompt": {"zh": "中文"}}, "ja", True, "中文"),  # 回退 zh
            ({"system_prompt": "純字串"}, "zh", True, "純字串"),
            ({"system_prompt": {"zh": "中文"}}, "zh", False, None),  # 明確停用
            ({}, "zh", True, None),  # 未設定
            ({"system_prompt": {}}, "zh", True, None),
            ({"system_prompt": {"zh": "   "}}, "zh", True, None),  # 空白視同未設定
            ({"system_prompt": None}, "zh", True, None),
            # 鍵存在但為 null：明確不要 prompt，不得跨語言回退（否則英文題目會收到中文 prompt）
            ({"system_prompt": {"zh": "中文", "en": None}}, "en", True, None),
            ({"system_prompt": {"zh": "中文", "en": ""}}, "en", True, None),
            # 鍵不存在且只定義了一種語言：回退到該語言（regex_match.yaml 的情況）
            ({"system_prompt": {"en": "only english"}}, "zh", True, "only english"),
            # 鍵不存在但定義了多種語言：回退 zh
            ({"system_prompt": {"zh": "中文", "ja": "日本語"}}, "ko", True, "中文"),
        ],
    )
    def test_resolution(self, cfg, lang, enabled, expected):
        assert resolve_system_prompt(cfg, lang, enabled) == expected


def make_model(evaluation):
    m = OpenAIModel.__new__(OpenAIModel)
    m.config = {
        "llm_api": {"api_key": "x", "base_url": "u"},
        "model": {"name": "m"},
        "evaluation": evaluation,
    }
    return m


class TestBuildMessages:
    """文字路徑：不再依評測方法白名單。"""

    @pytest.mark.parametrize("method", ["box", "math", "regex_match", "pattern", "custom_regex"])
    def test_system_prompt_sent_for_every_method_when_configured(self, method):
        m = make_model({"evaluation_method": method, "system_prompt": {"zh": "格式要求"}})
        msgs = m._build_messages("題目", "zh", method, True)
        assert [x["role"] for x in msgs] == ["system", "user"]
        assert msgs[0]["content"] == "格式要求"

    @pytest.mark.parametrize("method", ["box", "math", "regex_match", "pattern"])
    def test_no_system_message_when_not_configured(self, method):
        """未設定時的行為與修正前相同，不得平白多出一個空的 system message。"""
        m = make_model({"evaluation_method": method})
        assert [x["role"] for x in m._build_messages("題目", "zh", method, True)] == ["user"]

    def test_disabled_flag_wins(self):
        m = make_model({"evaluation_method": "box", "system_prompt": {"zh": "格式要求"}})
        assert [x["role"] for x in m._build_messages("題目", "zh", "box", False)] == ["user"]

    def test_shipped_regex_match_template_actually_sends_its_prompt(self):
        """回歸 #144：讀**實際出貨的範本**，不寫死語言。

        先前的版本把 prompt_lang 寫死成 "en" 才通過，掩蓋了真正的問題——
        範本只定義 en 鍵、又沒有 datasets_prompt_map，而 prompt_lang 預設是
        "zh"，所以使用者照官方範本設定仍然收不到 prompt。
        """
        import yaml

        tpl = pathlib.Path("twinkle_eval/templates/regex_match.yaml")
        ev = yaml.safe_load(tpl.read_text(encoding="utf-8"))["evaluation"]

        # 預設語言（未指定 datasets_prompt_map 時）
        assert resolve_system_prompt(ev, "zh", True), "預設語言下範本的 prompt 仍送不出"
        # 範本自己宣告的語言對應
        lang = ev["datasets_prompt_map"]["datasets/example/bbh/"]
        assert resolve_system_prompt(ev, lang, True)

    def test_box_with_no_system_prompt_sends_no_empty_system_message(self):
        """行為變更：舊版對 box/math 會送出 content 為空字串的 system message。

        空的 system block 在多數 chat template 下仍會被渲染，可能影響分數。
        新版不送，這是改善但屬行為變更，已記錄於 CHANGELOG。
        """
        m = make_model({"evaluation_method": "box"})
        assert m._build_messages("題目", "zh", "box", True) == [{"role": "user", "content": "題目"}]


class TestVisionMessages:
    """vision 路徑自己組 messages，是 #144 的另一半。"""

    def test_system_prompt_prepended(self):
        msgs = _build_vision_messages(
            "data:image/jpeg;base64,X", "Q", "auto", system_prompt="用 \\boxed{}"
        )
        assert [x["role"] for x in msgs] == ["system", "user"]
        assert msgs[0]["content"] == "用 \\boxed{}"

    def test_without_system_prompt_matches_previous_shape(self):
        msgs = _build_vision_messages("data:image/jpeg;base64,X", "Q", "auto")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert [c["type"] for c in msgs[0]["content"]] == ["image_url", "text"]

    def test_image_content_unchanged_when_system_prompt_added(self):
        """加入 system message 不得動到原本的 image_url / text 結構。"""
        a = _build_vision_messages("u", "Q", "high")
        b = _build_vision_messages("u", "Q", "high", system_prompt="p")
        assert a[0] == b[1]
