"""System prompt 的解析。

獨立成模組是為了避免循環 import：``core/config.py`` 需要匯入 ``models``
取得 ``LLMFactory``，而 ``models`` 又需要這個函式。本模組只依賴 typing。
"""

from typing import Any, Dict, Optional

_MISSING = object()


def resolve_system_prompt(
    eval_config: Dict[str, Any],
    prompt_lang: str,
    system_prompt_enabled: bool = True,
) -> Optional[str]:
    """從 evaluation 設定解析出要送出的 system prompt。

    語言鍵的解析規則刻意區分「鍵不存在」與「鍵存在但為空」：

    - 鍵**不存在** → 依序回退 ``zh``、再回退唯一已定義的語言。
      這讓只定義了 ``en`` 的設定（例如 ``templates/regex_match.yaml``）
      在預設 ``prompt_lang="zh"`` 下仍能送出 prompt。
    - 鍵**存在但為 None 或空白** → 視為該語言明確不要 prompt，回傳 None，
      **不跨語言回退**。否則 ``{zh: "中文", en: null}`` 搭配英文資料集
      會把中文 prompt 送給英文題目。

    Args:
        eval_config:           config 的 ``evaluation`` 區塊。
        prompt_lang:           語言代碼。
        system_prompt_enabled: 為 False 時一律不送。

    Returns:
        要送出的 system prompt；未設定、為空或被停用時回傳 None。
    """
    if not system_prompt_enabled:
        return None

    cfg = eval_config.get("system_prompt")
    if cfg is None:
        return None
    if not isinstance(cfg, dict):
        text = str(cfg).strip()
        return text or None

    value = cfg.get(prompt_lang, _MISSING)
    if value is _MISSING:
        value = cfg.get("zh", _MISSING)
    if value is _MISSING:
        defined = [v for v in cfg.values() if v is not None and str(v).strip()]
        value = defined[0] if len(defined) == 1 else _MISSING
    if value is _MISSING or value is None:
        return None

    text = str(value).strip()
    return text or None
