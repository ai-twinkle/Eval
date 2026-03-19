"""通用 Registry（工廠登錄表）實作。"""

from typing import Any, Dict, Generic, List, Type, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """通用的工廠登錄表，支援依名稱動態建立實例。

    使用方式：
        my_registry: Registry[MyABC] = Registry("my_registry")
        my_registry.register("foo", FooImpl)
        instance = my_registry.create("foo", config)
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._registry: Dict[str, Type[T]] = {}

    def register(self, key: str, cls: Type[T]) -> None:
        """向登錄表新增實作類別。

        Args:
            key: 識別名稱（config 中使用的字串）
            cls: 實作類別
        """
        self._registry[key] = cls

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        """依名稱建立實例。

        Args:
            key: 實作識別名稱
            *args: 傳遞給建構子的位置參數
            **kwargs: 傳遞給建構子的關鍵字參數

        Returns:
            T: 建立的實例

        Raises:
            KeyError: 若名稱不在登錄表中
        """
        if key not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"{self._name}: '{key}' 不存在。可用項目: {available}")
        return self._registry[key](*args, **kwargs)

    def get_available(self) -> List[str]:
        """回傳所有已登錄的名稱列表。"""
        return list(self._registry.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._registry
