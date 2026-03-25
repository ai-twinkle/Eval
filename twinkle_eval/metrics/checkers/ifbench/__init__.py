"""IFBench checkers — 58 種 OOD instruction checkers。

移植自 AllenAI IFBench (Apache 2.0):
https://github.com/allenai/IFBench
"""

from .instructions_registry import INSTRUCTION_DICT

__all__ = ["INSTRUCTION_DICT"]
