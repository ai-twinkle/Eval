# Apache 2.0 License — adapted from google-research/google-research/instruction_following_eval
from twinkle_eval.metrics.checkers.ifeval.evaluation_lib import (
    test_instruction_following_strict,
    test_instruction_following_loose,
)
from twinkle_eval.metrics.checkers.ifeval.instructions_registry import INSTRUCTION_DICT

__all__ = [
    "test_instruction_following_strict",
    "test_instruction_following_loose",
    "INSTRUCTION_DICT",
]
