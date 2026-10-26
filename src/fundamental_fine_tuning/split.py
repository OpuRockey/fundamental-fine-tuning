from __future__ import annotations

import hashlib
from collections import defaultdict

from .normalize import group_key
from .schemas import ChatExample


def _stable_bucket(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / (2**64)


def split_by_group(
    examples: list[ChatExample],
    *,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[ChatExample]]:
    ratios = [train_ratio, validation_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("split ratios cannot be negative")
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("train, validation, and test ratios must sum to 1")
    if train_ratio <= 0:
        raise ValueError("train ratio must be greater than zero")

    groups: dict[str, list[ChatExample]] = defaultdict(list)
    for example in examples:
        groups[group_key(example)].append(example)

    result: dict[str, list[ChatExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    validation_boundary = train_ratio
    test_boundary = train_ratio + validation_ratio

    for key in sorted(groups):
        bucket = _stable_bucket(key, seed)
        if bucket < validation_boundary:
            split_name = "train"
        elif bucket < test_boundary:
            split_name = "validation"
        else:
            split_name = "test"
        result[split_name].extend(groups[key])

    for split_name, members in result.items():
        members.sort(key=lambda example: group_key(example))
    return result
