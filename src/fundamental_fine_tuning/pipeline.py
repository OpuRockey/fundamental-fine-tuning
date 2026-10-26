from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from transformers import AutoTokenizer

from .io import read_jsonl, write_json, write_jsonl
from .normalize import (
    answer_signature,
    canonicalize_record,
    conflict_key,
    content_hash,
)
from .schemas import ChatExample
from .split import split_by_group
from .template import format_example


def _example_row(example: ChatExample, *, formatted_text: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "messages": [message.model_dump() for message in example.messages],
        "metadata": example.metadata,
    }
    if example.group_id:
        row["group_id"] = example.group_id
    if formatted_text is not None:
        row["formatted_text"] = formatted_text
    return row


def prepare_dataset(
    input_path: Path,
    output_dir: Path,
    report_dir: Path,
    *,
    model_id: str | None = None,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_records: int = 1_000_000,
    max_content_chars: int = 100_000,
) -> dict[str, Any]:
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    if max_records < 1 or max_content_chars < 1:
        raise ValueError("max_records and max_content_chars must be positive")

    raw_records, rejected = read_jsonl(input_path)
    if len(raw_records) > max_records:
        raise ValueError(
            f"Input has {len(raw_records)} parseable records; configured limit is {max_records}"
        )

    valid: list[ChatExample] = []
    for line_number, raw in raw_records:
        try:
            example = canonicalize_record(raw)
            total_chars = sum(len(message.content) for message in example.messages)
            if total_chars > max_content_chars:
                raise ValueError(
                    f"record has {total_chars} characters; limit is {max_content_chars}"
                )
            valid.append(example)
        except (ValueError, ValidationError, TypeError) as exc:
            rejected.append({
                "line_number": line_number,
                "reason": f"validation_error: {exc}",
                "record": raw,
            })

    # Exact conversation deduplication, preserving first occurrence.
    seen_hashes: set[str] = set()
    deduplicated: list[ChatExample] = []
    duplicate_count = 0
    for example in valid:
        digest = content_hash(example)
        if digest in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(digest)
        deduplicated.append(example)

    # Detect prompts/context with multiple different assistant answers.
    signatures_by_context: dict[str, set[str]] = defaultdict(set)
    examples_by_context: dict[str, list[ChatExample]] = defaultdict(list)
    for example in deduplicated:
        key = conflict_key(example)
        signatures_by_context[key].add(answer_signature(example))
        examples_by_context[key].append(example)

    conflict_keys = {
        key for key, signatures in signatures_by_context.items()
        if len(signatures) > 1
    }
    conflicts: list[dict[str, Any]] = []
    for key in sorted(conflict_keys):
        conflicts.append({
            "conflict_key": key,
            "examples": [
                _example_row(example)
                for example in examples_by_context[key]
            ],
            "action": "excluded_from_splits_for_manual_review",
        })

    clean = [
        example for example in deduplicated
        if conflict_key(example) not in conflict_keys
    ]

    splits = split_by_group(
        clean,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    tokenizer = None
    if model_id:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=False,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    formatted_counts: dict[str, int] = {}
    for split_name, examples in splits.items():
        rows = [_example_row(example) for example in examples]
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)

        if tokenizer is not None:
            formatted_rows = [
                _example_row(example, formatted_text=format_example(tokenizer, example))
                for example in examples
            ]
            write_jsonl(output_dir / f"{split_name}.formatted.jsonl", formatted_rows)
            formatted_counts[split_name] = len(formatted_rows)

    write_jsonl(output_dir / "rejected.jsonl", rejected)
    write_jsonl(output_dir / "conflicts.jsonl", conflicts)

    report = {
        "input_file": str(input_path),
        "model_id": model_id,
        "counts": {
            "parsed_records": len(raw_records),
            "rejected_records": len(rejected),
            "valid_before_deduplication": len(valid),
            "exact_duplicates_removed": duplicate_count,
            "unique_after_deduplication": len(deduplicated),
            "conflicting_context_groups": len(conflict_keys),
            "examples_excluded_due_to_conflicts": sum(
                len(examples_by_context[key]) for key in conflict_keys
            ),
            "examples_eligible_for_splits": len(clean),
            "splits": {name: len(examples) for name, examples in splits.items()},
            "formatted": formatted_counts,
        },
        "split_config": {
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "grouping": "group_id if present, otherwise normalized conversation context",
        },
        "warnings": [
            "This pipeline detects exact context conflicts, not semantic contradictions.",
            "Group-aware splitting reduces leakage but cannot guarantee semantic independence.",
            "Inspect rejected.jsonl and conflicts.jsonl before training.",
            "No truncation, token-length filtering, training, or loss masking is performed.",
        ],
    }
    write_json(report_dir / "report.json", report)

    input_sha256 = hashlib.sha256(input_path.read_bytes()).hexdigest()
    manifest = {
        "input_sha256": input_sha256,
        "report_file": str(report_dir / "report.json"),
        "model_id": model_id,
        "seed": seed,
        "output_files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
        "counts": report["counts"],
    }
    write_json(report_dir / "manifest.json", manifest)
    return report
