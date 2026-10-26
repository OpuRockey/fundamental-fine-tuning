import json
from pathlib import Path

from fundamental_fine_tuning.io import read_jsonl
from fundamental_fine_tuning.pipeline import prepare_dataset


def group_record(group_id: str, user: str, assistant: str) -> str:
    return json.dumps({
        "group_id": group_id,
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    })


def plain_record(user: str, assistant: str) -> str:
    return json.dumps({
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    })


def read_rows(path: Path) -> list[dict]:
    records, rejected = read_jsonl(path)
    assert rejected == []
    return [record[1] for record in records]


def test_prepare_dataset_end_to_end(tmp_path):
    input_path = tmp_path / "examples.jsonl"
    input_path.write_text(
        "\n".join([
            group_record("g1", "Question one", "Answer one"),
            plain_record("Question two", "Answer two"),
            group_record("g2", "Question three", "Answer three"),
            group_record("g1", "Question one", "Answer one "),  # duplicate after normalization
            plain_record("Conflicting prompt", "Answer A"),
            plain_record("Conflicting prompt", "Answer B"),  # same context, different answer
            json.dumps({"messages": [{"role": "user", "content": "no answer"}]}),
            "{ not valid json",
        ]) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "processed"
    report_dir = tmp_path / "reports"
    report = prepare_dataset(
        input_path=input_path,
        output_dir=output_dir,
        report_dir=report_dir,
    )

    counts = report["counts"]
    assert counts["parsed_records"] == 7  # 8 lines, one malformed
    assert counts["rejected_records"] == 2  # malformed JSON + missing assistant
    assert counts["valid_before_deduplication"] == 6
    assert counts["exact_duplicates_removed"] == 1
    assert counts["unique_after_deduplication"] == 5
    assert counts["conflicting_context_groups"] == 1
    assert counts["examples_excluded_due_to_conflicts"] == 2
    assert counts["examples_eligible_for_splits"] == 3
    assert sum(counts["splits"].values()) == 3

    rows_by_split = {
        name: read_rows(output_dir / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    assert not any(
        "formatted_text" in row for rows in rows_by_split.values() for row in rows
    )
    assert not list(output_dir.glob("*.formatted.jsonl"))

    # Grouped records must not cross splits.
    split_of_group: dict[str, str] = {}
    for name, rows in rows_by_split.items():
        for row in rows:
            group_id = row.get("group_id")
            if group_id is not None:
                assert split_of_group.setdefault(group_id, name) == name

    rejected_rows = read_rows(output_dir / "rejected.jsonl")
    assert len(rejected_rows) == 2
    assert any("invalid_jsonl" in row["reason"] for row in rejected_rows)
    assert any("validation_error" in row["reason"] for row in rejected_rows)
    assert not any(
        "Question one" in json.dumps(row) for row in rejected_rows
    )

    conflicts = read_rows(output_dir / "conflicts.jsonl")
    assert len(conflicts) == 1
    assert len(conflicts[0]["examples"]) == 2
    assert conflicts[0]["action"] == "excluded_from_splits_for_manual_review"


def test_prepare_dataset_rejects_zero_eligible_no_crash(tmp_path):
    input_path = tmp_path / "examples.jsonl"
    input_path.write_text(
        json.dumps({"messages": [{"role": "user", "content": "no answer"}]}) + "\n",
        encoding="utf-8",
    )
    report = prepare_dataset(
        input_path=input_path,
        output_dir=tmp_path / "processed",
        report_dir=tmp_path / "reports",
    )
    assert report["counts"]["examples_eligible_for_splits"] == 0
    assert all(
        (tmp_path / "processed" / f"{name}.jsonl").exists()
        for name in ("train", "validation", "test")
    )
