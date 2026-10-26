import json

from fundamental_fine_tuning.io import read_jsonl, write_json, write_jsonl


def test_write_then_read_jsonl_round_trips(tmp_path):
    path = tmp_path / "out.jsonl"
    rows = [{"a": 1}, {"b": [1, 2], "c": "ünicode"}]
    write_jsonl(path, rows)
    records, rejected = read_jsonl(path)
    assert [record[1] for record in records] == rows
    assert [record[0] for record in records] == [1, 2]
    assert rejected == []


def test_read_jsonl_skips_blank_lines_and_reports_bad_lines(tmp_path):
    path = tmp_path / "in.jsonl"
    path.write_text('{"a": 1}\n\nnot json\n[1, 2]\n', encoding="utf-8")
    records, rejected = read_jsonl(path)
    assert [record[1] for record in records] == [{"a": 1}]
    assert [row["line_number"] for row in rejected] == [3, 4]
    assert all(row["reason"].startswith("invalid_jsonl:") for row in rejected)


def test_write_json_sorts_keys_and_appends_newline(tmp_path):
    path = tmp_path / "out.json"
    write_json(path, {"b": 1, "a": 2})
    content = path.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert content.index('"a"') < content.index('"b"')
    assert json.loads(content) == {"b": 1, "a": 2}
