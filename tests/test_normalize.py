from fundamental_fine_tuning.normalize import (
    canonicalize_record,
    conflict_key,
    content_hash,
)


def test_instruction_record_converts_to_chat():
    example = canonicalize_record({
        "instruction": " Explain interfaces ",
        "input": "",
        "output": " A contract ",
    })
    assert [message.role for message in example.messages] == ["user", "assistant"]
    assert example.messages[0].content == "Explain interfaces"
    assert example.messages[1].content == "A contract"


def test_normalization_makes_whitespace_variants_equal():
    one = canonicalize_record({
        "messages": [
            {"role": "user", "content": "What   is PHP?"},
            {"role": "assistant", "content": "A language."},
        ]
    })
    two = canonicalize_record({
        "messages": [
            {"role": "user", "content": "What is PHP?"},
            {"role": "assistant", "content": "A language."},
        ]
    })
    assert content_hash(one) == content_hash(two)


def test_conflict_key_ignores_assistant_answer():
    one = canonicalize_record({
        "messages": [
            {"role": "user", "content": "What is PHP?"},
            {"role": "assistant", "content": "A language."},
        ]
    })
    two = canonicalize_record({
        "messages": [
            {"role": "user", "content": "What is PHP?"},
            {"role": "assistant", "content": "A framework."},
        ]
    })
    assert conflict_key(one) == conflict_key(two)
    assert content_hash(one) != content_hash(two)
