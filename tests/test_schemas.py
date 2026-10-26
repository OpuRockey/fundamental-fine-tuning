import pytest
from pydantic import ValidationError

from fundamental_fine_tuning.schemas import ChatExample, Message


def make_messages(*roles: str) -> list[Message]:
    return [Message(role=role, content=f"{role} content") for role in roles]


def test_minimal_user_assistant_is_valid():
    example = ChatExample(messages=make_messages("user", "assistant"))
    assert example.messages[0].role == "user"
    assert example.messages[1].role == "assistant"


def test_missing_assistant_is_rejected():
    with pytest.raises(ValidationError):
        ChatExample(messages=make_messages("user"))


def test_missing_user_is_rejected():
    with pytest.raises(ValidationError):
        ChatExample(messages=make_messages("system", "tool", "assistant"))


def test_conversation_must_end_with_assistant():
    with pytest.raises(ValidationError):
        ChatExample(messages=make_messages("user", "assistant", "user"))


def test_system_message_must_be_first():
    with pytest.raises(ValidationError):
        ChatExample(messages=make_messages("tool", "system", "user", "assistant"))


def test_at_most_one_system_message():
    with pytest.raises(ValidationError):
        ChatExample(messages=make_messages("system", "user", "assistant", "system"))


def test_user_must_be_followed_by_assistant_or_tool():
    with pytest.raises(ValidationError):
        ChatExample(messages=make_messages("user", "user", "assistant"))


def test_assistant_must_be_followed_by_user_or_tool():
    with pytest.raises(ValidationError):
        ChatExample(messages=make_messages("user", "assistant", "assistant"))


def test_blank_content_is_rejected():
    with pytest.raises(ValidationError):
        Message(role="assistant", content="   ")


def test_extra_fields_are_forbidden():
    with pytest.raises(ValidationError):
        ChatExample(messages=make_messages("user", "assistant"), extra=1)


def test_tool_turns_are_permitted():
    example = ChatExample(
        messages=make_messages("system", "user", "tool", "assistant"),
    )
    assert [message.role for message in example.messages] == [
        "system", "user", "tool", "assistant",
    ]
