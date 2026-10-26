from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    role: Role
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message content must not be blank")
        return value


class ChatExample(BaseModel):
    """Canonical single-example conversation schema."""

    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=2)
    group_id: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_conversation_shape(self) -> ChatExample:
        roles = [message.role for message in self.messages]
        if "assistant" not in roles:
            raise ValueError("conversation must contain at least one assistant message")
        if roles[-1] != "assistant":
            raise ValueError("conversation must end with an assistant message for SFT")
        if not any(role == "user" for role in roles):
            raise ValueError("conversation must contain at least one user message")

        # Conservative common SFT sequence: optional system, then user/assistant turns.
        # Tool calls/traces need a model-specific schema and validation policy.
        if roles.count("system") > 1:
            raise ValueError("at most one system message is supported")
        if "system" in roles and roles[0] != "system":
            raise ValueError("system message, when present, must be first")

        for index, role in enumerate(roles):
            if (
                role == "user"
                and index + 1 < len(roles)
                and roles[index + 1] not in {"assistant", "tool"}
            ):
                raise ValueError("user message must be followed by assistant or tool")
            if (
                role == "assistant"
                and index + 1 < len(roles)
                and roles[index + 1] not in {"user", "tool"}
            ):
                raise ValueError("assistant message must be followed by user or tool")
        return self
