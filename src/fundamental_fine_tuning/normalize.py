from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any

from .schemas import ChatExample, Message


def normalize_text(value: str) -> str:
    """Apply conservative, deterministic normalization without rewriting meaning."""
    value = unicodedata.normalize("NFC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    # Trim line edges and collapse excessive horizontal whitespace, preserving newlines.
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in value.split("\n")]
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def canonicalize_record(raw: dict[str, Any]) -> ChatExample:
    """Accept canonical messages or instruction/input/output records."""
    if "messages" in raw:
        source_messages = raw["messages"]
        if not isinstance(source_messages, list):
            raise ValueError("'messages' must be a list")
        messages = []
        for item in source_messages:
            if not isinstance(item, dict):
                raise ValueError("each message must be an object")
            messages.append(
                Message(
                    role=item.get("role"),
                    content=normalize_text(item.get("content", "")),
                )
            )
    elif "instruction" in raw and "output" in raw:
        instruction = normalize_text(str(raw["instruction"]))
        extra_input = normalize_text(str(raw.get("input") or ""))
        output = normalize_text(str(raw["output"]))
        if extra_input:
            user_content = f"{instruction}\n\nAdditional context:\n{extra_input}"
        else:
            user_content = instruction
        system = raw.get("system")
        messages = []
        if system and normalize_text(str(system)):
            messages.append(Message(role="system", content=normalize_text(str(system))))
        messages.extend([
            Message(role="user", content=user_content),
            Message(role="assistant", content=output),
        ])
    else:
        raise ValueError("record must contain 'messages' or both 'instruction' and 'output'")

    metadata_raw = raw.get("metadata") or {}
    if not isinstance(metadata_raw, dict):
        raise ValueError("'metadata' must be an object")
    metadata = {
        str(key): value
        for key, value in metadata_raw.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }

    group_id = raw.get("group_id")
    if group_id is not None:
        group_id = normalize_text(str(group_id))
        if not group_id:
            group_id = None

    return ChatExample(messages=messages, group_id=group_id, metadata=metadata)


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(example: ChatExample) -> str:
    payload = [
        {"role": message.role, "content": normalize_text(message.content)}
        for message in example.messages
    ]
    return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()


def conflict_key(example: ChatExample) -> str:
    """Hash system + user/tool context, excluding assistant answer text."""
    context = []
    for message in example.messages:
        if message.role == "assistant":
            continue
        context.append({"role": message.role, "content": normalize_text(message.content)})
    return hashlib.sha256(stable_json(context).encode("utf-8")).hexdigest()


def answer_signature(example: ChatExample) -> str:
    answers = [
        normalize_text(message.content)
        for message in example.messages
        if message.role == "assistant"
    ]
    return hashlib.sha256(stable_json(answers).encode("utf-8")).hexdigest()


def group_key(example: ChatExample) -> str:
    if example.group_id:
        return "source:" + example.group_id
    return "context:" + conflict_key(example)
