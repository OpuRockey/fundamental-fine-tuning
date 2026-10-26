from __future__ import annotations

from typing import Any

from .schemas import ChatExample


def format_example(tokenizer: Any, example: ChatExample) -> str:
    """Format a conversation using the selected tokenizer's official chat template."""
    messages = [
        {"role": message.role, "content": message.content}
        for message in example.messages
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(
            "Could not apply the tokenizer's chat template. Confirm that this tokenizer "
            "supports chat templates and that the conversation roles match its template."
        ) from exc
