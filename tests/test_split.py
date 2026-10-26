from fundamental_fine_tuning.normalize import group_key
from fundamental_fine_tuning.schemas import ChatExample, Message
from fundamental_fine_tuning.split import split_by_group


def make_example(group_id: str, prompt: str) -> ChatExample:
    return ChatExample(
        group_id=group_id,
        messages=[
            Message(role="user", content=prompt),
            Message(role="assistant", content="Answer."),
        ],
    )


def test_group_ids_do_not_cross_splits():
    examples = [make_example(f"group-{i}", f"Question {i}") for i in range(100)]
    splits = split_by_group(examples, seed=7)
    membership = {}
    for split_name, split_examples in splits.items():
        for example in split_examples:
            membership.setdefault(group_key(example), set()).add(split_name)
    assert all(len(split_names) == 1 for split_names in membership.values())


def test_split_ratios_must_sum_to_one():
    examples = [make_example("a", "Question")]
    try:
        split_by_group(
            examples,
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.2,
        )
    except ValueError as exc:
        assert "sum to 1" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
