# LLM Dataset Pipeline (Days 1–4)

A Python starter project for preparing instruction/response and conversational JSONL datasets for supervised fine-tuning (SFT). Uses `uv`, Pydantic validation, deterministic normalization, duplicate/conflict detection, group-aware train/validation/test splits, and Hugging Face chat templates.

## Requirements

- Windows 11
- Python 3.13+
- `uv`
- Internet access for the first tokenizer download, unless cached

This project prepares data only. It does not train a model, perform RAG, or decide whether fine-tuning is the right product solution. Those are separate design decisions.

## Setup (PowerShell)

```powershell
uv sync --extra dev
Copy-Item .env.example .env
```

Create your input at `data/raw/examples.jsonl`, one JSON object per line. Two supported shapes:

```json
{"messages":[{"role":"user","content":"What is PHP?"},{"role":"assistant","content":"PHP is a server-side scripting language."}]}
{"instruction":"Explain a PHP interface.","input":"","output":"An interface defines a contract that classes implement."}
```

You may mix these two shapes in one file. `system` messages are optional. Supported roles are `system`, `user`, `assistant`, and `tool`; role order is validated conservatively, so tool-heavy agent traces may need a purpose-built schema.

## Run

```powershell
uv run llm-data prepare data/raw/examples.jsonl --output-dir data/processed --report-dir data/reports
```

To apply the template from a small public instruct model:

```powershell
uv run llm-data prepare data/raw/examples.jsonl `
  --output-dir data/processed `
  --report-dir data/reports `
  --model-id Qwen/Qwen2.5-0.5B-Instruct
```

First run may download model tokenizer files. For offline use, pass a local tokenizer directory as `--model-id`.

To use `MODEL_ID` from `.env` as the default tokenizer source (the CLI `--model-id` flag overrides it), run through uv's env-file option:

```powershell
uv run --env-file .env llm-data prepare data/raw/examples.jsonl
```

Tuning options on `prepare`:
- `--train-ratio`, `--validation-ratio`, `--test-ratio`: split proportions (must sum to 1; default 0.8/0.1/0.1)
- `--seed`: seed for deterministic split assignment (default 42)
- `--max-records`: cap on parseable input records (default 1,000,000)
- `--max-content-chars`: cap on total message characters per record (default 100,000)

Outputs:
- `train.jsonl`, `validation.jsonl`, `test.jsonl`: canonical structured examples
- `train.formatted.jsonl`, etc.: examples with `formatted_text` from the model's chat template (when `--model-id` is supplied)
- `rejected.jsonl`: invalid rows and reasons (contains input data; protect it)
- `conflicts.jsonl`: same normalized conversation context with differing assistant answers
- `manifest.json`: counts, split sizes, hashes, and configuration
- `report.json`: validation and deduplication summary

## Data format and leakage controls

- Input records receive a stable content hash. The hash is used to group identical conversations before splitting.
- Exact duplicate conversations are removed.
- Records with the same normalized user/system context but different assistant responses are reported as conflicts. By default, all examples sharing a conflict key are excluded from the split datasets; review them and correct the source.
- Splitting is deterministic for a given dataset and seed.
- Related examples sharing the same group key stay in the same split. For example, variants from the same source document should share a `group_id`.
- If records include `group_id`, it is used as the split group. Otherwise, the normalized prompt/context hash is used. This is a useful baseline, not a guarantee against every kind of semantic leakage.
- Keep test data isolated from prompt iteration and training. Review it before use.

## Security and production notes

- Do not commit private, customer, regulated, or secret data into the repository.
- Rejected rows and reports may contain sensitive source text. Restrict access and set retention rules.
- The tokenizer is loaded with `trust_remote_code=False`; do not enable remote code without a security review.
- For large corpora, replace in-memory processing with streaming or a database-backed pipeline.
- Have human review for correctness, licensing, privacy, safety, and task coverage before training.
- Tokenization/truncation policy and assistant-only loss masking belong in the later training stage; this project does not silently truncate examples.
