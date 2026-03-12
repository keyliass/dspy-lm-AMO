# dspy-lm-auth: GEPA Test Fork

[![CI](https://github.com/MaximeRivest/dspy-lm-auth/actions/workflows/ci.yml/badge.svg)](https://github.com/MaximeRivest/dspy-lm-auth/actions/workflows/ci.yml) [![PyPI version](https://img.shields.io/pypi/v/dspy-lm-auth.svg)](https://pypi.org/project/dspy-lm-auth/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MaximeRivest/dspy-lm-auth/blob/main/LICENSE)

This fork is used as a simple GEPA test base on in-house datasets.

The underlying package still does the same core job:
- patch `dspy.LM` so `codex/...` routes work in DSPy
- reuse Pi-style credentials from `~/.pi/agent/auth.json`
- let you use a stronger subscription-backed reflection model for GEPA

The difference in this fork is the intent:
- keep the auth helper
- keep the local-student plus stronger-reflection workflow
- add a small house dataset layer under [`datasets/`](datasets/)
- use the repo as a practical sandbox for GEPA iterations

## Repository Goal

This repository is meant to be a lightweight place to test:
- a local or cheap **student model**
- `codex/gpt-5.4` as the **reflection model**
- GEPA on small structured in-house datasets

The current starter dataset is:
- [`datasets/afso_requirements_gepa_v1.jsonl`](datasets/afso_requirements_gepa_v1.jsonl)

It is intentionally small and stable so prompt optimization can be tested quickly before moving to larger internal datasets.

## Install

```bash
uv sync --extra dev
```

Or with `pip`:

```bash
pip install -e .[dev]
```

## One-Time Login

If Pi credentials are already present in `~/.pi/agent/auth.json`, nothing else is needed.

Otherwise:

```python
import dspy_lm_auth

dspy_lm_auth.login("codex")
```

This starts the OAuth flow and stores the credential in Pi's auth file.

## Minimal GEPA Setup

The intended workflow is:
1. run a small student model locally
2. use `codex/gpt-5.4` as the reflection model
3. optimize a DSPy program against a small house dataset

Example setup:

```python
import dspy
import dspy_lm_auth

dspy_lm_auth.install()

student_lm = dspy.LM(
    "ollama_chat/qwen3.5:0.8b",
    api_base="http://127.0.0.1:11434",
    api_key="ollama",
    model_type="chat",
    think=False,
    temperature=0,
)

reflection_lm = dspy.LM("codex/gpt-5.4")

dspy.configure(lm=student_lm, adapter=dspy.JSONAdapter())
```

## Dataset Shape

The house dataset in this fork is JSONL.

Each row is designed for a simple GEPA-friendly task:
- input: `question` plus retrieved `context`
- output: short `answer` plus structured requirement fields

Typical fields:
- `id`
- `split`
- `category`
- `question`
- `context`
- `answer`
- `theme`
- `requirement_type`
- `value`
- `unit`
- `applies_to`
- `source_doc`
- `source_pages`

This shape is deliberate:
- it keeps scoring simple
- it gives GEPA clean textual feedback
- it is much easier to validate than open-ended summaries

## Simple GEPA Pattern

Below is a minimal pattern for using a local JSONL dataset.

```python
import json
from pathlib import Path

import dspy
import dspy_lm_auth


dspy_lm_auth.install()

student_lm = dspy.LM(
    "ollama_chat/qwen3.5:0.8b",
    api_base="http://127.0.0.1:11434",
    api_key="ollama",
    model_type="chat",
    think=False,
    temperature=0,
)

reflection_lm = dspy.LM("codex/gpt-5.4")

dspy.configure(lm=student_lm, adapter=dspy.JSONAdapter())


class RequirementAnswer(dspy.Signature):
    """Answer from retrieved project context with a short, structured result."""

    question: str = dspy.InputField()
    context: str = dspy.InputField()
    answer: str = dspy.OutputField()
    theme: str = dspy.OutputField()
    requirement_type: str = dspy.OutputField()
    value: str = dspy.OutputField()
    unit: str = dspy.OutputField()
    applies_to: str = dspy.OutputField()


program = dspy.Predict(RequirementAnswer)


def load_examples(path: str):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    examples = [
        dspy.Example(
            question=row["question"],
            context=row["context"],
            answer=row["answer"],
            theme=row["theme"],
            requirement_type=row["requirement_type"],
            value=row["value"],
            unit=row["unit"],
            applies_to=row["applies_to"],
        ).with_inputs("question", "context")
        for row in rows
    ]
    trainset = [x for x, row in zip(examples, rows) if row["split"] == "train"]
    valset = [x for x, row in zip(examples, rows) if row["split"] == "val"]
    return trainset, valset


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    checks = {
        "answer": gold.answer.strip().lower() == pred.answer.strip().lower(),
        "theme": gold.theme.strip().lower() == pred.theme.strip().lower(),
        "requirement_type": gold.requirement_type.strip().lower() == pred.requirement_type.strip().lower(),
        "value": gold.value.strip().lower() == pred.value.strip().lower(),
        "unit": gold.unit.strip().lower() == pred.unit.strip().lower(),
        "applies_to": gold.applies_to.strip().lower() == pred.applies_to.strip().lower(),
    }
    score = sum(checks.values()) / len(checks)
    failures = [name for name, ok in checks.items() if not ok]
    feedback = "All fields correct." if not failures else f"Incorrect fields: {', '.join(failures)}."
    return dspy.Prediction(score=score, feedback=feedback)


trainset, valset = load_examples("datasets/afso_requirements_gepa_v1.jsonl")

gepa = dspy.GEPA(
    metric=metric,
    reflection_lm=reflection_lm,
    auto="light",
)

optimized = gepa.compile(program, trainset=trainset, valset=valset)
```

## What This Fork Is For

This fork is a practical base for:
- quickly validating whether GEPA improves a structured QA prompt
- checking whether a local student model is good enough for a narrow internal task
- iterating on small datasets before scaling annotation effort

It is not trying to be a polished benchmark suite or a general-purpose RAG framework.

## If You Only Want The Auth Piece

You can still use the package on its own.

```python
import dspy
import dspy_lm_auth

dspy_lm_auth.install()

lm = dspy.LM("codex/gpt-5.4")
dspy.configure(lm=lm)

print(lm("hello")[0]["text"])
```

Or explicitly keep the original provider name and select the auth route:

```python
import dspy_lm_auth

lm = dspy_lm_auth.LM("openai/gpt-5.4", auth_provider="codex")
print(lm("hello")[0]["text"])
```

## Credential Resolution

API key credentials can be stored as:
- a literal value
- an environment variable name
- a shell lookup prefixed with `!`

Example:

```json
{
  "some-provider": {
    "type": "api_key",
    "key": "OPENAI_API_KEY"
  }
}
```

## Development

```bash
uv sync --extra dev
.venv\Scripts\python.exe -m pytest
```

## License

MIT
