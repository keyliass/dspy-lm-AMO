# dspy-lm-auth

[![CI](https://github.com/MaximeRivest/dspy-lm-auth/actions/workflows/ci.yml/badge.svg)](https://github.com/MaximeRivest/dspy-lm-auth/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/dspy-lm-auth.svg)](https://pypi.org/project/dspy-lm-auth/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MaximeRivest/dspy-lm-auth/blob/main/LICENSE)

Pi-style LM authentication helpers for DSPy.

`dspy-lm-auth` makes it easy to reuse Pi-style credentials with `dspy.LM`, including ChatGPT Codex subscription auth.

## What it does

- reuses Pi credentials from `~/.pi/agent/auth.json`
- resolves provider config values from:
  - literal strings
  - environment variable names
  - `!shell command` lookups
- supports OAuth login and token refresh flows for subscription-backed providers
- patches `dspy.LM` so model aliases and alternate auth routes work out of the box

## Current support

- OpenAI Codex / ChatGPT Plus or Pro subscription

## Install

```bash
pip install dspy-lm-auth
```

Or with `uv`:

```bash
uv pip install dspy-lm-auth
```

## Quick start

```python
import dspy
import dspy_lm_auth

# Optional: patch dspy.LM in place.
dspy_lm_auth.install()

# Reuse Pi's ChatGPT Codex login from ~/.pi/agent/auth.json.
lm = dspy.LM("codex/gpt-5.4")
dspy.configure(lm=lm)

print(lm("hello")[0]["text"])
```

You can also keep the original model string and apply the Codex auth route explicitly:

```python
import dspy_lm_auth

lm = dspy_lm_auth.LM("openai/gpt-5.4", auth_provider="codex")
print(lm("hello")[0]["text"])
```

## A very cheap laptop stack: uv + llama-cpp + baguettotron + dspy

People often ask for a free stack to try DSPy locally on a laptop.

A nice minimal stack is:

- `uv` for env management
- `llama-cpp-python[server]` for OpenAI-compatible local serving
- [Baguettotron-GGUF](https://huggingface.co/P1eIAS/Baguettotron-GGUF) as the local model
- DSPy for the program
- `dspy-lm-auth` for the Codex reflection model when you want a stronger optimizer / teacher model

Start a local OpenAI-compatible server with `llama-cpp`:

```bash
uv venv
source .venv/bin/activate
uv pip install "llama-cpp-python[server]" huggingface-hub dspy dspy-lm-auth

uv run python -m llama_cpp.server \
  --host 127.0.0.1 \
  --port 8000 \
  --hf_model_repo_id P1eIAS/Baguettotron-GGUF \
  --model Baguettotron-BF16.gguf
```

If that GGUF is too large for your machine, swap in a smaller GGUF from the same repo.

Then connect to it from DSPy:

```python
import dspy

local_lm = dspy.LM(
    "openai/local-model",
    api_base="http://localhost:8000/v1",
    api_key="",
    model_type="chat",
)

dspy.configure(lm=local_lm, adapter=dspy.JSONAdapter())
```

## Tiny GEPA demo: optimize a French→English translator

This is a small self-contained demo that:

- uses a local `llama-cpp` server running Baguettotron as the **student model**
- uses your ChatGPT Codex subscription via `dspy-lm-auth` as the **GEPA reflection model**
- optimizes a tiny translator on just **10 French→English examples**

```python
import dspy
import dspy_lm_auth

# Patch dspy.LM so `codex/...` works.
dspy_lm_auth.install()

# Student model: local llama-cpp server.
student_lm = dspy.LM(
    "openai/local-model",
    api_base="http://localhost:8000/v1",
    api_key="",
    model_type="chat",
)

# Reflection model: stronger model used by GEPA to improve the prompt.
reflection_lm = dspy.LM("codex/gpt-5.4")

# DSPy program inference uses the cheap local model.
dspy.configure(lm=student_lm, adapter=dspy.JSONAdapter())


class TranslateFrenchToEnglish(dspy.Signature):
    """Translate the French input into short, natural English."""

    french: str = dspy.InputField(desc="French sentence")
    english: str = dspy.OutputField(desc="Natural English translation")


translator = dspy.Predict(TranslateFrenchToEnglish)

pairs = [
    ("bonjour", "hello"),
    ("merci beaucoup", "thank you very much"),
    ("où est la gare ?", "where is the train station?"),
    ("je suis fatigué", "I am tired"),
    ("il fait très chaud aujourd'hui", "it is very hot today"),
    ("je ne comprends pas", "I do not understand"),
    ("pouvez-vous m'aider ?", "can you help me?"),
    ("j'aime apprendre le français", "I like learning French"),
    ("nous arrivons demain matin", "we are arriving tomorrow morning"),
    ("combien ça coûte ?", "how much does it cost?"),
]

examples = [
    dspy.Example(french=fr, english=en).with_inputs("french")
    for fr, en in pairs
]

trainset = examples[:8]
valset = examples[8:]


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    guess = pred.english.strip()
    target = gold.english.strip()

    exact = guess.lower() == target.lower()
    score = 1.0 if exact else 0.0

    if exact:
        feedback = (
            "Exact match. Keep translations short, natural, and direct. "
            "Do not add explanations."
        )
    else:
        feedback = (
            f"Expected {target!r} but got {guess!r}. "
            "Prefer direct, idiomatic English. Preserve tense, pronouns, and politeness. "
            "Do not explain the translation or add extra words."
        )

    return dspy.Prediction(score=score, feedback=feedback)


gepa = dspy.GEPA(
    metric=metric,
    reflection_lm=reflection_lm,
    auto="light",
    track_stats=True,
)

optimized = gepa.compile(translator, trainset=trainset, valset=valset)

print("Optimized instruction:\n")
print(optimized.signature.instructions)
print()

print(optimized(french="je ne comprends pas").english)
print(optimized(french="combien ça coûte ?").english)
```

Notes:

- `student_lm` is the cheap local model you serve yourself.
- `reflection_lm` is the stronger model GEPA uses to improve the prompt.
- `auto="light"` keeps the demo small enough for a laptop workflow.
- if your local server requires a specific model name, replace `openai/local-model` with the one exposed by your `llama-cpp` server.

## Login

If you do not already have credentials stored in Pi's auth file:

```python
import dspy_lm_auth

# Starts the OAuth flow and writes credentials to ~/.pi/agent/auth.json.
dspy_lm_auth.login("codex")
```

## Credential resolution

API key credentials can be stored as:

- a literal value
- an environment variable name
- a shell lookup prefixed with `!`

Examples:

```json
{
  "some-provider": {
    "type": "api_key",
    "key": "OPENAI_API_KEY"
  }
}
```

```json
{
  "some-provider": {
    "type": "api_key",
    "key": "!op read op://Private/openai/api_key --no-newline"
  }
}
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src tests
```

## Roadmap

The package is structured so more Pi-like providers can be added later, for example:

- Anthropic subscription auth
- GitHub Copilot
- Gemini CLI
- Antigravity

## License

MIT
