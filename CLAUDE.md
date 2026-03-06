# AutoRAC

AI-assisted RAC encoding infrastructure. Fully self-contained -- `pip install autorac` is all you need.

## Quick start

```bash
pip install autorac
autorac encode "26 USC 21"
```

This runs the full pipeline: analyze statute, encode subsections, validate against oracles (PolicyEngine/TAXSIM), run 4 LLM reviewers, and log everything.

## Two backends

### 1. CLI backend (default)

Uses Claude Code CLI subprocess. Works with Max subscription. No API billing.

```bash
autorac encode "26 USC 32"                  # default --backend cli
autorac encode "26 USC 32" --backend cli    # explicit
```

### 2. API backend

Uses Claude API directly (anthropic SDK). Requires `ANTHROPIC_API_KEY`. Works on Modal or any server.

```bash
autorac encode "26 USC 32" --backend api
```

```python
from autorac import Orchestrator

orchestrator = Orchestrator(backend="api", model="claude-opus-4-6")
run = await orchestrator.encode("26 USC 21")
```

For batch encoding:

```python
from autorac import AgentSDKBackend, EncoderRequest
from pathlib import Path

backend = AgentSDKBackend()  # Requires ANTHROPIC_API_KEY

requests = [
    EncoderRequest(
        citation=f"26 USC {section}",
        statute_text=texts[section],
        output_path=Path(f"rac-us/statute/26/{section}.rac"),
    )
    for section in sections
]

responses = await backend.encode_batch(requests, max_concurrent=10)
```

## Architecture

```
+-----------------------------------------------------------------+
|                          AutoRAC                                  |
+-----------------------------------------------------------------+
|  Orchestrator (self-contained, embedded prompts)                  |
|    Phase 1: Analysis (subsection tree, encoding order)           |
|    Phase 2: Encoding (parallel per-subsection)                   |
|    Phase 3: Oracle validation (PE + TAXSIM)                      |
|    Phase 4: LLM review (4 reviewers in parallel)                 |
|    Phase 5: Logging and reporting                                |
+-----------------------------------------------------------------+
|  Embedded Prompts (src/autorac/prompts/)                          |
|    encoder.py, validator.py, reviewers.py                        |
+-----------------------------------------------------------------+
|  Encoder Backends                                                 |
|    ClaudeCodeBackend (subprocess)                                |
|    AgentSDKBackend (API, parallelization)                        |
+-----------------------------------------------------------------+
|  3-Tier Validator Pipeline                                        |
|    Tier 1: CI (rac pytest) - instant, catches syntax             |
|    Tier 2: Oracles (PE/TAXSIM) - fast ~10s, comparison data     |
|    Tier 3: LLM reviewers - uses oracle context to diagnose       |
+-----------------------------------------------------------------+
|  Encoding DB (SQLite)                                             |
+-----------------------------------------------------------------+
```

## Components

- `src/autorac/prompts/` - Embedded agent prompts (encoder, validator, 4 reviewers)
- `src/autorac/harness/orchestrator.py` - Main pipeline orchestrator
- `src/autorac/harness/backends.py` - Encoder backends (ClaudeCode, AgentSDK)
- `src/autorac/harness/encoding_db.py` - SQLite encoding session logging
- `src/autorac/harness/validator_pipeline.py` - Parallel validator execution
- `src/autorac/harness/encoder_harness.py` - Low-level encoder harness
- `src/autorac/harness/metrics.py` - Calibration computation

## Commands

```bash
# Setup
pip install -e .

# Encode a statute (full pipeline)
autorac encode "26 USC 21"

# Validate a .rac file
autorac validate path/to/file.rac

# Run tests
pytest tests/ -v
```

## Related repos

- **rac** - DSL parser, executor, runtime
- **rac-us** - US statute encodings
- **rac-validators** - External calculator validation (PolicyEngine, TAXSIM)
