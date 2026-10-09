# Multi-Agent Workflow Engine

A workflow engine written directly, without LangGraph or CrewAI. Independent steps run together. A reviewer can send a draft back, and the loop stops at `max_revisions`.

## Overview

Several roles share one dependency graph. `retrieve` and `outline` have no dependency on each other, so they run in one layer. `analyse`, `draft`, and `review` follow. If the review rejects the draft, the engine clears the outputs that must be recomputed and runs the draft again with the feedback on the blackboard.

The scripted backend needs no API key. Set `ANTHROPIC_API_KEY` to swap in the hosted model. The orchestration code does not change.

## Features

- Topological layers and a thread pool for steps that do not depend on each other
- Cycles and unknown dependencies fail when the graph is built
- Bounded revision loops via `Revision(target, feedback)`
- Retries up to `max_attempts`, with every attempt traced
- A thread-safe blackboard that rejects a second write to the same key
- Per-step status, attempt, duration, and token count

## Technology Stack

- Python
- FastAPI
- pytest

## Architecture

Agents build a prompt, call a backend, and return. Order, retry, and review live in the engine, so the same agent can be unit-tested and reused on another graph.

The review parser extracts the first JSON object, then looks for an explicit approval word. If it still cannot tell, it does not approve. A confused reviewer must not accept a draft by accident.

Tool steps sit on the same graph as model steps. Retrieval costs zero tokens and still has retry, tracing, and dependencies.

```
retrieve ─┐
          ├─→ analyse ─→ draft ─→ review ─┐
outline  ─┘                 ↑              │
                            └── revise ────┘
```

| Step | Kind | Job |
| --- | --- | --- |
| `retrieve` | tool | Search the document set. No model call |
| `outline` | model | Plan the report sections |
| `analyse` | model | Findings from the retrieved passages only |
| `draft` | model | Write from the findings and the outline |
| `review` | model | Score against a checklist. Approve or send back |

## Installation

```bash
python -m venv .venv
```

Windows: `.venv\Scripts\activate`. Linux or macOS: `source .venv/bin/activate`.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python demo.py
uvicorn workflow.api:app --reload
```

The demo runs the scripted backend. In the recorded sample the first draft was 620 words, over a 400 word limit. The writer cut it, and the second review approved it. An unrelated office note in the corpus was not retrieved.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Active backend and registered tools |
| `GET` | `/workflow` | The graph: steps, dependencies, execution layers |
| `POST` | `/run` | Run the workflow and return the report plus the full trace |

## Testing

```bash
pytest tests -q
```

59 tests cover the blackboard, topology, parallel execution, retries, the revision loop, tools, agents, review parsing, the pipeline, and the API.

## Limitations

The scripted backend is deterministic stand-in text, not a language model. The hosted backend is used only when the key is present. This repository is the engine and one research-and-report pipeline, not a general agent product.

## License

MIT
