# Multi-Agent Workflow Engine

Several agents with different roles, executed over a dependency graph:
independent steps run in parallel, failures are retried in isolation, and a
reviewer can send the run **backwards** to be redrafted.

That backwards edge is the point. A linear chain of LLM calls is a pipeline; a
graph that can revisit an earlier step based on a later one's judgement is a
workflow.

**Written from scratch — no LangGraph, no CrewAI.** The scheduling, the
revision loop, and the tracing are about 300 lines, and the whole thing runs
with no API key.

---

## The pipeline

```
retrieve ─┐
          ├─→ analyse ─→ draft ─→ review ─┐
outline  ─┘                 ↑              │
                            └── revise ────┘   (bounded by max_revisions)
```

| Step | Agent | Does |
|---|---|---|
| `retrieve` | tool | Searches the corpus — no model call |
| `outline` | LLM | Plans the report's sections |
| `analyse` | LLM | Extracts findings from retrieved passages only |
| `draft` | LLM | Writes the report from findings + outline |
| `review` | LLM | Grades against a checklist; approves or sends back |

`retrieve` and `outline` do not depend on each other, so the engine puts them in
the same layer and runs them concurrently.

---

## What a run looks like

```
$ python demo.py

backend: scripted
layers:
  0: outline, retrieve  (parallel)
  1: analyse
  2: draft
  3: review

trace:
  [  ok] outline       outliner     0ms     61tok
  [  ok] retrieve      retriever    0ms      0tok
  [  ok] analyse       analyst      0ms    307tok
  [  ok] draft         writer       0ms   1187tok
  [  ok] review        reviewer     0ms   1246tok — requested revision
  [  ok] draft         writer       0ms    347tok
  [  ok] review        reviewer     0ms    341tok

retrieved 3 of 4 passages
revisions: 1
  feedback: The draft runs to 620 words, over the 400 word limit.
            Cut the repeated context and keep one sentence per finding.

verdict: approved=True score=9
total: 3489 tokens in 3ms
by agent: {'reviewer': 1587, 'writer': 1534, 'analyst': 307, 'outliner': 61, 'retriever': 0}
```

The reviewer rejected a 620-word first draft, the writer saw that feedback and
produced 85 words, and the second review passed. The irrelevant corpus entry
(an office memo) never made it into retrieval.

---

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

python demo.py                          # full run with trace
pytest tests -q                         # 59 tests
uvicorn workflow.api:app --reload       # API on http://localhost:8000
```

No API key needed. Set `ANTHROPIC_API_KEY` to swap the scripted backend for
Claude — the orchestration code does not change.

---

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Active backend and registered tools |
| `GET` | `/workflow` | The graph: steps, dependencies, execution layers |
| `POST` | `/run` | Run the workflow; returns the report **and the full trace** |

`/workflow` exists so a caller can inspect the shape before committing to a
run, and `/run` returns the per-step trace alongside the result — a workflow you
cannot see inside of is one you cannot debug.

---

## Engine features

**Layered parallel execution.** Steps are topologically sorted into layers;
every step in a layer is independent of the others, so the layer runs on a
thread pool. Cycles and unknown dependencies are caught when the graph is built,
not halfway through a run.

**Bounded revision loops.** A step returns `Revision(target, feedback)` to send
the run back. The engine rewinds to that step's layer, clears the outputs that
are about to be recomputed, and puts the feedback on the blackboard so the
retried step sees *why*. `max_revisions` caps it — without that, a reviewer that
never approves loops forever.

**Retries that isolate failure.** A step that raises is retried up to
`max_attempts`. Every attempt is traced, including the failures. If all attempts
fail nothing is written under the step's output key, so dependents see a missing
key rather than a half-built value.

**Write-once shared state.** The blackboard is thread-safe and rejects a second
write to the same key, which turns "two steps claim the same output" from a
silent race into an immediate error.

**Tracing.** Per step: status, attempt number, duration, tokens. Per run: total
spend, tokens by agent, retry counts. "Which agent burned the budget" is a
question a multi-agent system gets asked constantly.

---

## Design decisions worth explaining

**Agents are thin; the engine owns control flow.** An agent renders a prompt,
calls the backend, returns a result. Ordering, retries, and revision live in the
engine. That keeps agents unit-testable and means the same agent works in a
different graph without modification.

**Reviewer parsing fails closed.** Models wrap JSON in prose or fences, so the
verdict parser extracts the first JSON object rather than requiring a bare one,
and falls back to looking for an explicit approval word. When it cannot tell,
it withholds approval — a garbled reviewer must never auto-pass a draft.

**Prompts are joined, not dedented.** Interpolating multi-line content into a
`textwrap.dedent` block leaves the first line indented and the rest flush, so
dedent finds no common prefix and silently does nothing. This broke
line-anchored parsing during development; prompts are now assembled by joining
lines.

**Tool steps sit in the same graph as LLM steps.** Retrieval is a tool call, not
a model call, and costs zero tokens — but it still gets retries, tracing, and
dependency ordering.

---

## Tests

59 tests across the engine and pipeline:

| Area | Covers |
|---|---|
| Blackboard | write-once, missing keys, concurrent writes |
| Topology | layering, cycles, unknown deps, duplicate names |
| Execution | parallelism (timed), retries, isolation, conditions |
| Revision loop | rerun, feedback delivery, revision cap, bad target |
| Tools | registry, validation, error messages |
| Agents | prompts, parsing, usage, review verdicts |
| Verdict parsing | bare JSON, fenced JSON, prose, unparsable |
| Pipeline | end-to-end, retrieval filtering, loop actually fires |
| API | health, graph description, run, validation |

---

## Layout

```
agent-workflow/
├── workflow/
│   ├── state.py       # thread-safe write-once blackboard
│   ├── llm.py         # backend protocol: scripted / Claude
│   ├── tools.py       # tool registry with argument validation
│   ├── agents.py      # Agent, ToolAgent, ReviewAgent
│   ├── graph.py       # topological layers, parallel exec, retries, revision
│   ├── tracing.py     # per-step and per-run trace
│   ├── pipelines.py   # the concrete research-and-report workflow
│   └── api.py         # FastAPI layer
├── tests/test_workflow.py
└── demo.py
```

## License

MIT
