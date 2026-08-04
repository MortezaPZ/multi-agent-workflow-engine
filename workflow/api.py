"""HTTP layer over the workflow engine."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .graph import GraphError, StepFailed
from .llm import resolve_backend
from .pipelines import build_workflow
from .state import Blackboard
from .tools import default_registry


class Passage(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=20000)


class RunRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    corpus: list[Passage] = Field(min_length=1, max_length=200)
    max_revisions: int = Field(default=2, ge=0, le=5)


class StepResponse(BaseModel):
    step: str
    agent: str
    status: str
    attempt: int
    duration_ms: int
    tokens: int
    error: str = ''
    note: str = ''


class RunResponse(BaseModel):
    topic: str
    report: str
    approved: bool
    score: int | None = None
    feedback: str
    findings: list[str]
    passages_used: int
    revisions: int
    trace: list[StepResponse]
    summary: dict


def create_app() -> FastAPI:
    app = FastAPI(
        title='Multi-Agent Workflow Engine',
        description=(
            'Runs a research-and-report workflow across several agents, with '
            'parallel execution, retries, and a reviewer that can send the run '
            'back for revision.'
        ),
        version='1.0.0',
    )

    @app.get('/health')
    def health() -> dict:
        backend = resolve_backend()
        return {
            'status': 'ok',
            'backend': backend.name,
            'tools': default_registry().names,
        }

    @app.get('/workflow')
    def describe_workflow() -> dict:
        """Expose the graph so a caller can see the shape before running it."""
        workflow = build_workflow(resolve_backend())
        return {
            'steps': [
                {
                    'name': step.name,
                    'agent': step.runnable.name,
                    'depends_on': step.depends_on,
                    'produces': step.produces,
                    'max_attempts': step.max_attempts,
                }
                for step in workflow.steps
            ],
            'layers': [
                [step.name for step in layer] for layer in workflow.layers
            ],
            'tools': default_registry().describe(),
        }

    @app.post('/run', response_model=RunResponse)
    def run(request: RunRequest) -> RunResponse:
        workflow = build_workflow(
            resolve_backend(), max_revisions=request.max_revisions
        )

        board = Blackboard()
        board.put('topic', request.topic)
        board.put('corpus', [p.model_dump() for p in request.corpus])

        try:
            board, trace = workflow.run(board)
        except StepFailed as exc:
            raise HTTPException(502, str(exc)) from exc
        except GraphError as exc:
            raise HTTPException(500, str(exc)) from exc

        verdict = board.get_or('verdict') or {}
        return RunResponse(
            topic=request.topic,
            report=board.get_or('report', ''),
            approved=bool(verdict.get('approved')),
            score=verdict.get('score'),
            feedback=verdict.get('feedback', ''),
            findings=board.get_or('findings', []),
            passages_used=len(board.get_or('passages', [])),
            revisions=board.get_or('revision_round', 0),
            trace=[
                StepResponse(
                    step=s.step,
                    agent=s.agent,
                    status=s.status,
                    attempt=s.attempt,
                    duration_ms=s.duration_ms,
                    tokens=s.tokens,
                    error=s.error,
                    note=s.note,
                )
                for s in trace.steps
            ],
            summary=trace.summary(),
        )

    return app


app = create_app()
