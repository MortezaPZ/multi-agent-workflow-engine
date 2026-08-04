"""Agents: a role, a prompt, and access to the shared blackboard.

An agent is deliberately thin. It renders a prompt from the board, calls the
backend, and returns a result. All control flow — ordering, retries, revision
loops — lives in the engine, so an agent stays testable in isolation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .graph import Revision
from .llm import LLMBackend
from .state import Blackboard
from .tools import ToolRegistry


@dataclass
class Agent:
    """An LLM-backed worker with a role and a prompt template."""

    name: str
    role: str
    backend: LLMBackend
    prompt: Callable[[Blackboard], str]
    reads: list[str] = field(default_factory=list)
    parse: Callable[[str], object] | None = None
    last_usage: tuple[int, int] = (0, 0)

    @property
    def system(self) -> str:
        return self.role

    def run(self, board: Blackboard) -> object:
        for key in self.reads:
            if not board.has(key):
                raise KeyError(f'{self.name} needs "{key}", which is not on the board.')

        completion = self.backend.complete(self.system, self.prompt(board))
        self.last_usage = (completion.input_tokens, completion.output_tokens)

        return self.parse(completion.text) if self.parse else completion.text


@dataclass
class ToolAgent:
    """A step that calls registered tools instead of a model.

    Real workflows mix both. Keeping tool steps in the same graph means
    retries, tracing, and dependencies work identically for them.
    """

    name: str
    registry: ToolRegistry
    tool: str
    arguments: Callable[[Blackboard], dict]
    last_usage: tuple[int, int] = (0, 0)

    def run(self, board: Blackboard) -> object:
        self.last_usage = (0, 0)
        return self.registry.call(self.tool, **self.arguments(board))


@dataclass
class ReviewAgent:
    """Grades another step's output and can send the run back to revise it.

    Returning a `Revision` is how a workflow becomes iterative: the writer sees
    the reviewer's feedback on its next attempt and produces something
    different, rather than retrying the identical prompt.
    """

    name: str
    role: str
    backend: LLMBackend
    prompt: Callable[[Blackboard], str]
    revise_target: str
    reads: list[str] = field(default_factory=list)
    last_usage: tuple[int, int] = (0, 0)

    def run(self, board: Blackboard) -> object:
        for key in self.reads:
            if not board.has(key):
                raise KeyError(f'{self.name} needs "{key}", which is not on the board.')

        completion = self.backend.complete(self.role, self.prompt(board))
        self.last_usage = (completion.input_tokens, completion.output_tokens)

        verdict = parse_verdict(completion.text)
        if verdict['approved']:
            return verdict
        return Revision(target=self.revise_target, feedback=verdict['feedback'])


def parse_verdict(text: str) -> dict:
    """Read a reviewer's decision, tolerating prose around the JSON.

    Models wrap JSON in explanation or fences often enough that requiring a
    bare object makes the reviewer the flakiest part of the run.
    """
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return {
                'approved': bool(data.get('approved', False)),
                'feedback': str(data.get('feedback', '')).strip(),
                'score': data.get('score'),
            }
        except json.JSONDecodeError:
            pass

    # No parsable JSON: fall back to looking for an explicit approval word so a
    # chatty reviewer does not deadlock the workflow.
    approved = bool(re.search(r'\bAPPROVED\b', text, re.IGNORECASE))
    return {
        'approved': approved,
        'feedback': text.strip()[:500],
        'score': None,
    }
