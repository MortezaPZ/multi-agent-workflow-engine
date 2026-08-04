"""Run traces.

Every step records what it did, how long it took, and what it cost. Without
this a multi-agent run is a black box: you see the final answer but not which
agent burned the budget or which one kept failing and retrying.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

StepStatus = Literal['completed', 'failed', 'skipped']


@dataclass
class StepTrace:
    step: str
    agent: str
    status: StepStatus
    attempt: int
    duration_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ''
    note: str = ''

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class RunTrace:
    steps: list[StepTrace] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)

    def record(self, trace: StepTrace) -> None:
        self.steps.append(trace)

    @property
    def duration_ms(self) -> int:
        return int((time.perf_counter() - self.started_at) * 1000)

    @property
    def total_tokens(self) -> int:
        return sum(step.tokens for step in self.steps)

    @property
    def failed_steps(self) -> list[StepTrace]:
        return [step for step in self.steps if step.status == 'failed']

    def by_agent(self) -> dict[str, int]:
        """Token spend per agent — the usual answer to 'why was this expensive'."""
        totals: dict[str, int] = {}
        for step in self.steps:
            totals[step.agent] = totals.get(step.agent, 0) + step.tokens
        return dict(sorted(totals.items(), key=lambda kv: kv[1], reverse=True))

    def retries(self) -> dict[str, int]:
        """How many extra attempts each step needed."""
        extra: dict[str, int] = {}
        for step in self.steps:
            if step.attempt > 1:
                extra[step.step] = max(extra.get(step.step, 0), step.attempt - 1)
        return extra

    def summary(self) -> dict:
        return {
            'steps': len(self.steps),
            'completed': sum(1 for s in self.steps if s.status == 'completed'),
            'failed': len(self.failed_steps),
            'skipped': sum(1 for s in self.steps if s.status == 'skipped'),
            'duration_ms': self.duration_ms,
            'total_tokens': self.total_tokens,
            'tokens_by_agent': self.by_agent(),
            'retries': self.retries(),
        }

    def render(self) -> str:
        """One line per step, for terminal output."""
        icons = {'completed': 'ok', 'failed': 'FAIL', 'skipped': 'skip'}
        lines = []
        for step in self.steps:
            mark = icons[step.status]
            suffix = f' attempt {step.attempt}' if step.attempt > 1 else ''
            detail = f' — {step.error}' if step.error else (
                f' — {step.note}' if step.note else ''
            )
            lines.append(
                f'  [{mark:>4}] {step.step:<22} {step.agent:<12} '
                f'{step.duration_ms:>5}ms {step.tokens:>6}tok{suffix}{detail}'
            )
        return '\n'.join(lines)
