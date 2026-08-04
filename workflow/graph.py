"""The workflow engine: a DAG of steps executed in dependency order.

Independent steps run in parallel, failures are isolated and retried, and a
step may send the run *backwards* to an earlier step — that revision loop is
what separates an agent workflow from a plain pipeline.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable

from .state import Blackboard
from .tracing import RunTrace, StepTrace


class GraphError(ValueError):
    """Raised when a workflow is not a valid DAG."""


class StepFailed(RuntimeError):
    """Raised by a step to signal a retryable failure."""


@dataclass
class Revision:
    """Returned by a step to send the run back to an earlier step.

    `feedback` is written to the blackboard so the target step can see why it
    is being re-run, which is what makes the second attempt different from the
    first.
    """

    target: str
    feedback: str


@runtime_checkable
class Runnable(Protocol):
    """Anything a step can execute. Agents satisfy this; so do plain functions."""

    name: str

    def run(self, board: Blackboard) -> object:
        ...


@dataclass
class Step:
    name: str
    runnable: Runnable
    depends_on: list[str] = field(default_factory=list)
    produces: str = ''
    max_attempts: int = 2
    # A step whose condition returns False is skipped along with nothing else;
    # its dependents still run and must tolerate the missing key.
    condition: Callable[[Blackboard], bool] | None = None

    def __post_init__(self) -> None:
        if not self.produces:
            self.produces = self.name
        if self.max_attempts < 1:
            raise GraphError(f'Step "{self.name}" needs at least one attempt.')


def topological_layers(steps: list[Step]) -> list[list[Step]]:
    """Group steps into layers that can each run in parallel.

    Every step in layer N depends only on steps in layers < N, so the whole
    layer is safe to execute concurrently.
    """
    by_name = {step.name: step for step in steps}
    if len(by_name) != len(steps):
        raise GraphError('Step names must be unique.')

    for step in steps:
        for dependency in step.depends_on:
            if dependency not in by_name:
                raise GraphError(
                    f'Step "{step.name}" depends on unknown step "{dependency}".'
                )

    remaining = {step.name: set(step.depends_on) for step in steps}
    layers: list[list[Step]] = []

    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps)
        if not ready:
            # Nothing is unblocked but steps remain: the dependencies form a cycle.
            raise GraphError(
                'Workflow contains a dependency cycle involving: '
                + ', '.join(sorted(remaining))
            )
        layers.append([by_name[name] for name in ready])
        for name in ready:
            del remaining[name]
        for deps in remaining.values():
            deps.difference_update(ready)

    return layers


class Workflow:
    def __init__(
        self,
        steps: list[Step],
        max_revisions: int = 2,
        max_workers: int = 4,
    ) -> None:
        self.steps = steps
        self.layers = topological_layers(steps)
        self.max_revisions = max_revisions
        self.max_workers = max_workers

    @property
    def step_names(self) -> list[str]:
        return [step.name for step in self.steps]

    def run(self, board: Blackboard | None = None) -> tuple[Blackboard, RunTrace]:
        board = board or Blackboard()
        trace = RunTrace()
        revisions = 0
        layer_index = 0

        while layer_index < len(self.layers):
            layer = self.layers[layer_index]
            revision = self._run_layer(layer, board, trace)

            if revision is None:
                layer_index += 1
                continue

            # A step asked to go back. Rewind to the target's layer and let the
            # steps between here and there run again with the feedback in hand.
            if revisions >= self.max_revisions:
                trace.record(
                    StepTrace(
                        step=revision.target,
                        agent='engine',
                        status='skipped',
                        attempt=1,
                        duration_ms=0,
                        note=f'revision limit ({self.max_revisions}) reached',
                    )
                )
                layer_index += 1
                continue

            revisions += 1
            target_layer = self._layer_of(revision.target)
            board.put('revision_feedback', revision.feedback, overwrite=True)
            board.put('revision_round', revisions, overwrite=True)
            self._clear_from(layer_index, target_layer, board)
            layer_index = target_layer

        return board, trace

    def _layer_of(self, step_name: str) -> int:
        for index, layer in enumerate(self.layers):
            if any(step.name == step_name for step in layer):
                return index
        raise GraphError(f'Cannot revise unknown step "{step_name}".')

    def _clear_from(self, current: int, target: int, board: Blackboard) -> None:
        """Drop outputs of steps that are about to be recomputed."""
        for index in range(target, current + 1):
            for step in self.layers[index]:
                if board.has(step.produces):
                    board.put(step.produces, None, overwrite=True)

    def _run_layer(
        self, layer: list[Step], board: Blackboard, trace: RunTrace
    ) -> Revision | None:
        if len(layer) == 1:
            results = [self._run_step(layer[0], board, trace)]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                results = list(
                    pool.map(lambda s: self._run_step(s, board, trace), layer)
                )

        # If several steps in a parallel layer request a revision, honour the
        # first in declaration order so the run stays deterministic.
        for result in results:
            if isinstance(result, Revision):
                return result
        return None

    def _run_step(
        self, step: Step, board: Blackboard, trace: RunTrace
    ) -> Revision | None:
        if step.condition is not None and not step.condition(board):
            trace.record(
                StepTrace(
                    step=step.name,
                    agent=step.runnable.name,
                    status='skipped',
                    attempt=1,
                    duration_ms=0,
                    note='condition not met',
                )
            )
            return None

        last_error = ''
        for attempt in range(1, step.max_attempts + 1):
            started = time.perf_counter()
            try:
                result = step.runnable.run(board)
            except Exception as exc:  # noqa: BLE001 - isolate step failures
                last_error = str(exc)
                trace.record(
                    StepTrace(
                        step=step.name,
                        agent=step.runnable.name,
                        status='failed',
                        attempt=attempt,
                        duration_ms=int((time.perf_counter() - started) * 1000),
                        error=last_error,
                    )
                )
                continue

            usage = getattr(step.runnable, 'last_usage', (0, 0))
            trace.record(
                StepTrace(
                    step=step.name,
                    agent=step.runnable.name,
                    status='completed',
                    attempt=attempt,
                    duration_ms=int((time.perf_counter() - started) * 1000),
                    input_tokens=usage[0],
                    output_tokens=usage[1],
                    note='requested revision' if isinstance(result, Revision) else '',
                )
            )

            if isinstance(result, Revision):
                return result

            board.put(step.produces, result, overwrite=True)
            return None

        # Every attempt failed. Record nothing under `produces` so dependent
        # steps see a missing key rather than a half-built value.
        raise StepFailed(
            f'Step "{step.name}" failed after {step.max_attempts} attempts: {last_error}'
        )
