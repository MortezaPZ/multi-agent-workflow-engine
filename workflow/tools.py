"""Tool registry.

Tools are plain functions with a schema. Keeping them behind a registry means
the engine can list what is available, validate arguments before calling, and
report a bad call as a normal step failure rather than a stack trace.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolError(RuntimeError):
    """Raised when a tool is missing or called incorrectly."""


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]
    required: list[str] = field(default_factory=list)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self, name: str, description: str, required: list[str] | None = None
    ) -> Callable:
        def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._tools:
                raise ToolError(f'Tool "{name}" is already registered.')
            self._tools[name] = Tool(name, description, handler, required or [])
            return handler

        return decorator

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise ToolError(
                f'Unknown tool "{name}". Available: {", ".join(self.names) or "none"}'
            )
        tool = self._tools[name]

        missing = [key for key in tool.required if key not in kwargs]
        if missing:
            raise ToolError(f'Tool "{name}" is missing arguments: {missing}')

        return tool.handler(**kwargs)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> list[dict]:
        return [
            {'name': t.name, 'description': t.description, 'required': t.required}
            for t in sorted(self._tools.values(), key=lambda t: t.name)
        ]


def default_registry() -> ToolRegistry:
    """The tools the demo pipeline uses."""
    registry = ToolRegistry()

    @registry.register(
        'search_corpus',
        'Find passages in the local corpus that mention the query terms.',
        required=['query', 'corpus'],
    )
    def search_corpus(query: str, corpus: list[dict], limit: int = 5) -> list[dict]:
        terms = {t for t in re.findall(r'[a-z]{3,}', query.lower())}
        scored: list[tuple[int, dict]] = []
        for entry in corpus:
            body = f"{entry.get('title', '')} {entry.get('body', '')}".lower()
            overlap = sum(1 for term in terms if term in body)
            if overlap:
                scored.append((overlap, entry))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [entry for _, entry in scored[:limit]]

    @registry.register(
        'summarise_numbers',
        'Compute count, mean, median, min and max for a list of numbers.',
        required=['values'],
    )
    def summarise_numbers(values: list[float]) -> dict:
        numbers = [float(v) for v in values if isinstance(v, (int, float))]
        if not numbers:
            raise ToolError('summarise_numbers needs at least one numeric value.')
        return {
            'count': len(numbers),
            'mean': round(statistics.fmean(numbers), 3),
            'median': round(statistics.median(numbers), 3),
            'min': min(numbers),
            'max': max(numbers),
        }

    @registry.register(
        'word_count',
        'Count the words in a piece of text.',
        required=['text'],
    )
    def word_count(text: str) -> int:
        return len(re.findall(r'\S+', text))

    return registry
