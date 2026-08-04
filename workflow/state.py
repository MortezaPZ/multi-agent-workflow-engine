"""Shared state passed between agents.

Agents never mutate each other's output. Each step writes its result under its
own key, and later steps read what earlier ones produced. That makes a run
reproducible from its trace and keeps two parallel agents from racing.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any


class StateError(KeyError):
    """Raised when a step reads a key that no step has produced."""


@dataclass
class Blackboard:
    """Thread-safe key/value store shared by every step in a run.

    Parallel branches write concurrently, so writes take a lock. Keys are
    write-once by default: a second write to the same key is a bug in the graph
    (two steps claiming the same output) rather than an intentional update.
    """

    _values: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def put(self, key: str, value: Any, *, overwrite: bool = False) -> None:
        with self._lock:
            if key in self._values and not overwrite:
                raise StateError(
                    f'"{key}" was already written by another step. '
                    'Two steps cannot claim the same output key.'
                )
            self._values[key] = value

    def get(self, key: str) -> Any:
        with self._lock:
            if key not in self._values:
                raise StateError(f'No step has produced "{key}".')
            return self._values[key]

    def get_or(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._values.get(key, default)

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._values

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._values)

    def keys(self) -> list[str]:
        with self._lock:
            return sorted(self._values)
