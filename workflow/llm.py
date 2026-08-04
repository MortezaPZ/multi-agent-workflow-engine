"""LLM backends.

`ScriptedBackend` makes the whole engine testable and lets the demo run with no
API key: it answers from rules instead of a model. `ClaudeBackend` is the real
one. Everything above this module sees only the `LLMBackend` protocol, so the
orchestration logic never changes when the backend does.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised when a completion cannot be produced."""


@dataclass
class Completion:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ''


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def complete(self, system: str, prompt: str) -> Completion:
        ...


def estimate_tokens(text: str) -> int:
    """Rough token count for the scripted backend's usage reporting."""
    return max(1, len(text) // 4)


@dataclass
class ScriptedBackend:
    """Deterministic backend driven by (pattern -> responder) rules.

    Rules are tried in order and the first whose pattern matches the prompt
    wins, so a test can pin one agent's behaviour and let the rest fall through
    to the default.
    """

    name: str = 'scripted'
    rules: list[tuple[str, Callable[[str], str] | str]] = field(default_factory=list)
    default: Callable[[str], str] | str = 'OK'
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system: str, prompt: str) -> Completion:
        self.calls.append((system, prompt))

        responder: Callable[[str], str] | str = self.default
        for pattern, candidate in self.rules:
            if re.search(pattern, prompt, re.IGNORECASE | re.DOTALL):
                responder = candidate
                break

        text = responder(prompt) if callable(responder) else responder
        return Completion(
            text=text,
            input_tokens=estimate_tokens(system + prompt),
            output_tokens=estimate_tokens(text),
            model=self.name,
        )


class ClaudeBackend:
    """Real completions via the Anthropic API."""

    name = 'claude'

    def __init__(self, model: str = 'claude-opus-5', max_tokens: int = 2048) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise LLMError('The anthropic package is not installed.') from exc

        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str) -> Completion:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{'role': 'user', 'content': prompt}],
        )

        # A refusal is a 200 with empty content — check before indexing.
        if message.stop_reason == 'refusal':
            raise LLMError('The model declined this request.')

        text = ''.join(
            block.text for block in message.content if block.type == 'text'
        )
        return Completion(
            text=text.strip(),
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
            model=self.model,
        )


def resolve_backend(name: str | None = None) -> LLMBackend:
    """Use Claude when a key is present, otherwise stay scripted."""
    choice = (name or os.environ.get('LLM_BACKEND') or '').lower()
    if not choice:
        choice = 'claude' if os.environ.get('ANTHROPIC_API_KEY') else 'scripted'

    if choice == 'scripted':
        from .pipelines import demo_backend

        return demo_backend()
    if choice == 'claude':
        return ClaudeBackend(os.environ.get('LLM_MODEL', 'claude-opus-5'))
    raise LLMError(f'Unknown backend "{choice}".')
