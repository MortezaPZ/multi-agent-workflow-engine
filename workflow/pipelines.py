"""The research-and-report pipeline.

    retrieve ─┐
              ├─→ analyse ─→ draft ─→ review ─┐
    outline  ─┘                 ↑              │
                                └── revise ────┘

`retrieve` and `outline` are independent, so the engine runs them in parallel.
`review` can send the run back to `draft`, which is what makes this a workflow
rather than a chain.
"""

from __future__ import annotations

import json
import re

from .agents import Agent, ReviewAgent, ToolAgent
from .graph import Step, Workflow
from .llm import LLMBackend, ScriptedBackend
from .state import Blackboard
from .tools import ToolRegistry, default_registry

RESEARCHER = (
    'You are a research analyst. Extract only claims supported by the supplied '
    'passages. Never introduce facts that are not present in them.'
)
OUTLINER = (
    'You are an editor. Produce a tight outline of the sections a short report '
    'on the topic should contain. One line per section.'
)
WRITER = (
    'You are a technical writer. Write a concise report from the findings and '
    'outline supplied. Do not invent numbers.'
)
REVIEWER = (
    'You are a reviewer. Grade the draft against the checklist and reply with '
    'JSON: {"approved": bool, "score": 0-10, "feedback": "..."}. '
    'Withhold approval if any checklist item fails.'
)


def build_workflow(
    backend: LLMBackend,
    registry: ToolRegistry | None = None,
    max_revisions: int = 2,
) -> Workflow:
    registry = registry or default_registry()

    retrieve = ToolAgent(
        name='retriever',
        registry=registry,
        tool='search_corpus',
        arguments=lambda board: {
            'query': board.get('topic'),
            'corpus': board.get('corpus'),
        },
    )

    outline = Agent(
        name='outliner',
        role=OUTLINER,
        backend=backend,
        reads=['topic'],
        prompt=lambda board: f'Topic: {board.get("topic")}\n\nOutline the report.',
    )

    analyse = Agent(
        name='analyst',
        role=RESEARCHER,
        backend=backend,
        reads=['passages', 'topic'],
        # Prompts are assembled by joining lines rather than with dedent on an
        # f-string: interpolating multi-line content into an indented block
        # leaves the first line indented and the rest flush, which defeats
        # dedent and breaks any line-anchored parsing downstream.
        prompt=lambda board: '\n'.join(
            [
                f'Topic: {board.get("topic")}',
                '',
                'Passages:',
                _render_passages(board.get('passages')),
                '',
                'List the findings this evidence supports, one per line.',
            ]
        ),
        parse=_parse_findings,
    )

    draft = Agent(
        name='writer',
        role=WRITER,
        backend=backend,
        reads=['findings', 'outline', 'topic'],
        prompt=lambda board: '\n'.join(
            [
                f'Topic: {board.get("topic")}',
                '',
                'Outline:',
                str(board.get('outline')),
                '',
                'Findings:',
                _render_findings(board.get('findings')),
                _render_feedback(board),
                'Write the report.',
            ]
        ),
    )

    review = ReviewAgent(
        name='reviewer',
        role=REVIEWER,
        backend=backend,
        revise_target='draft',
        reads=['report', 'findings'],
        prompt=lambda board: '\n'.join(
            [
                'Checklist:',
                '1. Every number in the draft appears in the findings.',
                '2. The report covers each outline section.',
                '3. It is under 400 words.',
                '',
                'Findings:',
                _render_findings(board.get('findings')),
                '',
                'Draft:',
                str(board.get('report')),
            ]
        ),
    )

    return Workflow(
        steps=[
            # retrieve and outline have no dependencies, so they share layer 0
            # and the engine runs them concurrently.
            Step('retrieve', retrieve, produces='passages'),
            Step('outline', outline, produces='outline'),
            Step('analyse', analyse, depends_on=['retrieve'], produces='findings'),
            Step('draft', draft, depends_on=['analyse', 'outline'], produces='report'),
            Step('review', review, depends_on=['draft'], produces='verdict'),
        ],
        max_revisions=max_revisions,
    )


def _render_passages(passages) -> str:
    if not passages:
        return '(no passages found)'
    return '\n'.join(
        f'- [{p.get("title", "untitled")}] {p.get("body", "")}' for p in passages
    )


def _render_findings(findings) -> str:
    if not findings:
        return '(none)'
    if isinstance(findings, str):
        return findings
    return '\n'.join(f'- {f}' for f in findings)


def _render_feedback(board: Blackboard) -> str:
    feedback = board.get_or('revision_feedback')
    if not feedback:
        return ''
    round_number = board.get_or('revision_round', 1)
    return (
        f'\nReviewer feedback from round {round_number} — address it in this '
        f'revision:\n{feedback}\n'
    )


def _parse_findings(text: str) -> list[str]:
    lines = [
        re.sub(r'^[-*\d.\s]+', '', line).strip()
        for line in text.splitlines()
        if line.strip()
    ]
    return [line for line in lines if len(line) > 3]


# --------------------------------------------------------------------------
# Scripted backend used by the demo and the tests.
# --------------------------------------------------------------------------

def demo_backend() -> ScriptedBackend:
    """A backend that plays the four roles convincingly enough to demo.

    The writer deliberately produces a too-long draft on its first attempt so
    the revision loop actually fires — a demo where review always passes proves
    nothing about the engine.
    """
    state = {'drafts': 0}

    def write(prompt: str) -> str:
        state['drafts'] += 1
        findings = re.findall(r'^- (.+)$', prompt, re.MULTILINE)
        body = ' '.join(findings) or 'No findings were available.'

        if state['drafts'] == 1:
            # First pass: padded well past the 400 word limit on purpose, so the
            # revision loop has something real to catch.
            filler = (
                ' This section expands on the point at length, restating the '
                'context and reiterating the supporting detail for emphasis.'
            ) * 30
            return f'# Report\n\n{body}{filler}'
        return f'# Report\n\n{body}\n\nPrepared from the findings above.'

    def review(prompt: str) -> str:
        draft = prompt.split('Draft:', 1)[-1]
        words = len(re.findall(r'\S+', draft))
        if words > 400:
            return json.dumps(
                {
                    'approved': False,
                    'score': 4,
                    'feedback': (
                        f'The draft runs to {words} words, over the 400 word limit. '
                        'Cut the repeated context and keep one sentence per finding.'
                    ),
                }
            )
        return json.dumps(
            {'approved': True, 'score': 9, 'feedback': f'Within limits at {words} words.'}
        )

    def analyse(prompt: str) -> str:
        bodies = re.findall(r'^- \[.*?\] (.+)$', prompt, re.MULTILINE)
        sentences: list[str] = []
        for body in bodies:
            sentences.extend(
                s.strip() for s in re.split(r'(?<=[.!?])\s+', body) if len(s.strip()) > 20
            )
        return '\n'.join(f'- {s}' for s in sentences[:6]) or '- No usable evidence.'

    return ScriptedBackend(
        rules=[
            (r'Checklist:', review),
            (r'Write the report\.', write),
            (r'List the findings', analyse),
            (
                r'Outline the report',
                '1. Background\n2. Key findings\n3. Implications',
            ),
        ],
        default='OK',
    )
