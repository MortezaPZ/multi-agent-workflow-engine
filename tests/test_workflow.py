import threading
import time

import pytest

from workflow.agents import Agent, ReviewAgent, ToolAgent, parse_verdict
from workflow.graph import (
    GraphError,
    Revision,
    Step,
    StepFailed,
    Workflow,
    topological_layers,
)
from workflow.llm import ScriptedBackend, estimate_tokens
from workflow.pipelines import build_workflow, demo_backend
from workflow.state import Blackboard, StateError
from workflow.tools import ToolError, ToolRegistry, default_registry

CORPUS = [
    {
        'title': 'quarterly-report',
        'body': (
            'Revenue for the quarter reached 4.2 million pounds. '
            'Churn fell to 3 percent across the subscriber base.'
        ),
    },
    {
        'title': 'support-review',
        'body': (
            'Average first response time was 4 hours during the period. '
            'Escalations dropped by a fifth compared with the prior quarter.'
        ),
    },
    {
        'title': 'unrelated-notes',
        'body': 'The office kitchen refit is scheduled for the spring.',
    },
]


class Constant:
    """Minimal runnable for engine tests."""

    def __init__(self, name, value=None, fail_times=0, delay=0.0):
        self.name = name
        self.value = value if value is not None else name
        self.fail_times = fail_times
        self.delay = delay
        self.calls = 0
        self.last_usage = (0, 0)

    def run(self, board):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.calls <= self.fail_times:
            raise RuntimeError(f'{self.name} transient failure {self.calls}')
        return self.value


class TestBlackboard:
    def test_put_then_get(self):
        board = Blackboard()
        board.put('topic', 'churn')
        assert board.get('topic') == 'churn'

    def test_reading_a_missing_key_names_it(self):
        with pytest.raises(StateError, match='findings'):
            Blackboard().get('findings')

    def test_double_write_is_rejected(self):
        board = Blackboard()
        board.put('report', 'first')
        with pytest.raises(StateError, match='already written'):
            board.put('report', 'second')

    def test_overwrite_is_explicit(self):
        board = Blackboard()
        board.put('report', 'first')
        board.put('report', 'second', overwrite=True)
        assert board.get('report') == 'second'

    def test_get_or_returns_default(self):
        assert Blackboard().get_or('missing', 'fallback') == 'fallback'

    def test_concurrent_writes_all_land(self):
        board = Blackboard()

        def writer(i):
            board.put(f'key{i}', i)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(board.keys()) == 24


class TestTopology:
    def test_independent_steps_share_a_layer(self):
        layers = topological_layers(
            [Step('a', Constant('a')), Step('b', Constant('b'))]
        )
        assert len(layers) == 1
        assert {s.name for s in layers[0]} == {'a', 'b'}

    def test_dependencies_create_ordered_layers(self):
        layers = topological_layers(
            [
                Step('c', Constant('c'), depends_on=['a', 'b']),
                Step('a', Constant('a')),
                Step('b', Constant('b')),
            ]
        )
        assert [s.name for s in layers[0]] == ['a', 'b']
        assert [s.name for s in layers[1]] == ['c']

    def test_cycle_is_reported(self):
        with pytest.raises(GraphError, match='cycle'):
            topological_layers(
                [
                    Step('a', Constant('a'), depends_on=['b']),
                    Step('b', Constant('b'), depends_on=['a']),
                ]
            )

    def test_unknown_dependency_is_reported(self):
        with pytest.raises(GraphError, match='unknown step'):
            topological_layers([Step('a', Constant('a'), depends_on=['ghost'])])

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(GraphError, match='unique'):
            topological_layers([Step('a', Constant('a')), Step('a', Constant('a'))])

    def test_zero_attempts_is_rejected(self):
        with pytest.raises(GraphError, match='at least one attempt'):
            Step('a', Constant('a'), max_attempts=0)


class TestExecution:
    def test_results_land_under_the_produces_key(self):
        workflow = Workflow([Step('a', Constant('a', 'value'), produces='out')])
        board, _ = workflow.run()
        assert board.get('out') == 'value'

    def test_dependent_step_sees_earlier_output(self):
        class Reader:
            name = 'reader'
            last_usage = (0, 0)

            def run(self, board):
                return f'saw:{board.get("first")}'

        workflow = Workflow(
            [
                Step('one', Constant('one', 'A'), produces='first'),
                Step('two', Reader(), depends_on=['one'], produces='second'),
            ]
        )
        board, _ = workflow.run()
        assert board.get('second') == 'saw:A'

    def test_independent_steps_run_concurrently(self):
        slow_a = Constant('a', delay=0.25)
        slow_b = Constant('b', delay=0.25)

        started = time.perf_counter()
        Workflow(
            [Step('a', slow_a, produces='a'), Step('b', slow_b, produces='b')]
        ).run()
        elapsed = time.perf_counter() - started

        # Sequentially this is 0.5s; in parallel it should be well under.
        assert elapsed < 0.4

    def test_transient_failure_is_retried(self):
        flaky = Constant('flaky', 'recovered', fail_times=1)
        board, trace = Workflow(
            [Step('flaky', flaky, produces='out', max_attempts=3)]
        ).run()

        assert board.get('out') == 'recovered'
        assert flaky.calls == 2
        assert trace.retries() == {'flaky': 1}

    def test_exhausted_retries_raise(self):
        workflow = Workflow(
            [Step('bad', Constant('bad', fail_times=5), max_attempts=2)]
        )
        with pytest.raises(StepFailed, match='after 2 attempts'):
            workflow.run()

    def test_failed_attempts_are_traced(self):
        workflow = Workflow(
            [Step('flaky', Constant('f', 'ok', fail_times=1), max_attempts=2)]
        )
        _, trace = workflow.run()
        assert len(trace.failed_steps) == 1

    def test_condition_skips_a_step(self):
        skipped = Constant('skipped')
        _, trace = Workflow(
            [Step('s', skipped, condition=lambda board: False)]
        ).run()

        assert skipped.calls == 0
        assert trace.steps[0].status == 'skipped'


class TestRevisionLoop:
    def _workflow(self, approve_after: int, max_revisions: int = 2):
        attempts = {'n': 0}

        class Writer:
            name = 'writer'
            last_usage = (0, 0)

            def run(self, board):
                attempts['n'] += 1
                return f'draft-{attempts["n"]}'

        class Reviewer:
            name = 'reviewer'
            last_usage = (0, 0)

            def run(self, board):
                if attempts['n'] >= approve_after:
                    return {'approved': True}
                return Revision(target='draft', feedback=f'redo #{attempts["n"]}')

        return (
            Workflow(
                [
                    Step('draft', Writer(), produces='report'),
                    Step('review', Reviewer(), depends_on=['draft'], produces='verdict'),
                ],
                max_revisions=max_revisions,
            ),
            attempts,
        )

    def test_revision_reruns_the_target_step(self):
        workflow, attempts = self._workflow(approve_after=2)
        board, _ = workflow.run()

        assert attempts['n'] == 2
        assert board.get('report') == 'draft-2'
        assert board.get('verdict') == {'approved': True}

    def test_feedback_reaches_the_board(self):
        workflow, _ = self._workflow(approve_after=2)
        board, _ = workflow.run()
        assert 'redo' in board.get('revision_feedback')
        assert board.get('revision_round') == 1

    def test_revision_limit_stops_an_endless_loop(self):
        workflow, attempts = self._workflow(approve_after=99, max_revisions=2)
        _, trace = workflow.run()

        # Initial draft plus exactly max_revisions re-drafts.
        assert attempts['n'] == 3
        assert any('revision limit' in s.note for s in trace.steps)

    def test_revising_an_unknown_step_is_reported(self):
        class BadReviewer:
            name = 'reviewer'
            last_usage = (0, 0)

            def run(self, board):
                return Revision(target='nonexistent', feedback='x')

        workflow = Workflow([Step('review', BadReviewer(), produces='v')])
        with pytest.raises(GraphError, match='unknown step'):
            workflow.run()


class TestTools:
    def test_search_finds_relevant_entries(self):
        hits = default_registry().call(
            'search_corpus', query='revenue and churn', corpus=CORPUS
        )
        titles = [h['title'] for h in hits]
        assert 'quarterly-report' in titles
        assert 'unrelated-notes' not in titles

    def test_number_summary(self):
        result = default_registry().call('summarise_numbers', values=[2, 4, 6, 8])
        assert result['count'] == 4
        assert result['mean'] == 5.0
        assert result['median'] == 5.0

    def test_summary_rejects_empty_input(self):
        with pytest.raises(ToolError, match='at least one numeric'):
            default_registry().call('summarise_numbers', values=[])

    def test_unknown_tool_lists_alternatives(self):
        with pytest.raises(ToolError, match='Available'):
            default_registry().call('does_not_exist')

    def test_missing_argument_is_named(self):
        with pytest.raises(ToolError, match='missing arguments'):
            default_registry().call('search_corpus', query='x')

    def test_duplicate_registration_is_rejected(self):
        registry = ToolRegistry()
        registry.register('t', 'first')(lambda: None)
        with pytest.raises(ToolError, match='already registered'):
            registry.register('t', 'second')(lambda: None)

    def test_registry_describes_itself(self):
        described = default_registry().describe()
        assert {d['name'] for d in described} == {
            'search_corpus',
            'summarise_numbers',
            'word_count',
        }


class TestAgents:
    def test_agent_calls_the_backend_and_returns_text(self):
        backend = ScriptedBackend(default='the answer')
        agent = Agent(
            name='a',
            role='role',
            backend=backend,
            prompt=lambda board: f'Q: {board.get("topic")}',
            reads=['topic'],
        )
        board = Blackboard()
        board.put('topic', 'churn')

        assert agent.run(board) == 'the answer'
        assert 'churn' in backend.calls[0][1]

    def test_agent_reports_missing_input(self):
        agent = Agent(
            name='a',
            role='r',
            backend=ScriptedBackend(),
            prompt=lambda board: '',
            reads=['topic'],
        )
        with pytest.raises(KeyError, match='topic'):
            agent.run(Blackboard())

    def test_agent_records_usage(self):
        agent = Agent(
            name='a',
            role='r',
            backend=ScriptedBackend(default='hello there'),
            prompt=lambda board: 'prompt',
        )
        agent.run(Blackboard())
        assert agent.last_usage[0] > 0
        assert agent.last_usage[1] > 0

    def test_parser_shapes_the_output(self):
        agent = Agent(
            name='a',
            role='r',
            backend=ScriptedBackend(default='- one\n- two'),
            prompt=lambda board: 'x',
            parse=lambda text: [
                line.lstrip('- ') for line in text.splitlines() if line.strip()
            ],
        )
        assert agent.run(Blackboard()) == ['one', 'two']

    def test_tool_agent_invokes_the_tool(self):
        agent = ToolAgent(
            name='t',
            registry=default_registry(),
            tool='word_count',
            arguments=lambda board: {'text': board.get('body')},
        )
        board = Blackboard()
        board.put('body', 'one two three')
        assert agent.run(board) == 3

    def test_review_agent_approves(self):
        agent = ReviewAgent(
            name='rev',
            role='r',
            backend=ScriptedBackend(default='{"approved": true, "feedback": "good"}'),
            prompt=lambda board: 'x',
            revise_target='draft',
        )
        assert agent.run(Blackboard())['approved'] is True

    def test_review_agent_requests_revision(self):
        agent = ReviewAgent(
            name='rev',
            role='r',
            backend=ScriptedBackend(
                default='{"approved": false, "feedback": "too long"}'
            ),
            prompt=lambda board: 'x',
            revise_target='draft',
        )
        result = agent.run(Blackboard())
        assert isinstance(result, Revision)
        assert result.target == 'draft'
        assert 'too long' in result.feedback


class TestVerdictParsing:
    def test_bare_json(self):
        assert parse_verdict('{"approved": true, "score": 8}')['approved'] is True

    def test_json_wrapped_in_prose(self):
        text = 'Here is my assessment:\n```json\n{"approved": false, "feedback": "no"}\n```\nHope that helps.'
        verdict = parse_verdict(text)
        assert verdict['approved'] is False
        assert verdict['feedback'] == 'no'

    def test_prose_approval_is_recognised(self):
        assert parse_verdict('This draft is APPROVED.')['approved'] is True

    def test_unparsable_text_withholds_approval(self):
        # Failing closed matters: a garbled reviewer must not auto-pass a draft.
        assert parse_verdict('mmm not sure about this one')['approved'] is False


class TestPipeline:
    def _run(self, max_revisions=2):
        workflow = build_workflow(demo_backend(), max_revisions=max_revisions)
        board = Blackboard()
        board.put('topic', 'quarterly revenue and churn performance')
        board.put('corpus', CORPUS)
        return workflow.run(board)

    def test_pipeline_produces_an_approved_report(self):
        board, _ = self._run()

        assert board.get('verdict')['approved'] is True
        assert board.get('report').startswith('# Report')

    def test_retrieval_filters_the_corpus(self):
        board, _ = self._run()
        titles = [p['title'] for p in board.get('passages')]
        assert 'unrelated-notes' not in titles

    def test_findings_are_extracted_from_passages(self):
        board, _ = self._run()
        findings = board.get('findings')
        assert findings
        assert any('evenue' in f or 'hurn' in f for f in findings)

    def test_the_review_loop_actually_fires(self):
        # The scripted writer over-writes on its first pass, so a run that
        # never revises would mean the loop is broken.
        board, trace = self._run()
        assert board.get_or('revision_round') == 1
        assert any(s.note == 'requested revision' for s in trace.steps)

    def test_parallel_layer_is_recognised(self):
        workflow = build_workflow(demo_backend())
        assert {s.name for s in workflow.layers[0]} == {'retrieve', 'outline'}

    def test_trace_accounts_for_every_step(self):
        _, trace = self._run()
        completed = {s.step for s in trace.steps if s.status == 'completed'}
        assert {'retrieve', 'outline', 'analyse', 'draft', 'review'} <= completed

    def test_trace_reports_token_spend_per_agent(self):
        _, trace = self._run()
        by_agent = trace.by_agent()
        assert by_agent
        assert trace.total_tokens == sum(by_agent.values())

    def test_summary_is_serialisable(self):
        _, trace = self._run()
        summary = trace.summary()
        assert summary['completed'] >= 5
        assert summary['duration_ms'] >= 0

    def test_render_produces_one_line_per_step(self):
        _, trace = self._run()
        assert len(trace.render().splitlines()) == len(trace.steps)


class TestApi:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from workflow.api import create_app

        return TestClient(create_app())

    def test_health_lists_tools(self, client):
        body = client.get('/health').json()
        assert body['status'] == 'ok'
        assert 'search_corpus' in body['tools']

    def test_workflow_description_exposes_layers(self, client):
        body = client.get('/workflow').json()
        assert body['layers'][0] == ['outline', 'retrieve']
        assert {s['name'] for s in body['steps']} == {
            'retrieve',
            'outline',
            'analyse',
            'draft',
            'review',
        }

    def test_run_returns_an_approved_report(self, client):
        response = client.post(
            '/run',
            json={
                'topic': 'quarterly revenue and churn',
                'corpus': [
                    {'title': c['title'], 'body': c['body']} for c in CORPUS
                ],
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body['approved'] is True
        assert body['report'].startswith('# Report')
        assert body['revisions'] == 1
        assert len(body['trace']) >= 5

    def test_run_rejects_an_empty_corpus(self, client):
        response = client.post(
            '/run', json={'topic': 'something', 'corpus': []}
        )
        assert response.status_code == 422

    def test_run_rejects_a_short_topic(self, client):
        response = client.post(
            '/run',
            json={'topic': 'x', 'corpus': [{'title': 't', 'body': 'b'}]},
        )
        assert response.status_code == 422

    def test_zero_revisions_returns_the_first_draft(self, client):
        response = client.post(
            '/run',
            json={
                'topic': 'quarterly revenue and churn',
                'corpus': [
                    {'title': c['title'], 'body': c['body']} for c in CORPUS
                ],
                'max_revisions': 0,
            },
        )

        body = response.json()
        assert body['revisions'] == 0
        assert body['approved'] is False


class TestBackend:
    def test_rules_match_in_order(self):
        backend = ScriptedBackend(
            rules=[(r'first', 'A'), (r'first|second', 'B')], default='C'
        )
        assert backend.complete('', 'the first thing').text == 'A'
        assert backend.complete('', 'the second thing').text == 'B'
        assert backend.complete('', 'neither').text == 'C'

    def test_calls_are_recorded(self):
        backend = ScriptedBackend()
        backend.complete('sys', 'prompt')
        assert backend.calls == [('sys', 'prompt')]

    def test_token_estimate_is_never_zero(self):
        assert estimate_tokens('') == 1
