"""Run the research-and-report workflow end to end and print the trace."""

from workflow.llm import resolve_backend
from workflow.pipelines import build_workflow
from workflow.state import Blackboard

CORPUS = [
    {
        'title': 'quarterly-report',
        'body': (
            'Revenue for the quarter reached 4.2 million pounds, up from 3.8 '
            'million in the prior period. Churn fell to 3 percent across the '
            'subscriber base, the lowest level recorded in two years.'
        ),
    },
    {
        'title': 'support-review',
        'body': (
            'Average first response time was 4 hours during the period. '
            'Escalations to the regional team dropped by a fifth compared with '
            'the prior quarter.'
        ),
    },
    {
        'title': 'headcount-note',
        'body': (
            'The support team grew from 11 to 14 people during the quarter, '
            'which the review links to the improvement in response times.'
        ),
    },
    {
        'title': 'facilities-memo',
        'body': 'The office kitchen refit is scheduled for the spring.',
    },
]

TOPIC = 'quarterly revenue, churn and support performance'


def main() -> None:
    backend = resolve_backend()
    workflow = build_workflow(backend, max_revisions=2)

    print(f'backend: {backend.name}')
    print('layers:')
    for index, layer in enumerate(workflow.layers):
        names = ', '.join(step.name for step in layer)
        parallel = '  (parallel)' if len(layer) > 1 else ''
        print(f'  {index}: {names}{parallel}')
    print()

    board = Blackboard()
    board.put('topic', TOPIC)
    board.put('corpus', CORPUS)

    board, trace = workflow.run(board)

    print('trace:')
    print(trace.render())
    print()

    print(f'retrieved {len(board.get("passages"))} of {len(CORPUS)} passages')
    print(f'findings:  {len(board.get("findings"))}')
    revisions = board.get_or('revision_round', 0)
    print(f'revisions: {revisions}')
    if revisions:
        print(f'  feedback: {board.get("revision_feedback")}')
    print()

    verdict = board.get('verdict')
    print(f'verdict: approved={verdict["approved"]} score={verdict.get("score")}')
    print(f'         {verdict["feedback"]}')
    print()

    summary = trace.summary()
    print(f'total: {summary["total_tokens"]} tokens in {summary["duration_ms"]}ms')
    print(f'by agent: {summary["tokens_by_agent"]}')
    print()

    print('--- report ---')
    print(board.get('report'))


if __name__ == '__main__':
    main()
