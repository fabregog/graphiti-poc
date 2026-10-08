"""Validate through the HTTP API that search returns only the latest state.

Prerequisites:
    uv run python poc_temporal.py --mode cardinality   # load the POC graph
    uv run uvicorn api:app --port 8010                 # expose /search

Run:  uv run python validate_search.py [--base-url http://localhost:8010]
"""

import argparse

import httpx

STATE_QUESTION = 'What is the current state of order ORD-1001?'
CARS_QUESTION = 'Which cars does Juan Perez have?'
TOYOTA_QUESTION = 'What do you know about Toyota Corolla ABC-123?'


class Checks:
    def __init__(self) -> None:
        self.failed = 0

    def expect(self, condition: bool, description: str) -> None:
        print(f'   [{"PASS" if condition else "FAIL"}] {description}')
        self.failed += not condition


def search(client: httpx.Client, query: str, **params) -> list[dict]:
    response = client.get('/search', params={'query': query, **params})
    response.raise_for_status()
    results = response.json()['results']
    for rank, r in enumerate(results, start=1):
        print(
            f'      {rank}. {r["relation"]:<9} {r["source"]} -> {r["target"]:<24} '
            f'valid_at={(r["valid_at"] or "-")[:10]} invalid_at={(r["invalid_at"] or "-")[:10]:<10} {r["status"]}'
        )
    return results


def targets(results: list[dict], source: str, relation: str) -> list[str]:
    return [r['target'] for r in results if r['source'] == source and r['relation'] == relation]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://localhost:8010')
    base_url = parser.parse_args().base_url

    checks = Checks()
    with httpx.Client(base_url=base_url, timeout=60) as client:
        print(f'\n1) Graphiti default search (include_history=true): "{STATE_QUESTION}"')
        results = search(client, STATE_QUESTION, include_history='true')
        states = targets(results, 'Order ORD-1001', 'HAS_STATE')
        checks.expect(
            {'PENDING', 'SHIPPED'} <= set(states),
            'default search also returns invalidated states (it does NOT return only the last one)',
        )
        if 'DELIVERED' in states:
            print(f'      note: the current state DELIVERED is ranked #{states.index("DELIVERED") + 1} of '
                  f'{len(states)}; ranking is by relevance, not recency')

        print(f'\n2) Current-only search (default): "{STATE_QUESTION}"')
        results = search(client, STATE_QUESTION)
        checks.expect(
            targets(results, 'Order ORD-1001', 'HAS_STATE') == ['DELIVERED'],
            'only the last state (DELIVERED) is returned',
        )
        checks.expect(all(r['invalid_at'] is None for r in results), 'no invalidated fact is returned')

        print('\n3) Point-in-time search (as_of)')
        for as_of, expected in [
            ('2026-01-05', []),
            ('2026-01-12', ['PENDING']),
            ('2026-01-17', ['SHIPPED']),
            ('2026-01-20', ['DELIVERED']),  # boundary: SHIPPED.invalid_at == DELIVERED.valid_at
        ]:
            print(f'   as_of={as_of}')
            results = search(client, STATE_QUESTION, as_of=as_of)
            checks.expect(
                targets(results, 'Order ORD-1001', 'HAS_STATE') == expected,
                f'state valid on {as_of} is {expected or "none (order did not exist yet)"}',
            )

        print(f'\n4) Current-only search on a multi-valued relation: "{CARS_QUESTION}"')
        results = search(client, CARS_QUESTION)
        checks.expect(
            sorted(targets(results, 'Juan Perez', 'HAS_CAR')) == ['Mazda CX-5 XYZ-789', 'Toyota Corolla ABC-123'],
            'both current cars are returned (current-only does not mean single result)',
        )

        print(f'\n5) Asking about one specific car: "{TOYOTA_QUESTION}"')
        print('   scope=related (default)')
        results = search(client, TOYOTA_QUESTION)
        checks.expect(
            'Mazda CX-5 XYZ-789' in targets(results, 'Juan Perez', 'HAS_CAR'),
            'related scope also returns the other car (semantically close neighbour fact)',
        )
        print('   scope=entity')
        results = search(client, TOYOTA_QUESTION, scope='entity')
        checks.expect(
            [r['target'] for r in results] == ['Toyota Corolla ABC-123'],
            'entity scope returns only the Toyota fact',
        )
        print(f'   scope=entity: "{CARS_QUESTION}"')
        results = search(client, CARS_QUESTION, scope='entity')
        checks.expect(
            sorted(targets(results, 'Juan Perez', 'HAS_CAR')) == ['Mazda CX-5 XYZ-789', 'Toyota Corolla ABC-123'],
            'entity scope anchored on Juan Perez still returns both cars',
        )

    print('\n' + ('RESULT: all checks passed' if checks.failed == 0 else f'RESULT: {checks.failed} check(s) FAILED'))
    raise SystemExit(1 if checks.failed else 0)


if __name__ == '__main__':
    main()
