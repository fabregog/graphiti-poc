"""Validate that multi-entity graph context stays attributable to each entity.

Loads a small graph into group `context-demo` where two cars share an owner and each car has
its own insurance policy, then asks about both cars through the API:

    Juan Perez -HAS_CAR-> Toyota Corolla ABC-123 -HAS_COLOR-> Red
               |          Policy POL-100 -COVERS-> Toyota,  POL-100 -HAS_STATE-> ACTIVE
               '-HAS_CAR-> Mazda CX-5 XYZ-789     -HAS_COLOR-> Blue
                          Policy POL-200 -COVERS-> Mazda,   POL-200 -HAS_STATE-> ACTIVE -> CANCELLED

"Policy POL-200 has state CANCELLED" never mentions the Mazda, and traversing through the shared
owner reaches the other car's facts: exactly where a flat list of facts becomes ambiguous.

Prerequisite:  uv run uvicorn api:app --port 8010
Run:           uv run python validate_context.py [--base-url http://localhost:8010]
"""

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

import httpx

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from cardinality_ingest import Cardinality, add_triplet_with_cardinality
from foundry_graphiti import build_graphiti

GROUP_ID = 'context-demo'
QUESTION = (
    'Compare the Toyota Corolla ABC-123 and the Mazda CX-5 XYZ-789: '
    'what color is each one and is its insurance policy active?'
)

TOYOTA, MAZDA = 'Toyota Corolla ABC-123', 'Mazda CX-5 XYZ-789'
# (source label, source, relation, target label, target, cardinality, valid_at)
TRIPLES = [
    ('Person', 'Juan Perez', 'HAS_CAR', 'Car', TOYOTA, Cardinality.MANY, '2025-03-01'),
    ('Person', 'Juan Perez', 'HAS_CAR', 'Car', MAZDA, Cardinality.MANY, '2026-06-15'),
    ('Person', 'Juan Perez', 'LIVES_IN', 'City', 'Madrid', Cardinality.ONE, '2024-01-01'),
    ('Car', TOYOTA, 'HAS_COLOR', 'Color', 'Red', Cardinality.ONE, '2025-03-01'),
    ('Car', MAZDA, 'HAS_COLOR', 'Color', 'Blue', Cardinality.ONE, '2026-06-15'),
    ('Policy', 'Policy POL-100', 'COVERS', 'Car', TOYOTA, Cardinality.MANY, '2025-03-01'),
    ('Policy', 'Policy POL-200', 'COVERS', 'Car', MAZDA, Cardinality.MANY, '2026-06-15'),
    ('Policy', 'Policy POL-100', 'HAS_STATE', 'State', 'ACTIVE', Cardinality.ONE, '2025-03-01'),
    ('Policy', 'Policy POL-200', 'HAS_STATE', 'State', 'ACTIVE', Cardinality.ONE, '2026-06-15'),
    ('Policy', 'Policy POL-200', 'HAS_STATE', 'State', 'CANCELLED', Cardinality.ONE, '2026-09-01'),
]


def entity(label: str, name: str) -> EntityNode:
    return EntityNode(
        uuid=str(uuid5(NAMESPACE_URL, f'{GROUP_ID}/{label}/{name}')), name=name, labels=[label], group_id=GROUP_ID
    )


async def seed() -> None:
    graphiti = build_graphiti()
    try:
        await clear_data(graphiti.driver, group_ids=[GROUP_ID])
        for src_label, src, relation, tgt_label, tgt, cardinality, valid_at in TRIPLES:
            source, target = entity(src_label, src), entity(tgt_label, tgt)
            phrase = relation.lower().replace('_', ' ')
            edge = EntityEdge(
                group_id=GROUP_ID,
                source_node_uuid=source.uuid,
                target_node_uuid=target.uuid,
                name=relation,
                fact=f'{src} {phrase} {tgt}',
                valid_at=datetime.fromisoformat(valid_at).replace(tzinfo=timezone.utc),
                created_at=utc_now(),
            )
            await add_triplet_with_cardinality(graphiti, source, edge, target, cardinality)
    finally:
        await graphiti.close()
    print(f'Seeded {len(TRIPLES)} triples into group "{GROUP_ID}"')


class Checks:
    def __init__(self) -> None:
        self.failed = 0

    def expect(self, condition: bool, description: str) -> None:
        print(f'   [{"PASS" if condition else "FAIL"}] {description}')
        self.failed += not condition


def facts_of(response: dict, entity_name: str) -> dict[str, dict]:
    block = next((e for e in response['entities'] if e['entity'] == entity_name), {'facts': []})
    return {f['fact']: f for f in block['facts']}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://localhost:8010')
    base_url = parser.parse_args().base_url

    logging.basicConfig(level=logging.ERROR)
    asyncio.run(seed())

    checks = Checks()
    params = {'query': QUESTION, 'group_id': GROUP_ID, 'depth': 2}
    with httpx.Client(base_url=base_url, timeout=120) as client:
        print(f'\n1) GET /context  "{QUESTION}"\n')
        ctx = client.get('/context', params=params).raise_for_status().json()
        print('\n'.join('   ' + line for line in ctx['prompt_context'].splitlines()), '\n')

        checks.expect(
            sorted(e['entity'] for e in ctx['entities']) == sorted([TOYOTA, MAZDA]),
            'one context block per car the question asks about',
        )
        toyota, mazda = facts_of(ctx, TOYOTA), facts_of(ctx, MAZDA)
        checks.expect(
            toyota.get('Policy POL-100 has state ACTIVE', {}).get('via') == ['Policy POL-100'],
            'Toyota block: "POL-100 has state ACTIVE" is tagged [via Policy POL-100]',
        )
        checks.expect(
            mazda.get('Policy POL-200 has state CANCELLED', {}).get('via') == ['Policy POL-200'],
            'Mazda block: "POL-200 has state CANCELLED" is tagged [via Policy POL-200]',
        )
        checks.expect(
            not any('POL-200' in f for f in toyota) and not any('POL-100' in f for f in mazda),
            "neither block contains the other car's policy",
        )
        checks.expect(
            'Policy POL-200 has state ACTIVE' not in mazda,
            "POL-200's invalidated ACTIVE state is not in the context",
        )
        checks.expect(
            toyota.get(f'Juan Perez has car {MAZDA}', {}).get('via') == ['Juan Perez'],
            'Toyota block: the other car appears only as [via Juan Perez] (owner context), not [direct]',
        )

        print(f'\n2) GET /answer  "{QUESTION}"\n')
        ans = client.get('/answer', params=params).raise_for_status().json()
        print(f'   answer: {ans["answer"]}')
        per_entity = {a['entity']: a['answer'] for a in ans['per_entity']}
        for name, text in per_entity.items():
            print(f'   - {name}: {text}')
        print()

        toyota_answer = next((t for n, t in per_entity.items() if 'Toyota' in n), '').lower()
        mazda_answer = next((t for n, t in per_entity.items() if 'Mazda' in n), '').lower()
        checks.expect(
            'red' in toyota_answer and 'active' in toyota_answer and 'cancel' not in toyota_answer,
            'Toyota: red, policy active, nothing about a cancellation',
        )
        checks.expect(
            'blue' in mazda_answer and 'cancel' in mazda_answer,
            'Mazda: blue, policy cancelled',
        )

    print('\n' + ('RESULT: all checks passed' if checks.failed == 0 else f'RESULT: {checks.failed} check(s) FAILED'))
    raise SystemExit(1 if checks.failed else 0)


if __name__ == '__main__':
    main()
