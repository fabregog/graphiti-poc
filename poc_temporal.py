"""POC: how Graphiti sets valid_at / invalid_at when ingesting fact triples.

Scenario 1 - HAS_STATE (replacing relation)
    An order moves PENDING -> SHIPPED -> DELIVERED. Each new HAS_STATE should
    invalidate the previous one: old.invalid_at == new.valid_at.

Scenario 2 - HAS_CAR (accumulating relation)
    A person buys a second car. The first HAS_CAR must stay valid
    (invalid_at stays null). Re-ingesting the first car again must reuse the
    existing edge instead of creating a duplicate.

Modes:
    graphiti     Graphiti.add_triplet: the LLM decides whether the new fact contradicts
                 existing ones (default).
    cardinality  Contradiction decided by the relation's declared cardinality
                 (see cardinality_ingest.py); Graphiti's temporal logic is reused.

Run:  uv run python poc_temporal.py [--mode graphiti|cardinality]
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, uuid5

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from cardinality_ingest import Cardinality, add_triplet_with_cardinality
from foundry_graphiti import build_graphiti

GROUP_ID = 'temporal-poc'

# Only used by --mode cardinality.
RELATION_CARDINALITY = {
    'HAS_STATE': Cardinality.ONE,
    'HAS_CAR': Cardinality.MANY,
}


def entity(label: str, name: str) -> EntityNode:
    # Deterministic UUIDs: the same business key always maps to the same node, as it
    # would when triples come from a system of record.
    return EntityNode(
        uuid=str(uuid5(NAMESPACE_URL, f'{GROUP_ID}/{label}/{name}')),
        name=name,
        labels=[label],
        group_id=GROUP_ID,
    )


def at(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc)


async def ingest(
    graphiti: Graphiti,
    mode: str,
    source: EntityNode,
    relation: str,
    target: EntityNode,
    fact: str,
    valid_at: datetime,
) -> None:
    edge = EntityEdge(
        group_id=GROUP_ID,
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name=relation,
        fact=fact,
        valid_at=valid_at,
        created_at=utc_now(),
    )
    if mode == 'cardinality':
        result = await add_triplet_with_cardinality(
            graphiti, source, edge, target, RELATION_CARDINALITY[relation]
        )
    else:
        result = await graphiti.add_triplet(source, edge, target)

    resolved, invalidated = result.edges[0], result.edges[1:]
    outcome = 'NEW edge' if resolved.uuid == edge.uuid else 'REUSED existing edge (duplicate)'
    print(f'\n>> ingest ({source.name})-[{relation}]->({target.name})  valid_at={valid_at:%Y-%m-%d}')
    print(f'   {mode}: {outcome}; invalidated {len(invalidated)} edge(s)')
    for e in invalidated:
        print(f'     - "{e.fact}"  invalid_at={e.invalid_at:%Y-%m-%d}')


@dataclass
class EdgeRow:
    target: str
    fact: str
    valid_at: datetime | None
    invalid_at: datetime | None
    expired_at: datetime | None
    created_at: datetime | None

    @property
    def is_current(self) -> bool:
        return self.invalid_at is None


async def edges_of(graphiti: Graphiti, source: EntityNode, relation: str) -> list[EdgeRow]:
    # Graphiti stores every fact as a RELATES_TO relationship; the triple's predicate
    # (HAS_STATE, HAS_CAR, ...) lives in the `name` property.
    records, _, _ = await graphiti.driver.execute_query(
        """
        MATCH (s:Entity {uuid: $uuid})-[e:RELATES_TO {name: $relation}]->(t:Entity)
        RETURN t.name AS target, e.fact AS fact, e.valid_at AS valid_at,
               e.invalid_at AS invalid_at, e.expired_at AS expired_at, e.created_at AS created_at
        ORDER BY e.valid_at
        """,
        uuid=source.uuid,
        relation=relation,
        routing_='r',
    )

    def native(value):
        return value.to_native() if value is not None else None

    return [
        EdgeRow(
            target=r['target'],
            fact=r['fact'],
            valid_at=native(r['valid_at']),
            invalid_at=native(r['invalid_at']),
            expired_at=native(r['expired_at']),
            created_at=native(r['created_at']),
        )
        for r in records
    ]


def print_edges(title: str, rows: list[EdgeRow]) -> None:
    def day(d: datetime | None) -> str:
        return f'{d:%Y-%m-%d}' if d else '-'

    def ts(d: datetime | None) -> str:
        return f'{d:%Y-%m-%d %H:%M:%S}' if d else '-'

    print(f'\n   {title}')
    print(f'   {"target":<22} {"valid_at":<11} {"invalid_at":<11} {"expired_at (UTC)":<20} status')
    for r in rows:
        status = 'CURRENT' if r.is_current else 'INVALIDATED'
        print(
            f'   {r.target:<22} {day(r.valid_at):<11} {day(r.invalid_at):<11} '
            f'{ts(r.expired_at):<20} {status}'
        )


class Checks:
    def __init__(self) -> None:
        self.failed = 0

    def expect(self, condition: bool, description: str) -> None:
        print(f'   [{"PASS" if condition else "FAIL"}] {description}')
        self.failed += not condition


async def scenario_has_state(graphiti: Graphiti, mode: str, checks: Checks) -> None:
    print('\n' + '=' * 78)
    print('SCENARIO 1: HAS_STATE - a new state replaces (invalidates) the previous one')
    print('=' * 78)

    order = entity('Order', 'Order ORD-1001')
    timeline = [
        ('PENDING', at('2026-01-10')),
        ('SHIPPED', at('2026-01-15')),
        ('DELIVERED', at('2026-01-20')),
    ]

    for step, (state_name, valid_at) in enumerate(timeline, start=1):
        state = entity('State', state_name)
        await ingest(
            graphiti, mode, order, 'HAS_STATE', state,
            fact=f'{order.name} has state {state_name}',
            valid_at=valid_at,
        )
        rows = await edges_of(graphiti, order, 'HAS_STATE')
        print_edges(f'HAS_STATE edges after step {step}:', rows)

        checks.expect(len(rows) == step, f'{step} HAS_STATE edge(s) exist (history is kept, not deleted)')
        current = [r for r in rows if r.is_current]
        checks.expect(
            [r.target for r in current] == [state_name],
            f'only {state_name} is current (invalid_at is null)',
        )
        for older, newer in zip(rows, rows[1:]):
            checks.expect(
                older.invalid_at == newer.valid_at,
                f'{older.target}.invalid_at == {newer.target}.valid_at ({newer.valid_at:%Y-%m-%d})',
            )
            checks.expect(older.expired_at is not None, f'{older.target}.expired_at is set (ingestion time)')


async def scenario_has_car(graphiti: Graphiti, mode: str, checks: Checks) -> None:
    print('\n' + '=' * 78)
    print('SCENARIO 2: HAS_CAR - a new car is added; the previous one stays valid')
    print('=' * 78)

    person = entity('Person', 'Juan Perez')
    corolla = entity('Car', 'Toyota Corolla ABC-123')
    cx5 = entity('Car', 'Mazda CX-5 XYZ-789')

    await ingest(graphiti, mode, person, 'HAS_CAR', corolla,
                 fact=f'{person.name} has car {corolla.name}', valid_at=at('2025-03-01'))
    rows = await edges_of(graphiti, person, 'HAS_CAR')
    print_edges('HAS_CAR edges after step 1:', rows)
    checks.expect(len(rows) == 1 and rows[0].is_current, 'Corolla edge created and current')

    await ingest(graphiti, mode, person, 'HAS_CAR', cx5,
                 fact=f'{person.name} has car {cx5.name}', valid_at=at('2026-06-15'))
    rows = await edges_of(graphiti, person, 'HAS_CAR')
    print_edges('HAS_CAR edges after step 2:', rows)
    checks.expect(len(rows) == 2, '2 HAS_CAR edges exist')
    checks.expect(all(r.is_current for r in rows), 'both cars are current (invalid_at is null on both)')
    checks.expect(all(r.expired_at is None for r in rows), 'neither edge was expired')

    # The same triple arrives again in a later ingestion batch.
    await ingest(graphiti, mode, person, 'HAS_CAR', corolla,
                 fact=f'{person.name} has car {corolla.name}', valid_at=at('2026-09-01'))
    rows = await edges_of(graphiti, person, 'HAS_CAR')
    print_edges('HAS_CAR edges after step 3 (Corolla re-ingested):', rows)
    checks.expect(len(rows) == 2, 'still 2 HAS_CAR edges (re-ingested triple was deduplicated)')
    checks.expect(all(r.is_current for r in rows), 'both cars still current')
    corolla_row = next(r for r in rows if r.target == corolla.name)
    checks.expect(
        corolla_row.valid_at == at('2025-03-01'),
        'Corolla keeps its original valid_at (2025-03-01), not the re-ingestion date',
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['graphiti', 'cardinality'], default='graphiti')
    mode = parser.parse_args().mode

    logging.basicConfig(level=logging.WARNING)
    # graphiti's queries read an optional `reference_time` property that these edges don't have.
    logging.getLogger('neo4j.notifications').setLevel(logging.ERROR)
    graphiti = build_graphiti()
    checks = Checks()
    try:
        await graphiti.build_indices_and_constraints()
        await clear_data(graphiti.driver, group_ids=[GROUP_ID])

        print(f'Mode: {mode}')
        await scenario_has_state(graphiti, mode, checks)
        await scenario_has_car(graphiti, mode, checks)
    finally:
        await graphiti.close()

    print('\n' + '=' * 78)
    print('RESULT: all checks passed' if checks.failed == 0 else f'RESULT: {checks.failed} check(s) FAILED')
    if checks.failed and mode == 'graphiti':
        print('Note: in graphiti mode the LLM decides contradictions; a HAS_CAR failure means it judged the')
        print('      second car to contradict the first. Compare with --mode cardinality (see README).')
    print('Explore the graph: http://localhost:7475  (queries in queries.cypher)')
    raise SystemExit(1 if checks.failed else 0)


if __name__ == '__main__':
    asyncio.run(main())
