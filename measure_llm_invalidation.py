"""Measure how often Graphiti.add_triplet wrongly invalidates a multi-valued relation.

Each trial runs in its own throwaway group_id: ingest HAS_CAR Toyota, then HAS_CAR Mazda,
and record whether the LLM marked the Toyota fact as contradicted. Trials are repeated
with and without unrelated order-state history in the same group, because the candidate
facts the LLM sees come from a hybrid search over the whole group.

Run:  uv run python measure_llm_invalidation.py [--trials 10]
"""

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from uuid import uuid4

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

from foundry_graphiti import build_graphiti


async def add(graphiti: Graphiti, group_id: str, source: EntityNode, relation: str,
              target: EntityNode, fact: str, day: int):
    edge = EntityEdge(
        group_id=group_id,
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name=relation,
        fact=fact,
        valid_at=datetime(2026, 1, day, tzinfo=timezone.utc),
        created_at=utc_now(),
    )
    return await graphiti.add_triplet(source, edge, target)


async def trial(graphiti: Graphiti, with_history: bool) -> bool:
    group_id = f'measure-{uuid4().hex[:10]}'

    def node(label: str, name: str) -> EntityNode:
        return EntityNode(name=name, labels=[label], group_id=group_id)

    try:
        if with_history:
            order = node('Order', 'Order ORD-1001')
            for day, state in ((1, 'PENDING'), (2, 'SHIPPED'), (3, 'DELIVERED')):
                await add(graphiti, group_id, order, 'HAS_STATE', node('State', state),
                          f'Order ORD-1001 has state {state}', day)
        person = node('Person', 'Juan Perez')
        await add(graphiti, group_id, person, 'HAS_CAR', node('Car', 'Toyota Corolla ABC-123'),
                  'Juan Perez has car Toyota Corolla ABC-123', 10)
        result = await add(graphiti, group_id, person, 'HAS_CAR', node('Car', 'Mazda CX-5 XYZ-789'),
                           'Juan Perez has car Mazda CX-5 XYZ-789', 20)
        return len(result.edges) > 1  # edges[1:] are the invalidated edges
    finally:
        await clear_data(graphiti.driver, group_ids=[group_id])


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    trials = parser.parse_args().trials

    logging.basicConfig(level=logging.ERROR)
    graphiti = build_graphiti()
    try:
        await graphiti.build_indices_and_constraints()
        for with_history in (False, True):
            results = await asyncio.gather(*(trial(graphiti, with_history) for _ in range(trials)))
            label = 'with order history in group' if with_history else 'empty group'
            print(f'{label:<30} Toyota HAS_CAR invalidated by Mazda HAS_CAR in {sum(results)}/{trials} trials')
    finally:
        await graphiti.close()


if __name__ == '__main__':
    asyncio.run(main())
