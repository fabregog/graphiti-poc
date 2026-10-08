"""Deterministic alternative to Graphiti.add_triplet for structured fact triples.

Graphiti.add_triplet asks the LLM whether the new fact contradicts existing facts, so a
multi-valued relation (HAS_CAR) is sometimes invalidated as if it were single-valued.
For triples coming from a system of record the relation's cardinality is already known,
so here the contradiction decision is a rule. The temporal bookkeeping
(resolve_edge_contradictions) and persistence (add_nodes_and_edges_bulk) are Graphiti's own,
so valid_at / invalid_at / expired_at follow exactly the same semantics.
"""

from enum import Enum

from graphiti_core import Graphiti
from graphiti_core.edges import EntityEdge, create_entity_edge_embeddings
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import AddTripletResults
from graphiti_core.nodes import EntityNode, create_entity_node_embeddings
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.edge_operations import resolve_edge_contradictions


class Cardinality(Enum):
    ONE = 'one'  # a new value replaces the current one (e.g. HAS_STATE)
    MANY = 'many'  # values accumulate (e.g. HAS_CAR)


async def add_triplet_with_cardinality(
    graphiti: Graphiti,
    source: EntityNode,
    edge: EntityEdge,
    target: EntityNode,
    cardinality: Cardinality,
) -> AddTripletResults:
    """Same contract as Graphiti.add_triplet; edges[0] is the resolved edge, edges[1:] the invalidated ones."""
    nodes = [await _existing_or(graphiti, source), await _existing_or(graphiti, target)]

    current = [
        e
        for e in await EntityEdge.get_by_node_uuid(graphiti.driver, source.uuid)
        if e.source_node_uuid == source.uuid and e.name == edge.name and e.invalid_at is None
    ]

    # The same triple arriving again keeps the original edge (and its valid_at).
    for e in current:
        if e.target_node_uuid == target.uuid:
            return AddTripletResults(edges=[e], nodes=nodes)

    candidates = current if cardinality is Cardinality.ONE else []

    # Late-arriving triple: if a newer value is already current, the new edge is born
    # already superseded (same rule Graphiti applies in resolve_extracted_edge).
    newer = [c for c in candidates if c.valid_at and edge.valid_at and c.valid_at > edge.valid_at]
    if newer:
        edge.invalid_at = min(c.valid_at for c in newer)
        edge.expired_at = utc_now()

    invalidated = resolve_edge_contradictions(edge, candidates)

    # Edges loaded from the graph come back without embeddings; re-embed so saving
    # them does not wipe the stored vectors (Graphiti.add_triplet does the same).
    edges = [edge, *invalidated]
    await create_entity_edge_embeddings(graphiti.embedder, edges)
    await create_entity_node_embeddings(graphiti.embedder, nodes)
    await add_nodes_and_edges_bulk(graphiti.driver, [], [], nodes, edges, graphiti.embedder)
    return AddTripletResults(edges=edges, nodes=nodes)


async def _existing_or(graphiti: Graphiti, node: EntityNode) -> EntityNode:
    try:
        return await EntityNode.get_by_uuid(graphiti.driver, node.uuid)
    except NodeNotFoundError:
        return node
