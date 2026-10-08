"""Search API over the POC graph.

    GET /search?query=...                        facts valid now (latest state only)
    GET /search?query=...&as_of=2026-01-17       facts valid at that instant
    GET /search?query=...&include_history=true   Graphiti's default search: current AND invalidated facts
    GET /search?query=...&relation=HAS_STATE     restrict to one or more relation names
    GET /search?query=...&scope=entity           only facts connected to the entity the question is about

scope=related (default) is Graphiti's hybrid search: the top facts related to the question,
which includes neighbouring facts (asking about one car also returns the owner's other cars).

    GET /context?query=...&depth=2               per-entity graph context, ready for an LLM prompt
    GET /answer?query=...&depth=2                LLM answer built from that context

Run:  uv run uvicorn api:app --port 8010   (docs at http://localhost:8010/docs)
"""

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_filters import ComparisonOperator, DateFilter, SearchFilters
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now

from foundry_graphiti import build_graphiti
from graph_context import EntityAnswer, EntityContext, answer, render, resolve_entities, traverse
from poc_temporal import GROUP_ID


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graphiti = build_graphiti()
    yield
    await app.state.graphiti.close()


app = FastAPI(title='Graphiti temporal POC - search', lifespan=lifespan)


class Fact(BaseModel):
    fact: str
    relation: str
    source: str
    target: str
    valid_at: datetime | None
    invalid_at: datetime | None
    expired_at: datetime | None
    status: str  # CURRENT / INVALIDATED, relative to now


class SearchResponse(BaseModel):
    query: str
    as_of: datetime | None  # None when include_history=true
    entity: str | None  # entity the results were anchored to (scope=entity)
    results: list[Fact]  # in Graphiti's relevance order


def valid_at_instant(as_of: datetime, relations: list[str] | None) -> SearchFilters:
    # valid_at <= as_of AND (invalid_at IS NULL OR invalid_at > as_of)
    return SearchFilters(
        edge_types=relations,
        valid_at=[[DateFilter(date=as_of, comparison_operator=ComparisonOperator.less_than_equal)]],
        invalid_at=[
            [DateFilter(comparison_operator=ComparisonOperator.is_null)],
            [DateFilter(date=as_of, comparison_operator=ComparisonOperator.greater_than)],
        ],
    )


@app.get('/search', response_model=SearchResponse)
async def search(
    query: str,
    as_of: datetime | None = Query(None, description='Instant to evaluate validity at (default: now; naive = UTC)'),
    include_history: bool = Query(False, description='Return invalidated facts too (Graphiti default behaviour)'),
    relation: list[str] | None = Query(None, description='Relation names to keep, e.g. HAS_STATE'),
    scope: Literal['related', 'entity'] = Query(
        'related', description='entity: only facts connected to the entity the question is about'
    ),
    group_id: str = GROUP_ID,
    limit: int = Query(10, ge=1, le=50),
) -> SearchResponse:
    if include_history and as_of is not None:
        raise HTTPException(400, 'as_of cannot be combined with include_history=true')

    if include_history:
        search_filter = SearchFilters(edge_types=relation)
    else:
        as_of = ensure_utc(as_of) or utc_now()
        search_filter = valid_at_instant(as_of, relation)

    graphiti = app.state.graphiti

    entity: EntityNode | None = None
    if scope == 'entity':
        # The entity is the top hit of Graphiti's node search (BM25 + embeddings on names).
        nodes = (await graphiti.search_(query, config=NODE_HYBRID_SEARCH_RRF, group_ids=[group_id])).nodes
        if not nodes:
            return SearchResponse(query=query, as_of=as_of, entity=None, results=[])
        entity = nodes[0]
        search_filter.edge_uuids = [
            e.uuid for e in await EntityEdge.get_by_node_uuid(graphiti.driver, entity.uuid)
        ]

    edges = await graphiti.search(query, group_ids=[group_id], num_results=limit, search_filter=search_filter)

    node_uuids = list({u for e in edges for u in (e.source_node_uuid, e.target_node_uuid)})
    names = {n.uuid: n.name for n in await EntityNode.get_by_uuids(graphiti.driver, node_uuids)}
    now = utc_now()

    return SearchResponse(
        query=query,
        as_of=as_of,
        entity=entity.name if entity else None,
        results=[
            Fact(
                fact=e.fact,
                relation=e.name,
                source=names.get(e.source_node_uuid, e.source_node_uuid),
                target=names.get(e.target_node_uuid, e.target_node_uuid),
                valid_at=e.valid_at,
                invalid_at=e.invalid_at,
                expired_at=e.expired_at,
                status='CURRENT' if e.invalid_at is None or e.invalid_at > now else 'INVALIDATED',
            )
            for e in edges
        ],
    )


class ContextResponse(BaseModel):
    query: str
    as_of: datetime
    entities: list[EntityContext]
    prompt_context: str  # what is sent to the LLM


class AnswerResponse(ContextResponse):
    answer: str
    per_entity: list[EntityAnswer]


async def build_context(query: str, as_of: datetime | None, depth: int, group_id: str) -> ContextResponse:
    graphiti = app.state.graphiti
    as_of = ensure_utc(as_of) or utc_now()
    entities = await resolve_entities(graphiti, query, group_id)
    contexts = [await traverse(graphiti, e, group_id, as_of, depth) for e in entities]
    return ContextResponse(query=query, as_of=as_of, entities=contexts, prompt_context=render(contexts, as_of))


@app.get('/context', response_model=ContextResponse)
async def context(
    query: str,
    as_of: datetime | None = Query(None, description='Instant to evaluate validity at (default: now; naive = UTC)'),
    depth: int = Query(2, ge=1, le=3, description='Traversal hops from each entity'),
    group_id: str = GROUP_ID,
) -> ContextResponse:
    return await build_context(query, as_of, depth, group_id)


@app.get('/answer', response_model=AnswerResponse)
async def answer_question(
    query: str,
    as_of: datetime | None = Query(None, description='Instant to evaluate validity at (default: now; naive = UTC)'),
    depth: int = Query(2, ge=1, le=3, description='Traversal hops from each entity'),
    group_id: str = GROUP_ID,
) -> AnswerResponse:
    ctx = await build_context(query, as_of, depth, group_id)
    result = await answer(app.state.graphiti, query, ctx.prompt_context)
    return AnswerResponse(**ctx.model_dump(), answer=result.answer, per_entity=result.per_entity)
