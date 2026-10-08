"""Per-entity graph context for an LLM.

When a question is about several entities and graph traversal pulls in neighbouring facts,
a flat list of facts loses which entity each fact belongs to, and how it is connected.
Here every fact is kept under the entity it was reached from, tagged with its path:

    <ENTITY name="Mazda CX-5 XYZ-789">
    [direct] Policy POL-200 covers Mazda CX-5 XYZ-789 (since 2026-06-15)
    [via Policy POL-200] Policy POL-200 has state CANCELLED (since 2026-09-01)
    </ENTITY>

"Policy POL-200 has state CANCELLED" never mentions the Mazda; the path is what ties it to it.
"""

import asyncio
from datetime import datetime

from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import ModelSize
from graphiti_core.nodes import EntityNode
from graphiti_core.prompts.models import Message
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF


class ContextFact(BaseModel):
    fact: str
    relation: str
    hops: int  # 1 = the fact involves the entity itself
    via: list[str]  # intermediate entities between the entity and this fact
    valid_at: datetime | None
    invalid_at: datetime | None


class EntityContext(BaseModel):
    entity: str
    facts: list[ContextFact]


class EntityAnswer(BaseModel):
    entity: str
    answer: str


class Answer(BaseModel):
    answer: str = Field(description='Direct answer to the question')
    per_entity: list[EntityAnswer] = Field(
        description='For each entity block, what the answer says about that entity, using only that block'
    )


class _MentionedEntities(BaseModel):
    names: list[str] = Field(description='Entities the question asks about, exactly as written in the question')


async def resolve_entities(graphiti: Graphiti, question: str, group_id: str) -> list[EntityNode]:
    """Entities the question is about: LLM extracts the mentions, node search resolves each one."""
    response = await graphiti.llm_client.generate_response(
        [
            Message(role='system', content='You identify which entities a question is about.'),
            Message(
                role='user',
                content='List the entities (people, vehicles, orders, policies, places, ...) that this '
                f'question asks about, exactly as written. Do not add entities it does not mention.\n\n'
                f'Question: {question}',
            ),
        ],
        response_model=_MentionedEntities,
        model_size=ModelSize.small,
        prompt_name='poc.resolve_entities',
    )

    async def top_node(name: str) -> EntityNode | None:
        nodes = (await graphiti.search_(name, config=NODE_HYBRID_SEARCH_RRF, group_ids=[group_id])).nodes
        return nodes[0] if nodes else None

    resolved = await asyncio.gather(*(top_node(n) for n in response['names'] or [question]))
    return list({n.uuid: n for n in resolved if n is not None}.values())


# Depth is interpolated (Cypher cannot parameterise variable-length bounds); callers validate it.
_TRAVERSAL = """
MATCH path = (a:Entity {uuid: $uuid})-[rels:RELATES_TO*1..%d]-(:Entity)
WHERE all(r IN rels WHERE r.group_id = $group_id
                      AND r.valid_at <= $as_of
                      AND (r.invalid_at IS NULL OR r.invalid_at > $as_of))
WITH last(rels) AS e, length(path) AS hops, [n IN nodes(path)[1..-1] | n.name] AS via
ORDER BY hops
WITH e, head(collect({hops: hops, via: via})) AS shortest
RETURN e.fact AS fact, e.name AS relation, shortest.hops AS hops, shortest.via AS via,
       e.valid_at AS valid_at, e.invalid_at AS invalid_at
ORDER BY hops, relation, fact
"""


async def traverse(
    graphiti: Graphiti, entity: EntityNode, group_id: str, as_of: datetime, depth: int
) -> EntityContext:
    """Facts valid at `as_of` within `depth` hops of the entity, each with its shortest path."""
    records, _, _ = await graphiti.driver.execute_query(
        _TRAVERSAL % depth, uuid=entity.uuid, group_id=group_id, as_of=as_of, routing_='r'
    )
    return EntityContext(
        entity=entity.name,
        facts=[
            ContextFact(
                fact=r['fact'],
                relation=r['relation'],
                hops=r['hops'],
                via=r['via'],
                valid_at=r['valid_at'].to_native() if r['valid_at'] else None,
                invalid_at=r['invalid_at'].to_native() if r['invalid_at'] else None,
            )
            for r in records
        ],
    )


def render(contexts: list[EntityContext], as_of: datetime) -> str:
    lines = [
        'Facts are grouped by the entity the question is about.',
        '[direct] facts involve that entity itself. [via X] facts are about a related entity X reached '
        'through the graph: they describe X, not the entity of the block.',
        f'All facts are valid as of {as_of:%Y-%m-%d}.',
    ]
    for context in contexts:
        lines.append(f'\n<ENTITY name="{context.entity}">')
        for f in context.facts:
            tag = 'direct' if f.hops == 1 else 'via ' + ' -> '.join(f.via)
            since = f' (since {f.valid_at:%Y-%m-%d})' if f.valid_at else ''
            lines.append(f'[{tag}] {f.fact}{since}')
        lines.append('</ENTITY>')
    return '\n'.join(lines)


async def answer(graphiti: Graphiti, question: str, context: str) -> Answer:
    response = await graphiti.llm_client.generate_response(
        [
            Message(
                role='system',
                content='Answer questions using only the knowledge-graph context provided. Attribute each '
                'fact to the entity block it appears in, and respect the [direct] / [via X] tags. If the '
                'context does not contain the answer, say so.',
            ),
            Message(role='user', content=f'{context}\n\nQuestion: {question}'),
        ],
        response_model=Answer,
        model_size=ModelSize.medium,
        prompt_name='poc.answer',
    )
    return Answer(**response)
