# Graphiti temporal POC: `valid_at` / `invalid_at` with fact triples

This POC ingests fact triples with Graphiti (graphiti-core 0.30.2), Neo4j 5.26 and Azure AI Foundry (`gpt-6-luna` + `text-embedding-3-large-863247`). It checks how Graphiti fills `valid_at` / `invalid_at` in two cases:

| # | Relation | Desired behaviour |
|---|----------|-------------------|
| 1 | `HAS_STATE` (single-valued) | A new state invalidates the previous one: `old.invalid_at == new.valid_at` |
| 2 | `HAS_CAR` (multi-valued) | A new car is added and the previous car stays valid. Re-ingesting a car that is already stored reuses its edge. |

## Run it

```bash
docker compose up -d                                  # Neo4j on bolt://localhost:7688, browser http://localhost:7475
uv sync
uv run python poc_temporal.py                         # Graphiti native (LLM decides contradictions)
uv run python poc_temporal.py --mode cardinality      # rule-based contradictions (deterministic)
uv run python measure_llm_invalidation.py --trials 12 # how often native mode gets HAS_CAR wrong

uv run uvicorn api:app --port 8010                    # search API, docs at http://localhost:8010/docs
uv run python validate_search.py                      # checks /search returns only the latest state
uv run python validate_context.py                     # seeds "context-demo", checks /context and /answer
```

Configuration lives in `.env` (see `.env.example`). Each run clears and rebuilds the `temporal-poc` group. After a run, open the Neo4j browser (user `neo4j`, password `graphiti-poc-pass`) and use the queries in [queries.cypher](queries.cypher): full history, current facts, a point-in-time query and a visual graph.

## Results

| Scenario | `--mode graphiti` (native `add_triplet`) | `--mode cardinality` |
|---|---|---|
| 1. `HAS_STATE` replaces the previous state | Passed in 5/5 runs, plus 16/16 isolated trials | Passed in 3/3 runs |
| 2. `HAS_CAR` keeps the previous car | **Failed in 3/5 runs** | Passed in 3/3 runs |
| Run time | ~30 s (several LLM calls per triple) | ~3.5 s (embeddings only) |

`measure_llm_invalidation.py` with 12 trials gave these rates of the second car wrongly invalidating the first:

| Group contents | Wrong invalidations |
|---|---|
| Only the two car facts | 2/12 |
| Also contains the unrelated order-state history | **10/12** |

Example output of scenario 1, which behaves the same in both modes:

```
target      valid_at    invalid_at  expired_at (UTC)     status
PENDING     2026-01-10  2026-01-15  2026-10-07 20:46:22  INVALIDATED
SHIPPED     2026-01-15  2026-01-20  2026-10-07 20:46:22  INVALIDATED
DELIVERED   2026-01-20  -           -                    CURRENT
```

## Search: does it return only the last state?

**Not by default.** `Graphiti.search()` returns current *and* invalidated facts, ranked by relevance (hybrid BM25 + embeddings, merged with RRF), not by recency. For "What is the current state of order ORD-1001?" it returned the three states with the current one **last**:

```
1. SHIPPED    valid_at=2026-01-15 invalid_at=2026-01-20 INVALIDATED
2. PENDING    valid_at=2026-01-10 invalid_at=2026-01-15 INVALIDATED
3. DELIVERED  valid_at=2026-01-20 invalid_at=-          CURRENT
```

Taking the top hit would therefore give a wrong answer. To return only what is true, `api.py` passes Graphiti's own `SearchFilters` so the filter runs inside the Neo4j query, before the result limit:

```
valid_at <= as_of AND (invalid_at IS NULL OR invalid_at > as_of)      -- as_of defaults to now
```

| Request | Result for ORD-1001 |
|---|---|
| `GET /search?query=...` | `DELIVERED` only |
| `GET /search?query=...&as_of=2026-01-17` | `SHIPPED` (state on that date) |
| `GET /search?query=...&include_history=true` | All three (Graphiti default) |
| `GET /search?query=...&relation=HAS_STATE` | Restricts to the given relation names; combinable with the above |
| `GET /search?query=...&scope=entity` | Only facts connected to the entity the question is about (see below) |

Validity intervals are half-open, `[valid_at, invalid_at)`: on 2026-01-20 the state is `DELIVERED`, not `SHIPPED`. "Current only" does not mean a single result: for `HAS_CAR` it returns both current cars. `validate_search.py` checks all of this through the HTTP API and passes 11/11.

The `status` field in the response is relative to *now*. With `as_of=2026-01-17`, `SHIPPED` is returned because it was valid then, and it is still labelled `INVALIDATED`.

`validate_search.py` expects the graph loaded with `--mode cardinality`. After a native run that wrongly invalidated the Toyota, check 4 fails because of ingestion, not search.

Graphiti caveat: in 0.30.2, the date-filter parameters are named only by their position inside an AND group (`$invalid_at_0`). Two OR groups that both carry a date would therefore overwrite each other's value. The filter above has a single dated condition per field, so it is not affected.

### Why asking about one car also returns the other car

For "What do you know about Toyota Corolla ABC-123?" the default search also returns the Mazda. Graphiti's search returns the *top facts related to the question*; it does not return an exact answer. It merges two retrievers:

| Fact | Keyword match (BM25) | Embedding similarity (Neo4j score) | Returned |
|---|---|---|---|
| Juan Perez has car Toyota Corolla ABC-123 | yes (2.46) | 0.762 | #1 |
| Juan Perez has car Mazda CX-5 XYZ-789 | no shared words | 0.657, above the 0.6 cutoff | #2 |
| Order ORD-1001 has state DELIVERED | no | 0.567 | no |

- **The similarity cutoff is loose.** Neo4j reports cosine as `(1 + cos) / 2`, so Graphiti's `0.6` cutoff equals a raw cosine of only **0.2**. The Mazda fact has a raw cosine of 0.315: weakly related ("a car owned by Juan"), but enough to pass. For the bare query `Toyota Corolla ABC-123`, even the order's DELIVERED state passes.
- **RRF merges by rank, not score.** Anything either retriever finds is returned, up to the limit, however large the score gap.

Raising the cutoff is not a reliable fix. The Mazda scores 0.725 for the bare query but 0.657 for the question, so no single threshold separates related from unrelated facts.

**`scope=entity`** resolves the entity the question is about and keeps only the facts connected to it. It takes the top hit of Graphiti's node search (BM25 plus embeddings on entity names), then passes that entity's edge UUIDs as a `SearchFilters.edge_uuids` filter, alongside the temporal filter. The response's `entity` field shows which entity was chosen.

| Question (`scope=entity`) | Entity | Facts returned |
|---|---|---|
| What do you know about Toyota Corolla ABC-123? | Toyota Corolla ABC-123 | Toyota only |
| Which cars does Juan Perez have? | Juan Perez | Both cars |
| What is the current state of order ORD-1001? | Order ORD-1001 | DELIVERED |
| Does Juan Perez own the Mazda CX-5 XYZ-789? | Mazda CX-5 XYZ-789 | Mazda only |

Limitation: only the single top entity is used. A question with no clear entity is anchored to whatever node ranks first, so check `entity` in the response. `scope=related` remains the default because it is what you want when feeding facts to an LLM, where neighbouring context helps.

## Context for an LLM when a question covers several entities

Say a question asks about two cars, and graph traversal pulls in related facts for both. How does the LLM know which fact belongs to which car? Only if the context says so. A flat list of facts loses that. Each Graphiti fact names both of its entities ("Juan Perez has car Toyota…"), but facts two hops away usually don't name the entity in question. "Policy POL-200 has state CANCELLED" never mentions the Mazda, and with many facts the LLM would have to work out those links itself.

`GET /context` keeps that structure:

1. **Find the entities.** The LLM lists the entities the question mentions, and Graphiti's node search maps each one to a graph node.
2. **Traverse from each entity** up to `depth` hops (default 2), following only facts valid at `as_of`. Each fact keeps its shortest path.
3. **Render one block per entity.** `[direct]` marks facts that involve the entity itself. `[via X]` marks facts about a neighbour X, with the path. Facts reachable from both entities appear in both blocks.

```
<ENTITY name="Toyota Corolla ABC-123">
[direct] Policy POL-100 covers Toyota Corolla ABC-123 (since 2025-03-01)
[direct] Toyota Corolla ABC-123 has color Red (since 2025-03-01)
[via Juan Perez] Juan Perez has car Mazda CX-5 XYZ-789 (since 2026-06-15)
[via Policy POL-100] Policy POL-100 has state ACTIVE (since 2025-03-01)
...
</ENTITY>
<ENTITY name="Mazda CX-5 XYZ-789">
[direct] Policy POL-200 covers Mazda CX-5 XYZ-789 (since 2026-06-15)
[direct] Mazda CX-5 XYZ-789 has color Blue (since 2026-06-15)
[via Policy POL-200] Policy POL-200 has state CANCELLED (since 2026-09-01)
...
</ENTITY>
```

The prompt explains the tags: a `[via X]` fact describes X, not the entity of the block. `GET /answer` sends this context to gpt-6-luna and returns an overall answer plus a per-entity answer. For "Compare the Toyota and the Mazda: color, and is its policy active?" it answered: *Toyota: red, POL-100 active. Mazda: blue, POL-200 cancelled.* `validate_context.py` passed 8/8 checks in 4/4 runs. The checks cover block attribution, the `via` paths, the other car appearing only as `[via Juan Perez]`, and the invalidated `POL-200 ACTIVE` state being excluded.

Design notes:

- **Depth is the main lever.** At depth 2, each car already sees the other car through the shared owner. Without the `[via Juan Perez]` tag, that fact is the one most likely to be misattributed.
- **Hub nodes.** Value-like nodes such as `State: ACTIVE` or `Color: Red` are shared by many entities. At depth 3, a policy would reach every other policy in the same state. Keep depth low, or stop traversal at value nodes, before scaling up.
- **Volume.** The traversal returns every valid fact within `depth` hops; it does not rank them by relevance. For large neighbourhoods, rank or cap them, for example with `/search` filtered to the traversed edge UUIDs.

## How Graphiti decides (verified in graphiti-core 0.30.2 source)

**Four timestamps on every edge.** Graphiti is bi-temporal:

- `valid_at` / `invalid_at` are *event time*: when the fact was true in the real world. You supply `valid_at` from your triple.
- `created_at` / `expired_at` are *transaction time*: when Graphiti stored and retired the fact. `expired_at` is the wall clock at the moment of invalidation.

**Nothing is deleted.** An invalidated edge stays in the graph with `invalid_at` and `expired_at` set, so the full history can be queried. Graphiti's search does **not** filter out invalidated edges by default. To get only current facts, filter `invalid_at IS NULL` or pass `SearchFilters`.

**Invalidation is an LLM judgment on the fact text.** `add_triplet` gathers two lists:

- edges between the same two nodes, used for duplicate detection;
- a hybrid search over the **whole `group_id`**, used as invalidation candidates.

It then asks the LLM (prompt `dedupe_edges.resolve_edge`) which candidates the new fact *contradicts*. Graphiti has no concept of relation cardinality, and the relation name (`HAS_CAR`) is not what is compared; the `fact` sentence is. This explains the results:

- "Order ORD-1001 has state SHIPPED" versus "... has state PENDING" is reliably judged a contradiction.
- "Juan Perez has car Mazda" versus "Juan Perez has car Toyota" is *sometimes* judged a contradiction. This happens far more often when other "replacement-style" facts appear among the candidates. Those candidates can be any fact in the group, even ones attached to unrelated nodes.

**Applying the contradiction (`resolve_edge_contradictions`).** If an old edge has `valid_at < new.valid_at`, it gets `invalid_at = new.valid_at` and `expired_at = now`. If the new triple is older than a contradicting current edge (a late arrival), the new edge is stored already invalidated.

**Re-ingesting the same triple.** When the fact text matches an edge between the same nodes exactly, Graphiti reuses that edge and keeps its original `valid_at`. It does not check whether that edge is invalidated. So once `HAS_CAR Toyota` has been wrongly invalidated, re-sending it does **not** bring it back. This was observed in the failing runs.

**`valid_at` should always come from your data.** If an edge has no `valid_at`, Graphiti asks the LLM to infer one from the fact text, using *now* as the reference time.

## Recommendation

Your triples come from structured data, so you already know each relation's cardinality. Do not leave contradiction detection to the LLM. `cardinality_ingest.py` keeps Graphiti's data model, temporal rules (`resolve_edge_contradictions`) and persistence (`add_nodes_and_edges_bulk`). It replaces only the LLM's "is this a contradiction?" decision with a declared rule:

```python
RELATION_CARDINALITY = {'HAS_STATE': Cardinality.ONE, 'HAS_CAR': Cardinality.MANY}
await add_triplet_with_cardinality(graphiti, source, edge, target, RELATION_CARDINALITY[edge.name])
```

- **`ONE`**: the current edges for the same `(source, relation)` are the only invalidation candidates.
- **`MANY`**: there are no candidates, so nothing is invalidated.
- **Same triple while still current**: the existing edge is reused and keeps its original `valid_at`.
- **Same triple after it was invalidated**: a new edge is created, starting a new validity interval (for example, a state returning to `PENDING`). The POC does not exercise this case.

Native `add_triplet` / `add_episode` remain useful for unstructured text, where cardinality is not known up front.

## Integration notes for Foundry and gpt-6-luna

- **LLM client.** `foundry_graphiti.FoundryReasoningLLMClient` is used instead of Graphiti's `AzureOpenAILLMClient`, for two reasons:
  - In 0.30.2, `AzureOpenAILLMClient._handle_structured_response` returns a dict, while the base class unpacks a 3-tuple. That breaks every structured call.
  - Graphiti only treats `o1*`/`o3*`/`gpt-5*` as reasoning models. For `gpt-6-luna` it would send `max_tokens`, which the deployment rejects, and fall back to reasoning effort `minimal`, which the model does not support (`none`/`low`/…/`max` only). The client always uses the Responses API with `LLM_REASONING_EFFORT` (default `low`).
- **Small model.** `small_model` must also be set to the deployment, because edge dedup and contradiction run on the "small" model.
- **Endpoint.** It is the resource's OpenAI v1 URL (`https://<resource>.services.ai.azure.com/openai/v1/`), not the `/api/projects/...` project URL.
- **Embeddings.** They are stored at full 3072 dimensions. That is fine as long as every vector comes from the same model.
- **`httpx`.** It is added explicitly because graphiti-core imports it but openai 3.x no longer pulls it in.

## Files

| File | Purpose |
|---|---|
| `poc_temporal.py` | The two scenarios, with PASS/FAIL checks and `--mode graphiti\|cardinality` |
| `cardinality_ingest.py` | Deterministic `add_triplet_with_cardinality` |
| `foundry_graphiti.py` | Graphiti factory for Foundry and Neo4j, plus the reasoning-model LLM client |
| `measure_llm_invalidation.py` | Measures the native false-invalidation rate for `HAS_CAR` |
| `api.py` | FastAPI `GET /search` (`as_of`, `include_history`, `relation`, `scope`), `GET /context`, `GET /answer` |
| `validate_search.py` | Calls `/search` and checks that only the latest state is returned |
| `graph_context.py` | Entity resolution, per-entity traversal with paths, prompt rendering, LLM answer |
| `validate_context.py` | Seeds the `context-demo` graph and checks `/context` and `/answer` |
| `queries.cypher` | Neo4j browser queries for history, current state and point-in-time views |
| `docker-compose.yml` | Neo4j for this POC on ports 7475/7688, so it does not clash with other local instances |
