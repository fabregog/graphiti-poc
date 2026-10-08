// Graphiti stores every fact as (:Entity)-[:RELATES_TO {name: <predicate>}]->(:Entity).
// Open http://localhost:7475 (user neo4j / graphiti-poc-pass) and run any of these.

// 1. Full temporal history of every fact in the POC
MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
WHERE e.group_id = 'temporal-poc'
RETURN s.name AS source, e.name AS relation, t.name AS target,
       e.valid_at AS valid_at, e.invalid_at AS invalid_at,
       e.created_at AS created_at, e.expired_at AS expired_at,
       CASE WHEN e.invalid_at IS NULL THEN 'CURRENT' ELSE 'INVALIDATED' END AS status
ORDER BY source, relation, valid_at;

// 2. What is true NOW (current facts only)
MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
WHERE e.group_id = 'temporal-poc' AND e.invalid_at IS NULL
RETURN s.name AS source, e.name AS relation, t.name AS target, e.valid_at AS since
ORDER BY source, relation;

// 3. Point-in-time query: what was true on a given date (event time)
WITH datetime('2026-01-17T00:00:00Z') AS asOf
MATCH (s:Entity)-[e:RELATES_TO]->(t:Entity)
WHERE e.group_id = 'temporal-poc'
  AND e.valid_at <= asOf
  AND (e.invalid_at IS NULL OR e.invalid_at > asOf)
RETURN s.name AS source, e.name AS relation, t.name AS target;

// 4. Visual graph of the POC (current and invalidated edges)
MATCH p = (s:Entity)-[e:RELATES_TO]->(t:Entity)
WHERE e.group_id = 'temporal-poc'
RETURN p;
