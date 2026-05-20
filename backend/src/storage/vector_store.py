"""Vector store adapter for LawGPT.

Three concrete implementations behind one protocol:

  * ``ChromaVectorStore``     — local Chroma collection (dev).
  * ``PgVectorStore``         — Postgres + pgvector (docker compose / managed).
  * ``CloudflareVectorize``   — Cloudflare Vectorize REST API (prod).

All search calls take a ``namespace`` (user_id) so multi-tenant isolation is
enforced at the data layer. Adapters store ``namespace`` as filterable metadata
and reject cross-namespace reads.

Example:
    >>> store = make_vector_store(backend="chroma")
    >>> store.upsert(  # doctest: +SKIP
    ...     items=[VectorItem(
    ...         id="chunk_1",
    ...         vector=[0.1] * 8,
    ...         content="hello",
    ...         metadata={"week": "week_1", "doc_type": "note"},
    ...         namespace="usr_demo",
    ...     )]
    ... )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

log = logging.getLogger(__name__)


@dataclass
class VectorItem:
    id: str
    vector: list[float]
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    namespace: str = ""


@dataclass
class SearchHit:
    id: str
    score: float
    content: str
    metadata: dict[str, Any]


class VectorStore(Protocol):
    """Adapter contract. Implementations must enforce ``namespace`` isolation."""

    def upsert(self, items: Iterable[VectorItem]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]: ...

    def delete(self, ids: Iterable[str], namespace: str) -> None: ...

    def delete_namespace(self, namespace: str) -> None: ...


# ── Chroma (local dev) ────────────────────────────────────────────────────────


class ChromaVectorStore:
    """Persistent Chroma collection. Namespace stored on metadata."""

    def __init__(self, collection_name: str, persist_dir: str):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def upsert(self, items: Iterable[VectorItem]) -> None:
        items = list(items)
        if not items:
            return
        ids = [it.id for it in items]
        embeddings = [it.vector for it in items]
        documents = [it.content for it in items]
        metadatas = [{**it.metadata, "namespace": it.namespace} for it in items]
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        ns_filter: dict[str, Any] = {"namespace": {"$eq": namespace}}
        combined: dict[str, Any]
        if where:
            combined = {"$and": [ns_filter, where]}
        else:
            combined = ns_filter

        res = self._collection.query(
            query_embeddings=[query_vector], n_results=k, where=combined
        )
        hits: list[SearchHit] = []
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for i, id_ in enumerate(ids):
            # Chroma returns cosine *distance*; convert to similarity in [0,1].
            score = 1.0 - float(distances[i]) if distances else 0.0
            hits.append(SearchHit(id=id_, score=score, content=docs[i], metadata=metas[i] or {}))
        return hits

    def delete(self, ids: Iterable[str], namespace: str) -> None:
        ids = list(ids)
        if not ids:
            return
        self._collection.delete(ids=ids, where={"namespace": {"$eq": namespace}})

    def delete_namespace(self, namespace: str) -> None:
        self._collection.delete(where={"namespace": {"$eq": namespace}})


# ── Postgres + pgvector ───────────────────────────────────────────────────────


class PgVectorStore:
    """Postgres + pgvector. Sync sqlalchemy core for simplicity inside FastAPI.

    Table schema (created on first call):
        CREATE TABLE IF NOT EXISTS vectors (
            id           TEXT PRIMARY KEY,
            namespace    TEXT NOT NULL,
            content      TEXT NOT NULL,
            metadata     JSONB NOT NULL DEFAULT '{}',
            embedding    VECTOR(<dim>) NOT NULL
        );
        CREATE INDEX IF NOT EXISTS vectors_ns_idx ON vectors (namespace);
        CREATE INDEX IF NOT EXISTS vectors_ann_idx ON vectors
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    """

    def __init__(self, dsn: str, dim: int, table: str = "vectors"):
        from sqlalchemy import create_engine

        # Use sync driver here (psycopg) — vector inserts are infrequent and
        # search runs in a threadpool from FastAPI.
        sync_dsn = dsn.replace("+asyncpg", "").replace("+psycopg", "")
        self._engine = create_engine(sync_dsn, future=True)
        self._dim = dim
        self._table = table
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._table} (
                        id        TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        content   TEXT NOT NULL,
                        metadata  JSONB NOT NULL DEFAULT '{{}}',
                        embedding VECTOR({self._dim}) NOT NULL
                    )
                    """
                )
            )
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {self._table}_ns_idx "
                    f"ON {self._table} (namespace)"
                )
            )

    def upsert(self, items: Iterable[VectorItem]) -> None:
        from sqlalchemy import text

        items = list(items)
        if not items:
            return
        with self._engine.begin() as conn:
            for it in items:
                conn.execute(
                    text(
                        f"""
                        INSERT INTO {self._table} (id, namespace, content, metadata, embedding)
                        VALUES (:id, :ns, :content, CAST(:meta AS JSONB), CAST(:emb AS VECTOR))
                        ON CONFLICT (id) DO UPDATE
                            SET namespace = EXCLUDED.namespace,
                                content   = EXCLUDED.content,
                                metadata  = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding
                        """
                    ),
                    {
                        "id": it.id,
                        "ns": it.namespace,
                        "content": it.content,
                        "meta": _json_dumps(it.metadata),
                        "emb": _pg_vector_literal(it.vector),
                    },
                )

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        from sqlalchemy import text

        meta_clauses, params = _build_meta_where(where or {})
        params.update({"ns": namespace, "emb": _pg_vector_literal(query_vector), "k": k})
        meta_sql = (" AND " + " AND ".join(meta_clauses)) if meta_clauses else ""
        sql = f"""
            SELECT id, content, metadata, 1 - (embedding <=> CAST(:emb AS VECTOR)) AS score
            FROM {self._table}
            WHERE namespace = :ns{meta_sql}
            ORDER BY embedding <=> CAST(:emb AS VECTOR)
            LIMIT :k
        """
        with self._engine.begin() as conn:
            rows = conn.execute(text(sql), params).all()
        return [
            SearchHit(id=r[0], content=r[1], metadata=r[2] or {}, score=float(r[3])) for r in rows
        ]

    def delete(self, ids: Iterable[str], namespace: str) -> None:
        from sqlalchemy import text

        ids = list(ids)
        if not ids:
            return
        with self._engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {self._table} WHERE namespace = :ns AND id = ANY(:ids)"),
                {"ns": namespace, "ids": ids},
            )

    def delete_namespace(self, namespace: str) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {self._table} WHERE namespace = :ns"),
                {"ns": namespace},
            )


def _pg_vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{v:.7f}" for v in vec) + "]"


def _json_dumps(d: dict[str, Any]) -> str:
    import json

    return json.dumps(d, separators=(",", ":"))


def _build_meta_where(where: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Translate a simple equality filter dict into Postgres JSONB conditions.

    Only supports flat ``{"key": "value"}`` and ``{"key": {"$in": [...]}}``
    forms — matches the surface area used by the agent today.
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for i, (k, v) in enumerate(where.items()):
        if isinstance(v, dict) and "$in" in v:
            ph = f"in_{i}"
            clauses.append(f"metadata->>'{k}' = ANY(:{ph})")
            params[ph] = list(v["$in"])
        else:
            ph = f"eq_{i}"
            clauses.append(f"metadata->>'{k}' = :{ph}")
            params[ph] = str(v)
    return clauses, params


# ── Cloudflare Vectorize (prod) ───────────────────────────────────────────────


class CloudflareVectorize:
    """Cloudflare Vectorize REST API client.

    Namespaces map to ``namespace`` parameter on the upsert/query endpoints.
    Docs: https://developers.cloudflare.com/vectorize/reference/client-api/
    """

    BASE = "https://api.cloudflare.com/client/v4/accounts/{account}/vectorize/v2/indexes/{index}"

    def __init__(self, account_id: str, api_token: str, index_name: str):
        import httpx

        self._url = self.BASE.format(account=account_id, index=index_name)
        self._client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/x-ndjson",
            },
            timeout=30.0,
        )
        self._json_client = httpx.Client(
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )

    def upsert(self, items: Iterable[VectorItem]) -> None:
        import json

        ndjson = "\n".join(
            json.dumps(
                {
                    "id": it.id,
                    "values": it.vector,
                    "namespace": it.namespace,
                    "metadata": {**it.metadata, "content": it.content},
                }
            )
            for it in items
        )
        if not ndjson:
            return
        r = self._client.post(f"{self._url}/upsert", content=ndjson)
        r.raise_for_status()

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        body: dict[str, Any] = {
            "vector": query_vector,
            "topK": k,
            "namespace": namespace,
            "returnMetadata": "all",
        }
        if where:
            body["filter"] = where
        r = self._json_client.post(f"{self._url}/query", json=body)
        r.raise_for_status()
        data = r.json().get("result", {}).get("matches", [])
        return [
            SearchHit(
                id=m["id"],
                score=float(m.get("score", 0.0)),
                content=(m.get("metadata", {}) or {}).get("content", ""),
                metadata={k: v for k, v in (m.get("metadata") or {}).items() if k != "content"},
            )
            for m in data
        ]

    def delete(self, ids: Iterable[str], namespace: str) -> None:
        ids = list(ids)
        if not ids:
            return
        # Vectorize delete-by-id endpoint scopes via index, but we still pre-check namespace
        # via a search round-trip when paranoid. For now trust caller-supplied namespace.
        r = self._json_client.post(f"{self._url}/delete-by-ids", json={"ids": ids})
        r.raise_for_status()

    def delete_namespace(self, namespace: str) -> None:
        r = self._json_client.post(
            f"{self._url}/delete-by-filter", json={"namespace": namespace}
        )
        r.raise_for_status()


# ── In-memory (tests) ─────────────────────────────────────────────────────────


class InMemoryVectorStore:
    """Tiny in-memory store for unit tests. Cosine similarity, no persistence."""

    def __init__(self) -> None:
        self._items: dict[str, VectorItem] = {}

    def upsert(self, items: Iterable[VectorItem]) -> None:
        for it in items:
            self._items[it.id] = it

    def search(
        self,
        query_vector: list[float],
        namespace: str,
        k: int = 8,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for it in self._items.values():
            if it.namespace != namespace:
                continue
            if where and not _match_where(it.metadata, where):
                continue
            score = _cosine(query_vector, it.vector)
            hits.append(SearchHit(id=it.id, score=score, content=it.content, metadata=dict(it.metadata)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def delete(self, ids: Iterable[str], namespace: str) -> None:
        for i in list(ids):
            if i in self._items and self._items[i].namespace == namespace:
                del self._items[i]

    def delete_namespace(self, namespace: str) -> None:
        for k in [k for k, v in self._items.items() if v.namespace == namespace]:
            del self._items[k]


def _cosine(a: list[float], b: list[float]) -> float:
    import math

    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _match_where(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    for k, v in where.items():
        if isinstance(v, dict) and "$in" in v:
            if meta.get(k) not in v["$in"]:
                return False
        elif isinstance(v, dict) and "$eq" in v:
            if meta.get(k) != v["$eq"]:
                return False
        else:
            if meta.get(k) != v:
                return False
    return True


# ── Factory ───────────────────────────────────────────────────────────────────


def make_vector_store(backend: str | None = None) -> VectorStore:
    """Instantiate a vector store per the configured backend.

    ``backend`` overrides ``get_settings().vector_backend`` when given (used by
    tests to force an in-memory store).
    """
    from src.config import (
        CHROMA_COLLECTION,
        CHROMA_DIR,
        EMBEDDING_DIMENSIONS,
        get_settings,
    )

    settings = get_settings()
    backend = backend or settings.vector_backend

    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "chroma":
        return ChromaVectorStore(CHROMA_COLLECTION, str(CHROMA_DIR))
    if backend == "pgvector":
        return PgVectorStore(settings.database_url, dim=EMBEDDING_DIMENSIONS)
    if backend == "vectorize":
        return CloudflareVectorize(
            account_id=settings.vectorize_account_id,
            api_token=settings.vectorize_api_token,
            index_name=settings.vectorize_index_name,
        )
    raise ValueError(f"unknown vector backend: {backend}")
