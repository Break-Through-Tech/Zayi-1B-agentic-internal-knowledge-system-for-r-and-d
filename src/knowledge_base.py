"""ChromaDB knowledge base over the chunked papers (Task 5).

Each chunking configuration gets its OWN collection, named by the chunks'
config fingerprint — so experiments with different chunk sizes/overlaps can
coexist and be compared by retrieval quality without ever mixing or
colliding (chunk ids are only unique within a config).
"""

from functools import lru_cache

import chromadb

from src.embedding import DEFAULT_MODEL, embed_query, embed_texts

COLLECTION_PREFIX = "papers"
_UPSERT_BATCH = 500

# Local cross-encoder for reranking (challenge stretch goal #2): re-scores
# the top candidates with a model that reads query and passage TOGETHER,
# which separates near-misses better than embedding distance alone.
DEFAULT_RERANKER = "ms-marco-MiniLM-L-12-v2"


@lru_cache(maxsize=2)
def _get_ranker(model_name=DEFAULT_RERANKER):
    from flashrank import Ranker

    return Ranker(model_name=model_name)


def collection_name_for(chunks):
    """`papers_<config_id>` — refuses a mixed-config chunk list."""
    config_ids = {c["metadata"]["config_id"] for c in chunks}
    if len(config_ids) != 1:
        raise ValueError(
            f"chunks span {len(config_ids)} configs ({sorted(config_ids)}); "
            "build one collection per config"
        )
    return f"{COLLECTION_PREFIX}_{config_ids.pop()}"


def build_knowledge_base(chunks, client=None, path="chroma", model_name=DEFAULT_MODEL):
    """Embed the chunks and upsert them (ids, texts, metadata, vectors) into
    a per-config ChromaDB collection. Re-running with the same chunks is
    idempotent. Returns the collection."""
    if client is None:
        client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(
        collection_name_for(chunks),
        # the embedding model is recorded so a query can never silently use
        # a different model than the one that built the vectors
        metadata={"hnsw:space": "cosine", "embedding_model": model_name},
    )
    embeddings = embed_texts([c["text"] for c in chunks], model_name)
    for i in range(0, len(chunks), _UPSERT_BATCH):
        batch = chunks[i : i + _UPSERT_BATCH]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=embeddings[i : i + _UPSERT_BATCH].tolist(),
        )
    return collection


def search(collection, query, k=5, model_name=DEFAULT_MODEL, where=None, rerank=False):
    """Semantic search; returns result dicts with the fields needed for a
    cited answer: text, similarity, paper id/title, and the page range.

    With `rerank=True`, a wider candidate set is fetched and re-scored by a
    local cross-encoder (FlashRank) before returning the top k — results
    then carry a `rerank_score` and are ordered by it."""
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with is not None and built_with != model_name:
        raise ValueError(
            f"collection was built with {built_with!r} but query uses "
            f"{model_name!r}; mixed embedding spaces return garbage"
        )
    n_candidates = max(k * 4, 16) if rerank else k
    result = collection.query(
        query_embeddings=[embed_query(query, model_name).tolist()],
        n_results=min(n_candidates, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    rows = []
    for chunk_id, text, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        rows.append(
            {
                "chunk_id": chunk_id,
                "similarity": round(1.0 - distance, 4),  # cosine space
                "paper_id": metadata["paper_id"],
                "paper_title": metadata["paper_title"],
                "page_start": metadata["page_start"],
                "page_end": metadata["page_end"],
                "text": text,
            }
        )
    if rerank:
        rows = _rerank(query, rows)[:k]
    return rows


def _rerank(query, rows, model_name=DEFAULT_RERANKER):
    from flashrank import RerankRequest

    ranked = _get_ranker(model_name).rerank(
        RerankRequest(
            query=query,
            passages=[{"id": r["chunk_id"], "text": r["text"]} for r in rows],
        )
    )
    by_id = {r["chunk_id"]: r for r in rows}
    return [
        dict(by_id[item["id"]], rerank_score=round(float(item["score"]), 4))
        for item in ranked
    ]
