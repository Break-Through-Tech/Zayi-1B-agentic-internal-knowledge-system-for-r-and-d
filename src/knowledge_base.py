"""ChromaDB knowledge base over the chunked papers (Task 5).

Each chunking configuration gets its OWN collection, named by the chunks'
config fingerprint — so experiments with different chunk sizes/overlaps can
coexist and be compared by retrieval quality without ever mixing or
colliding (chunk ids are only unique within a config).
"""

import hashlib
from functools import lru_cache

import chromadb

from src.embedding import DEFAULT_MODEL, embed_query, embed_texts

COLLECTION_PREFIX = "papers"
_UPSERT_BATCH = 500

# Local cross-encoder for reranking (challenge stretch goal #2): re-scores
# the top candidates with a model that reads query and passage TOGETHER,
# which separates near-misses better than embedding distance alone.
# CAVEAT: the cross-encoder reads query+passage as ONE 512-token sequence,
# so with default-size (≈510-token) chunks roughly half the pairs get their
# tails truncated — reranking is most trustworthy on 256–384-token chunk
# collections. Measured 2026-09-20; see Progress.md notes.
DEFAULT_RERANKER = "ms-marco-MiniLM-L-12-v2"
_RERANKER_MAX_LENGTH = 512  # the cross-encoder's hard limit


@lru_cache(maxsize=2)
def _get_ranker(model_name=DEFAULT_RERANKER):
    from flashrank import Ranker

    return Ranker(model_name=model_name, max_length=_RERANKER_MAX_LENGTH)


def _corpus_hash(chunks):
    """Deterministic identity of the exact indexed content, so a collection
    can tell whether it was built from the same chunk texts."""
    digest = hashlib.sha1()
    for chunk in chunks:
        digest.update(chunk["chunk_id"].encode())
        digest.update(b"\x00")
        digest.update(chunk["text"].encode())
        digest.update(b"\x01")
    return digest.hexdigest()[:12]


def _require_cosine(collection):
    """`search` scores results as 1 - distance, which is only meaningful in
    cosine space — and Chroma keeps whatever space a same-named collection
    was FIRST created with, ignoring later creation metadata. Anything
    non-cosine is rejected rather than silently mis-scored."""
    config = getattr(collection, "configuration", None)
    space = (config.get("hnsw") or {}).get("space") if isinstance(config, dict) else None
    if space is not None and space != "cosine":
        raise ValueError(
            f"collection {collection.name!r} uses {space!r} distance, not "
            "cosine; delete and rebuild it (similarity scores would be wrong)"
        )


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
    """Embed the chunks and store them (ids, texts, metadata, vectors) in a
    per-config ChromaDB collection with REPLACEMENT semantics: after a build,
    the collection contains exactly the given chunks — ids that a previous
    build wrote but this corpus no longer produces are deleted (plain upsert
    would leave them searchable as stale results).

    Refuses to rebuild an existing collection with a different embedding
    model (the vectors would be overwritten while queries embed elsewhere —
    silent garbage). Records model, corpus hash, and chunk config in the
    collection metadata; ChromaDB ignores creation metadata for existing
    collections, so it is set explicitly via modify() after every build."""
    if not chunks:
        raise ValueError("no chunks to index")
    if client is None:
        client = chromadb.PersistentClient(path=path)
    collection = client.get_or_create_collection(
        collection_name_for(chunks), metadata={"hnsw:space": "cosine"}
    )
    _require_cosine(collection)
    existing = collection.metadata or {}
    existing_model = existing.get("embedding_model")
    if existing_model is not None and existing_model != model_name:
        raise ValueError(
            f"collection {collection.name!r} was built with {existing_model!r}; "
            f"rebuilding with {model_name!r} would corrupt it — delete the "
            "collection first, or use the model it was built with"
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

    current_ids = {c["chunk_id"] for c in chunks}
    stale_ids = [i for i in collection.get(include=[])["ids"] if i not in current_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)

    sample = chunks[0]["metadata"]
    provenance = {
        # modify() rejects hnsw:* keys (the distance function is fixed at
        # creation), so only non-hnsw keys are (re)written here
        key: value
        for key, value in existing.items()
        if not key.startswith("hnsw:")
    }
    provenance.update(
        embedding_model=model_name,
        corpus_hash=_corpus_hash(chunks),
        config_id=sample["config_id"],
        chunk_size=sample["chunk_size"],
        chunk_overlap=sample["chunk_overlap"],
        scope=sample["scope"],
    )
    collection.modify(metadata=provenance)
    return collection


def search(collection, query, k=5, model_name=DEFAULT_MODEL, where=None, rerank=False):
    """Semantic search; returns result dicts with the fields needed for a
    cited answer: text, similarity, paper id/title, and the page range.

    With `rerank=True`, a wider candidate set is fetched and re-scored by a
    local cross-encoder (FlashRank) before returning the top k — results
    then carry a `rerank_score` and are ordered by it."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    _require_cosine(collection)
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with is not None and built_with != model_name:
        raise ValueError(
            f"collection was built with {built_with!r} but query uses "
            f"{model_name!r}; mixed embedding spaces return garbage"
        )
    total = collection.count()
    if total == 0:
        return []
    n_candidates = max(k * 4, 16) if rerank else k
    result = collection.query(
        query_embeddings=[embed_query(query, model_name).tolist()],
        n_results=min(n_candidates, total),
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
    if rerank and rows:  # the cross-encoder errors on an empty passage list
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
