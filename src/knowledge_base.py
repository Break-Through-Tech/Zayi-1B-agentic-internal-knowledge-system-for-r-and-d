"""ChromaDB knowledge base over the chunked papers (Task 5).

Each chunking configuration gets its OWN collection, named by the chunks'
config fingerprint — so experiments with different chunk sizes/overlaps can
coexist and be compared by retrieval quality without ever mixing or
colliding (chunk ids are only unique within a config).
"""

import chromadb

from src.embedding import DEFAULT_MODEL, embed_query, embed_texts

COLLECTION_PREFIX = "papers"
_UPSERT_BATCH = 500


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
        collection_name_for(chunks), metadata={"hnsw:space": "cosine"}
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


def search(collection, query, k=5, model_name=DEFAULT_MODEL, where=None):
    """Semantic search; returns result dicts with the fields needed for a
    cited answer: text, similarity, paper id/title, and the page range."""
    result = collection.query(
        query_embeddings=[embed_query(query, model_name).tolist()],
        n_results=k,
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
    return rows
