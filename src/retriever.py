"""LangChain retriever over the ChromaDB knowledge base (October, Task 2).

Wraps `src.knowledge_base.search` in LangChain's standard `BaseRetriever`
interface so the RAG chain (and any other LangChain component) can use it
directly. Each result is a `Document` whose metadata carries everything a
citation needs: paper title, arXiv id, and page range.

`get_retriever()` is the one-liner entry point: it opens the persisted
store, reusing the existing collection when it already matches the current
corpus and building it (≈1 min, first run only) when it doesn't.
"""

from pathlib import Path
from typing import Any

import chromadb
import chromadb.errors
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from src.chunking import ChunkingConfig, build_chunks
from src.cleaning import clean_pages
from src.data_io import load_pages
from src.embedding import DEFAULT_MODEL
from src.knowledge_base import (
    build_knowledge_base,
    collection_name_for,
    is_up_to_date,
    search,
)

# Repo-root store, so scripts and notebooks share one index regardless of
# the working directory they run from.
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent.parent / "chroma"

# Baseline settings. Reranking lifted or matched hit@1 in every config of
# the September sweep; k=5 gives the generator a few corroborating excerpts.
DEFAULT_K = 5
DEFAULT_RERANK = True


def row_to_document(row):
    """`search()` result dict -> LangChain Document (text + citation metadata)."""
    metadata = {key: value for key, value in row.items() if key != "text"}
    return Document(page_content=row["text"], metadata=metadata, id=row["chunk_id"])


class PaperRetriever(BaseRetriever):
    """Semantic search over one knowledge-base collection, returning
    Documents ranked best-first (by rerank score when `rerank` is on)."""

    collection: Any
    k: int = Field(default=DEFAULT_K, ge=1)
    rerank: bool = DEFAULT_RERANK
    model_name: str = DEFAULT_MODEL
    where: dict | None = None  # optional Chroma metadata filter, e.g. {"paper_id": "2106.09685"}

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        rows = search(
            self.collection,
            query,
            k=self.k,
            model_name=self.model_name,
            where=self.where or None,  # chroma rejects an empty filter {}
            rerank=self.rerank,
        )
        return [row_to_document(row) for row in rows]


def load_knowledge_base(
    config=None,
    path=DEFAULT_CHROMA_PATH,
    client=None,
    model_name=DEFAULT_MODEL,
    pages=None,
    rebuild=False,
):
    """Return the collection for `config` (default: 512 tokens / 64 overlap /
    page scope), building it only when it is missing or stale.

    Chunking is cheap (tokenizer only), so the chunks are always rebuilt
    from the source data and compared against the stored corpus hash; the
    expensive embedding step runs only on a mismatch. `pages` defaults to
    the cleaned curated corpus; custom `pages` must go to their own store
    (`path` or `client`), because the collection name depends only on the
    chunking config — they would otherwise replace the shared repo index.
    `rebuild=True` forces a re-embed even when the collection looks current."""
    if pages is not None and client is None and Path(path) == DEFAULT_CHROMA_PATH:
        raise ValueError(
            "custom pages would overwrite the shared repo index; "
            "pass a separate path= or client="
        )
    if client is None:
        client = chromadb.PersistentClient(path=str(path))
    if pages is None:
        pages = clean_pages(load_pages())
    chunks = build_chunks(pages, config or ChunkingConfig())
    if not rebuild:
        try:
            existing = client.get_collection(collection_name_for(chunks))
        except chromadb.errors.NotFoundError:
            existing = None
        if existing is not None and is_up_to_date(existing, chunks, model_name):
            return existing
    return build_knowledge_base(chunks, client=client, model_name=model_name)


def get_retriever(
    k=DEFAULT_K,
    rerank=DEFAULT_RERANK,
    config=None,
    path=DEFAULT_CHROMA_PATH,
    client=None,
    model_name=DEFAULT_MODEL,
    where=None,
    rebuild=False,
):
    """Ready-to-use retriever over the curated papers:

        from src.retriever import get_retriever
        docs = get_retriever().invoke("How does LoRA reduce trainable parameters?")
        docs[0].metadata["paper_title"], docs[0].metadata["page_start"]
    """
    collection = load_knowledge_base(
        config=config, path=path, client=client, model_name=model_name, rebuild=rebuild
    )
    return PaperRetriever(
        collection=collection, k=k, rerank=rerank, model_name=model_name, where=where
    )
