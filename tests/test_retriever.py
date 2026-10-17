import chromadb
import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

import src.retriever as retriever_module
from src.chunking import ChunkingConfig, build_chunks
from src.knowledge_base import build_knowledge_base, is_up_to_date
from src.retriever import PaperRetriever, get_retriever, load_knowledge_base, row_to_document

CONFIG = ChunkingConfig(chunk_size=128, chunk_overlap=0)


def _page(paper_id, number, text):
    return {
        "paper_id": paper_id,
        "paper_title": f"Paper {paper_id}",
        "category": "cs.CL",
        "pdf_url": f"https://example.org/{paper_id}",
        "page_number": number,
        "text": text,
    }


def _pages(castle_fact="thick stone walls"):
    castle = "".join(
        f"Medieval castles were defended with moats, drawbridges and {castle_fact}, point {i}.\n"
        for i in range(30)
    )
    gradient = "".join(
        f"Gradient descent updates model parameters along the negative gradient, step {i}.\n"
        for i in range(30)
    )
    return [_page("castles", 3, castle), _page("optim", 7, gradient)]


@pytest.fixture()
def client(tmp_path):
    # an on-disk store per test: EphemeralClient instances share one
    # in-process store, which would leak collections between tests
    return chromadb.PersistentClient(path=str(tmp_path))


@pytest.fixture()
def collection(client):
    return build_knowledge_base(build_chunks(_pages(), CONFIG), client=client)


def test_row_to_document_keeps_citation_metadata():
    row = {
        "chunk_id": "c1",
        "similarity": 0.8,
        "paper_id": "2106.09685",
        "paper_title": "LoRA",
        "page_start": 4,
        "page_end": 5,
        "text": "low-rank update",
    }
    doc = row_to_document(row)
    assert doc.page_content == "low-rank update"
    assert doc.id == "c1"
    assert doc.metadata == {key: value for key, value in row.items() if key != "text"}


def test_invoke_returns_cited_documents_best_first(collection):
    docs = PaperRetriever(collection=collection, k=2, rerank=False).invoke(
        "How were medieval castles defended?"
    )
    assert len(docs) == 2
    assert all(isinstance(doc, Document) for doc in docs)
    top = docs[0]
    assert top.metadata["paper_id"] == "castles"
    assert top.metadata["paper_title"] == "Paper castles"
    assert top.metadata["page_start"] == top.metadata["page_end"] == 3
    assert top.id == top.metadata["chunk_id"]
    assert "castles" in top.page_content
    similarities = [doc.metadata["similarity"] for doc in docs]
    assert similarities == sorted(similarities, reverse=True)
    assert "rerank_score" not in top.metadata


def test_rerank_adds_scores_and_respects_k(collection):
    docs = PaperRetriever(collection=collection, k=1).invoke(
        "How does gradient descent optimize parameters?"
    )
    assert len(docs) == 1
    assert docs[0].metadata["paper_id"] == "optim"
    assert "rerank_score" in docs[0].metadata


def test_where_filter_restricts_papers(collection):
    docs = PaperRetriever(
        collection=collection, k=5, rerank=False, where={"paper_id": "optim"}
    ).invoke("How were medieval castles defended?")
    assert docs and {doc.metadata["paper_id"] for doc in docs} == {"optim"}


def test_batch_runs_several_queries(collection):
    retriever = PaperRetriever(collection=collection, k=1, rerank=False)
    results = retriever.batch(["castle moats", "negative gradient step"])
    assert [docs[0].metadata["paper_id"] for docs in results] == ["castles", "optim"]


def test_invalid_k_rejected_and_empty_collection_returns_nothing(client):
    empty = client.create_collection("empty", metadata={"hnsw:space": "cosine"})
    with pytest.raises(ValidationError):
        PaperRetriever(collection=empty, k=0)
    assert PaperRetriever(collection=empty, rerank=False).invoke("anything") == []


def test_load_builds_once_then_reuses(client, monkeypatch):
    pages = _pages()
    first = load_knowledge_base(CONFIG, client=client, pages=pages)
    assert first.count() == len(build_chunks(pages, CONFIG))

    def fail(*args, **kwargs):
        raise AssertionError("an up-to-date collection must not be re-embedded")

    monkeypatch.setattr(retriever_module, "build_knowledge_base", fail)
    again = load_knowledge_base(CONFIG, client=client, pages=pages)
    assert again.name == first.name
    assert again.count() == first.count()


def test_load_rebuilds_when_corpus_changes(client):
    load_knowledge_base(CONFIG, client=client, pages=_pages())
    new_pages = _pages(castle_fact="boiling oil poured from murder holes")
    rebuilt = load_knowledge_base(CONFIG, client=client, pages=new_pages)
    assert is_up_to_date(rebuilt, build_chunks(new_pages, CONFIG))
    docs = PaperRetriever(collection=rebuilt, k=1, rerank=False).invoke("boiling oil")
    assert "boiling oil" in docs[0].page_content


def test_load_rebuild_flag_forces_build(client, monkeypatch):
    pages = _pages()
    load_knowledge_base(CONFIG, client=client, pages=pages)
    calls = []
    real_build = retriever_module.build_knowledge_base

    def counting_build(*args, **kwargs):
        calls.append(1)
        return real_build(*args, **kwargs)

    monkeypatch.setattr(retriever_module, "build_knowledge_base", counting_build)
    load_knowledge_base(CONFIG, client=client, pages=pages, rebuild=True)
    assert calls == [1]


def test_load_refuses_collection_built_with_another_model(client):
    pages = _pages()
    load_knowledge_base(CONFIG, client=client, pages=pages)
    with pytest.raises(ValueError, match="built with"):
        load_knowledge_base(CONFIG, client=client, pages=pages, model_name="some/other-model")


def test_get_retriever_wires_settings_through(collection, monkeypatch):
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        return collection

    monkeypatch.setattr(retriever_module, "load_knowledge_base", fake_load)
    retriever = get_retriever(k=3, rerank=False, config=CONFIG, where={"paper_id": "optim"})
    assert seen["config"] is CONFIG
    assert (retriever.k, retriever.rerank, retriever.where) == (3, False, {"paper_id": "optim"})
    assert retriever.collection is collection
    assert retriever.invoke("anything")[0].metadata["paper_id"] == "optim"


def test_load_rebuilds_when_only_metadata_changes(client):
    pages = _pages()
    load_knowledge_base(CONFIG, client=client, pages=pages)
    fixed = [dict(page, paper_title="Corrected Title") for page in pages]  # same texts
    rebuilt = load_knowledge_base(CONFIG, client=client, pages=fixed)
    docs = PaperRetriever(collection=rebuilt, k=1, rerank=False).invoke("castle moats")
    assert docs[0].metadata["paper_title"] == "Corrected Title"


def test_interrupted_build_is_never_up_to_date(client, monkeypatch):
    import src.knowledge_base as kb

    old_chunks = build_chunks(_pages(), CONFIG)
    collection = build_knowledge_base(old_chunks, client=client)
    assert is_up_to_date(collection, old_chunks)

    def crash(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(kb, "embed_texts", crash)
    with pytest.raises(KeyboardInterrupt):
        build_knowledge_base(build_chunks(_pages("boiling oil"), CONFIG), client=client)
    # contents may now be a mix of old and new — must not be trusted as either
    assert not is_up_to_date(client.get_collection(collection.name), old_chunks)


def test_missing_vectors_are_not_up_to_date(client):
    chunks = build_chunks(_pages(), CONFIG)
    collection = build_knowledge_base(chunks, client=client)
    collection.delete(ids=[chunks[0]["chunk_id"]])
    assert not is_up_to_date(collection, chunks)


def test_empty_where_means_no_filter(collection):
    docs = PaperRetriever(collection=collection, k=1, rerank=False, where={}).invoke("castle moats")
    assert docs[0].metadata["paper_id"] == "castles"


def test_custom_pages_cannot_overwrite_the_shared_index():
    with pytest.raises(ValueError, match="shared repo index"):
        load_knowledge_base(CONFIG, pages=_pages())


def test_get_retriever_passes_model_and_rebuild(collection, monkeypatch):
    seen = {}

    def fake_load(**kwargs):
        seen.update(kwargs)
        return collection

    monkeypatch.setattr(retriever_module, "load_knowledge_base", fake_load)
    retriever = get_retriever(model_name="BAAI/bge-small-en-v1.5", rebuild=True)
    assert seen["model_name"] == retriever.model_name == "BAAI/bge-small-en-v1.5"
    assert seen["rebuild"] is True


def test_concurrent_first_calls_load_the_model_once(monkeypatch):
    import threading
    import time
    from functools import lru_cache

    import src.embedding as embedding

    loads = []

    @lru_cache(maxsize=2)
    def slow_load(name):
        loads.append(name)
        time.sleep(0.05)
        return object()

    monkeypatch.setattr(embedding, "_load_model", slow_load)
    threads = [threading.Thread(target=embedding.get_model, args=("m",)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert loads == ["m"]
