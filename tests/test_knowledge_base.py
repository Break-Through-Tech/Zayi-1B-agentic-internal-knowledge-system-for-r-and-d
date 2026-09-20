import chromadb
import numpy as np
import pytest

from src.chunking import ChunkingConfig, build_chunks
from src.embedding import embed_query, embed_texts
from src.knowledge_base import build_knowledge_base, collection_name_for, search


def _page(paper_id, number, text):
    return {
        "paper_id": paper_id,
        "paper_title": f"Paper {paper_id}",
        "category": "cs.CL",
        "pdf_url": f"https://example.org/{paper_id}",
        "page_number": number,
        "text": text,
    }


@pytest.fixture(scope="module")
def chunks():
    castle = "".join(
        f"Medieval castles were defended with moats, drawbridges and thick stone walls, point {i}.\n"
        for i in range(30)
    )
    gradient = "".join(
        f"Gradient descent updates model parameters along the negative gradient, step {i}.\n"
        for i in range(30)
    )
    pages = [_page("castles", 1, castle), _page("optim", 1, gradient)]
    return build_chunks(pages, ChunkingConfig(chunk_size=128, chunk_overlap=0))


@pytest.fixture()
def client():
    return chromadb.EphemeralClient()


def test_embeddings_are_normalized_and_deterministic():
    texts = ["retrieval augmented generation", "low rank adaptation"]
    first = embed_texts(texts)
    second = embed_texts(texts)
    assert first.shape == (2, 384)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
    assert np.allclose(first, second)
    assert not np.allclose(first[0], first[1])


def test_query_embedding_differs_from_passage_embedding():
    text = "how are castles defended?"
    assert not np.allclose(embed_query(text), embed_texts([text])[0])


def test_collection_name_embeds_config_and_rejects_mixed(chunks):
    name = collection_name_for(chunks)
    assert name == f"papers_{chunks[0]['metadata']['config_id']}"
    other = dict(chunks[0], metadata=dict(chunks[0]["metadata"], config_id="deadbeef"))
    with pytest.raises(ValueError, match="configs"):
        collection_name_for(chunks + [other])


def test_build_and_search_roundtrip(chunks, client):
    collection = build_knowledge_base(chunks, client=client)
    assert collection.count() == len(chunks)

    results = search(collection, "How were medieval castles defended?", k=3)
    assert results[0]["paper_id"] == "castles"
    assert results[0]["page_start"] == 1 and results[0]["page_end"] == 1
    assert results[0]["paper_title"] == "Paper castles"
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)
    assert all(-1.0 <= s <= 1.0 for s in similarities)

    other = search(collection, "How does gradient descent optimize parameters?", k=1)
    assert other[0]["paper_id"] == "optim"


def test_rebuild_is_idempotent(chunks, client):
    build_knowledge_base(chunks, client=client)
    collection = build_knowledge_base(chunks, client=client)
    assert collection.count() == len(chunks)


def test_rebuild_removes_stale_chunks(chunks, client):
    # regression: upsert-only rebuilds left chunks from a previous corpus
    # searchable (4 stale ReAct chunks in the real store)
    build_knowledge_base(chunks, client=client)
    smaller = chunks[:-2]
    collection = build_knowledge_base(smaller, client=client)
    assert collection.count() == len(smaller)
    stored = set(collection.get(include=[])["ids"])
    assert stored == {c["chunk_id"] for c in smaller}


def test_build_refuses_different_model_on_existing_collection(chunks, client):
    # regression: rebuilding with another model silently overwrote vectors
    # while collection metadata kept claiming the old model
    build_knowledge_base(chunks, client=client)
    with pytest.raises(ValueError, match="would corrupt"):
        build_knowledge_base(chunks, client=client, model_name="some/other-model")


def test_build_records_provenance_metadata(chunks, client):
    collection = build_knowledge_base(chunks, client=client)
    meta = collection.metadata
    assert meta["embedding_model"]
    assert meta["corpus_hash"]
    assert meta["config_id"] == chunks[0]["metadata"]["config_id"]
    assert meta["chunk_size"] == chunks[0]["metadata"]["chunk_size"]
    # rebuilding from the same chunks keeps the same corpus hash
    assert build_knowledge_base(chunks, client=client).metadata["corpus_hash"] == meta["corpus_hash"]


def test_build_and_search_reject_non_cosine_collection(chunks, client):
    # regression: a pre-existing same-named collection keeps its original
    # distance space (chroma ignores creation metadata), so build stamped an
    # L2 collection and search scored it as if cosine
    from src.knowledge_base import collection_name_for

    # EphemeralClient instances share one in-process store, so use a config
    # no other test builds to get a collection name that's genuinely new
    unique = build_chunks(
        [_page("castles", 1, "castle walls and moats sentence.\n" * 20)],
        ChunkingConfig(chunk_size=192, chunk_overlap=0),
    )
    client.create_collection(name=collection_name_for(unique))  # chroma default: l2
    with pytest.raises(ValueError, match="cosine"):
        build_knowledge_base(unique, client=client)
    l2 = client.get_collection(collection_name_for(unique))
    with pytest.raises(ValueError, match="cosine"):
        search(l2, "anything", k=1)


def test_build_rejects_empty_chunks(client):
    with pytest.raises(ValueError, match="no chunks"):
        build_knowledge_base([], client=client)


def test_empty_results_and_bad_k_are_handled(chunks, client):
    # regression: reranking an empty filtered candidate set crashed in ONNX
    collection = build_knowledge_base(chunks, client=client)
    assert search(collection, "anything", k=2, where={"paper_id": "nope"}, rerank=True) == []
    with pytest.raises(ValueError, match="k must be"):
        search(collection, "anything", k=0)
    empty = client.get_or_create_collection(
        "empty_test", metadata={"hnsw:space": "cosine"}
    )
    assert search(empty, "anything", k=3) == []


def test_search_rejects_mismatched_embedding_model(chunks, client):
    collection = build_knowledge_base(chunks, client=client)
    with pytest.raises(ValueError, match="mixed embedding spaces"):
        search(collection, "anything", model_name="some/other-model")


def test_rerank_orders_by_cross_encoder_and_respects_k(chunks, client):
    collection = build_knowledge_base(chunks, client=client)
    results = search(collection, "How were medieval castles defended?", k=2, rerank=True)
    assert len(results) == 2
    assert results[0]["paper_id"] == "castles"
    scores = [r["rerank_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_evaluate_retrieval_reports_hits_and_misses(chunks, client):
    from src.retrieval_eval import evaluate_retrieval

    collection = build_knowledge_base(chunks, client=client)
    examples = [
        ("How were medieval castles defended?", "castles"),
        ("How does gradient descent optimize parameters?", "optim"),
        ("How does gradient descent optimize parameters?", "castles"),  # forced miss
    ]
    report = evaluate_retrieval(collection, examples, ks=(1, 2))
    assert report["n_queries"] == 3
    assert report["hit@1"] == round(2 / 3, 3)
    assert report["hit@1"] <= report["hit@2"] <= 1.0
    assert len(report["misses_at_1"]) == 1
    assert report["misses_at_1"][0]["expected"] == "castles"
    assert 0.0 <= report["mrr@2"] <= 1.0  # depth-suffixed: observed to max(ks)
    assert "mrr" not in report  # unqualified name would overstate the metric
    assert "page_hit@1" not in report  # no page labels given
    assert len(report["details"]) == 3


def test_evaluate_retrieval_page_level(chunks, client):
    from src.retrieval_eval import evaluate_retrieval

    collection = build_knowledge_base(chunks, client=client)
    examples = [
        {"query": "How were medieval castles defended?", "expected_paper_id": "castles", "expected_pages": [1]},
        {"query": "How were medieval castles defended?", "expected_paper_id": "castles", "expected_pages": [99]},
        ("How does gradient descent optimize parameters?", "optim"),  # tuple, unlabeled
    ]
    report = evaluate_retrieval(collection, examples, ks=(1,))
    assert report["n_page_labeled"] == 2
    # page 1 is where the castle chunks live; page 99 can never hit
    assert report["page_hit@1"] == 0.5
    assert report["page_mrr@1"] == 0.5  # ranks: 1 and never -> (1.0 + 0.0) / 2
    assert report["hit@1"] == 1.0  # paper-level all correct
    labeled = [d for d in report["details"] if d["expected_pages"]]
    assert labeled[0]["page_rank"] == 1 and labeled[1]["page_rank"] is None


def test_evaluate_retrieval_validates_inputs(chunks, client):
    from src.retrieval_eval import evaluate_retrieval

    collection = build_knowledge_base(chunks, client=client)
    with pytest.raises(ValueError, match="no examples"):
        evaluate_retrieval(collection, [])
    with pytest.raises(ValueError, match="ks"):
        evaluate_retrieval(collection, [("q", "castles")], ks=())


def test_metadata_filter(chunks, client):
    collection = build_knowledge_base(chunks, client=client)
    results = search(
        collection, "defensive structures", k=2, where={"paper_id": "optim"}
    )
    assert results and all(r["paper_id"] == "optim" for r in results)
