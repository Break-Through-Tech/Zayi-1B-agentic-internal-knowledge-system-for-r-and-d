import pytest

from src.chunking import ChunkingConfig, build_chunks, chunk_stats, count_tokens
from src.cleaning import clean_pages
from src.data_io import load_papers, iter_pages


@pytest.fixture(scope="module")
def one_paper_pages():
    # Real data keeps the tests honest about extraction artifacts;
    # one paper keeps them fast.
    papers = load_papers()
    return clean_pages(list(iter_pages([papers[0]])))


def _normalized(text):
    return " ".join(text.split())


def test_respects_token_limit(one_paper_pages):
    config = ChunkingConfig(chunk_size=256, chunk_overlap=32)
    chunks = build_chunks(one_paper_pages, config)
    assert chunks
    assert all(c["metadata"]["token_count"] <= 256 for c in chunks)


def test_no_text_lost_without_overlap(one_paper_pages):
    config = ChunkingConfig(chunk_size=512, chunk_overlap=0)
    chunks = build_chunks(one_paper_pages, config)
    rejoined = _normalized(" ".join(c["text"] for c in chunks))
    original = _normalized(" ".join(p["text"] for p in one_paper_pages))
    assert rejoined == original


def test_overlap_repeats_text_between_consecutive_chunks(one_paper_pages):
    config = ChunkingConfig(chunk_size=256, chunk_overlap=64, scope="paper")
    chunks = build_chunks(one_paper_pages, config)
    tail = _normalized(chunks[0]["text"])[-40:]
    assert tail and tail in _normalized(chunks[0]["text"] + " " + chunks[1]["text"])
    # the second chunk starts with content from the end of the first
    first_words = set(_normalized(chunks[0]["text"]).split()[-30:])
    second_start = set(_normalized(chunks[1]["text"]).split()[:30])
    assert first_words & second_start


def test_page_scope_chunks_cite_single_pages(one_paper_pages):
    chunks = build_chunks(one_paper_pages, ChunkingConfig(scope="page"))
    page_numbers = {p["page_number"] for p in one_paper_pages}
    for c in chunks:
        assert c["metadata"]["page_start"] == c["metadata"]["page_end"]
        assert c["metadata"]["page_start"] in page_numbers


def test_paper_scope_page_ranges_are_ordered(one_paper_pages):
    chunks = build_chunks(one_paper_pages, ChunkingConfig(scope="paper"))
    for c in chunks:
        assert c["metadata"]["page_start"] <= c["metadata"]["page_end"]
    starts = [c["metadata"]["page_start"] for c in chunks]
    assert starts == sorted(starts)


def test_metadata_is_flat_scalars_for_chromadb(one_paper_pages):
    chunks = build_chunks(one_paper_pages, ChunkingConfig())
    for c in chunks:
        for key, value in c["metadata"].items():
            assert isinstance(value, (str, int, float, bool)), (key, value)


def test_upstream_metadata_passes_through(one_paper_pages):
    pages = [dict(p, section="Introduction") for p in one_paper_pages[:2]]
    chunks = build_chunks(pages, ChunkingConfig())
    assert all(c["metadata"]["section"] == "Introduction" for c in chunks)
    assert all(c["metadata"]["cleaner"] for c in chunks)


def test_chunk_ids_unique_and_deterministic(one_paper_pages):
    config = ChunkingConfig(chunk_size=384, chunk_overlap=48)
    first = build_chunks(one_paper_pages, config)
    second = build_chunks(one_paper_pages, config)
    ids = [c["chunk_id"] for c in first]
    assert len(ids) == len(set(ids))
    assert first == second


def test_rejects_unknown_scope(one_paper_pages):
    with pytest.raises(ValueError):
        build_chunks(one_paper_pages, ChunkingConfig(scope="chapter"))


def test_stats_shape(one_paper_pages):
    chunks = build_chunks(one_paper_pages, ChunkingConfig())
    stats = chunk_stats(chunks)
    assert stats["n_chunks"] == len(chunks)
    assert stats["n_papers"] == 1
    assert stats["tokens_min"] <= stats["tokens_mean"] <= stats["tokens_max"]


def test_count_tokens_matches_tokenizer_limit_semantics():
    assert count_tokens("retrieval augmented generation") >= 3
