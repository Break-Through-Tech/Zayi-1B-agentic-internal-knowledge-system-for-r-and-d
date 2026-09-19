import pytest

from src.chunking import (
    ChunkingConfig,
    build_chunks,
    chunk_stats,
    count_tokens,
    get_tokenizer,
)
from src.cleaning import clean_pages
from src.data_io import load_papers, iter_pages


@pytest.fixture(scope="module")
def one_paper_pages():
    # Real data keeps the tests honest about extraction artifacts;
    # one paper keeps them fast.
    papers = load_papers()
    return clean_pages(list(iter_pages([papers[0]])))


def _page(number, text, **extra):
    return {
        "paper_id": "P1",
        "paper_title": "Synthetic Paper",
        "category": "cs.CL",
        "pdf_url": "https://example.org/p1",
        "page_number": number,
        "text": text,
        **extra,
    }


def _normalized(text):
    return " ".join(text.split())


def test_respects_model_limit_including_special_tokens(one_paper_pages):
    # regression: chunks used to hit 513-514 tokens once [CLS]/[SEP] were added
    tokenizer = get_tokenizer()
    for size in (256, 512):
        config = ChunkingConfig(chunk_size=size, chunk_overlap=32)
        chunks = build_chunks(one_paper_pages, config)
        assert chunks
        assert all(len(tokenizer.encode(c["text"])) <= size for c in chunks)


def test_no_text_lost_without_overlap(one_paper_pages):
    config = ChunkingConfig(chunk_size=512, chunk_overlap=0)
    chunks = build_chunks(one_paper_pages, config)
    rejoined = _normalized(" ".join(c["text"] for c in chunks))
    original = _normalized(" ".join(p["text"] for p in one_paper_pages))
    assert rejoined == original


def test_consecutive_chunks_genuinely_share_overlap_text(one_paper_pages):
    config = ChunkingConfig(chunk_size=256, chunk_overlap=64, scope="paper")
    chunks = build_chunks(one_paper_pages, config)
    # the start of chunk 2 must be duplicated text from inside chunk 1
    lead = _normalized(chunks[1]["text"])[:60]
    assert lead in _normalized(chunks[0]["text"])


def test_page_scope_chunks_cite_single_pages(one_paper_pages):
    chunks = build_chunks(one_paper_pages, ChunkingConfig(scope="page"))
    page_numbers = {p["page_number"] for p in one_paper_pages}
    for c in chunks:
        assert c["metadata"]["page_start"] == c["metadata"]["page_end"]
        assert c["metadata"]["page_start"] in page_numbers


def test_paper_scope_cites_the_actual_source_pages():
    # three pages with distinct content: every chunk's cited page range must
    # actually contain the chunk's text
    pages = [
        _page(1, "".join(f"alpha retrieval sentence number {i}.\n" for i in range(40))),
        _page(2, "".join(f"bravo evaluation sentence number {i}.\n" for i in range(40))),
        _page(3, "".join(f"charlie training sentence number {i}.\n" for i in range(40))),
    ]
    chunks = build_chunks(pages, ChunkingConfig(chunk_size=128, chunk_overlap=0, scope="paper"))
    page_texts = {p["page_number"]: p["text"] for p in pages}
    for c in chunks:
        cited = "\n".join(
            page_texts[n]
            for n in range(c["metadata"]["page_start"], c["metadata"]["page_end"] + 1)
        )
        assert _normalized(c["text"]) in _normalized(cited)
    # marker words only exist on their own pages, so citations must reach them
    assert any("charlie" in c["text"] and c["metadata"]["page_end"] == 3 for c in chunks)
    assert all(c["metadata"]["page_start"] >= 2 for c in chunks if "bravo" in c["text"] and "alpha" not in c["text"])


@pytest.mark.parametrize(
    "text,size,overlap",
    [
        ("the attention mechanism computes queries keys values.\n" * 30, 128, 0),
        # regressions: repeated text with nonzero overlap used to raise
        # "could not locate a chunk in its source paper"
        ("data model.\n" * 60, 20, 5),
        ("data model.\n" * 60, 40, 32),
        ("the attention mechanism computes queries keys values over sequences.\n" * 120, 512, 64),
    ],
)
def test_paper_scope_handles_identical_repeated_pages(text, size, overlap):
    pages = [_page(1, text), _page(2, text)]
    chunks = build_chunks(
        pages, ChunkingConfig(chunk_size=size, chunk_overlap=overlap, scope="paper")
    )
    assert chunks[-1]["metadata"]["page_end"] == 2
    assert 2 in {c["metadata"]["page_start"] for c in chunks}
    # every citation must contain its chunk's text
    both = _normalized(text + "\n" + text)
    single = _normalized(text)
    for c in chunks:
        cited = both if c["metadata"]["page_start"] != c["metadata"]["page_end"] else single
        assert _normalized(c["text"]) in cited


def test_paper_scope_metadata_comes_from_chunk_start_page():
    # regression: pass-through metadata used to always come from page 1
    pages = [
        _page(1, "".join(f"alpha intro sentence number {i}.\n" for i in range(40)), section="Introduction"),
        _page(2, "".join(f"bravo method sentence number {i}.\n" for i in range(40)), section="Methods"),
    ]
    chunks = build_chunks(pages, ChunkingConfig(chunk_size=128, chunk_overlap=0, scope="paper"))
    starting_on_page_2 = [c for c in chunks if c["metadata"]["page_start"] == 2]
    assert starting_on_page_2
    assert all(c["metadata"]["section"] == "Methods" for c in starting_on_page_2)


def test_ids_never_collide_across_configs(one_paper_pages):
    # regression: 256- and 512-token runs used to produce identical ids for
    # different text, silently corrupting a reused ChromaDB collection
    a = build_chunks(one_paper_pages, ChunkingConfig(chunk_size=256, chunk_overlap=32))
    b = build_chunks(one_paper_pages, ChunkingConfig(chunk_size=512, chunk_overlap=64))
    assert not ({c["chunk_id"] for c in a} & {c["chunk_id"] for c in b})
    assert a[0]["metadata"]["config_id"] != b[0]["metadata"]["config_id"]


def test_metadata_scalar_contract_is_enforced():
    good = _page(1, "some sentence here.\n" * 20)
    build_chunks([good], ChunkingConfig(chunk_size=128, chunk_overlap=0))  # fine
    with pytest.raises(ValueError, match="authors"):
        build_chunks(
            [_page(1, "some sentence here.\n" * 20, authors=["a", "b"])],
            ChunkingConfig(chunk_size=128, chunk_overlap=0),
        )
    with pytest.raises(ValueError, match="section"):
        build_chunks(
            [_page(1, "some sentence here.\n" * 20, section=None)],
            ChunkingConfig(chunk_size=128, chunk_overlap=0),
        )


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


def test_default_data_prefers_ocr_corrected_source():
    # the corrected file (OCR-fixed ReAct figure pages) must be picked up
    # when present, so the fixes actually reach the chunks
    from src.data_io import DEFAULT_DATA_PATH, load_pages

    if DEFAULT_DATA_PATH.name == "curated_papers_text_corrected.json":
        react_p2 = next(
            p for p in load_pages()
            if p["paper_id"] == "2210.03629" and p["page_number"] == 2
        )
        assert not any(ord(c) < 32 and c not in "\n\t" for c in react_p2["text"])


def test_team_cleaned_adapter_matches_chunker_schema():
    from src.data_io import load_team_cleaned_pages

    pages = load_team_cleaned_pages()
    assert len(pages) == 218
    required = {"paper_id", "paper_title", "category", "pdf_url", "page_number", "text", "cleaner"}
    assert required <= set(pages[0])
    # adapter output chunks without errors and keeps citations
    chunks = build_chunks(pages[:5], ChunkingConfig(chunk_size=256, chunk_overlap=0))
    assert chunks and chunks[0]["metadata"]["cleaner"] == "team-task3-v1"


def test_count_tokens_counts_content_only():
    tokenizer = get_tokenizer()
    text = "retrieval augmented generation"
    assert count_tokens(text) == len(tokenizer.encode(text)) - tokenizer.num_special_tokens_to_add()
