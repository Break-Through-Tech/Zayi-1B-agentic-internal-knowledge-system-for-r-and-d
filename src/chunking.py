"""Token-aware text chunking (Task 4).

Splits cleaned page records into retrieval-sized chunks whose token counts
are measured with the embedding model's own tokenizer — the limit that
actually matters when the chunks are embedded in Task 5 (models silently
truncate past their max sequence length).

Chunk metadata is flat scalars only, which is what ChromaDB accepts.
"""

from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default target embedding model (512-token limit, small, strong retrieval
# quality). If Task 5 picks a different model, pass its name in the config so
# chunk sizes are measured with the right tokenizer.
DEFAULT_TOKENIZER = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64  # tokens
    tokenizer_name: str = DEFAULT_TOKENIZER
    # "page": chunks never cross page boundaries (every chunk cites exactly
    # one page). "paper": pages are joined per paper before splitting, so
    # sentences spanning page breaks stay intact; chunks then cite a page range.
    scope: str = "page"


@lru_cache(maxsize=4)
def get_tokenizer(name=DEFAULT_TOKENIZER):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def count_tokens(text, tokenizer_name=DEFAULT_TOKENIZER):
    tokenizer = get_tokenizer(tokenizer_name)
    return len(tokenizer.encode(text, add_special_tokens=False))


def _make_splitter(config):
    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=lambda t: count_tokens(t, config.tokenizer_name),
    )


def build_chunks(pages, config=ChunkingConfig()):
    """Chunk page records into a flat list of chunk dicts:

        {"chunk_id": str, "text": str, "metadata": {flat scalars}}

    Pages must already be cleaned; any extra scalar fields on the page
    records (e.g. "cleaner") pass through into chunk metadata untouched, so
    fields added upstream in Task 3 (like section labels) flow in for free.
    """
    if config.scope not in ("page", "paper"):
        raise ValueError(f"unknown scope: {config.scope!r}")

    splitter = _make_splitter(config)
    chunks = []

    by_paper = {}
    for page in pages:
        by_paper.setdefault(page["paper_id"], []).append(page)

    for paper_pages in by_paper.values():
        paper_pages = sorted(paper_pages, key=lambda p: p["page_number"])
        if config.scope == "page":
            pieces = _split_within_pages(paper_pages, splitter)
        else:
            pieces = _split_across_pages(paper_pages, splitter)

        for index, (text, first_page, page_start, page_end) in enumerate(pieces):
            metadata = {
                key: value
                for key, value in first_page.items()
                if key not in ("text", "page_number")
            }
            metadata.update(
                page_start=page_start,
                page_end=page_end,
                chunk_index=index,
                token_count=count_tokens(text, config.tokenizer_name),
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
                scope=config.scope,
            )
            chunks.append(
                {
                    "chunk_id": f"{first_page['paper_id']}_c{index:04d}",
                    "text": text,
                    "metadata": metadata,
                }
            )
    return chunks


def _split_within_pages(paper_pages, splitter):
    """Each chunk comes from exactly one page."""
    for page in paper_pages:
        for text in splitter.split_text(page["text"]):
            yield text, page, page["page_number"], page["page_number"]


def _split_across_pages(paper_pages, splitter):
    """Join the paper's pages, split, then map each chunk back to the page
    range it spans via character offsets."""
    spans = []  # (start_offset, end_offset, page_number)
    parts = []
    offset = 0
    for page in paper_pages:
        parts.append(page["text"])
        spans.append((offset, offset + len(page["text"]), page["page_number"]))
        offset += len(page["text"]) + 1  # +1 for the joining newline
    full_text = "\n".join(parts)

    search_from = 0
    for text in splitter.split_text(full_text):
        start = full_text.index(text, search_from)
        end = start + len(text)
        search_from = start + 1  # overlapping chunks start after the previous one
        pages_hit = [p for s, e, p in spans if s < end and e > start]
        yield text, paper_pages[0], pages_hit[0], pages_hit[-1]


def chunk_stats(chunks):
    """Summary numbers for verifying a chunking run."""
    token_counts = [c["metadata"]["token_count"] for c in chunks]
    papers = {c["metadata"]["paper_id"] for c in chunks}
    return {
        "n_chunks": len(chunks),
        "n_papers": len(papers),
        "tokens_min": min(token_counts),
        "tokens_mean": round(sum(token_counts) / len(token_counts), 1),
        "tokens_max": max(token_counts),
        "under_100_tokens": sum(1 for t in token_counts if t < 100),
    }
