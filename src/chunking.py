"""Token-aware text chunking (Task 4).

Splits cleaned page records into retrieval-sized chunks whose token counts
are measured with the embedding model's own tokenizer — the limit that
actually matters when the chunks are embedded in Task 5 (models silently
truncate past their max sequence length). `chunk_size` is the model's max
sequence length: the splitter budget subtracts the special tokens ([CLS],
[SEP]) the tokenizer adds, so encoded chunks genuinely fit.

Chunk metadata is flat scalars only (enforced), which is what ChromaDB
accepts. Chunk ids embed a fingerprint of the chunking configuration so
chunks from different configs can never collide inside one collection.
"""

import hashlib
from dataclasses import dataclass
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Default target embedding model (512-token limit, small, strong retrieval
# quality). If Task 5 picks a different model, pass its name in the config so
# chunk sizes are measured with the right tokenizer.
DEFAULT_TOKENIZER = "BAAI/bge-small-en-v1.5"

_SCALAR_TYPES = (str, int, float, bool)


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 512  # tokens, including the model's special tokens
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
    """Content tokens only (no [CLS]/[SEP])."""
    tokenizer = get_tokenizer(tokenizer_name)
    return len(tokenizer.encode(text, add_special_tokens=False))


def config_fingerprint(config, cleaner_version="raw"):
    """Short stable id for (chunking config, cleaning pass). Baked into every
    chunk id so runs with different settings can never silently mix or
    collide when upserted into the same ChromaDB collection."""
    key = "|".join(
        [
            config.tokenizer_name,
            str(config.chunk_size),
            str(config.chunk_overlap),
            config.scope,
            cleaner_version,
        ]
    )
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def _make_splitter(config):
    content_budget = config.chunk_size - get_tokenizer(
        config.tokenizer_name
    ).num_special_tokens_to_add()
    return RecursiveCharacterTextSplitter(
        chunk_size=content_budget,
        chunk_overlap=config.chunk_overlap,
        length_function=lambda t: count_tokens(t, config.tokenizer_name),
        add_start_index=True,
    )


def build_chunks(pages, config=ChunkingConfig()):
    """Chunk page records into a flat list of chunk dicts:

        {"chunk_id": str, "text": str, "metadata": {flat scalars}}

    Pages must already be cleaned; any extra scalar fields on the page
    records (e.g. "cleaner") pass through into chunk metadata untouched, so
    fields added upstream in Task 3 (like section labels) flow in for free.
    In paper scope, pass-through metadata comes from the page where the
    chunk STARTS. Non-scalar metadata values are rejected loudly rather
    than stored broken in ChromaDB.
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
            pieces = _split_across_pages(paper_pages, splitter, config)

        for index, (text, source_page, page_start, page_end) in enumerate(pieces):
            fingerprint = config_fingerprint(
                config, source_page.get("cleaner", "raw")
            )
            metadata = {
                key: value
                for key, value in source_page.items()
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
                config_id=fingerprint,
            )
            for key, value in metadata.items():
                if not isinstance(value, _SCALAR_TYPES) or value is None:
                    raise ValueError(
                        f"metadata field {key!r} is {type(value).__name__}; "
                        "ChromaDB metadata must be str/int/float/bool"
                    )
            chunks.append(
                {
                    "chunk_id": f"{source_page['paper_id']}_{fingerprint}_c{index:04d}",
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


def _split_across_pages(paper_pages, splitter, config):
    """Chunks that may span page boundaries, with provenance tracked by
    construction: every page is split into line pieces that carry their page
    number, and pieces are merged into chunks under the token budget with
    token overlap carried between consecutive chunks. Because a chunk is
    assembled FROM tagged pieces, its page range is exact by definition —
    there is no re-locating of chunk text afterwards (searching for a chunk's
    text in the joined paper is ambiguous whenever text repeats, and broke in
    two separate ways before this design).
    """
    budget = config.chunk_size - get_tokenizer(
        config.tokenizer_name
    ).num_special_tokens_to_add()

    pieces = []  # (text, token_count, page_record)
    for page in paper_pages:
        for line in page["text"].split("\n"):
            if not line.strip():
                continue
            line_tokens = count_tokens(line, config.tokenizer_name)
            if line_tokens > budget:
                for part in splitter.split_text(line):
                    pieces.append(
                        (part, count_tokens(part, config.tokenizer_name), page)
                    )
            else:
                pieces.append((line, line_tokens, page))

    current = []  # accumulated (text, token_count, page_record)
    current_tokens = 0
    for piece in pieces:
        piece_tokens = piece[1]
        if current and current_tokens + piece_tokens > budget:
            yield _emit(current)
            # carry the trailing pieces that fit the overlap into the next chunk
            tail = []
            tail_tokens = 0
            for text, tokens, page in reversed(current):
                if tail_tokens + tokens > config.chunk_overlap:
                    break
                tail.insert(0, (text, tokens, page))
                tail_tokens += tokens
            if tail_tokens + piece_tokens > budget:
                tail, tail_tokens = [], 0
            current, current_tokens = tail, tail_tokens
        current.append(piece)
        current_tokens += piece_tokens
    if current:
        yield _emit(current)


def _emit(pieces):
    text = "\n".join(piece_text for piece_text, _, _ in pieces)
    first_page = pieces[0][2]
    last_page = pieces[-1][2]
    return text, first_page, first_page["page_number"], last_page["page_number"]


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
