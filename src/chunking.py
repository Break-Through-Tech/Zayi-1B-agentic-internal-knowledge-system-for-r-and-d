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
            pieces = _split_across_pages(paper_pages, splitter)

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


def _split_across_pages(paper_pages, splitter):
    """Join the paper's pages, split, then map each chunk back to the page
    range it spans via verified character offsets.

    The splitter reports each chunk's start offset; every offset is verified
    against the actual text (and repaired by a bounded forward search if the
    splitter's report is off) before pages are assigned, so a chunk's cited
    span always contains the chunk's exact text. For byte-identical repeated
    passages the chosen occurrence can be ambiguous, but any cited page
    genuinely contains the text.
    """
    spans = []  # (start_offset, end_offset, page_record)
    parts = []
    offset = 0
    for page in paper_pages:
        parts.append(page["text"])
        spans.append((offset, offset + len(page["text"]), page))
        offset += len(page["text"]) + 1  # +1 for the joining newline
    full_text = "\n".join(parts)

    previous_start, previous_end = -1, -1
    for doc in splitter.create_documents([full_text]):
        text = doc.page_content
        start = doc.metadata.get("start_index", -1)
        valid = (
            start > previous_start
            and start + len(text) > previous_end
            and full_text.startswith(text, start)
        )
        if not valid:
            # walk occurrences forward until one keeps chunk order intact
            start = full_text.find(text, previous_start + 1)
            while start != -1 and start + len(text) <= previous_end:
                start = full_text.find(text, start + 1)
            if start == -1:
                raise ValueError(
                    "could not locate a chunk in its source paper; "
                    "use scope='page' for this corpus"
                )
        previous_start, previous_end = start, start + len(text)

        pages_hit = [
            record
            for span_start, span_end, record in spans
            if span_start < previous_end and span_end > previous_start
        ]
        yield text, pages_hit[0], pages_hit[0]["page_number"], pages_hit[-1][
            "page_number"
        ]


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
