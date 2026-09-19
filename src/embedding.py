"""Embeddings for chunks and queries (Task 5).

The embedding model is the same one whose tokenizer sizes the chunks in
`chunking.py` (bge-small-en-v1.5, 512-token input limit) — chunks built with
the default config are guaranteed to fit its input window, so nothing is
silently truncated at embedding time.
"""

from functools import lru_cache

# Must stay in sync with chunking.DEFAULT_TOKENIZER: the chunker sizes chunks
# with this model's tokenizer so they fit its input limit.
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# bge models are trained to embed retrieval QUERIES with this prefix;
# passages are embedded without it.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def get_model(name=DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def embed_texts(texts, model_name=DEFAULT_MODEL, batch_size=64):
    """Normalized passage embeddings, one row per text (cosine-ready)."""
    model = get_model(model_name)
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(query, model_name=DEFAULT_MODEL):
    """Normalized embedding for a search query (with the bge query prefix)."""
    return embed_texts([QUERY_INSTRUCTION + query], model_name)[0]
