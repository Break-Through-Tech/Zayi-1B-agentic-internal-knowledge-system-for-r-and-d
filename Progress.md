# Progress Tracker

## Current Status

Tasks 1–3 completed. Task 4 implemented in two variants (consolidation pending). Task 5 started.

## Completed

### Task 1: Set Up Development Environment & Review Dataset

- Reviewed the project repository structure and confirmed the provided dataset resources.
- Confirmed that `data/curated_papers_text.json` contains 10 research papers and 218 extracted pages.
- Confirmed that all papers contain the same six top-level metadata fields.
- Confirmed that every page record contains `page_number` and `text`, with sequential page numbering.
- Found no missing or null metadata values, empty pages, duplicate paper IDs, or duplicate PDF URLs.
- Documented known PDF text-extraction artifacts, including control characters, ligatures, broken line wraps, and mixed tables, equations, and figures.
- Observed that section structure is present implicitly in extracted text but is not represented as structured metadata.

### Task 2: Conduct Exploratory Data Analysis (EDA)

- Recorded the analysis in the project notebook.
- Analyzed paper metadata, page-level structure, document lengths, missing values, and formatting quality.
- Confirmed the following dataset measurements:

| Measure | Result |
| --- | ---: |
| Papers analyzed | 10 |
| Total pages | 218 |
| Extracted words | 121,577 |
| Extracted characters | 799,909 |
| Average pages per paper | 21.8 |
| Minimum pages per paper | 8 |
| Maximum pages per paper | 34 |
| Average words per page | 557.69 |

- Found no empty pages.
- Identified one page with fewer than 100 words and four pages with more than 1,000 words.
- Confirmed that all five word-count outliers occur in the RAG survey paper and are primarily reference-heavy pages rather than corrupted or missing data.
- Documented the implication that page boundaries should not automatically be treated as retrieval chunks.
- Recommended preserving paper ID, title, category, and page number during preprocessing and chunking to support accurate retrieval and citations.

### Task 3: Clean and Preprocess Research Paper Text

- Text cleaning and preprocessing landed in two outputs:
  - `data/cleaned_papers_text.json` — per-page records with `section` and `citation_key` fields (218 records).
  - `data/processed/cleaned_papers.json` — flat per-page records (218 records); currently the input to the chunking and embedding notebooks.
- Ligature artifacts removed in both outputs.
- Open items to resolve as a team:
  - Choose one canonical cleaned file so Tasks 4–5 build on the same input.
  - Some control characters remain in `data/processed/cleaned_papers.json`.
  - Hyphenated compounds are sometimes fused during de-hyphenation (e.g. "state-of-the-art" → "state-of-theart", "cross-encoder" → "crossencoder").
  - The `section` values in `data/cleaned_papers_text.json` are unreliable (they frequently capture table/algorithm text rather than section names).

## In Progress

- **Task 4 — Text Chunking (two implementations, consolidation pending):**
  - `notebooks/03_text_chunking.ipynb` on main: word-based chunking (400 words / 80 overlap), size experiments, and page-fidelity/phrase-retention checks.
  - `task4-text-chunking` branch: token-aware chunking as importable modules (`src/`) with 31 tests and its own notebook — chunk sizes measured with the embedding model's tokenizer so every chunk fits the model's input limit; page or paper scope; config-fingerprinted chunk ids.
  - Plan: agree on one approach (or merge the two) before the ChromaDB knowledge base is finalized.
- **Task 5 — Embeddings & ChromaDB Knowledge Base (started):**
  - `notebooks/04_embeddings_chromadb.ipynb` on main: end-to-end first version — embeds the word-based chunks with `all-MiniLM-L6-v2` into a persistent ChromaDB at `data/chroma_db`, with sample semantic-search queries and result checks.
  - A token-aware variant building on the branch pipeline is in progress on a separate branch.

## Blockers

- Team decision needed: canonical cleaned JSON, and which chunking implementation feeds the final knowledge base.

## Next Steps

- Consolidate the two Task 4 implementations and re-generate the knowledge base from the agreed chunks.
- Align chunk sizes with the embedding model's token limit (see Notes).
- Decide whether `data/chroma_db/` (binary store, ~10 MB) should stay in git or be gitignored and rebuilt locally from the notebook.
- Prepare for the November evaluation: retrieval accuracy against the ground-truth Q&A pairs at page level.

## Notes

- Reference-heavy pages may introduce noise into semantic retrieval.
- Word counts and model token counts diverge significantly on this corpus: measured with the bge-small tokenizer, ~62% of the 400-word chunks exceed 512 tokens (max ~1,979). Embedding models silently truncate input past their limit (`all-MiniLM-L6-v2` at 256 tokens, `bge-small-en-v1.5` at 512), so over-length chunks are only partially represented by their embeddings. Sizing chunks in tokens (with the same tokenizer as the embedding model) avoids this.
