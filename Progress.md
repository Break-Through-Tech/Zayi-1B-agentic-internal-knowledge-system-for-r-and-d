# Progress Tracker

## Current Status

Tasks 1–4 completed. Task 5 initial implementation done. Remaining for the milestone: settle on the canonical cleaned data file and validate retrieval once the ground-truth Q&A set is available.

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

### Task 4: Implement and Test Text Chunking

- Consolidated on the token-aware implementation (agreed with Brayden): chunk sizes are measured with the embedding model's own tokenizer so every chunk fits the model's input limit, including special tokens. The earlier word-based version (400 words / 80 overlap) established the experiment framework and remains in git history; word-based sizing was retired because ~62% of 400-word chunks exceeded the 512-token model limit (see Notes).
- Implementation lives in importable modules (`src/cleaning.py`, `src/chunking.py`) with a pytest suite, so later tasks import the same code the notebook demonstrates.
- Configurable chunk size / overlap / scope ("page": each chunk cites exactly one page; "paper": chunks may span pages and cite a page range). Chunk ids embed a config fingerprint so different experimental configs can never collide in the vector store.
- `notebooks/03_text_chunking.ipynb` documents cleaning before/after, full-corpus chunking stats, a 12-config comparison, and a same-passage chunk-size comparison.

### Task 5: Generate Embeddings & Build ChromaDB Knowledge Base (initial implementation)

- Initial implementation done: `src/embedding.py` + `src/knowledge_base.py` embed the chunks with `bge-small-en-v1.5` (512-token input limit — matches the chunk budget) and store text + metadata + vectors in ChromaDB, one collection per chunking config.
- `notebooks/04_knowledge_base.ipynb` builds the knowledge base (557 vectors) and runs sample semantic-search queries: 5/6 return the expected paper at rank 1, with page-level citations on every result (the miss retrieved Self-RAG content from the RAG survey paper, which covers it).
- The ChromaDB store is generated locally (`chroma/`, gitignored) — rebuilt from the notebook in about a minute, so no binary store lives in git.

## In Progress

- 

## Blockers

- Team decision still open: which cleaned JSON is canonical — `data/cleaned_papers_text.json` vs `data/processed/cleaned_papers.json` (see Task 3 open items). The chunking pipeline can consume either via `src/data_io.py`.

## Next Steps

- Pick the canonical cleaned file and re-point the pipeline input at it.
- When the ground-truth Q&A set arrives (November): measure retrieval accuracy per chunking config (collections are per-config precisely so configs can be compared head-to-head) and tune chunk size / overlap / k.
- October milestone: baseline RAG pipeline (LangChain + Gemini) on top of `src/knowledge_base.search`.

## Notes

- Chunk-config sweep (2026-09-19, 20 self-labeled queries in `data/eval_queries_sample.json`, 6 configs × rerank on/off): all configs score hit@1 0.85–0.95 and hit@3 0.95–1.0 — differences are 1–2 queries, i.e. within noise at this sample size. Two consistent signals: FlashRank reranking lifted or matched hit@1 in every config (weakest config 0.85 → 0.95 with rerank), and smaller chunks (256–384) edge out 512 without rerank. Decision: keep the current default until the official ground-truth Q&A set arrives, then re-run this sweep for the final call; use `rerank=True` in the October generator.

- Reference-heavy pages may introduce noise into semantic retrieval.
- Word counts and model token counts diverge significantly on this corpus: measured with the bge-small tokenizer, ~62% of the 400-word chunks exceed 512 tokens (max ~1,979). Embedding models silently truncate input past their limit (`all-MiniLM-L6-v2` at 256 tokens, `bge-small-en-v1.5` at 512), so over-length chunks are only partially represented by their embeddings. Sizing chunks in tokens (with the same tokenizer as the embedding model) avoids this.
