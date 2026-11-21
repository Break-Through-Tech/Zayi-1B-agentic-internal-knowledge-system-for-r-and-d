# Progress Tracker

## Current Status

*Updated 2026-11-14.*

- **Milestone 1 (September) — complete:** Tasks 1–5 done (data review, EDA, cleaning, chunking, embeddings + ChromaDB knowledge base).
- **Milestone 2 (October, Baseline RAG Pipeline) — complete:** retriever (#12), baseline LangChain + Gemini pipeline (#13), citation handling + API retries (#14) and the end-to-end evaluation (#15) are on `main`.
- **Milestone 3 (November, Refining & Handoff) — in progress:** Task 1 (retrieval accuracy, #20) is done; the Gradio UI (Task 6, #19) has its first complete version on `main`. See the November table under In Progress.
- Still open from September: choose the canonical cleaned data file; validate retrieval against the ground-truth Q&A set when it arrives (November).

## Completed — Milestone 1 (September)

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
  - `data/processed/cleaned_papers.json` — flat per-page records (218 records); consumable by the chunking pipeline via the adapter in `src/data_io.py`.
  - Note: until the team picks a canonical cleaned file, the pipeline chunks the figure-text-corrected curated source (`data/curated_papers_text_corrected.json`) through the interim cleaner in `src/cleaning.py`.
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

### Task 5: Generate Embeddings & Build ChromaDB Knowledge Base

- Done: `src/embedding.py` + `src/knowledge_base.py` embed the chunks with `bge-small-en-v1.5` (512-token input limit — matches the chunk budget) and store text + metadata + vectors in ChromaDB, one collection per chunking config.
- `notebooks/04_knowledge_base.ipynb` builds the knowledge base (553 vectors from the corrected source) and runs sample semantic-search queries with page-level citations on every result.
- The ChromaDB store is generated locally (`chroma/`, gitignored) — rebuilt from the notebook in about a minute, so no binary store lives in git. Rebuilds have **replacement semantics**: chunks a previous build wrote that the current corpus no longer produces are deleted (plain upsert left 4 stale pre-OCR chunks searchable — caught in external code review and fixed).
- Collections carry provenance metadata (embedding model, corpus hash, chunk config), rebuilding an existing collection with a different embedding model is refused, and `search` refuses queries with a mismatched model — mixed embedding spaces fail loudly instead of returning garbage.
- Retrieval quality is **measured**, not assumed: `src/retrieval_eval.py` computes deterministic paper-level hit@k and MRR, and **page-level hit@k** for examples labeled with `expected_pages` — the format the November ground-truth Q&A set plugs into. `data/eval_queries_sample.json` holds a 20-query paper-labeled smoke-test set (2 per paper).
- The config sweep is committed as runnable code: `scripts/run_retrieval_sweep.py` writes machine-readable results to `data/eval/sweep_results.json` (latest run included).
- Optional local cross-encoder reranking (stretch goal 2): `search(..., rerank=True)` re-scores candidates with FlashRank — it lifted or matched paper hit@1 in every tested config. **Caveat:** the cross-encoder reads query+passage as one 512-token sequence, so with default ~510-token chunks roughly half the pairs get their tails truncated; reranking is most trustworthy on 256–384-token chunk collections (which also score best pre-rerank).
- Test suite (September): 49 passing tests across data loading, cleaning, chunking, embeddings, knowledge base, and evaluation.

## In Progress

### Milestone 3 (November) — Refining, Dependability & Handoff

| Task | Issue | Assignees | Due | Status |
| --- | --- | --- | --- | --- |
| 1. Improve Retrieval Accuracy & Reranking | #20 | Deyon, Brayden | 11/14 | **Done** — on `main` |
| 2. Improve Answer Generation & Context Quality | #21 | Brayden, Matthew, Ourania | 11/18 | Open |
| 3. Enhance Source Citation Precision & System Resilience | #22 | Matthew, Tharun | 11/21 | Open (retry and citation fixes from Task 1 already on `main`) |
| 4. Complete Python RAG Architecture & Technical Documentation | #17 | Tharun, Atai, Deyon | 11/25 | To do |
| 5. Final Evaluation, Testing & Project Handoff | #18 | Matthew, Deyon, Brayden, Ourania, Tharun | 11/28 | To do |
| 6. Add UI (Gradio) | #19 | Deyon, Ourania, Matthew | 12/5 | **Done** — on `main`, including triage memos with a review step |

- **Task 1 (Retrieval Accuracy & Reranking, #20) — done (Deyon):** page labels for all 40 ground-truth questions (`data/eval/ground_truth_pages.json`), so retrieval is now measured at the page level, not just the paper level. With reranking the answer page is in the top 5 for 36/40 questions vs 31/40 without. The rerank pool is configurable and now defaults to 40 candidates; 512/64 chunks and k=5 stay. Full results and the remaining misses: `data/eval/retrieval_tuning_report.md` (reproduce with `python scripts/run_gt_retrieval_eval.py`, no API calls). Also fixed: Gemini 429 retries never fired (the client hides the status code), and invalid citations are now flagged in the result's `citations` key instead of raising.
- **Task 6 (Gradio UI, #19) — built by Deyon; all three parts done.** Run `python scripts/app.py` → http://127.0.0.1:7860 (code in `src/ui.py`, tests in `tests/test_ui.py`). Of the issue's three parts, two are done:
  - [x] **Interactive RAG queries** — chat-style page: ask a question, see each step (search → write answer → check citations, with timings), get the answer with Markdown and math rendered. *Answer* mode calls Gemini; *Search only* shows the retrieved excerpts with no API call (useful under the free-tier limit). Settings for number of excerpts (3–10) and reranking.
  - [x] **Source citations** — every `S1`-style citation is a link to its source card (paper, pages, full excerpt, "Open in PDF" at the right page); uncited sources are dimmed and invalid citations are flagged.
  - [x] **Technical memo review** — *Memo* mode drafts a structured triage memo (`src/memo.py`: Bottom line, Key findings, How it works, Evidence and results, Limitations and open questions; every claim cited). Automatic checks run first (all sections present, citations point at retrieved excerpts, every section cited or marked not covered). The reviewer edits the draft in place, adds notes, then **Approve** or **Request changes** downloads the memo stamped with the decision, date, notes and whether it was edited (the brief's human-in-the-loop approval step; "approved with no edits" is the first-pass approval rate the brief asks for). One API call per memo, same as an answer. Tests: `tests/test_memo.py`, `tests/test_ui.py`.
  - Also: errors (missing key, daily quota, rate limit) show as messages instead of crashing; works at phone width; 144 tests passing. New dependencies: `gradio`, `markdown-it-py` — run `pip install -r requirements.txt`.

### Milestone 2 (October) — Baseline RAG Pipeline

| Task | Issue | Assignees | Due | Status |
| --- | --- | --- | --- | --- |
| 1. Validate September Knowledge Base & Retrieval Setup | #11 | Matthew, Claire | 10/14 | To do (`tests/test_retriever.py`, `tests/test_knowledge_base.py` and notebook 04's sample searches already cover most checks) |
| 2. Implement ChromaDB Retriever | #12 | Brayden, Matthew, Atai, Deyon | 10/17 | **Done** — on `main` |
| 3. Build Baseline RAG Pipeline with LangChain & Gemini | #13 | Atai, Brayden, Deyon | 10/24 | **Done** — on `main` |
| 4. Add Citation Handling & API Reliability | #14 | Tharun, Claire, Matthew | 10/27 | **Done** — on `main` |
| 5. Test & Document End-to-End Baseline RAG System | #15 | Claire, Tharun | 10/31 | **Done** — `data/eval/evaluation_report.md` |

- **Task 2 (Implement ChromaDB Retriever, #12) — done:** `src/retriever.py` wraps `knowledge_base.search` as a LangChain `BaseRetriever` (`PaperRetriever`). Each result is a `Document` with the chunk text and citation metadata (paper title, arXiv id, page range, similarity, rerank score). `get_retriever()` opens the local store and re-embeds only when the collection is missing or stale (embedding model, chunk text, and chunk metadata are all checked; an interrupted build is never reused). Defaults: k=5, reranking on. On the real corpus the correct paper ranks first for 19/20 sample queries.
- **Task 3 (Baseline RAG with LangChain & Gemini, #13):** `src/rag.py` builds a linear LCEL chain: question → retriever → numbered excerpts → grounded prompt → LLM → answer with inline [S1], [S2]... citations (S-prefixed so they can't be confused with the papers' own reference numbers) + a matching source list. Gemini (`gemini-3.5-flash`) by default, OpenRouter as fallback; keys come from a local `.env` (see `.env.example`). If nothing is retrieved, the LLM is not called. CLI: `scripts/ask.py` (`--retrieve-only` works without a key).
- Tasks 4–5 (#14, #15) — done: retry/backoff with tenacity and citation validation in `src/rag.py`; 40-question end-to-end evaluation in `data/eval/evaluation_report.md`.

## Blockers

- Team decision still open: which cleaned JSON is canonical — `data/cleaned_papers_text.json` vs `data/processed/cleaned_papers.json` (see Task 3 open items). The chunking pipeline can consume either via `src/data_io.py`.

## Next Steps

- Pick the canonical cleaned file and re-point the pipeline input at it.
- When the ground-truth Q&A set arrives (November): measure retrieval accuracy per chunking config (collections are per-config precisely so configs can be compared head-to-head) and tune chunk size / overlap / k.
- October: citation handling/API reliability (#14) and end-to-end testing (#15) on top of the baseline RAG chain. Retriever entry point for all of these: `src.retriever.get_retriever()`.

## Notes

- Chunk-config sweep (rerun 2026-09-20 from a clean store via `scripts/run_retrieval_sweep.py`; results in `data/eval/sweep_results.json`): all 6 configs score paper hit@1 0.85–0.95 and hit@3 0.95–1.0 — differences are 1–2 queries, i.e. within noise at this sample size. Two consistent signals: FlashRank reranking lifted or matched hit@1 in every config (weakest 0.85 → 0.95), and smaller chunks (256–384) edge out 512 without rerank. Decision: keep the current default until the official ground-truth Q&A set arrives, then re-run this sweep (with page labels) for the final call; use `rerank=True` in the October generator, ideally over a 256–384-token collection given the reranker's 512-token pair limit.

- Reference-heavy pages may introduce noise into semantic retrieval.

  **Update — reference-page filtering tested:** Built scripts/test_reference_filtering.py to evaluate whether removing bibliography/reference sections improves retrieval quality. The script detects the start of each paper's References/Bibliography section via heading match, removes all subsequent pages, rebuilds the ChromaDB index, and compares retrieval metrics against the unfiltered corpus.

  Result: removing reference pages reduced the indexed corpus from 553 chunks to 257 chunks (~54% reduction) but produced no change in hit@1 (0.850), hit@3 (0.950), or mrr@3 (0.900) on the sample evaluation set.

  Decision: do not adopt reference-page filtering at this time. The experiment found no measurable retrieval benefit on the current query set, so the added preprocessing complexity is not justified.

  Caveat: results are based on the current ~20-query evaluation set and should be revalidated when the larger ground-truth Q&A dataset becomes available

- Word counts and model token counts diverge significantly on this corpus: measured with the bge-small tokenizer, ~62% of the 400-word chunks exceed 512 tokens (max ~1,979). Embedding models silently truncate input past their limit (`all-MiniLM-L6-v2` at 256 tokens, `bge-small-en-v1.5` at 512), so over-length chunks are only partially represented by their embeddings. Sizing chunks in tokens (with the same tokenizer as the embedding model) avoids this.
