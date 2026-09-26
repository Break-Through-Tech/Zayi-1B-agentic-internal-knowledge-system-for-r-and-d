# Agentic Internal Knowledge System for R&D

### 👥 Team Members

| Name | GitHub Handle | Contribution |
|---|---|---|
| Ranidnu-W | @Ranidnu-W | Core project implementation, token-aware chunking, embeddings, ChromaDB knowledge base, retrieval evaluation, collection lifecycle fixes, reranking work, and repo-level maintenance |
| MatthewZ1 | @MatthewZ1 | Corrected corrupted figure text, fixed ReAct paper OCR issues, updated data-source handling, and contributed to the corrected dataset workflow |
| Brayden Uglione | @BrantisIsHacking | Updated text-cleaning and chunking notebooks, consolidated data-cleaning files, supported testing, and developed task delegation and progress tracking |
| Tharun Kumar Malla Dinakaran | @Thxrunn | Developed EDA and text preprocessing notebooks to analyze, clean, validate, and prepare the research paper dataset for the RAG pipeline. |
| Atai Kydyrov | @atai20 | Findings formatting, markdown cleanup, notebook updates, and research-paper analysis documentation |
| joseambrosioo | @joseambrosioo | Challenge project overview updates and project coordination documentation |
| hari-bttai | @hari-bttai | Initial repository commit and project scaffolding |

---

## 🎯 Project Highlights

- Built a local retrieval-first research assistant for Zayi’s R&D workflow using a curated set of ArXiv ML papers.
- Created a reproducible pipeline for cleaning paper text, chunking documents with token-aware sizing, embedding them, and storing them in ChromaDB.
- Implemented evaluation logic for retrieval performance using hit@k and MRR so that chunking strategies can be compared systematically.
- Added optional local reranking support via FlashRank to improve result relevance in the knowledge base.
- Organized the project around modular code, notebooks, dataset assets, tests, and an evaluation sweep script tied to the milestone workflow.

---

A local retrieval-first research assistant for Zayi’s R&D workflow. This repository turns a curated set of ArXiv ML papers into a searchable knowledge base using cleaned text, token-aware chunking, embeddings, and ChromaDB. The system is designed to support document retrieval, citation-grounded answers, and future agentic workflows built on top of the same retrieval layer.

## Project context

- Host organization: Zayi
- Challenge advisor: Jose Ambrosio
- Program: Break Through Tech AI Studio - Fall 2026
- Goal: build an internal knowledge system that lets engineers quickly query dense research papers, find the right passages and page ranges, and draft structured technical memos with citations.

This project follows the repository’s milestone structure:

- Task 1: environment setup and dataset review
- Task 2: exploratory data analysis
- Task 3: cleaning and preprocessing
- Task 4: text chunking
- Task 5: embeddings and ChromaDB knowledge base

## What is implemented

The current codebase contains the actual retrieval stack used for experimentation:

- Data loading and normalization from curated ArXiv paper JSON files
- Text cleaning and preprocessing utilities
- Token-aware chunking using the embedding model tokenizer instead of raw word counts
- BAAI/bge-small-en-v1.5 embeddings for both document and query passages
- ChromaDB collection creation with per-config storage and provenance metadata
- Retrieval evaluation with hit@k and MRR metrics
- Optional local reranking through FlashRank
- Sweep script to benchmark multiple chunking configurations

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── Challenge-Project-Overview.md
├── Progress.md
├── FindingsOnResearchPapers.md
├── Getting-Started-for-Fellows.md
├── data/
│   ├── curated_papers_text.json
│   ├── curated_papers_text_corrected.json
│   ├── cleaned_papers_text.json
│   ├── eval_queries_sample.json
│   ├── eval/
│   │   └── sweep_results.json
│   ├── images/
│   └── processed/
│       └── cleaned_papers.json
├── notebooks/
│   ├── 0.5_corrupted_text_fix.ipynb
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_text_chunking.ipynb
│   └── 04_knowledge_base.ipynb
├── scripts/
│   └── run_retrieval_sweep.py
├── src/
│   ├── __init__.py
│   ├── chunking.py
│   ├── cleaning.py
│   ├── data_io.py
│   ├── embedding.py
│   ├── knowledge_base.py
│   └── retrieval_eval.py
├── tests/
│   ├── test_chunking.py
│   ├── test_cleaning.py
│   └── test_knowledge_base.py
└── chroma/            # local vector store, generated at runtime
```

## Data and source materials

The project ingests a curated subset of ML research papers from ArXiv. The data is stored locally in the repository under `data/`.

Primary dataset files:

- `data/curated_papers_text.json`: original paper JSON
- `data/curated_papers_text_corrected.json`: corrected version preferred when present
- `data/cleaned_papers_text.json`: cleaned per-page output from Task 3
- `data/processed/cleaned_papers.json`: flat cleaned records used by pipeline adapters
- `data/eval_queries_sample.json`: sample labeled retrieval questions used for evaluation

The loader in `src/data_io.py` prefers the corrected dataset automatically:

- `DEFAULT_DATA_PATH` = `data/curated_papers_text_corrected.json` if it exists
- otherwise it falls back to `data/curated_papers_text.json`

## Environment setup

From the repository root:

```bash
cd /workspaces/Zayi-1B-agentic-internal-knowledge-system-for-r-and-d
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want Python imports from the project package to resolve in local runs and tests, set the project root on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD"
```

Optional but useful for notebooks:

```bash
python -m ipykernel install --user --name zayi-rag
```

## Installation notes

The project currently depends on:

- `langchain-text-splitters`
- `transformers`
- `sentence-transformers`
- `chromadb`
- `flashrank`
- `pandas`, `matplotlib`, `jupyter`, `ipykernel`
- `pytesseract`, `Pillow`
- `pytest`

The locked dependency set is maintained in `requirements.txt`.

## Project workflow

### 1. Load and inspect the data

The loader module normalizes raw paper data into flat page records:

```python
from src.data_io import load_pages
pages = load_pages()
print(len(pages))
```

This is used across the notebooks and downstream chunking pipeline.

### 2. Clean the text

The cleaning pipeline handles textual artifacts such as ligature cleanup, broken formatting, and noisy OCR-like patterns, then passes cleaned records to the chunker.

Relevant files:

- `src/cleaning.py`
- `notebooks/02_text_preprocessing.ipynb`

### 3. Chunk the documents

The chunking implementation in `src/chunking.py` uses the tokenizer from the embedding model to measure tokens and keep chunks within the model limit.

Key design details:

- `ChunkingConfig(chunk_size, chunk_overlap, tokenizer_name, scope)`
- default tokenizer: `BAAI/bge-small-en-v1.5`
- supported scopes:
  - `page`: chunk stays within a single page
  - `paper`: chunk may span page boundaries while tracking page ranges
- each chunk carries metadata including page numbers and config fingerprint

Example:

```python
from src.chunking import ChunkingConfig, build_chunks
from src.data_io import load_pages
from src.cleaning import clean_pages

cleaned = clean_pages(load_pages())
chunks = build_chunks(cleaned, ChunkingConfig(chunk_size=256, chunk_overlap=32, scope="page"))
print(len(chunks))
```

### 4. Build the knowledge base

The vector store is created with ChromaDB in `src/knowledge_base.py`.

Configuration highlights:

- one Chroma collection per chunking configuration
- embeddings use `BAAI/bge-small-en-v1.5`
- collection metadata stores model name, corpus hash, and config info
- existing collections are checked to prevent silent cross-model corruption
- stale entries are removed when a collection is rebuilt

Example:

```python
from src.data_io import load_pages
from src.cleaning import clean_pages
from src.chunking import ChunkingConfig, build_chunks
from src.knowledge_base import build_knowledge_base

pages = clean_pages(load_pages())
chunks = build_chunks(pages, ChunkingConfig(chunk_size=256, chunk_overlap=32, scope="page"))
collection = build_knowledge_base(chunks)
print(collection.name)
```

### 5. Query the knowledge base

Search is exposed through `src/knowledge_base.py`:

```python
from src.knowledge_base import search

results = search(collection, "What is the role of retrieval in RAG?", k=5, rerank=True)
for r in results:
    print(r["paper_title"], r["page_start"], r["page_end"], r["similarity"])
```

This supports:

- semantic retrieval over embedded chunks
- optional reranking with FlashRank
- citation-friendly result metadata such as page numbers and paper title

### 6. Run the evaluation sweep

The script `scripts/run_retrieval_sweep.py` builds several chunking configurations, evaluates retrieval against the sample labeled query set, prints a summary table, and saves results to `data/eval/sweep_results.json`.

From the repo root:

```bash
python scripts/run_retrieval_sweep.py
```

The sweep compares different chunk sizes, overlaps, and page-vs-paper scope, and also tests reranking on and off.

## Notebooks

The notebook flow is:

- `notebooks/0.5_corrupted_text_fix.ipynb`: OCR/corruption repair work on the raw text
- `notebooks/01_data_exploration.ipynb`: dataset inspection and EDA
- `notebooks/02_text_preprocessing.ipynb`: cleaning and normalization
- `notebooks/03_text_chunking.ipynb`: chunking experiments and stats
- `notebooks/04_knowledge_base.ipynb`: embedding, ChromaDB indexing, and sample searches

## Testing

The repository includes a small local test suite for cleaning, chunking, and knowledge-base behavior.

Run tests from the repo root with:

```bash
export PYTHONPATH="$PWD"
pytest -q
```

This is the recommended invocation for the current repository layout because the project package lives under `src/`.

## Current project status

The repository is currently positioned as a working retrieval prototype for the AI Studio milestone flow:

- data and preprocessing pipeline are in place
- token-aware chunking is implemented and validated
- a ChromaDB knowledge base is built per config
- retrieval quality is measured with a deterministic eval script
- optional reranking and future RAG orchestration are supported by the architecture

The project’s current notes indicate that the team is still deciding on the canonical cleaned dataset and preparing for the official ground-truth Q&A evaluation set planned for the next milestone phase.

## Stretch goals and next milestones

The repository document set references the following directions:

1. Agentic self-correction with LangGraph
2. Local reranking with FlashRank
3. Deterministic evaluation using rule-based metrics
4. Browser UI prototyping with Gradio

The next milestone direction is to layer a baseline RAG pipeline on top of the working knowledge-base search layer, using Gemini or similar LLM APIs for final answer generation.

## License

This project repository does not currently contain a project-specific license file. Before public distribution, confirm the appropriate license with the challenge advisor and add the corresponding file if needed.

## References

Useful project references include:

- Retrieval-Augmented Generation for Large Language Models: A Survey
- Dense Passage Retrieval for Open-Domain Question Answering
- Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
- FlashAttention
- LoRA
- RAGAS
- ChromaDB documentation
- LangChain RAG documentation

## Notes

The implementation is deliberately built for local experimentation and reproducibility. It emphasizes:

- deterministic chunk configuration
- clean provenance tracking of chunk metadata
- embedding-model-aware tokenizer sizing
- local evaluation before adding heavier API-driven generation layers

This keeps the retrieval pipeline testable, auditable, and easy to compare across multiple chunking strategies.
