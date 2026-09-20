"""Sweep chunking configs (and rerank on/off) against the labeled query set.

Rebuilds one ChromaDB collection per config from the current default data
source, evaluates each with src.retrieval_eval, prints a table, and writes
machine-readable results to data/eval/sweep_results.json.

Run from the repo root:  python scripts/run_retrieval_sweep.py
"""

import json
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

import chromadb

from src.chunking import ChunkingConfig, build_chunks
from src.cleaning import clean_pages
from src.data_io import DEFAULT_DATA_PATH, load_pages
from src.knowledge_base import build_knowledge_base
from src.retrieval_eval import evaluate_retrieval

QUERIES_PATH = REPO / "data" / "eval_queries_sample.json"
RESULTS_PATH = REPO / "data" / "eval" / "sweep_results.json"

CONFIGS = [
    ChunkingConfig(chunk_size=256, chunk_overlap=32, scope="page"),
    ChunkingConfig(chunk_size=384, chunk_overlap=48, scope="page"),
    ChunkingConfig(chunk_size=512, chunk_overlap=64, scope="page"),
    ChunkingConfig(chunk_size=512, chunk_overlap=0, scope="page"),
    ChunkingConfig(chunk_size=512, chunk_overlap=64, scope="paper"),
    ChunkingConfig(chunk_size=384, chunk_overlap=48, scope="paper"),
]


def main():
    examples = json.load(open(QUERIES_PATH))
    cleaned = clean_pages(load_pages())
    client = chromadb.PersistentClient(path=str(REPO / "chroma"))

    rows = []
    print(f"{len(examples)} labeled queries | source: {DEFAULT_DATA_PATH.name}")
    print(f"{'config':<16}{'chunks':>7}{'rerank':>8}{'hit@1':>8}{'hit@3':>8}{'mrr':>8}")
    for config in CONFIGS:
        chunks = build_chunks(cleaned, config)
        collection = build_knowledge_base(chunks, client=client)
        label = f"{config.scope}-{config.chunk_size}/{config.chunk_overlap}"
        for rerank in (False, True):
            report = evaluate_retrieval(collection, examples, ks=(1, 3), rerank=rerank)
            print(
                f"{label:<16}{len(chunks):>7}{str(rerank):>8}"
                f"{report['hit@1']:>8}{report['hit@3']:>8}{report['mrr@3']:>8}"
            )
            rows.append(
                {
                    "scope": config.scope,
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "n_chunks": len(chunks),
                    "rerank": rerank,
                    "hit@1": report["hit@1"],
                    "hit@3": report["hit@3"],
                    "mrr@3": report["mrr@3"],
                    "misses_at_1": [m["expected"] for m in report["misses_at_1"]],
                    "collection": collection.name,
                    "corpus_hash": collection.metadata["corpus_hash"],
                }
            )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": date.today().isoformat(),
        "source_file": DEFAULT_DATA_PATH.name,
        "queries_file": QUERIES_PATH.name,
        "n_queries": len(examples),
        "note": "paper-level labels only; page-level metrics await the official ground-truth set",
        "results": rows,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {RESULTS_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
