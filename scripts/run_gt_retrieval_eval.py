"""Retrieval evaluation on the official 40-question ground-truth set (November).

Joins data/ground_truth_qa.json with the page labels in
data/eval/ground_truth_pages.json and measures, per chunking config and
rerank setting, paper-level and page-level hit@k and MRR. Page-level is the
one that matters: a page hit means a chunk from a page that contains the
answer evidence was retrieved, not just some chunk of the right paper. (It
does not check that the chunk itself holds the answer, so it is an upper
bound — slightly looser for small chunks, which split a page into several.)

No LLM calls — free to run. Results go to data/eval/gt_retrieval_results.json.

Run from the repo root:  python scripts/run_gt_retrieval_eval.py
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

import chromadb

from src.chunking import ChunkingConfig
from src.data_io import DEFAULT_DATA_PATH
from src.retrieval_eval import evaluate_retrieval
from src.retriever import load_knowledge_base

GT_PATH = REPO / "data" / "ground_truth_qa.json"
PAGES_PATH = REPO / "data" / "eval" / "ground_truth_pages.json"
RESULTS_PATH = REPO / "data" / "eval" / "gt_retrieval_results.json"

CONFIGS = [
    ChunkingConfig(chunk_size=256, chunk_overlap=32, scope="page"),
    ChunkingConfig(chunk_size=384, chunk_overlap=48, scope="page"),
    ChunkingConfig(chunk_size=512, chunk_overlap=64, scope="page"),  # October default
]
# None = embedding similarity only (October eval); ints = rerank that many
# candidates. 20 was the app default before 2026-11-09; the app now uses 40.
RERANK_POOLS = [None, 20, 40, 80]
KS = (1, 3, 5, 10)


def load_examples():
    questions = json.loads(GT_PATH.read_text(encoding="utf-8"))
    labels = json.loads(PAGES_PATH.read_text(encoding="utf-8"))["labels"]
    if len(labels) != len(questions):
        raise ValueError(f"{len(labels)} page labels for {len(questions)} questions")
    examples = []
    for item, label in zip(questions, labels):
        if label["question"] != item["question"] or label["paper_id"] != item["paper_id"]:
            raise ValueError(f"page label {label['index']} does not match its question")
        examples.append(
            {
                "query": item["question"],
                "expected_paper_id": item["paper_id"],
                "expected_pages": label["expected_pages"],
            }
        )
    return examples


def main():
    examples = load_examples()
    client = chromadb.PersistentClient(path=str(REPO / "chroma"))

    rows = []
    header = "".join(f"{f'p@{k}':>7}" for k in KS) + "".join(f"{f'paper@{k}':>9}" for k in (1, 5))
    print(f"{len(examples)} questions | source: {DEFAULT_DATA_PATH.name}")
    print(f"{'config':<12}{'chunks':>7}{'rerank':>8}{header}{'p_mrr':>7}{'s/q':>6}")
    for config in CONFIGS:
        collection = load_knowledge_base(config, client=client)  # re-embeds only if stale
        n_chunks = collection.count()
        label = f"{config.chunk_size}/{config.chunk_overlap}"
        for pool in RERANK_POOLS:
            start = time.perf_counter()
            report = evaluate_retrieval(
                collection, examples, ks=KS, rerank=pool is not None, rerank_pool=pool
            )
            per_query = (time.perf_counter() - start) / len(examples)
            depth = max(KS)
            print(
                f"{label:<12}{n_chunks:>7}{str(pool or 'off'):>8}"
                + "".join(f"{report[f'page_hit@{k}']:>7}" for k in KS)
                + "".join(f"{report[f'hit@{k}']:>9}" for k in (1, 5))
                + f"{report[f'page_mrr@{depth}']:>7}{per_query:>6.2f}"
            )
            rows.append(
                {
                    "chunk_size": config.chunk_size,
                    "chunk_overlap": config.chunk_overlap,
                    "scope": config.scope,
                    "n_chunks": n_chunks,
                    "rerank_pool": pool,
                    "seconds_per_query": round(per_query, 3),
                    **{
                        key: value
                        for key, value in report.items()
                        if key.startswith(("hit@", "page_hit@", "mrr@", "page_mrr@"))
                    },
                    "page_rank": [d["page_rank"] for d in report["details"]],
                    "paper_rank": [d["paper_rank"] for d in report["details"]],
                    "collection": collection.name,
                }
            )

    payload = {
        "date": date.today().isoformat(),
        "source_file": DEFAULT_DATA_PATH.name,
        "questions_file": GT_PATH.name,
        "page_labels_file": PAGES_PATH.name,
        "n_questions": len(examples),
        "note": (
            "page_rank / paper_rank: per question (ground-truth order), 1-based rank "
            f"of the first page/paper hit within the top {max(KS)}, null = not found"
        ),
        "results": rows,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {RESULTS_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
