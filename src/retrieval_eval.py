"""Deterministic retrieval evaluation (groundwork for the November eval and
challenge stretch goal #3 — no LLM-as-judge, no API calls).

Examples are (query, expected_paper_id) tuples, or dicts that may add page
labels:

    {"query": ..., "expected_paper_id": ..., "expected_pages": [3, 4]}

Paper-level hit rates are always computed; when an example carries
`expected_pages`, page-level hits are also computed — a result is a page hit
when its paper matches AND its cited page range [page_start, page_end]
contains one of the expected pages. The November ground-truth Q&A pairs are
page-level, so they plug in as dicts with `expected_pages`.
"""

from src.knowledge_base import search


def _normalize(example):
    if isinstance(example, dict):
        return example["query"], example["expected_paper_id"], example.get("expected_pages")
    query, expected_paper = example
    return query, expected_paper, None


def evaluate_retrieval(collection, examples, ks=(1, 3), rerank=False):
    """Hit rates at each k in `ks` (paper-level always; page-level over the
    examples that carry page labels), paper-level MRR, and per-query details
    for error analysis. Also lists the queries that missed at rank 1."""
    if not examples:
        raise ValueError("no examples to evaluate")
    if not ks:
        raise ValueError("ks must contain at least one cutoff")
    max_k = max(ks)
    paper_hits = {k: 0 for k in ks}
    page_hits = {k: 0 for k in ks}
    n_page_labeled = 0
    reciprocal_ranks = []
    details = []
    misses_at_1 = []

    for example in examples:
        query, expected_paper, expected_pages = _normalize(example)
        rows = search(collection, query, k=max_k, rerank=rerank)

        paper_rank = next(
            (i + 1 for i, r in enumerate(rows) if r["paper_id"] == expected_paper), None
        )
        reciprocal_ranks.append(1.0 / paper_rank if paper_rank else 0.0)
        for k in ks:
            if paper_rank is not None and paper_rank <= k:
                paper_hits[k] += 1

        page_rank = None
        if expected_pages:
            n_page_labeled += 1
            page_rank = next(
                (
                    i + 1
                    for i, r in enumerate(rows)
                    if r["paper_id"] == expected_paper
                    and any(r["page_start"] <= p <= r["page_end"] for p in expected_pages)
                ),
                None,
            )
            for k in ks:
                if page_rank is not None and page_rank <= k:
                    page_hits[k] += 1

        if paper_rank != 1:
            misses_at_1.append(
                {"query": query, "expected": expected_paper, "got": [r["paper_id"] for r in rows]}
            )
        details.append(
            {
                "query": query,
                "expected_paper_id": expected_paper,
                "expected_pages": expected_pages,
                "paper_rank": paper_rank,
                "page_rank": page_rank,
                "retrieved": [
                    {
                        "chunk_id": r["chunk_id"],
                        "paper_id": r["paper_id"],
                        "page_start": r["page_start"],
                        "page_end": r["page_end"],
                    }
                    for r in rows
                ],
            }
        )

    n = len(examples)
    report = {"n_queries": n, "n_page_labeled": n_page_labeled, "rerank": rerank}
    for k in ks:
        report[f"hit@{k}"] = round(paper_hits[k] / n, 3)  # paper-level
    report["mrr"] = round(sum(reciprocal_ranks) / n, 3)  # paper-level
    if n_page_labeled:
        for k in ks:
            report[f"page_hit@{k}"] = round(page_hits[k] / n_page_labeled, 3)
    report["misses_at_1"] = misses_at_1
    report["details"] = details
    return report
