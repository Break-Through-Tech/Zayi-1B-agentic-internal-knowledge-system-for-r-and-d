"""Deterministic retrieval evaluation (groundwork for the November eval and
challenge stretch goal #3 — no LLM-as-judge, no API calls).

Works on labeled examples of (query, expected paper id). When the official
ground-truth Q&A set arrives, its pairs plug straight in — and because
collections are per-config, chunking configs can be compared head-to-head
with the same function.
"""

from src.knowledge_base import search


def evaluate_retrieval(collection, examples, ks=(1, 3), rerank=False):
    """Hit rate at each k in `ks`: the fraction of queries whose expected
    paper appears in the top k results. Returns the rates plus the queries
    that missed at rank 1 (with what was retrieved instead)."""
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    misses_at_1 = []
    for query, expected in examples:
        papers = [r["paper_id"] for r in search(collection, query, k=max_k, rerank=rerank)]
        for k in ks:
            if expected in papers[:k]:
                hits[k] += 1
        if not papers or papers[0] != expected:
            misses_at_1.append({"query": query, "expected": expected, "got": papers})
    n = len(examples)
    report = {"n_queries": n, "rerank": rerank}
    for k in ks:
        report[f"hit@{k}"] = round(hits[k] / n, 3)
    report["misses_at_1"] = misses_at_1
    return report
