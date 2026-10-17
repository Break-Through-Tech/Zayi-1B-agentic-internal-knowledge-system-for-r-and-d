"""Validate the September knowledge base before building the RAG generator

Checks, in order: cleaned data -> chunks -> embedding model -> stored ChromaDB
collection (counts, provenance, record identity, stored vectors) -> sample
similarity searches -> labeled eval set. Prints PASS/FAIL per check and exits
non-zero on failure.

The script only READS the existing collection. If it is missing or stale, the
relevant checks FAIL; pass --build to create or refresh it (replacement
semantics, so stale chunks are removed) and validate the result.

Run from the repo root:
    python scripts/validate_knowledge_base.py
    python scripts/validate_knowledge_base.py --build
    python scripts/validate_knowledge_base.py --chunk-size 384 --chunk-overlap 48
"""
import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import chromadb
import numpy as np

from src.chunking import ChunkingConfig, build_chunks, get_tokenizer
from src.cleaning import clean_pages
from src.data_io import DEFAULT_DATA_PATH, load_pages
from src.embedding import DEFAULT_MODEL, embed_texts
from src.knowledge_base import (
    _corpus_hash,
    build_knowledge_base,
    collection_name_for,
    search,
)
from src.retrieval_eval import evaluate_retrieval

# (label, path, needs_remap): "needs_remap" sets are loaded with different
# field names (question/paper_id) and are translated to the query/
# expected_paper_id shape evaluate_retrieval() expects.
EVAL_SETS = [
    ("eval_queries_sample.json (internal sample, 20 Qs)", REPO / "data" / "eval_queries_sample.json", False),
    ("ground_truth_qa.json (official set, 40 Qs)", REPO / "data" / "ground_truth_qa.json", True),
]


def load_eval_examples(path, needs_remap):
    raw = json.load(open(path))
    if needs_remap:
        return [{"query": item["question"], "expected_paper_id": item["paper_id"]} for item in raw]
    return raw

SAMPLE_QUERIES = [
    ("What is retrieval-augmented generation and how does it combine parametric and non-parametric memory?", "2005.11401"),
    ("How does low-rank adaptation reduce the number of trainable parameters?", "2106.09685"),
    ("How does FlashAttention reduce reads and writes to GPU high bandwidth memory?", "2205.14135"),
    ("How do reflection tokens let a model decide when to retrieve?", "2310.11511"),
    ("Why do language models struggle with information in the middle of long contexts?", "2307.03172"),
    ("How does a retrieval evaluator trigger corrective actions such as web search?", "2401.15884"),
    ("Which metrics measure faithfulness and context relevance without reference answers?", "2309.15217"),
    ("How is a dense passage retriever trained with in-batch negatives?", "2004.04906"),
    ("How does ReAct interleave reasoning traces with actions?", "2210.03629"),
    ("What are the naive, advanced and modular RAG paradigms?", "2312.10997"),
]

REQUIRED_META = {
    "paper_id", "paper_title", "category", "pdf_url", "page_start",
    "page_end", "chunk_index", "token_count", "config_id", "cleaner",
}

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def finish():
    print("\n" + ("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED: {failures}"))
    sys.exit(1 if failures else 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    parser.add_argument("--scope", choices=["page", "paper"], default="page")
    parser.add_argument("--store", type=Path, default=REPO / "chroma")
    parser.add_argument("--build", action="store_true", help="create or refresh the collection before validating")
    args = parser.parse_args()
    config = ChunkingConfig(args.chunk_size, args.chunk_overlap, scope=args.scope)
    print(f"config: chunk_size={config.chunk_size} overlap={config.chunk_overlap} scope={config.scope} model={DEFAULT_MODEL}")

    # 1. cleaned data
    print("\n1. Cleaned data")
    print(f"  source: {DEFAULT_DATA_PATH.resolve()}")
    print("  (validated via the interim cleaner src.cleaning.clean_pages, not data/processed/cleaned_papers.json)")
    pages = clean_pages(load_pages())
    papers = {p["paper_id"] for p in pages}
    check("218 pages from 10 papers", len(pages) == 218 and len(papers) == 10, f"{len(pages)} pages, {len(papers)} papers")
    all_text = "\n".join(p["text"] for p in pages)
    check("no control characters left", not re.search(r"[\x00-\x08\x0b-\x1f\x7f]", all_text))
    check("no ligatures left", not re.search("[\ufb00-\ufb04]", all_text))
    check("no empty pages", all(p["text"].strip() for p in pages))

    # 2. chunks
    print("\n2. Chunks")
    chunks = build_chunks(pages, config)
    tokenizer = get_tokenizer()
    over = [c for c in chunks if len(tokenizer.encode(c["text"])) > config.chunk_size]
    check("every chunk fits the model limit", not over, f"{len(chunks)} chunks, {len(over)} over {config.chunk_size}")
    check("chunk ids unique", len({c["chunk_id"] for c in chunks}) == len(chunks))
    check("all chunk metadata fields present", all(REQUIRED_META <= set(c["metadata"]) for c in chunks))
    max_page = {}
    for p in pages:
        max_page[p["paper_id"]] = max(max_page.get(p["paper_id"], 0), p["page_number"])
    bad = [c for c in chunks if not 1 <= c["metadata"]["page_start"] <= c["metadata"]["page_end"] <= max_page[c["metadata"]["paper_id"]]]
    check("page ranges valid for their paper", not bad)
    check("every paper has chunks", {c["metadata"]["paper_id"] for c in chunks} == papers)

    # 3. embedding model (fresh vectors; stored vectors are checked in step 4)
    print("\n3. Embedding model")
    vecs = embed_texts([c["text"] for c in chunks[:8]])
    check("model returns 384-dim normalized vectors", vecs.shape[1] == 384 and np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5), f"shape {vecs.shape}")

    # 4. stored collection
    print(f"\n4. ChromaDB collection ({args.store})")
    client = chromadb.PersistentClient(path=str(args.store))
    name = collection_name_for(chunks)
    if args.build:
        print(f"  --build: creating or refreshing {name} (stale chunks are replaced)")
        collection = build_knowledge_base(chunks, client=client)
    else:
        # list_collections() returns Collection objects in some chromadb
        # versions and plain names in others; handle both rather than
        # depending on a version-specific "not found" exception class
        existing = {getattr(c, "name", c) for c in client.list_collections()}
        check(f"collection {name} exists", name in existing, "" if name in existing else "not found: rerun with --build to create it")
        if name not in existing:
            finish()
        collection = client.get_collection(name)

    meta = collection.metadata or {}
    print(f"  collection {name}; stored config: " + str({k: meta.get(k) for k in ("embedding_model", "chunk_size", "chunk_overlap", "scope")}))
    check("stored count matches built chunk count", collection.count() == len(chunks), f"{collection.count()} stored vs {len(chunks)} built")
    check("stored corpus hash matches current data", meta.get("corpus_hash") == _corpus_hash(chunks), "mismatch means the store is stale: rerun with --build")
    check("embedding model recorded matches the code", meta.get("embedding_model") == DEFAULT_MODEL, f"stored {meta.get('embedding_model')!r}, code {DEFAULT_MODEL!r}")
    check(
        "stored chunk config matches",
        (meta.get("chunk_size"), meta.get("chunk_overlap"), meta.get("scope")) == (config.chunk_size, config.chunk_overlap, config.scope),
    )

    stored = collection.get(include=["metadatas", "documents", "embeddings"])
    stored_ids = stored["ids"]
    expected = {c["chunk_id"]: c for c in chunks}
    missing, extra = set(expected) - set(stored_ids), set(stored_ids) - set(expected)
    check("stored ids match the ids built from current data", not missing and not extra, f"missing {len(missing)}, extra {len(extra)}")
    text_bad = [i for i, d in zip(stored_ids, stored["documents"]) if i in expected and d != expected[i]["text"]]
    meta_bad = [i for i, m in zip(stored_ids, stored["metadatas"]) if i in expected and m != expected[i]["metadata"]]
    check("stored text matches the intended chunk text", not text_bad, f"{len(text_bad)} differ")
    check("stored metadata matches the intended metadata", not meta_bad, f"{len(meta_bad)} differ")

    emb = np.asarray(stored["embeddings"], dtype=float)
    check("one stored vector per record", emb.ndim == 2 and emb.shape[0] == len(stored_ids), f"shape {emb.shape}")
    check("stored vectors are 384-dim and finite", emb.ndim == 2 and emb.shape[1] == 384 and bool(np.isfinite(emb).all()))
    check("stored vectors are normalized", bool(np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)))
    # re-embed a spread of chunks and compare with what is stored, to catch
    # vectors that are well-formed but belong to different text
    sample = [c for c in chunks[:: max(1, len(chunks) // 10)] if c["chunk_id"] in set(stored_ids)][:10]
    if sample:
        idx = {i: n for n, i in enumerate(stored_ids)}
        fresh = embed_texts([c["text"] for c in sample])
        sims = [float(fresh[n] @ emb[idx[c["chunk_id"]]]) for n, c in enumerate(sample)]
        check("stored vectors match re-embedded text", min(sims) > 0.99, f"min cosine {min(sims):.4f} over {len(sample)} chunks")

    # 5. sample searches
    print("\n5. Sample similarity searches (top-3)")
    hits = 0
    for question, expected_paper in SAMPLE_QUERIES:
        results = search(collection, question, k=3)
        top = [r["paper_id"] for r in results]
        ok = expected_paper in top
        hits += ok
        print(f"\n  Q: {question}\n  expected {expected_paper} -> {'FOUND' if ok else 'MISSING'} (rank {top.index(expected_paper) + 1 if ok else '-'})")
        for r in results:
            snippet = " ".join(r["text"].split())[:90]
            print(f"    {r['similarity']:.3f}  {r['paper_id']}  p.{r['page_start']}-{r['page_end']}  {r['paper_title'][:40]}  | {snippet}...")
    check("expected paper in top-3 for most sample queries", hits >= 8, f"{hits}/{len(SAMPLE_QUERIES)}; paper-level only, read the snippets for relevance")
    first = search(collection, SAMPLE_QUERIES[0][0], k=1)[0]
    check("results carry citation fields", all(first.get(k) not in (None, "") for k in ("paper_title", "page_start", "page_end", "chunk_id", "text")))

    # 6. labeled eval sets -- both the team's internal sample and the
    # official ground-truth set, so a regression against either is visible
    print("\n6. Labeled eval sets")
    for label, path, needs_remap in EVAL_SETS:
        examples = load_eval_examples(path, needs_remap)
        report = evaluate_retrieval(collection, examples, ks=(1, 3))
        print(f"\n  {label}")
        print(f"  {report['n_queries']} queries ({config.scope}-{config.chunk_size}/{config.chunk_overlap}): hit@1={report['hit@1']} hit@3={report['hit@3']} mrr@3={report['mrr@3']}")
        check(f"hit@3 >= 0.90 on {label}", report["hit@3"] >= 0.90, "a miss is a signal to investigate, not proof the store is broken")
        for miss in report["misses_at_1"]:
            print(f"    miss@1: expected {miss['expected']}, got {miss['got']}")

    finish()


if __name__ == "__main__":
    main()
