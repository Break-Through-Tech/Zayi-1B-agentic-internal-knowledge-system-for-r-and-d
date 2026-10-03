"""
test_reference_filtering.py

Tests the hypothesis from Progress.md's Notes: "Reference-heavy pages may
introduce noise into semantic retrieval." Builds two ChromaDB stores from
the same cleaned corpus -- one with reference/bibliography pages included
(current default), one with them filtered out before chunking -- then runs
the real evaluate_retrieval() against both and prints a side-by-side
comparison.

The two variants are kept in SEPARATE on-disk Chroma stores (not just
separate collections) because config_fingerprint() doesn't encode whether
reference pages were filtered -- both variants produce the identical
config_id, so collection_name_for() would give them the same name. Separate
`path=` values avoid that collision entirely.

Usage (run from repo root):
    python scripts/test_reference_filtering.py \
        --input data/curated_papers_text_corrected.json \
        --eval-queries data/eval_queries_sample.json \
        --chunk-size 512 --chunk-overlap 64 --scope page --ks 1 3
"""

import sys
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import chromadb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


from src.chunking import ChunkingConfig, build_chunks, chunk_stats
from src.knowledge_base import build_knowledge_base
from src.retrieval_eval import evaluate_retrieval
from src.cleaning import clean_pages

# --------------------------------------------------------------------------
# Reference-page detection (heuristic -- tune against corpus)
# --------------------------------------------------------------------------
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works cited)\s*$",
    re.IGNORECASE | re.MULTILINE
)

CITATION_LINE_RE = re.compile(
    r"^\s*(\[\d+\]|\(\d+\)|\d+\.)\s",
    re.IGNORECASE
)

REFERENCE_LINE_RATIO = 0.5


def is_reference_page(text: str) -> bool:
    if REFERENCE_HEADING_RE.search(text):
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return False
    citation_lines = sum(1 for ln in lines if CITATION_LINE_RE.match(ln))
    return (citation_lines / len(lines)) >= REFERENCE_LINE_RATIO


def load_pages(input_path: Path):
    """Loads the corpus and flattens it into the page-record shape
    build_chunks()/knowledge_base.py expect: one dict per page, each
    carrying its own paper_id/paper_title, rather than pages nested
    under each paper. Assumes the raw corpus shape (list of papers, each
    with an "id", "title", and a "pages" list of {page_number, text})."""
    with open(input_path, "r", encoding="utf-8") as f:
        papers = json.load(f)

    pages = []
    for paper in papers:
        for page in paper["pages"]:
            pages.append(
                {
                    "paper_id": paper["id"],
                    "paper_title": paper["title"],
                    "page_number": page["page_number"],
                    "text": page["text"],
                }
            )
    return pages


def split_reference_pages_v1(pages):
    """Flags reference/bibliography pages, with one stateful rule beyond
    is_reference_page(): once a paper's references section has started
    (via a heading match), every later page of that SAME paper is also
    treated as references -- even if its citation-line ratio is too low
    to trigger on its own. This matters because multi-page bibliographies
    are common (e.g. the RAG paper's references run 6 pages), and wrapped,
    multi-line citation entries push the per-page bracket-line ratio below
    threshold on continuation pages that have no heading of their own.
 
    KNOWN LIMITATION (superseded by v2, kept here for before/after
    comparison only -- not the default): is_reference_page() checks the
    heading against the WHOLE page text, so when a page has body content
    followed by the References heading partway down (common -- see v2's
    docstring), this drops the body content on that page too, along with
    the bibliography. v2 has the identical heading-triggers-everything-
    after behavior but a clearer docstring naming this limitation; v1 is
    retained only so --detector can show the two side by side."""
    kept, dropped = [], []
    in_references = {}  # paper_id -> bool, once True stays True
    for page in sorted(pages, key=lambda p: (p["paper_id"], p["page_number"])):
        pid = page["paper_id"]
        already_in_refs = in_references.get(pid, False)
        flagged = already_in_refs or is_reference_page(page.get("text", ""))
        in_references[pid] = flagged
        (dropped if flagged else kept).append(page)
    return kept, dropped

def has_reference_heading(text: str) -> bool:
    return bool(REFERENCE_HEADING_RE.search(text))

def split_reference_pages_v2(pages):
    """
    This operates at whole-page granularity: if the References heading
    appears partway through a page (common in this corpus -- see the
    "Flagged pages by paper" report this script prints), the preceding
    body text on that page is also removed along with the bibliography.
    This trade-off is deliberate for this experiment: splitting a page at
    the heading and keeping only its body half would be more precise, but
    introduces a second kind of transformation (content surgery, not just
    page inclusion/exclusion) that makes the experiment's unit of analysis
    harder to reason about. Whole-page removal keeps the comparison simple
    and reproducible at the cost of some lost body text on heading pages.
    """
    kept, dropped = [], []
    in_references = {}

    for page in sorted(pages, key=lambda p: (p["paper_id"], p["page_number"])):
        pid = page["paper_id"]
        text = page.get("text", "")

        matches = REFERENCE_HEADING_RE.findall(text)

        if matches:
            print(
                f"HEADING MATCH: {pid} p.{page.get('page_number')}: "
                f"{matches}"
    )

        if has_reference_heading(text):
            in_references[pid] = True

        flagged = in_references.get(pid, False)
        (dropped if flagged else kept).append(page)

    return kept, dropped

# --------------------------------------------------------------------------
# Build one variant end to end: pages -> chunks -> ChromaDB collection
# --------------------------------------------------------------------------

def build_variant(pages, store_path, config):
    chunks = build_chunks(pages, config=config)
    stats = chunk_stats(chunks)
    client = chromadb.PersistentClient(path=str(store_path))
    collection = build_knowledge_base(chunks, client=client, path=str(store_path))
    return collection, stats


# --------------------------------------------------------------------------
# Compare eval reports
# --------------------------------------------------------------------------

_METRIC_PREFIXES = ("hit@", "mrr@", "page_hit@", "page_mrr@")


def print_comparison(report_with, report_without):
    keys = [
        k
        for k in report_with
        if isinstance(report_with[k], (int, float)) and k.startswith(_METRIC_PREFIXES)
    ]
    print(f"\n{'metric':<14}{'with refs':>12}{'without refs':>15}{'delta':>10}")
    for key in keys:
        w = report_with[key]
        wo = report_without.get(key)
        if wo is None:
            continue
        print(f"{key:<14}{w:>12.3f}{wo:>15.3f}{wo - w:>+10.3f}")

    # Queries that flip from a miss to a hit (or vice versa) at rank 1 are
    # the most actionable signal -- more so than the aggregate hit@k delta.
    misses_with = {m["query"] for m in report_with["misses_at_1"]}
    misses_without = {m["query"] for m in report_without["misses_at_1"]}
    fixed = misses_with - misses_without
    broken = misses_without - misses_with
    if fixed:
        print(f"\nFixed by dropping reference pages ({len(fixed)}):")
        for q in fixed:
            print(f"  + {q}")
    if broken:
        print(f"\nBroken by dropping reference pages ({len(broken)}):")
        for q in broken:
            print(f"  - {q}")
    if not fixed and not broken:
        print("\nNo queries flipped at rank 1 -- reference pages aren't "
              "changing top-1 retrieval for this query set.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/curated_papers_text_corrected.json"))
    parser.add_argument("--eval-queries", type=Path, default=Path("data/eval_queries_sample.json"))
    parser.add_argument("--chunk-size", type=int, default=512, help="tokens, per ChunkingConfig")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="tokens")
    parser.add_argument("--scope", choices=["page", "paper"], default="page")
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--store-dir", type=Path, default=Path("chroma_experiments"))
    parser.add_argument(
        "--detector",
        choices=["v1", "v2"],
        default="v2",
        help="Reference-page detection strategy. v2 is the maintained "
             "default; v1 is kept only for before/after comparison -- see "
             "its docstring for the bug v2 fixes.",
    )
    args = parser.parse_args()

    print(f"Loading pages from {args.input} ...")
    pages = clean_pages(load_pages(args.input))
    print(f"  {len(pages)} total pages")

    #kept, dropped = split_reference_pages(pages)
    print(f"\nUsing detector: {args.detector}")
    if args.detector == "v1":
        kept, dropped = split_reference_pages_v1(pages)
    else:
        kept, dropped = split_reference_pages_v2(pages)

    print(f"  {len(dropped)} pages flagged as reference/bibliography pages")
    print(f"  {len(kept)} pages retained")
    if dropped:
        sample = dropped[0]
        preview = sample.get("text", "")[:120].replace("\n", " ")
        print(f"  Example flagged page ({sample.get('paper_id', '?')}, "
              f"p.{sample.get('page_number', '?')}): \"{preview}...\"")

    refs_by_paper = defaultdict(list)

    for page in dropped:
        refs_by_paper[page["paper_id"]].append(
            page["page_number"]
        )

    print("\nFlagged pages by paper:")

    for pid, page_nums in sorted(refs_by_paper.items()):
        print(
            f"  {pid}: "
            f"{sorted(page_nums)} "
            f"({len(page_nums)} pages)"
        )

    config = ChunkingConfig(
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap, scope=args.scope
    )

    print("\nBuilding WITH-references store ...")
    kb_with, stats_with = build_variant(pages, args.store_dir / "with_references", config)
    print(f"  {stats_with}")

    print("Building WITHOUT-references store ...")
    kb_without, stats_without = build_variant(kept, args.store_dir / "without_references", config)
    print(f"  {stats_without}")

    with open(args.eval_queries, "r", encoding="utf-8") as f:
        examples = json.load(f)

    print(f"\nRunning eval (ks={args.ks}, rerank={args.rerank}) against both stores ...")
    report_with = evaluate_retrieval(kb_with, examples, ks=tuple(args.ks), rerank=args.rerank)
    report_without = evaluate_retrieval(kb_without, examples, ks=tuple(args.ks), rerank=args.rerank)

    print_comparison(report_with, report_without)




if __name__ == "__main__":
    main()
