"""End-to-end RAG evaluation on the 40 ground-truth questions.

One LLM call per question. The defaults reproduce the October baseline
(no reranking, results in data/eval/rag_evaluation.json). The final
evaluation runs the tuned pipeline into its own file:

    python scripts/evaluate_rag.py --limit 40 --rerank \\
        --output data/eval/final_rag_evaluation.json --resume

Results are saved after every question. `--resume` keeps the questions that
already succeeded in the output file and only runs the rest, so a run can be
spread over several days of free-tier quota (20 requests a day). The run
stops at the first daily-quota error instead of recording 40 failures.

Per question it records paper-level hit@5 (any excerpt from the right
paper), page-level hit@5 (an excerpt from a page holding the answer
evidence, per data/eval/ground_truth_pages.json), whether the answer cited
such an excerpt, citation-label validity, and keyword recall. Answer
correctness is graded by hand afterwards (see generate_eval_report.py).
"""

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.rag import build_rag_chain, get_llm, is_daily_quota_error, validate_citations  # noqa: E402

QUESTIONS_PATH = REPO / "data" / "ground_truth_qa.json"
PAGES_PATH = REPO / "data" / "eval" / "ground_truth_pages.json"
DEFAULT_OUTPUT = REPO / "data" / "eval" / "rag_evaluation.json"


def load_page_labels(questions, path=PAGES_PATH):
    """Expected evidence pages per question, in question order. Fails loudly
    if the label file and the question file have drifted apart."""
    labels = json.loads(Path(path).read_text(encoding="utf-8"))["labels"]
    if len(labels) != len(questions):
        raise ValueError(f"{len(labels)} page labels for {len(questions)} questions")
    for i, (label, item) in enumerate(zip(labels, questions), start=1):
        if label["index"] != i or label["question"] != item["question"]:
            raise ValueError(f"page label {i} does not match question {i}")
    return [label["expected_pages"] for label in labels]


def evidence_sources(sources, paper_id, expected_pages):
    """Source numbers whose excerpt comes from a page holding the evidence."""
    pages = set(expected_pages)
    return [
        s["n"]
        for s in sources
        if s["paper_id"] == paper_id
        and any(p in pages for p in range(s["page_start"], s["page_end"] + 1))
    ]


def score(item, answer, sources, expected_pages):
    """Deterministic metrics for one answered question."""
    citations = validate_citations(answer, sources)
    on_evidence = evidence_sources(sources, item["paper_id"], expected_pages)
    keywords = item.get("key_keywords", [])
    matched = [k for k in keywords if k.lower() in answer.lower()]
    return {
        "retrieval_hit_at_5": any(s["paper_id"] == item["paper_id"] for s in sources),
        "page_hit_at_5": bool(on_evidence),
        "cited_evidence_page": bool(set(on_evidence) & set(citations["cited_sources"])),
        "citation_valid": citations["valid"],
        "citation_count": len(citations["cited_sources"]),
        "invalid_citations": citations["invalid_sources"],
        "keyword_recall": round(len(matched) / len(keywords), 3) if keywords else None,
        "matched_keywords": matched,
    }


def load_previous(path):
    """Successful records from an earlier run, keyed by question text."""
    path = Path(path)
    if not path.exists():
        return {}
    return {
        r["question"]: r
        for r in json.loads(path.read_text(encoding="utf-8"))
        if r.get("status") == "success"
    }


def same_config(previous, run_info):
    """Keep only earlier records made with the same settings (date aside),
    so a resumed run never mixes configurations. Returns (kept, n_dropped)."""
    key = {k: v for k, v in run_info.items() if k != "date"}
    kept = {
        q: r for q, r in previous.items()
        if {k: v for k, v in r.get("run", {}).items() if k != "date"} == key
    }
    return kept, len(previous) - len(kept)


def run(questions, page_labels, chain, output_path, run_info, previous=None, pause=5.0):
    """Evaluate `questions` with `chain`, saving after each one.

    Returns (results, stopped_early). Questions found in `previous` are kept
    as they are and cost no call."""
    previous = previous or {}
    results, stopped_early, calls = [], False, 0
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for i, (item, expected_pages) in enumerate(zip(questions, page_labels), start=1):
        if item["question"] in previous:
            results.append(previous[item["question"]])
            continue
        if stopped_early:
            continue
        if calls and pause:
            time.sleep(pause)  # stay under the per-minute rate limit
        calls += 1
        print(f"\nQuestion {i}/{len(questions)}: {item['question']}")
        record = {
            "index": i,
            "question": item["question"],
            "expected_answer": item["ground_truth_answer"],
            "expected_paper_id": item["paper_id"],
            "expected_pages": expected_pages,
            "run": run_info,
        }
        started = time.perf_counter()
        try:
            response = chain.invoke(item["question"])
        except Exception as error:
            if is_daily_quota_error(error):
                print("Daily API quota used up; stopping. Rerun with --resume after it resets.")
                stopped_early = True
                continue
            record.update(status="error", error=str(error))
            print("Error:", error)
        else:
            record.update(
                generated_answer=response["answer"],
                sources=response["sources"],
                seconds=round(time.perf_counter() - started, 1),
                status="success",
                **score(item, response["answer"], response["sources"], expected_pages),
            )
            print(
                f"paper hit {record['retrieval_hit_at_5']}, page hit {record['page_hit_at_5']}, "
                f"cited evidence {record['cited_evidence_page']}, citations valid {record['citation_valid']}"
            )
        results.append(record)
        output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results, stopped_early


def summarize(results, total):
    ok = [r for r in results if r["status"] == "success"]
    print("\nEVALUATION SUMMARY")
    print(f"Answered: {len(ok)}/{total}  (errors: {len(results) - len(ok)})")
    if not ok:
        return

    def rate(key):
        known = [r[key] for r in ok if key in r]
        return f"{sum(known)}/{len(known)}" if known else "n/a"

    for label, key in [
        ("Paper hit@5", "retrieval_hit_at_5"),
        ("Page hit@5", "page_hit_at_5"),
        ("Cited an evidence page", "cited_evidence_page"),
        ("Citations valid", "citation_valid"),
    ]:
        print(f"{label}: {rate(key)}")
    recalls = [r["keyword_recall"] for r in ok if r.get("keyword_recall") is not None]
    if recalls:
        print(f"Keyword recall: {sum(recalls) / len(recalls):.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--limit", type=int, default=2, help="first N questions (1-40, default 2)")
    parser.add_argument("--rerank", action="store_true", help="rerank with FlashRank (the app default)")
    parser.add_argument("--rerank-pool", type=int, default=40, help="candidates to rerank (default 40)")
    parser.add_argument("--k", type=int, default=5, help="excerpts per question (default 5)")
    parser.add_argument("--model", help="LLM model name (default from .env / provider default)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="results JSON path")
    parser.add_argument("--resume", action="store_true", help="skip questions already answered in --output")
    args = parser.parse_args()
    if not 1 <= args.limit <= 40:
        parser.error("--limit must be between 1 and 40")
    if args.rerank and args.rerank_pool < args.k:
        parser.error("--rerank-pool must be at least --k")
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output

    all_questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    page_labels = load_page_labels(all_questions)[: args.limit]
    questions = all_questions[: args.limit]
    previous = load_previous(output) if args.resume else {}
    if not args.resume and output.exists():
        print(f"Note: overwriting {output} (use --resume to keep finished questions).")

    from src.retriever import get_retriever

    print("Loading retriever...")
    retriever = get_retriever(k=args.k, rerank=args.rerank, rerank_pool=args.rerank_pool if args.rerank else None)
    llm = get_llm(model=args.model)
    run_info = {
        "date": date.today().isoformat(),
        "model": str(getattr(llm, "model", None) or getattr(llm, "model_name", "")).removeprefix("models/"),
        "k": args.k,
        "rerank": args.rerank,
        "rerank_pool": args.rerank_pool if args.rerank else None,
    }
    previous, mismatched = same_config(previous, run_info)
    if mismatched:
        print(f"Re-running {mismatched} question(s) answered with different settings.")
    print(f"Run: {run_info}; {len(previous)} question(s) already done")

    results, stopped = run(questions, page_labels, build_rag_chain(retriever, llm), output, run_info, previous)
    summarize(results, len(questions))
    print("Results saved to:", output)
    if stopped:
        sys.exit(2)


if __name__ == "__main__":
    main()
