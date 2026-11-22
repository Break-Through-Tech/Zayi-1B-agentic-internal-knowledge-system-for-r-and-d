"""Triage-memo evaluation: one memo per paper (data/eval/memo_topics.json).

Measures two of the brief's metrics for memos:
- Citation precision: share of [S#] citations that point at a retrieved
  excerpt, plus the three automatic pre-review checks from src/memo.py.
- First-pass approval rate: share of memos a reviewer would approve with no
  or minimal corrections. That part is manual: reviewers read the Markdown
  files written to data/eval/memos/ and record a verdict per memo in
  data/eval/memo_reviews.json (approve / minor / major); `--report` then
  combines both.

One LLM call per memo, final pipeline settings (rerank pool 40, k = 5).
Results are saved after every memo; `--resume` skips memos already done and
the run stops cleanly at the daily quota.

Run from the repo root:
    python scripts/evaluate_memos.py --resume      # draft the memos (10 calls)
    python scripts/evaluate_memos.py --report      # summary, no API calls
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.memo import MEMO_SECTIONS, _not_covered, draft_memo, memo_document, split_sections  # noqa: E402
from src.rag import is_daily_quota_error  # noqa: E402

EVAL = REPO / "data" / "eval"
TOPICS_PATH = EVAL / "memo_topics.json"
RESULTS_PATH = EVAL / "memo_evaluation.json"
REVIEWS_PATH = EVAL / "memo_reviews.json"
MEMO_DIR = EVAL / "memos"
VERDICTS = ("approve", "minor", "major")  # approve and minor count as first-pass approved


def memo_metrics(result, paper_id):
    """Deterministic metrics for one drafted memo."""
    cited = result["citations"]["cited_sources"]
    invalid = result["citations"]["invalid_sources"]
    sections = split_sections(result["memo"])
    return {
        "passed_checks": result["passed"],
        "checks": {c["name"]: c["ok"] for c in result["checks"]},
        "citations": len(cited),
        "invalid_citations": invalid,
        "citation_precision": round((len(cited) - len(invalid)) / len(cited), 3) if cited else None,
        "sections_not_covered": [s for s in MEMO_SECTIONS if _not_covered(sections.get(s, ""))],
        "sources_from_topic_paper": sum(s["paper_id"] == paper_id for s in result["sources"]),
    }


def run(topics, retriever, llm, results_path, run_info, previous=None, pause=5.0, memo_dir=None):
    """Draft one memo per topic, saving after each. Returns (results, stopped_early)."""
    previous = previous or {}
    memo_dir = Path(memo_dir or MEMO_DIR)
    memo_dir.mkdir(parents=True, exist_ok=True)
    results, stopped, calls = [], False, 0
    for i, t in enumerate(topics, start=1):
        if t["topic"] in previous:
            results.append(previous[t["topic"]])
            continue
        if stopped:
            continue
        if calls and pause:
            time.sleep(pause)
        calls += 1
        print(f"Memo {i}/{len(topics)}: {t['topic']}")
        started = time.perf_counter()
        try:
            docs = retriever.invoke(t["topic"])
            result = draft_memo(t["topic"], docs, llm)
        except Exception as error:
            if is_daily_quota_error(error):
                print("Daily API quota used up; stopping. Rerun with --resume after it resets.")
                stopped = True
                continue
            results.append({"index": i, **t, "run": run_info, "status": "error", "error": str(error)})
            print("Error:", error)
        else:
            path = memo_dir / f"{i:02d}-{t['paper_id']}.md"
            path.write_text(memo_document(result, when=datetime.now()) + "\n", encoding="utf-8")
            record = {
                "index": i, **t, "run": run_info, "status": "success",
                "seconds": round(time.perf_counter() - started, 1),
                "memo_file": path.name, "memo": result["memo"], "sources": result["sources"],
                **memo_metrics(result, t["paper_id"]),
            }
            results.append(record)
            print(f"  checks passed {record['passed_checks']}, citation precision {record['citation_precision']}")
        Path(results_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(results_path).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results, stopped


def report(results, reviews):
    """Summary dict over the successful memos, joined with any reviews."""
    ok = [r for r in results if r.get("status") == "success"]
    cited = sum(r["citations"] for r in ok)
    valid = sum(r["citations"] - len(r["invalid_citations"]) for r in ok)
    summary = {
        "memos": len(ok),
        "passed_all_checks": sum(r["passed_checks"] for r in ok),
        "citation_precision": round(valid / cited, 3) if cited else None,
        "citations": cited,
        "sections_not_covered": sum(len(r["sections_not_covered"]) for r in ok),
    }
    verdicts = [reviews[str(r["index"])]["verdict"] for r in ok if str(r["index"]) in reviews]
    unknown = set(verdicts) - set(VERDICTS)
    if unknown:
        raise ValueError(f"unknown review verdicts: {sorted(unknown)}")
    if verdicts:
        summary["reviewed"] = len(verdicts)
        summary.update({v: verdicts.count(v) for v in VERDICTS})
        summary["first_pass_approval_rate"] = round(
            (verdicts.count("approve") + verdicts.count("minor")) / len(verdicts), 3
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resume", action="store_true", help="skip memos already drafted")
    parser.add_argument("--report", action="store_true", help="summarize saved results and reviews; no API calls")
    parser.add_argument("--model", help="LLM model name (default from .env / provider default)")
    args = parser.parse_args()

    if args.report:
        results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
        reviews = json.loads(REVIEWS_PATH.read_text(encoding="utf-8"))["reviews"] if REVIEWS_PATH.exists() else {}
        print(json.dumps(report(results, reviews), indent=2))
        return

    topics = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))["topics"]
    previous = {}
    if args.resume and RESULTS_PATH.exists():
        previous = {r["topic"]: r for r in json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
                    if r.get("status") == "success"}

    from src.rag import get_llm
    from src.retriever import get_retriever

    llm = get_llm(model=args.model)
    run_info = {
        "date": date.today().isoformat(),
        "model": str(getattr(llm, "model", None) or getattr(llm, "model_name", "")).removeprefix("models/"),
        "k": 5, "rerank": True, "rerank_pool": 40,
    }
    retriever = get_retriever(k=5, rerank=True, rerank_pool=40)
    print(f"Run: {run_info}; {len(previous)} memo(s) already done")
    results, stopped = run(topics, retriever, llm, RESULTS_PATH, run_info, previous)
    print(json.dumps(report(results, {}), indent=2))
    if stopped:
        sys.exit(2)


if __name__ == "__main__":
    main()
