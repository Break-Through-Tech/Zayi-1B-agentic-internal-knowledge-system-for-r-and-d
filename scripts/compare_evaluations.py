"""Compare the October baseline with the final evaluation run (no API calls).

Scores both runs' saved answers with the same deterministic metrics from
scripts/evaluate_rag.py, so the comparison uses today's citation rules for
both, and adds the manual correctness grades where they exist:

    data/eval/rag_evaluation.json        + data/eval/manual_scores.json        (October)
    data/eval/final_rag_evaluation.json  + data/eval/final_manual_scores.json  (final)

Prints Markdown tables (summary and per question) and writes the numbers to
data/eval/final_comparison.json.

Run from the repo root:  python scripts/compare_evaluations.py
"""

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
EVAL = REPO / "data" / "eval"

_spec = importlib.util.spec_from_file_location("evaluate_rag", Path(__file__).with_name("evaluate_rag.py"))
evaluate_rag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluate_rag)

METRICS = [
    ("Paper hit@5", "retrieval_hit_at_5"),
    ("Page hit@5", "page_hit_at_5"),
    ("Cited an evidence page", "cited_evidence_page"),
    ("Citations valid", "citation_valid"),
]
RUBRIC = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def rescore(questions, page_labels, results):
    """Per-question metrics for a run, keyed by 1-based question index.
    Questions the run didn't answer successfully are left out."""
    by_question = {r["question"]: r for r in results if r.get("status") == "success"}
    scored = {}
    for i, (item, pages) in enumerate(zip(questions, page_labels), start=1):
        r = by_question.get(item["question"])
        if r is not None:
            scored[i] = evaluate_rag.score(item, r["generated_answer"], r["sources"], pages)
            scored[i]["seconds"] = r.get("seconds")
    return scored


def load_grades(path):
    path = Path(path)
    if not path.exists():
        return {}
    grades = json.loads(path.read_text(encoding="utf-8"))["scores"]
    unknown = set(grades.values()) - set(RUBRIC)
    if unknown:
        raise ValueError(f"unknown grades in {path.name}: {sorted(unknown)}")
    return {int(k): v for k, v in grades.items()}


def summarize(scored, grades, total):
    """Totals for one run. Rates are over the answered questions."""
    n = len(scored)
    summary = {"answered": n, "total": total}
    for _, key in METRICS:
        summary[key] = sum(s[key] for s in scored.values())
    recalls = [s["keyword_recall"] for s in scored.values() if s["keyword_recall"] is not None]
    summary["keyword_recall"] = round(sum(recalls) / len(recalls), 3) if recalls else None
    graded = {i: g for i, g in grades.items() if i in scored}
    if graded:
        counts = Counter(graded.values())
        summary["graded"] = len(graded)
        summary.update({g: counts.get(g, 0) for g in RUBRIC})
        summary["weighted_quality"] = round(sum(RUBRIC[g] for g in graded.values()) / len(graded), 3)
    return summary


def _cell(summary, key):
    if key not in summary or summary[key] is None:
        return "n/a"
    if key in ("keyword_recall", "weighted_quality"):
        return f"{summary[key]:.1%}"
    base = summary["graded"] if key in RUBRIC else summary["answered"]
    return f"{summary[key]}/{base}"


def markdown(before, after, scored_before, scored_after, grades_before, grades_after):
    rows = [("Answered", None)] + METRICS + [("Keyword recall", "keyword_recall")]
    rows += [("Correct", "correct"), ("Partial", "partial"), ("Incorrect", "incorrect"),
             ("Weighted answer quality", "weighted_quality")]
    lines = ["| Metric | October baseline | Final |", "|---|---:|---:|"]
    for label, key in rows:
        if key is None:
            lines.append(f"| {label} | {before['answered']}/{before['total']} | {after['answered']}/{after['total']} |")
        else:
            lines.append(f"| {label} | {_cell(before, key)} | {_cell(after, key)} |")

    lines += ["", "| Q | Page hit (Oct → final) | Cited evidence (Oct → final) | Grade (Oct → final) |", "|---:|---|---|---|"]
    mark = {True: "yes", False: "no", None: "-"}
    for i in sorted(set(scored_before) | set(scored_after)):
        b, a = scored_before.get(i, {}), scored_after.get(i, {})
        lines.append(
            f"| {i} | {mark[b.get('page_hit_at_5')]} → {mark[a.get('page_hit_at_5')]} "
            f"| {mark[b.get('cited_evidence_page')]} → {mark[a.get('cited_evidence_page')]} "
            f"| {grades_before.get(i, '-')} → {grades_after.get(i, '-')} |"
        )
    return "\n".join(lines)


def main():
    questions = json.loads(evaluate_rag.QUESTIONS_PATH.read_text(encoding="utf-8"))
    labels = evaluate_rag.load_page_labels(questions)
    runs = {}
    for name, results_file, grades_file in [
        ("october", "rag_evaluation.json", "manual_scores.json"),
        ("final", "final_rag_evaluation.json", "final_manual_scores.json"),
    ]:
        results = json.loads((EVAL / results_file).read_text(encoding="utf-8"))
        scored = rescore(questions, labels, results)
        grades = load_grades(EVAL / grades_file)
        runs[name] = {"scored": scored, "grades": grades, "summary": summarize(scored, grades, len(questions))}

    print(markdown(
        runs["october"]["summary"], runs["final"]["summary"],
        runs["october"]["scored"], runs["final"]["scored"],
        runs["october"]["grades"], runs["final"]["grades"],
    ))
    out = EVAL / "final_comparison.json"
    out.write_text(json.dumps(
        {name: {"summary": run["summary"], "per_question": run["scored"], "grades": run["grades"]}
         for name, run in runs.items()},
        indent=2,
    ), encoding="utf-8")
    print(f"\nSaved to {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
