"""Offline tests for scripts/evaluate_rag.py (no LLM, no retriever)."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_rag.py"
spec = importlib.util.spec_from_file_location("evaluate_rag", SCRIPT)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

ITEM = {
    "paper_id": "p1",
    "question": "What is X?",
    "ground_truth_answer": "X is a low-rank update.",
    "key_keywords": ["low-rank", "frozen"],
}


def source(n, paper_id="p1", start=3, end=None):
    return {"n": n, "paper_id": paper_id, "paper_title": "T", "page_start": start,
            "page_end": start if end is None else end, "chunk_id": f"c{n}"}


class QuotaError(Exception):
    code = 429


class FakeChain:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.asked = []

    def invoke(self, question):
        self.asked.append(question)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return {"answer": outcome, "sources": [source(1), source(2, start=7)]}


def questions(n):
    return [dict(ITEM, question=f"Q{i}?") for i in range(1, n + 1)]


def test_page_labels_match_the_question_file():
    qs = json.loads(ev.QUESTIONS_PATH.read_text(encoding="utf-8"))
    labels = ev.load_page_labels(qs)
    assert len(labels) == len(qs) == 40
    assert all(labels) and all(isinstance(p, int) for pages in labels for p in pages)
    with pytest.raises(ValueError, match="does not match"):
        ev.load_page_labels(list(reversed(qs)))


def test_evidence_sources_checks_paper_and_page_range():
    sources = [source(1, start=2, end=4), source(2, start=5), source(3, paper_id="p2", start=3)]
    assert ev.evidence_sources(sources, "p1", [3]) == [1]
    assert ev.evidence_sources(sources, "p1", [5, 9]) == [2]
    assert ev.evidence_sources(sources, "p1", [9]) == []


def test_score_separates_retrieving_from_citing_the_evidence():
    sources = [source(1, start=3), source(2, start=7)]
    cited_wrong = ev.score(ITEM, "It is a low-rank update [S2].", sources, [3])
    assert cited_wrong["retrieval_hit_at_5"] and cited_wrong["page_hit_at_5"]
    assert not cited_wrong["cited_evidence_page"]
    assert cited_wrong["citation_valid"] and cited_wrong["keyword_recall"] == 0.5

    cited_right = ev.score(ITEM, "Low-rank, frozen weights [S1].", sources, [3])
    assert cited_right["cited_evidence_page"] and cited_right["keyword_recall"] == 1.0

    invalid = ev.score(ITEM, "Something [S9].", sources, [3])
    assert not invalid["citation_valid"] and invalid["invalid_citations"] == [9]


def test_run_saves_each_answer_and_records_ordinary_errors(tmp_path):
    out = tmp_path / "r.json"
    chain = FakeChain(["Low-rank [S1].", RuntimeError("boom")])
    results, stopped = ev.run(questions(2), [[3], [3]], chain, out, {"model": "m"}, pause=0)
    assert not stopped
    assert [r["status"] for r in results] == ["success", "error"]
    assert results[0]["page_hit_at_5"] and results[0]["run"] == {"model": "m"}
    assert json.loads(out.read_text()) == results


def test_run_stops_at_daily_quota_and_resume_skips_finished(tmp_path):
    out = tmp_path / "r.json"
    quota = QuotaError("429 quota GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    first = FakeChain(["A [S1].", quota])
    results, stopped = ev.run(questions(3), [[3]] * 3, first, out, {}, pause=0)
    assert stopped and first.asked == ["Q1?", "Q2?"]  # Q3 never sent
    assert [r["question"] for r in results] == ["Q1?"]  # quota failure not recorded

    second = FakeChain(["B [S1].", "C [S1]."])
    results, stopped = ev.run(questions(3), [[3]] * 3, second, out, {}, ev.load_previous(out), pause=0)
    assert not stopped and second.asked == ["Q2?", "Q3?"]
    assert [r["question"] for r in results] == ["Q1?", "Q2?", "Q3?"]


def test_resume_drops_records_from_other_settings():
    previous = {
        "Q1?": {"run": {"date": "2026-11-20", "model": "m", "rerank": True}},
        "Q2?": {"run": {"date": "2026-11-20", "model": "m", "rerank": False}},
        "Q3?": {},  # October records carry no run info
    }
    kept, dropped = ev.same_config(previous, {"date": "2026-11-22", "model": "m", "rerank": True})
    assert list(kept) == ["Q1?"] and dropped == 2


# --- scripts/compare_evaluations.py -------------------------------------------

_cspec = importlib.util.spec_from_file_location(
    "compare_evaluations", SCRIPT.with_name("compare_evaluations.py")
)
cmp = importlib.util.module_from_spec(_cspec)
_cspec.loader.exec_module(cmp)


def test_rescore_uses_current_rules_and_skips_unanswered():
    qs = questions(3)
    results = [
        {"question": "Q1?", "status": "success", "generated_answer": "Low-rank [S1].",
         "sources": [source(1)], "citation_valid": True},  # stale stored fields are ignored
        {"question": "Q2?", "status": "success", "generated_answer": "No citation here.",
         "sources": [source(1)], "citation_valid": True},
        {"question": "Q3?", "status": "error"},
    ]
    scored = cmp.rescore(qs, [[3]] * 3, results)
    assert set(scored) == {1, 2}
    assert scored[1]["citation_valid"] and not scored[2]["citation_valid"]


def test_summary_counts_grades_only_for_answered_questions(tmp_path):
    scored = {1: ev.score(ITEM, "Low-rank [S1].", [source(1)], [3]),
              2: ev.score(ITEM, "Frozen [S1].", [source(1, start=8)], [3])}
    grades_file = tmp_path / "g.json"
    grades_file.write_text(json.dumps({"scores": {"1": "correct", "2": "partial", "3": "incorrect"}}))
    summary = cmp.summarize(scored, cmp.load_grades(grades_file), total=3)
    assert summary["answered"] == 2 and summary["page_hit_at_5"] == 1
    assert summary["graded"] == 2 and summary["incorrect"] == 0
    assert summary["weighted_quality"] == 0.75

    grades_file.write_text(json.dumps({"scores": {"1": "maybe"}}))
    with pytest.raises(ValueError, match="unknown grades"):
        cmp.load_grades(grades_file)
