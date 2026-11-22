"""Offline tests for scripts/evaluate_memos.py (fake retriever and LLM)."""

import importlib.util
import json
from pathlib import Path

import pytest
from langchain_core.language_models import GenericFakeChatModel

from test_memo import DOCS, GOOD  # same fixtures as the memo tests

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "evaluate_memos.py"
spec = importlib.util.spec_from_file_location("evaluate_memos", SCRIPT)
em = importlib.util.module_from_spec(spec)
spec.loader.exec_module(em)

TOPICS = [{"paper_id": "2205.14135", "topic": "FlashAttention"},
          {"paper_id": "2106.09685", "topic": "LoRA"}]


class QuotaError(Exception):
    code = 429


class FakeRetriever:
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.asked = []

    def invoke(self, topic):
        self.asked.append(topic)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
        return DOCS


def llm(*replies):
    return GenericFakeChatModel(messages=iter(replies))


def test_memo_files_and_metrics(tmp_path):
    bad = GOOD.replace("[S2]", "[S7]", 1)  # one citation outside the 2 retrieved excerpts
    out = tmp_path / "r.json"
    results, stopped = em.run(TOPICS, FakeRetriever(), llm(GOOD, bad), out, {"model": "m"},
                              pause=0, memo_dir=tmp_path / "memos")
    assert not stopped and [r["status"] for r in results] == ["success", "success"]
    good, worse = results
    assert good["passed_checks"] and good["citation_precision"] == 1.0
    assert good["sections_not_covered"] == ["Limitations and open questions"]
    assert good["sources_from_topic_paper"] == 2 and worse["sources_from_topic_paper"] == 0
    assert not worse["passed_checks"] and worse["invalid_citations"] == [7]
    assert (tmp_path / "memos" / "01-2205.14135.md").read_text().startswith("# Triage memo: FlashAttention")
    assert json.loads(out.read_text()) == results


def test_quota_stop_and_resume(tmp_path):
    out = tmp_path / "r.json"
    quota = QuotaError("429 GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    first = FakeRetriever([None, quota])
    results, stopped = em.run(TOPICS, first, llm(GOOD), out, {}, pause=0, memo_dir=tmp_path)
    assert stopped and [r["topic"] for r in results] == ["FlashAttention"]

    previous = {r["topic"]: r for r in results}
    second = FakeRetriever()
    results, stopped = em.run(TOPICS, second, llm(GOOD), out, {}, previous, pause=0, memo_dir=tmp_path)
    assert not stopped and second.asked == ["LoRA"] and len(results) == 2


def test_report_counts_minor_corrections_as_first_pass_approved():
    results = [
        {"index": 1, "status": "success", "passed_checks": True, "citations": 10,
         "invalid_citations": [], "sections_not_covered": []},
        {"index": 2, "status": "success", "passed_checks": False, "citations": 10,
         "invalid_citations": [9], "sections_not_covered": ["Bottom line"]},
        {"index": 3, "status": "error"},
    ]
    summary = em.report(results, {"1": {"verdict": "approve"}, "2": {"verdict": "minor"}})
    assert summary["memos"] == 2 and summary["passed_all_checks"] == 1
    assert summary["citation_precision"] == 0.95 and summary["sections_not_covered"] == 1
    assert summary["first_pass_approval_rate"] == 1.0 and summary["major"] == 0
    with pytest.raises(ValueError, match="unknown review verdicts"):
        em.report(results, {"1": {"verdict": "fine"}})
