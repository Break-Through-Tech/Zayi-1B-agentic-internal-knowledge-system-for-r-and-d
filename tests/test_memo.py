"""Triage memo drafting and automatic checks (src/memo.py). No API calls."""

from datetime import datetime

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel

from src.memo import (
    MEMO_SECTIONS,
    NO_CONTEXT_MEMO,
    NOT_COVERED,
    build_memo,
    check_memo,
    draft_memo,
    memo_body,
    memo_document,
    revise_memo,
    split_sections,
)


def _doc(n, paper_id="2205.14135", title="FlashAttention", page=3, text="Tiling avoids HBM reads."):
    return Document(
        page_content=text,
        id=f"c{n}",
        metadata={"chunk_id": f"c{n}", "paper_id": paper_id, "paper_title": title, "page_start": page, "page_end": page},
    )


DOCS = [_doc(1), _doc(2, page=7, text="FlashAttention is 3x faster on GPT-2.")]

GOOD = f"""## Bottom line
FlashAttention computes exact attention with fewer memory reads [S1].

## Key findings
- 3x faster training on GPT-2 [S2].
- Exact, not approximate [S1].

## How it works
Tiling keeps blocks in on-chip SRAM [S1].

## Evidence and results
GPT-2 trains 3x faster [S2].

## Limitations and open questions
{NOT_COVERED}
"""


class RecordingModel(GenericFakeChatModel):
    prompts: list = []

    def _generate(self, messages, *args, **kwargs):
        self.prompts.append(messages)
        return super()._generate(messages, *args, **kwargs)


def _llm(reply=GOOD):
    return RecordingModel(messages=iter([reply]), prompts=[])


def _check(memo, n=2):
    return {c["name"]: c for c in check_memo(memo, n)}


def test_split_sections_matches_loose_headings():
    memo = "Intro line\n# **Bottom Line:**\nA [S1].\n### key findings\n- B [S2]\n## Unknown heading\nC"
    sections = split_sections(memo)
    assert sections[None] == "Intro line"
    assert sections["Bottom line"] == "A [S1]."
    # text under an unknown heading stays with the section before it
    assert sections["Key findings"] == "- B [S2]\n## Unknown heading\nC"


def test_good_memo_passes_all_checks():
    checks = check_memo(GOOD, 2)
    assert [c["ok"] for c in checks] == [True, True, True]
    assert "not covered: Limitations and open questions" in checks[2]["detail"]


def test_missing_section_fails():
    memo = GOOD.replace("## How it works\nTiling keeps blocks in on-chip SRAM [S1].\n", "")
    check = _check(memo)["All sections present"]
    assert not check["ok"] and "How it works" in check["detail"]


def test_empty_section_counts_as_missing():
    memo = GOOD.replace("Tiling keeps blocks in on-chip SRAM [S1].", "")
    assert "How it works" in _check(memo)["All sections present"]["detail"]


def test_citation_outside_retrieved_excerpts_fails():
    check = _check(GOOD.replace("[S2]", "[S9]"))["Citations point at retrieved excerpts"]
    assert not check["ok"] and "S9" in check["detail"]


def test_range_citation_past_the_excerpts_fails():
    assert not _check(GOOD.replace("[S2]", "[S1-S3]"))["Citations point at retrieved excerpts"]["ok"]


def test_papers_own_reference_numbers_are_not_citations():
    memo = GOOD.replace("[S1]", "[12]").replace("[S2]", "[3]")
    check = _check(memo)["Citations point at retrieved excerpts"]
    assert not check["ok"] and check["detail"] == "no citations"


def test_uncited_section_fails():
    memo = GOOD.replace("GPT-2 trains 3x faster [S2].", "GPT-2 trains 3x faster.")
    check = _check(memo)["Every section cited or marked not covered"]
    assert not check["ok"] and check["detail"] == "uncited: Evidence and results"


def test_memo_without_sections_fails_every_check():
    checks = check_memo("A free-form paragraph [S1].", 2)
    assert [c["ok"] for c in checks] == [False, True, False]
    assert checks[2]["detail"] == "no sections found"


def test_draft_memo_sends_excerpts_and_section_list():
    llm = _llm()
    result = draft_memo("  FlashAttention  ", DOCS, llm)
    assert result["topic"] == "FlashAttention"
    assert result["passed"] and result["memo"] == GOOD.strip()
    assert [s["n"] for s in result["sources"]] == [1, 2]
    assert result["citations"]["invalid_sources"] == []
    system, human = llm.prompts[0]
    for section in MEMO_SECTIONS:
        assert section in system.content
    assert "[S2] FlashAttention (arXiv:2205.14135), p. 7" in human.content
    assert "Memo topic: FlashAttention" in human.content


def test_draft_memo_without_excerpts_skips_the_llm():
    llm = _llm()
    result = draft_memo("anything", [], llm)
    assert result["memo"] == NO_CONTEXT_MEMO and not result["passed"]
    assert llm.prompts == []


@pytest.mark.parametrize("topic", ["", "   ", None])
def test_draft_memo_rejects_empty_topic(topic):
    with pytest.raises(ValueError):
        draft_memo(topic, DOCS, _llm())


def test_build_memo_retrieves_with_the_topic():
    class Retriever:
        def invoke(self, q):
            self.q = q
            return DOCS

    retriever = Retriever()
    result = build_memo(" tiling ", retriever=retriever, llm=_llm())
    assert retriever.q == "tiling" and result["passed"]


def test_memo_document_has_title_status_body_and_linked_sources():
    result = draft_memo("FlashAttention", DOCS, _llm())
    doc = memo_document(result, when=datetime(2026, 11, 21, 9, 5))
    assert doc.startswith("# Triage memo: FlashAttention\n")
    assert "2026-11-21 09:05 from 2 paper excerpts. Not yet reviewed. Automatic checks: all passed.*" in doc
    assert "## Bottom line" in doc
    assert "- **[S2]** FlashAttention (arXiv:2205.14135), p. 7: https://arxiv.org/pdf/2205.14135#page=7" in doc


def test_memo_document_lists_failed_checks():
    result = draft_memo("FlashAttention", DOCS, _llm(GOOD.replace("[S2]", "[S9]")))
    doc = memo_document(result)
    assert not result["passed"]
    assert "Automatic checks failed: citations point at retrieved excerpts." in doc


def test_memo_body_strips_title_notices_and_sources():
    doc = memo_document(draft_memo("FlashAttention", DOCS, _llm()))
    doc = doc.replace("# Triage memo: FlashAttention\n", "# Triage memo: FlashAttention\n\n> **Review:** Approved\n")
    body = memo_body(doc)
    assert body.startswith("## Bottom line") and body.endswith(NOT_COVERED)
    assert "Sources" not in body and "Triage memo" not in body and "Review:" not in body
    assert memo_body(GOOD) == GOOD.strip()  # sections alone pass through


def test_revise_memo_sends_draft_notes_and_excerpts():
    llm = _llm(GOOD.replace("3x faster [S2]", "3x faster on GPT-2 at 1K context [S2]"))
    draft = memo_document(draft_memo("FlashAttention", DOCS, _llm()))
    result = revise_memo("FlashAttention", DOCS, draft, "  Add the context length.  ", llm)
    assert result["passed"] and "1K context" in result["memo"]
    human = llm.prompts[0][1].content
    assert "Reviewer's notes:\nAdd the context length." in human
    assert "Current draft (may include the reviewer's own edits):\n\n## Bottom line" in human
    assert "## Sources" not in human and "[S2] FlashAttention (arXiv:2205.14135), p. 7" in human


@pytest.mark.parametrize("notes, docs", [("", DOCS), ("   ", DOCS), (None, DOCS), ("fix it", [])])
def test_revise_memo_needs_notes_and_excerpts(notes, docs):
    with pytest.raises(ValueError):
        revise_memo("FlashAttention", docs, GOOD, notes, _llm())


def test_memo_document_labels_revisions():
    doc = memo_document(draft_memo("FlashAttention", DOCS, _llm()), revision=2)
    assert "*Revision 2 by the Zayi 1B Research Assistant" in doc
