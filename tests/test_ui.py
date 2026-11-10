from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel

from src import ui
from src.rag import MissingConfigError
from src.ui import (
    Backend,
    View,
    describe_error,
    link_citations,
    memo_markdown,
    render_sources,
    run_query,
)


def _doc(n, paper_id="2106.09685", title="LoRA", page=4, text="LoRA freezes the pretrained weights."):
    return Document(
        page_content=text,
        id=f"c{n}",
        metadata={
            "chunk_id": f"c{n}",
            "paper_id": paper_id,
            "paper_title": title,
            "page_start": page,
            "page_end": page,
        },
    )


DOCS = [_doc(1), _doc(2, "2005.11401", "RAG", 2, "RAG pairs a retriever with a generator.")]


class FakeRetriever:
    def __init__(self, docs, error=None):
        self.docs, self.error, self.queries = docs, error, []

    def invoke(self, question):
        self.queries.append(question)
        if self.error:
            raise self.error
        return list(self.docs)


def _backend(docs=DOCS, reply="LoRA freezes weights [S1].", search_error=None, llm_error=None):
    retriever = FakeRetriever(docs, search_error)
    calls = {"retriever": [], "llm": 0}

    def make_retriever(k, rerank):
        calls["retriever"].append((k, rerank))
        return retriever

    def make_llm():
        calls["llm"] += 1
        if llm_error:
            raise llm_error
        return GenericFakeChatModel(messages=iter([reply]))

    return Backend(make_retriever=make_retriever, make_llm=make_llm), calls


# ------------------------------------------------------------ link_citations


@pytest.mark.parametrize(
    "answer, expected",
    [
        ("Claim [S2].", "Claim [S2](#src-2)."),
        ("A [S1][S2]", "A [S1](#src-1)[S2](#src-2)"),
        ("A [S1, S2]", "A [S1](#src-1)[S2](#src-2)"),
        ("A [S1; 2]", "A [S1](#src-1)[S2](#src-2)"),
        ("A [S1-S3]", "A [S1](#src-1)–[S3](#src-3)"),
        ("Out of range [S7].", "Out of range \\[S7\\]."),
        ("Mixed [S1, S9]", "Mixed [S1](#src-1)\\[S9\\]"),
        ("The paper's own ref [12] and BART [32]", "The paper's own ref [12] and BART [32]"),
    ],
)
def test_link_citations(answer, expected):
    assert link_citations(answer, n_sources=3) == expected


# --------------------------------------------------------------- inline_math


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"update $\Delta W = BA$ and $r \ll d$.", r"update \(\Delta W = BA\) and \(r \ll d\)."),
        ("costs $5 and $10", "costs $5 and $10"),  # prices are not math
        ("$5-$10 range", "$5-$10 range"),
        ("price $5, rank $r$", r"price $5, rank \(r\)"),
        ("display $$x^2$$ stays", "display $$x^2$$ stays"),
        ("a $ b $ c", "a $ b $ c"),  # space inside the dollars: not math
    ],
)
def test_inline_math(text, expected):
    assert ui.inline_math(text) == expected


def test_answer_math_is_rewritten_for_rendering():
    backend, _ = _backend(reply=r"Update $\Delta W = BA$ [S1].")
    assert _final(run_query(backend, "Q?")).answer == r"Update \(\Delta W = BA\) [S1](#src-1)."


# ------------------------------------------------------------ render_sources


def test_source_cards_are_numbered_linked_and_escaped():
    docs = [_doc(1, title="<script>alert(1)</script>", page=7, text="x < y & <b>bold</b>")]
    out = render_sources(docs)
    assert 'id="src-1"' in out and ">S1<" in out
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<b>bold</b>" not in out and "x &lt; y &amp;" in out
    assert 'href="https://arxiv.org/pdf/2106.09685#page=7"' in out
    assert "p. 7" in out


def test_source_cards_mark_uncited_only_after_an_answer():
    assert "Not cited" not in render_sources(DOCS)  # search only
    out = render_sources(DOCS, cited={1})
    assert 'class="src-card cited" id="src-1"' in out
    assert 'class="src-card uncited" id="src-2"' in out
    assert out.count("Not cited") == 1


def test_long_excerpt_gets_preview_and_full_text():
    text = "word " * 200
    out = render_sources([_doc(1, text=text)])
    assert "Show full excerpt" in out and "…" in out
    assert "Show full excerpt" not in render_sources([_doc(1, text="short")])


def test_no_docs_message():
    assert "No matching excerpts" in render_sources([])


# ------------------------------------------------------------- memo export


def test_memo_markdown_lists_sources_and_flags_bad_citations():
    result = {
        "question": "What does\nLoRA freeze?",
        "answer": "It freezes weights [S1] and [S5].  ",
        "sources": [
            {"n": 1, "paper_title": "LoRA", "paper_id": "2106.09685", "page_start": 4, "page_end": 5, "chunk_id": "c1"}
        ],
        "citations": {"valid": False, "cited_sources": [1, 5], "invalid_sources": [5]},
    }
    memo = memo_markdown(result)
    assert memo.startswith("# What does LoRA freeze?\n")
    assert "from 1 paper excerpt." in memo
    assert "It freezes weights [S1] and [S5].\n" in memo
    assert "cites [S5], which is not one of the sources" in memo
    assert "- **[S1]** LoRA (arXiv:2106.09685), pp. 4–5 — https://arxiv.org/pdf/2106.09685#page=4" in memo


# ----------------------------------------------------------- describe_error


class _ApiError(Exception):
    def __init__(self, code):
        super().__init__(f"HTTP {code}")
        self.code = code


def _wrapped(code):
    try:
        try:
            raise _ApiError(code)
        except _ApiError as inner:
            raise RuntimeError("wrapper without status") from inner
    except RuntimeError as outer:
        return outer


def test_describe_error_messages():
    assert "Search only still works" in describe_error(MissingConfigError("No key."), "answer")
    assert "rate-limiting" in describe_error(_wrapped(429), "answer")  # status found through the wrapper
    daily = _ApiError(429)
    daily.args = ("429 RESOURCE_EXHAUSTED quotaId GenerateRequestsPerDayPerProjectPerModel-FreeTier",)
    assert "daily API quota is used up" in describe_error(daily, "answer")
    msg = describe_error(_ApiError(403), "answer")
    assert "API key was rejected" in msg and "OPENROUTER_API_KEY" in msg  # provider-neutral
    msg = describe_error(ValueError("boom " * 100), "search")
    assert msg.startswith("Search failed (ValueError): boom") and len(msg) < 260


# ---------------------------------------------------------------- run_query


def _final(views):
    """Merge View snapshots the way the UI applies them (None = unchanged)."""
    state = View()
    for v in views:
        for field in ("status", "answer", "sources", "memo_path"):
            value = getattr(v, field)
            if value is not None or field == "memo_path" and v.answer is not None:
                setattr(state, field, value)
    return state


def test_empty_question_does_nothing():
    backend, calls = _backend()
    views = list(run_query(backend, "   "))
    assert len(views) == 1 and "Type a question" in views[0].status
    assert views[0].answer is None and views[0].sources is None  # previous results kept
    assert calls == {"retriever": [], "llm": 0}


def test_search_only_never_builds_an_llm():
    backend, calls = _backend()
    final = _final(run_query(backend, "What is RAG?", k=4, rerank=False, generate=False))
    assert calls == {"retriever": [(4, False)], "llm": 0}
    assert "no API call" in final.status and "2 excerpts" in final.status
    assert final.answer == ui.SEARCH_ONLY_ANSWER
    assert 'id="src-2"' in final.sources and final.memo_path is None


def test_full_answer_flow(tmp_path):
    backend, calls = _backend(reply="LoRA freezes weights [S1].")
    views = list(run_query(backend, "  What does LoRA freeze?  "))
    assert [v.status.split('"')[1] for v in views] == ["status status-busy"] * 2 + ["status status-ok"]
    assert views[1].sources is not None and views[1].answer is None  # sources shown before the answer
    final = _final(views)
    assert final.answer == "LoRA freezes weights [S1](#src-1)."
    assert 'class="src-card uncited" id="src-2"' in final.sources
    memo = Path(final.memo_path)
    assert memo.suffix == ".md" and "# What does LoRA freeze?" in memo.read_text(encoding="utf-8")
    assert calls["llm"] == 1 and backend._make_retriever(5, True).queries == ["What does LoRA freeze?"]


def test_invalid_citation_warns_but_keeps_answer():
    backend, _ = _backend(reply="Claim [S1]. Made-up claim [S9].")
    final = _final(run_query(backend, "Q?"))
    assert "status-warn" in final.status and "[S9]" in final.status
    assert final.answer == "Claim [S1](#src-1). Made-up claim \\[S9\\]."
    assert final.memo_path


def test_search_failure_is_reported():
    backend, calls = _backend(search_error=RuntimeError("index missing"))
    final = _final(run_query(backend, "Q?"))
    assert "status-error" in final.status and "Search failed (RuntimeError): index missing" in final.status
    assert final.sources == ui.EMPTY_SOURCES and calls["llm"] == 0


def test_answer_failure_keeps_sources_visible():
    backend, _ = _backend(llm_error=MissingConfigError("No Gemini API key found."))
    views = list(run_query(backend, "Q?"))
    final = _final(views)
    assert "status-error" in final.status and "No Gemini API key" in final.status
    assert views[-1].sources is None and 'id="src-1"' in final.sources  # excerpts stay up
    assert final.answer == "" and final.memo_path is None


def test_no_docs_answers_without_llm_call():
    backend, calls = _backend(docs=[])
    final = _final(run_query(backend, "Q?"))
    assert "No relevant excerpts" in final.answer and calls["llm"] == 0  # no key needed
    assert "No matching excerpts" in final.sources
    assert "status-warn" in final.status and "nothing to answer from" in final.status
    assert " s · answer" not in final.status  # no fake "answer 0.0 s" timing
    assert final.memo_path is None  # nothing to download


def test_invalid_citation_wording_is_plural_aware():
    backend, _ = _backend(reply="A [S7]. B [S9].")
    status = _final(run_query(backend, "Q?")).status
    assert "The answer cites [S7], [S9], which are not among the retrieved sources" in status
    backend, _ = _backend(reply="A [S7].")
    assert "cites [S7], which is not one of the retrieved sources" in _final(run_query(backend, "Q?")).status


def test_memos_share_one_folder_and_never_overwrite():
    result = {"question": "Q?", "answer": "A", "sources": [],
              "citations": {"valid": True, "cited_sources": [], "invalid_sources": []}}
    a, b = Path(ui.write_memo(result)), Path(ui.write_memo(result))
    assert a != b and a.parent == b.parent and a.exists() and b.exists()


def test_range_citation_marks_every_source_cited():
    docs = [_doc(1), _doc(2), _doc(3)]
    backend, _ = _backend(docs=docs, reply="Claim [S1-S3].")
    sources = _final(run_query(backend, "Q?")).sources
    assert "uncited" not in sources and "Not cited" not in sources


# ------------------------------------------------------------------- layout


def test_app_builds_without_loading_models():
    backend, calls = _backend()
    papers = [{"id": "2106.09685", "title": "LoRA", "category": "PEFT"}]
    app = ui.build_app(backend, papers=papers)
    assert app.title == ui.APP_TITLE
    assert calls == {"retriever": [], "llm": 0}  # nothing runs until a question is asked
