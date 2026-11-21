from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel

from src import ui
from src.rag import NO_CONTEXT_ANSWER, MissingConfigError
from src.ui import (
    Backend,
    Turn,
    answer_html,
    describe_error,
    link_citations,
    memo_markdown,
    model_name,
    protect_math,
    render_sources,
    render_thread,
    render_turn,
    run_turn,
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
PAPERS = [{"id": "2106.09685", "title": "LoRA <v2>", "category": "PEFT"}]


class FakeRetriever:
    def __init__(self, docs, error=None):
        self.docs, self.error, self.queries = docs, error, []

    def invoke(self, question):
        self.queries.append(question)
        if self.error:
            raise self.error
        return list(self.docs)


class NamedFakeModel(GenericFakeChatModel):
    model: str = "models/gemini-test"


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
        return NamedFakeModel(messages=iter([reply]))

    return Backend(make_retriever=make_retriever, make_llm=make_llm), calls, retriever


def _run(backend, question="What does LoRA freeze?", **kw):
    """Run a turn; return it plus the phase and rendered HTML after each step."""
    turn = Turn(n=1, question=question, **kw)
    snapshots = [(t.phase, render_turn(t)) for t in run_turn(backend, turn)]
    return turn, snapshots


# ------------------------------------------------------------ answer text


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


def test_link_citations_use_the_turn_anchor():
    assert link_citations("A [S2]", 3, anchor="t4-s") == "A [S2](#t4-s2)"


@pytest.mark.parametrize(
    "text, tex",
    [
        (r"update $\Delta W = BA$ and $r \ll d$.", [(r"\Delta W = BA", False), (r"r \ll d", False)]),
        ("costs $5 and $10", []),  # prices are not math
        ("$5-$10 range", []),
        ("price $5, rank $r$", [("r", False)]),
        ("a $ b $ c", []),  # space inside the dollars: not math
        ("display $$x^2$$ here", [("x^2", True)]),
        (r"also \(a_1\) and \[b_2\]", [("a_1", False), ("b_2", True)]),
    ],
)
def test_protect_math(text, tex):
    out, found = protect_math(text)
    assert found == tex
    assert out.count("@@MATH") == len(tex)


def test_answer_html_renders_markdown_citations_and_math():
    out = answer_html(r"**LoRA** trains $A_{i}$ and $B$ [S1, S2]; see [S9].", 2, anchor="t1-s")
    assert "<strong>LoRA</strong>" in out
    assert '<span class="math">A_{i}</span>' in out  # underscore survives Markdown
    assert '<a href="#t1-s1">S1</a><a href="#t1-s2">S2</a>' in out
    assert "[S9]" in out and 'href="#t1-s9"' not in out  # out-of-range label is plain text


def test_answer_html_escapes_model_output():
    out = answer_html('<img src=x onerror=alert(1)> [x](javascript:alert(1)) $<b>$', 0)
    assert "<img" not in out and "&lt;img" in out
    assert 'href="javascript' not in out
    assert '<span class="math">&lt;b&gt;</span>' in out


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
    assert memo.startswith("# What does LoRA freeze?\n") and "Drafted by the Zayi 1B Research Assistant" in memo
    assert "from 1 paper excerpt." in memo
    assert "It freezes weights [S1] and [S5].\n" in memo
    assert "cites [S5], which is not one of the sources" in memo
    assert "- **[S1]** LoRA (arXiv:2106.09685), pp. 4–5 — https://arxiv.org/pdf/2106.09685#page=4" in memo


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


def test_model_name():
    assert model_name(SimpleNamespace(model="models/gemini-3.5-flash")) == "gemini-3.5-flash"
    assert model_name(SimpleNamespace(model_name="openai/gpt-x")) == "openai/gpt-x"
    assert model_name(None) == ""


# ------------------------------------------------------------ source cards


def test_source_cards_are_numbered_linked_and_escaped():
    docs = [_doc(1, title="<script>alert(1)</script>", page=7, text="x < y & <b>bold</b>")]
    out = render_sources(docs, anchor="t3-s")
    assert 'id="t3-s1"' in out and ">S1<" in out
    assert "<script>" not in out and "&lt;script&gt;" in out
    assert "<b>bold</b>" not in out and "x &lt; y &amp;" in out
    assert 'href="https://arxiv.org/pdf/2106.09685#page=7"' in out and "p. 7" in out


def test_source_cards_mark_uncited_only_after_an_answer():
    assert "Not cited" not in render_sources(DOCS)  # search only
    out = render_sources(DOCS, cited={1})
    assert 'class="src-card cited" id="src-1"' in out
    assert 'class="src-card uncited" id="src-2"' in out
    assert out.count("Not cited") == 1


def test_long_excerpt_gets_preview_and_full_text():
    out = render_sources([_doc(1, text="word " * 200)], anchor="t1-s")
    # the whole excerpt is in one paragraph (CSS clamps it); the toggle comes after it
    assert "Show full excerpt" in out and 'id="t1-s1-full"' in out
    assert out.index('class="src-text"') < out.index('class="src-full"')
    assert "…" not in out
    assert "Show full excerpt" not in render_sources([_doc(1, text="short")])


# --------------------------------------------------------------- query flow


def test_full_answer_flow():
    backend, calls, retriever = _backend(reply=r"LoRA freezes weights [S1], so $\Delta W = BA$.")
    turn, snaps = _run(backend, "  What does LoRA freeze?  ".strip())
    assert [phase for phase, _ in snaps] == ["search", "write", "done"]
    searching, writing, done = (html for _, html in snaps)
    assert "Searching the papers…" in searching and 'class="step pending"' in searching
    assert 'id="t1-s2"' in writing and 'class="dots"' in writing  # sources shown before the answer
    assert "Writing the answer…" in writing
    assert '<a href="#t1-s1">S1</a>' in done and r'<span class="math">\Delta W = BA</span>' in done
    assert 'class="src-card uncited" id="t1-s2"' in done
    assert "Used 2 excerpts from 2 papers" in done and "gemini-test" in done
    assert "think done" in done and " open" not in done.split("</summary>")[0]  # timeline collapses
    assert calls["llm"] == 1 and retriever.queries == ["What does LoRA freeze?"]
    assert turn.model == "gemini-test" and turn.memo.startswith("# What does LoRA freeze?")


def test_download_link_carries_the_memo():
    backend, _, _ = _backend()
    turn, snaps = _run(backend)
    done = snaps[-1][1]
    href = done.split('href="data:text/markdown;charset=utf-8,')[1].split('"')[0]
    assert unquote(href) == turn.memo
    assert "\n" not in href and "#" not in href and " " not in href  # percent-encoded, so "#" can't cut it short
    assert 'download="zayi-1b-what-does-lora-freeze.md"' in done


def test_search_only_never_builds_an_llm():
    backend, calls, _ = _backend()
    turn, snaps = _run(backend, k=4, rerank=False, generate=False)
    assert calls == {"retriever": [(4, False)], "llm": 0}
    done = snaps[-1][1]
    assert "Closest excerpts" in done and "Search only · no API call" in done
    assert "Not cited" not in done and "Download" not in done and 'class="prose"' not in done
    assert "embedding search, top 4" in done and "reranked" not in done


def test_invalid_citation_warns_but_keeps_answer():
    backend, _, _ = _backend(reply="Claim [S1]. Made-up claim [S9].")
    turn, snaps = _run(backend)
    done = snaps[-1][1]
    assert 'class="note warn"' in done and "[S9], which is not one of the retrieved excerpts" in done
    assert 'class="step failed">Checked the citations' in done
    assert '<a href="#t1-s1">S1</a>' in done and turn.memo


def test_search_failure_is_reported():
    backend, calls, _ = _backend(search_error=RuntimeError("index <missing>"))
    turn, snaps = _run(backend)
    done = snaps[-1][1]
    assert turn.phase == "error" and calls["llm"] == 0
    assert "Search failed (RuntimeError): index &lt;missing&gt;" in done and "Sources" not in done


def test_answer_failure_keeps_sources_visible():
    backend, _, _ = _backend(llm_error=MissingConfigError("No Gemini API key found."))
    turn, snaps = _run(backend)
    done = snaps[-1][1]
    assert turn.phase == "error" and 'class="note error"' in done and "No Gemini API key" in done
    assert 'id="t1-s1"' in done  # the excerpts stay up
    assert not turn.memo and "Download" not in done and "Answering failed" in done


def test_no_docs_answers_without_llm_call():
    backend, calls, _ = _backend(docs=[])
    turn, snaps = _run(backend)
    done = snaps[-1][1]
    assert calls["llm"] == 0  # no key needed
    assert NO_CONTEXT_ANSWER.split(".")[0] in done and "Found no matching excerpts" in done
    assert not turn.memo and "Download" not in done and "Sources" not in done


def test_backend_answers_empty_docs_without_an_llm():
    backend, calls, _ = _backend()
    result = backend.answer("Q?", [])
    assert calls["llm"] == 0 and result["answer"] == NO_CONTEXT_ANSWER and result["model"] == ""


def test_range_citation_marks_every_source_cited():
    backend, _, _ = _backend(docs=[_doc(1), _doc(2), _doc(3)], reply="Claim [S1-S3].")
    _, snaps = _run(backend)
    assert "uncited" not in snaps[-1][1]


def test_answer_without_citations_is_flagged_in_the_steps():
    backend, _, _ = _backend(reply="LoRA freezes weights.")
    _, snaps = _run(backend)
    assert 'class="step failed">Checked the citations <span class=io>the answer cites no excerpts' in snaps[-1][1]


def test_question_is_escaped():
    backend, _, _ = _backend(docs=[])
    _, snaps = _run(backend, question="<img src=x onerror=alert(1)>")
    assert "<img" not in snaps[-1][1] and "&lt;img" in snaps[-1][1]


# ------------------------------------------------------------------- thread


def test_empty_thread_shows_the_hero_with_papers():
    out = render_thread([], PAPERS)
    assert 'class="hero"' in out and "<b>1 research paper</b>" in out and "<h1>Zayi 1B</h1>" in out
    assert render_thread([], PAPERS * 10).count("<li>") == 10 and "<b>10 research papers</b>" in render_thread([], PAPERS * 10)
    assert "LoRA &lt;v2&gt;" in out and 'href="https://arxiv.org/abs/2106.09685"' in out


def test_thread_keeps_turn_ids_unique():
    backend, _, _ = _backend()
    turns = []
    for n in (1, 2):
        turn = Turn(n=n, question=f"Q{n}?")
        list(run_turn(backend, turn))
        turns.append(turn)
    out = render_thread(turns, PAPERS)
    assert 'id="t1-s1"' in out and 'id="t2-s1"' in out and 'href="#t2-s1"' in out
    assert 'class="hero"' not in out


def test_styles_are_complete_and_assets_exist():
    for placeholder in ("FONT", "LOGO_URL", "BG_URL", "ICON_", "CHIP_ICONS"):
        assert placeholder not in ui.CSS
    assert ui.LOGO.read_text(encoding="utf-8").startswith("<svg") and ui.BACKGROUND.stat().st_size > 50_000
    assert "@@" not in ui.CSS


def test_app_builds_without_loading_models():
    backend, calls, _ = _backend()
    app = ui.build_app(backend, papers=PAPERS)
    assert app.title == ui.APP_TITLE
    assert calls == {"retriever": [], "llm": 0}  # nothing runs until a question is asked


# ------------------------------------------------------------- memo review

MEMO_REPLY = """## Bottom line
LoRA freezes the base weights [S1].

## Key findings
- Freezes weights [S1].

## How it works
Low-rank updates [S1].

## Evidence and results
Not covered in the retrieved excerpts.

## Limitations and open questions
RAG is a separate method [S2]."""


def test_memo_turn_drafts_memo_and_shows_review_panel():
    backend, calls, _ = _backend(reply=MEMO_REPLY)
    turn, snapshots = _run(backend, "LoRA <for us>", as_memo=True)
    assert [p for p, _ in snapshots] == ["search", "write", "done"]
    assert "Writing the memo" in snapshots[1][1]
    assert calls["llm"] == 1 and turn.cited == {1, 2} and not turn.invalid
    assert [c["ok"] for c in turn.checks] == [True, True, True]
    out = snapshots[-1][1]
    assert 'class="prose memo"' in out and "<h2>Bottom line</h2>" in out
    assert "Wrote the memo" in out and "All sections present" in out
    assert 'id="t1-review"' in out and 'id="t1-edit"' in out and 'data-review="approve"' in out
    # the editable draft is the downloadable memo, escaped
    assert "# Triage memo: LoRA &lt;for us&gt;" in out and "<for us>" not in out
    assert 'download="zayi-1b-memo-lora-for-us.md"' in out and "Download draft" in out
    assert 'data-file="zayi-1b-memo-lora-for-us"' in out


def test_memo_turn_shows_failed_checks():
    backend, _, _ = _backend(reply="Just a paragraph citing [S7].")
    turn, snapshots = _run(backend, as_memo=True)
    out = snapshots[-1][1]
    assert [c["ok"] for c in turn.checks] == [False, False, False]
    assert 'class="step failed"' in out and "S7 not among the 2 excerpts" in out
    assert "Some automatic checks failed" in out
    assert "Automatic checks failed" in turn.memo


def test_memo_turn_without_excerpts_skips_llm_and_review():
    backend, calls, _ = _backend(docs=[])
    turn, snapshots = _run(backend, as_memo=True)
    assert calls["llm"] == 0 and "no memo was drafted" in turn.answer
    assert 'class="review"' not in snapshots[-1][1]


def test_answer_turns_have_no_review_panel():
    backend, _, _ = _backend()
    _, snapshots = _run(backend)
    assert 'class="review"' not in snapshots[-1][1] and "Download .md" in snapshots[-1][1]


def test_thread_script_is_valid_javascript(tmp_path):
    # THREAD_JS is a normal Python string: a "\\n" written as "\n" becomes a
    # raw newline inside a JS string literal and breaks the whole script.
    assert "split('\\n')" in ui.THREAD_JS and "join('\\n')" in ui.THREAD_JS
    node = __import__("shutil").which("node")
    if node is None:
        pytest.skip("node not installed")
    script = tmp_path / "thread.js"
    script.write_text("(function (element, watch) {" + ui.THREAD_JS + "})", encoding="utf-8")
    result = __import__("subprocess").run([node, "--check", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_revision_turn_reuses_excerpts_and_sends_notes():
    revised = MEMO_REPLY.replace("Not covered in the retrieved excerpts.", "Freezing saves memory [S1].")
    backend, calls, retriever = _backend(reply=revised)
    first, _ = _run(backend, "LoRA", as_memo=True)
    turn = Turn(n=2, question="LoRA", as_memo=True, docs=list(first.docs), revision=2, revises=1,
                notes="Fill in <evidence>.", draft_in=first.memo)
    snapshots = [(t.phase, render_turn(t)) for t in run_turn(backend, turn)]
    assert [p for p, _ in snapshots] == ["search", "write", "done"]
    assert len(retriever.queries) == 1  # only the first memo searched
    assert "Revising the memo" in snapshots[1][1]
    out = snapshots[-1][1]
    assert "Reused the 2 excerpts from memo 1" in out and "Wrote revision 2" in out
    assert '<span class="bubble-tag">Revise memo 1</span>Fill in &lt;evidence&gt;.' in out
    assert "Review revision 2" in out and "Freezing saves memory" in out
    assert "*Revision 2 by the" in turn.memo and 'download="zayi-1b-memo-lora-r2.md"' in out


def test_review_panel_has_approve_and_revise():
    backend, _, _ = _backend(reply=MEMO_REPLY)
    _, snapshots = _run(backend, "LoRA", as_memo=True)
    out = snapshots[-1][1]
    assert 'data-review="approve"' in out and 'data-review="revise"' in out and 'id="t1-tip"' in out
    assert "Request changes" not in out


def test_app_wires_the_revise_bridge():
    backend, _, _ = _backend()
    app = ui.build_app(backend, papers=PAPERS)
    ids = {getattr(b, "elem_id", None) for b in app.blocks.values()}
    assert {"revise-payload", "revise-go"} <= ids
    assert "#revise-payload textarea" in ui.THREAD_JS and "#revise-go" in ui.THREAD_JS


def test_revision_turn_validates_the_payload():
    backend, _, _ = _backend(reply=MEMO_REPLY)
    memo, _ = _run(backend, "LoRA", as_memo=True, k=7)
    answer = Turn(n=2, question="Q?")
    list(run_turn(backend, answer))
    turns = [memo, answer]
    good = ui.revision_turn('{"n": 1, "draft": "## Bottom line\\nX", "notes": " shorter "}', turns)
    assert (good.n, good.revises, good.revision, good.notes, good.k) == (3, 1, 2, "shorter", 7)
    assert good.as_memo and good.docs == memo.docs and good.docs is not memo.docs
    for bad in ["", "not json", "[]", '{"n": 1, "draft": "x"}', '{"n": 1, "draft": "x", "notes": "  "}',
                '{"n": 2, "draft": "x", "notes": "y"}', '{"n": 9, "draft": "x", "notes": "y"}', '{"n": "a", "draft": "x", "notes": "y"}']:
        assert ui.revision_turn(bad, turns) is None, bad
