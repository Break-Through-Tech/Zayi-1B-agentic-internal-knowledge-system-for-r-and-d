"""Gradio front end for the RAG pipeline (November, Task 6).

Ask a question, read a cited answer, and check every citation against the
excerpt it points to. "Search only" shows what the retriever finds without
calling the LLM, so exploring the papers doesn't use the daily API quota.

The query flow (`run_query`) is plain Python that yields `View` snapshots,
so it is unit-tested without a browser; `build_app` only wires it to
Gradio components. Launch with `python scripts/app.py`.
"""

import atexit
import html
import logging
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.rag import (
    CITATION_GROUP,
    NO_CONTEXT_ANSWER,
    MissingConfigError,
    _http_status,
    answer_from_docs,
    is_daily_quota_error,
    get_llm,
    page_label,
    source_label,
    sources_from,
)
from src.retriever import DEFAULT_K

log = logging.getLogger(__name__)

APP_TITLE = "Zayi Research Assistant"
PREVIEW_CHARS = 320

EXAMPLE_QUESTIONS = [
    "How does LoRA reduce the number of trainable parameters?",
    "What are reflection tokens in Self-RAG used for?",
    "How does FlashAttention reduce reads and writes to GPU memory?",
    "Where in a long context do language models use information best?",
    "How does CRAG decide whether retrieved documents can be trusted?",
]

EMPTY_ANSWER = "Ask a question to get an answer drawn from the papers, with citations."
EMPTY_SOURCES = '<p class="empty">The excerpts behind the answer will appear here.</p>'
SEARCH_ONLY_ANSWER = (
    "*Search only — no answer was generated and no API quota was used. "
    "The closest excerpts are listed under Sources.*"
)


# ---------------------------------------------------------------- rendering


def pdf_page_url(paper_id, page):
    return f"https://arxiv.org/pdf/{paper_id}#page={page}"


def link_citations(answer, n_sources):
    """Turn `[S2]`, `[S1, S3]`, `[S1-S3]` into Markdown links to the source
    cards (`#src-2`). Labels outside 1..n_sources stay plain text, so a
    citation that points nowhere never looks clickable."""

    def link(token):
        n = int(token.group(1))
        label = source_label(n)
        return f"[{label}](#src-{n})" if 1 <= n <= n_sources else f"\\[{label}\\]"

    def group(match):
        inner = match.group(0)[1:-1].strip()
        parts = re.split(r"\s*([,;\-–])\s*", inner)
        out = []
        for part in parts:
            if part in {",", ";"}:
                continue  # adjacent chips need no separator
            elif part in {"-", "–"}:
                out.append("–")
            else:
                out.append(re.sub(r"S?(\d+)", link, part))
        return "".join(out)

    return CITATION_GROUP.sub(group, answer)


def not_in_sources(invalid, where="the sources below"):
    """'cites [S7], which is not one of …' / 'cites [S7], [S9], which are not among …'"""
    labels = ", ".join(f"[{source_label(i)}]" for i in invalid)
    verb = "is not one of" if len(invalid) == 1 else "are not among"
    return f"the answer cites {labels}, which {verb} {where}"


# Inline math as Gemini writes it: $\Delta W = BA$. Pandoc's rule tells it
# apart from prices: no space just inside the dollars, and no digit right
# after the closing one ("$5 and $10" is not math).
_INLINE_MATH = re.compile(r"(?<![\\$])\$(?![\s$])([^$\n]+?)(?<!\s)\$(?![\d$])")


def inline_math(answer):
    """Rewrite inline $…$ math as \\(…\\), so the answer box renders it with
    KaTeX while a plain "$" (a price) stays literal text."""
    return _INLINE_MATH.sub(lambda m: f"\\({m.group(1)}\\)", answer)


def status_html(kind, text):
    """One-line status above the results. kind: busy | ok | warn | error."""
    return f'<div class="status status-{kind}" role="status">{html.escape(text)}</div>'


def render_sources(docs, cited=None):
    """Source cards, numbered like the prompt (S1 = docs[0]). `cited`: the
    source numbers the answer used, to mark the rest as not cited; None in
    search-only mode."""
    if not docs:
        return '<p class="empty">No matching excerpts were found.</p>'
    cards = []
    for source, doc in zip(sources_from(docs), docs):
        n = source["n"]
        text = " ".join(doc.page_content.split())
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS].rsplit(" ", 1)[0] + "…"
        meta = [f"arXiv:{source['paper_id']}", page_label(source)]
        badge = ""
        state = ""
        if cited is not None:
            state = " cited" if n in cited else " uncited"
            badge = '<span class="src-badge">Not cited</span>' if n not in cited else ""
        full = (
            f'<details><summary>Show full excerpt</summary><p>{html.escape(text)}</p></details>'
            if len(text) > len(preview)
            else ""
        )
        cards.append(
            f'<article class="src-card{state}" id="src-{n}">'
            f'<div class="src-head"><span class="src-tag">{source_label(n)}</span>'
            f'<div class="src-title">{html.escape(source["paper_title"])}</div>{badge}</div>'
            f'<div class="src-meta">{html.escape(" · ".join(meta))}</div>'
            f'<p class="src-preview">{html.escape(preview)}</p>{full}'
            f'<a class="src-link" href="{html.escape(pdf_page_url(source["paper_id"], source["page_start"]))}" '
            f'target="_blank" rel="noopener noreferrer">Open page {source["page_start"]} in the PDF ↗</a>'
            f"</article>"
        )
    return '<div class="src-list">' + "".join(cards) + "</div>"


def papers_html(papers):
    rows = "".join(
        f'<li><a href="https://arxiv.org/abs/{html.escape(p["id"])}" target="_blank" '
        f'rel="noopener noreferrer">{html.escape(p["title"])}</a>'
        f'<span class="paper-meta">{html.escape(p.get("category", ""))} · arXiv:{html.escape(p["id"])}</span></li>'
        for p in papers
    )
    return f'<ol class="papers">{rows}</ol>'


def memo_markdown(result, when=None):
    """The answer and its sources as a standalone Markdown file for review."""
    when = when or datetime.now()
    question = " ".join(result["question"].split())
    n = len(result["sources"])
    lines = [
        f"# {question}",
        "",
        f"*Drafted by the {APP_TITLE} on {when:%Y-%m-%d %H:%M} from {n} paper "
        f"excerpt{'s' if n != 1 else ''}. Check the cited pages before relying on it.*",
        "",
        "## Answer",
        "",
        result["answer"].strip(),
        "",
    ]
    invalid = result["citations"]["invalid_sources"]
    if invalid:
        lines += [f"> Note: {not_in_sources(invalid)}.", ""]
    if result["sources"]:
        lines += ["## Sources", ""]
        lines += [
            f"- **[{source_label(s['n'])}]** {s['paper_title']} (arXiv:{s['paper_id']}), "
            f"{page_label(s)} — {pdf_page_url(s['paper_id'], s['page_start'])}"
            for s in result["sources"]
        ]
    return "\n".join(lines) + "\n"


_memo_dir = None


def write_memo(result):
    """Write the memo into one per-process temp folder (removed at exit)."""
    global _memo_dir
    if _memo_dir is None:
        _memo_dir = Path(tempfile.mkdtemp(prefix="zayi-answers-"))
        atexit.register(shutil.rmtree, _memo_dir, True)
    stem = f"answer-{datetime.now():%Y%m%d-%H%M%S}"
    path, n = _memo_dir / f"{stem}.md", 1
    while path.exists():
        n += 1
        path = _memo_dir / f"{stem}-{n}.md"
    path.write_text(memo_markdown(result), encoding="utf-8")
    return str(path)


def describe_error(err, stage):
    """Short, actionable message for the status line (full trace goes to the log)."""
    if isinstance(err, MissingConfigError):
        return str(err) + " Search only still works without a key."
    status = _http_status(err)
    if is_daily_quota_error(err):
        return (
            "The daily API quota is used up (Gemini free tier: 20 requests a day). "
            "Search only still works; answers resume when the quota resets."
        )
    if status == 429:
        return "The LLM provider is rate-limiting requests. Wait a minute and try again; Search only still works."
    if status in (401, 403):
        return (
            "The API key was rejected. Check the key in .env "
            "(GOOGLE_API_KEY for Gemini, OPENROUTER_API_KEY for OpenRouter)."
        )
    detail = " ".join(str(err).split())[:200]
    what = "Search" if stage == "search" else "Answering"
    return f"{what} failed ({type(err).__name__}){': ' + detail if detail else ''}"


# --------------------------------------------------------------- query flow


class Backend:
    """Retrieval and generation behind the app. The defaults load the shared
    knowledge base once and build a fresh LLM client per answer (so .env
    edits apply without a restart); tests inject fakes."""

    def __init__(self, make_retriever=None, make_llm=None):
        self._make_retriever = make_retriever or self._default_retriever
        self._make_llm = make_llm or get_llm
        self._collection = None
        self._lock = threading.Lock()

    def _default_retriever(self, k, rerank):
        from src.retriever import PaperRetriever, load_knowledge_base

        with self._lock:  # first query may arrive while warm_up is loading
            if self._collection is None:
                self._collection = load_knowledge_base()
        return PaperRetriever(collection=self._collection, k=k, rerank=rerank)

    def warm_up(self):
        """Load the index, embedding model and reranker before the first question."""
        self.search("warm-up query", DEFAULT_K, True)

    def search(self, question, k, rerank):
        return self._make_retriever(int(k), bool(rerank)).invoke(question)

    def answer(self, question, docs):
        # no excerpts -> fixed "nothing found" reply; don't require an API key for it
        return answer_from_docs(question, docs, self._make_llm() if docs else None)


@dataclass
class View:
    """One snapshot of the results area. None = leave that part unchanged."""

    status: str | None = None
    answer: str | None = None
    sources: str | None = None
    memo_path: str | None = None


def run_query(backend, question, k=DEFAULT_K, rerank=True, generate=True):
    """Search, then (unless `generate` is False) answer — yielding a View
    after each step so the sources appear while the answer is written."""
    question = (question or "").strip()
    if not question:
        yield View(status=status_html("warn", "Type a question first."))
        return

    yield View(
        status=status_html("busy", "Searching the papers…"),
        answer="",
        sources='<p class="empty">Searching…</p>',
    )
    start = time.perf_counter()
    try:
        docs = backend.search(question, k, rerank)
    except Exception as err:
        log.exception("search failed")
        yield View(status=status_html("error", describe_error(err, "search")), answer="", sources=EMPTY_SOURCES)
        return
    search_s = time.perf_counter() - start
    n = len(docs)
    found = f"{n} excerpt{'s' if n != 1 else ''} · search {search_s:.1f} s"

    if not generate:
        yield View(status=status_html("ok", f"{found} · no API call"), answer=SEARCH_ONLY_ANSWER, sources=render_sources(docs))
        return

    if not docs:  # nothing to ground an answer in: say so, no LLM call, nothing to download
        yield View(status=status_html("warn", f"{found} · nothing to answer from"), answer=NO_CONTEXT_ANSWER, sources=render_sources(docs))
        return

    yield View(status=status_html("busy", f"{found} · writing the answer…"), sources=render_sources(docs))
    start = time.perf_counter()
    try:
        result = backend.answer(question, docs)
    except Exception as err:
        log.exception("answer failed")
        yield View(status=status_html("error", describe_error(err, "answer")), answer="")
        return
    answer_s = time.perf_counter() - start

    citations = result["citations"]
    if not citations["valid"]:
        problem = not_in_sources(citations["invalid_sources"], "the retrieved sources")
        status = status_html("warn", f"{problem[0].upper()}{problem[1:]}. Treat that claim with caution.")
    else:
        status = status_html("ok", f"{found} · answer {answer_s:.1f} s")
    yield View(
        status=status,
        answer=link_citations(inline_math(result["answer"].strip()), n),
        sources=render_sources(docs, cited=set(citations["cited_sources"])),
        memo_path=write_memo(result),
    )


# ------------------------------------------------------------------- layout

CSS = """
#app { max-width: 1120px; margin: 0 auto; }
#app .html-container { padding: 0 !important; }
.gradio-container { overflow: clip !important; }  /* "hidden" would break the sticky answer column */
.app-header { padding: 28px 0 6px; }
.app-header .eyebrow { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
  color: var(--body-text-color-subdued); margin: 0 0 6px; }
.app-header h1 { font-size: 28px; font-weight: 650; letter-spacing: -.01em; margin: 0 0 6px; }
.app-header p { color: var(--body-text-color-subdued); margin: 0; max-width: 680px; font-size: 15px; }
.hint { color: var(--body-text-color-subdued); font-size: 13px; display: flex; align-items: center; min-height: 100%; }
@media (max-width: 640px) { #actions { flex-wrap: wrap; } #actions .hint-block { flex: 1 1 100%; } }
.panel-title { font-size: 13px !important; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
  color: var(--body-text-color-subdued); margin: 4px 0 8px !important; }
.status { font-size: 14px; padding: 9px 12px; border-radius: 6px; border: 1px solid var(--border-color-primary);
  background: var(--background-fill-secondary); }
.status-busy::before { content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 8px;
  border: 2px solid var(--body-text-color-subdued); border-right-color: transparent; border-radius: 50%;
  vertical-align: -1px; animation: spin .8s linear infinite; }
@keyframes spin { to { transform: rotate(360deg); } }
.status-ok { color: var(--body-text-color-subdued); }
.status-warn { border-color: #d9a400; background: rgba(217,164,0,.10); }
.status-error { border-color: #d64545; background: rgba(214,69,69,.10); }
.answer { font-size: 15.5px; line-height: 1.65; }
@media (min-width: 900px) { #answer-col { position: sticky; top: 16px; align-self: flex-start; } }
.answer p:first-child { margin-top: 0; }
.answer a[href^="#src-"] { display: inline-block; font-size: 11.5px; font-weight: 600; line-height: 1;
  padding: 3px 5px; margin: 0 1px; border-radius: 4px; text-decoration: none; vertical-align: 2px;
  color: var(--color-accent); background: var(--color-accent-soft); }
.answer a[href^="#src-"]:hover { text-decoration: underline; }
.empty { color: var(--body-text-color-subdued); font-size: 14px; margin: 0; }
.src-list { display: flex; flex-direction: column; gap: 10px; }
.src-card { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 12px 14px;
  background: var(--block-background-fill); scroll-margin-top: 16px; transition: border-color .2s, box-shadow .2s; }
.src-card:target { border-color: var(--color-accent); box-shadow: 0 0 0 3px var(--color-accent-soft); }
.src-card.uncited { opacity: .62; }
.src-head { display: flex; gap: 10px; align-items: flex-start; }
.src-tag { flex: none; font-size: 11.5px; font-weight: 700; padding: 3px 6px; border-radius: 4px;
  color: var(--color-accent); background: var(--color-accent-soft); }
.src-title { font-weight: 600; font-size: 14px; line-height: 1.35; flex: 1; }
.src-badge { flex: none; font-size: 11px; color: var(--body-text-color-subdued); border: 1px solid var(--border-color-primary);
  border-radius: 999px; padding: 1px 7px; }
.src-meta { font-size: 12.5px; color: var(--body-text-color-subdued); margin: 4px 0 8px; font-variant-numeric: tabular-nums; }
.src-preview, .src-card details p { font-size: 13.5px; line-height: 1.55; margin: 0 0 6px; }
.src-card:has(details[open]) .src-preview { display: none; }
.src-card details summary { cursor: pointer; font-size: 12.5px; color: var(--body-text-color-subdued); margin-bottom: 6px; }
.src-link { font-size: 12.5px; text-decoration: none; color: var(--color-accent); }
.src-link:hover { text-decoration: underline; }
.papers { margin: 0; padding-left: 20px; font-size: 14px; }
.papers li { margin: 4px 0; }
.papers a { text-decoration: none; }
.paper-meta { color: var(--body-text-color-subdued); font-size: 12.5px; margin-left: 8px; }
.app-footer { color: var(--body-text-color-subdued); font-size: 12.5px; padding: 8px 0 24px; }
"""


def make_theme():
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        font=["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
        font_mono=["SFMono-Regular", "Menlo", "Consolas", "monospace"],
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        block_shadow="none",
        block_shadow_dark="none",
        button_primary_background_fill="*neutral_900",
        button_primary_background_fill_hover="*neutral_700",
        button_primary_text_color="white",
        button_primary_background_fill_dark="*neutral_100",
        button_primary_background_fill_hover_dark="*neutral_300",
        button_primary_text_color_dark="*neutral_900",
        color_accent_soft="*primary_50",
        color_accent_soft_dark="*neutral_800",
    )


def build_app(backend=None, papers=None):
    """The Gradio Blocks app. Pass theme/css to `launch` (Gradio 6):
    `build_app().launch(theme=make_theme(), css=CSS)`."""
    import gradio as gr

    backend = backend or Backend()
    if papers is None:
        from src.data_io import load_papers

        papers = load_papers()

    def submit(question, k, rerank, generate):
        for view in run_query(backend, question, k, rerank, generate):
            yield (
                gr.skip() if view.status is None else view.status,
                gr.skip() if view.answer is None else view.answer,
                gr.skip() if view.sources is None else view.sources,
                # the download follows the answer it belongs to
                gr.skip() if view.answer is None else gr.update(value=view.memo_path, visible=view.memo_path is not None),
            )

    with gr.Blocks(title=APP_TITLE, analytics_enabled=False) as app:
        with gr.Column(elem_id="app"):
            gr.HTML(
                '<header class="app-header"><p class="eyebrow">Zayi · Internal knowledge system</p>'
                f"<h1>{APP_TITLE}</h1>"
                f"<p>Ask about the {len(papers)} curated ML papers. Answers cite the excerpts they come "
                "from, so every claim can be checked against the paper.</p></header>"
            )
            question = gr.Textbox(
                show_label=False,
                placeholder="Ask a question about the papers…",
                lines=1,  # Enter submits; long questions grow to max_lines
                max_lines=4,
                autofocus=True,
            )
            with gr.Row(elem_id="actions"):
                ask_btn = gr.Button("Ask", variant="primary", scale=0, min_width=110)
                search_btn = gr.Button("Search only", variant="secondary", scale=0, min_width=130)
                gr.HTML(
                    '<span class="hint">Search only shows the matching excerpts without an API call.</span>',
                    elem_classes="hint-block",
                )
            with gr.Accordion("Retrieval settings", open=False):
                with gr.Row():
                    k = gr.Slider(3, 10, value=DEFAULT_K, step=1, label="Excerpts to use")
                    rerank = gr.Checkbox(
                        value=True,
                        label="Rerank with cross-encoder",
                        info="Finds the right page more often; adds about 3 s per question.",
                    )
            gr.Examples(EXAMPLE_QUESTIONS, inputs=question, label="Example questions")
            status = gr.HTML("")
            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_id="answer-col"):
                    gr.HTML('<h2 class="panel-title">Answer</h2>')
                    answer = gr.Markdown(
                        EMPTY_ANSWER,
                        elem_classes="answer",
                        # inline math arrives as \(…\) via inline_math(); a bare "$" stays text
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "\\(", "right": "\\)", "display": False},
                        ],
                    )
                    download = gr.DownloadButton("Download as Markdown", visible=False, size="sm", scale=0)
                with gr.Column(scale=4):
                    gr.HTML('<h2 class="panel-title">Sources</h2>')
                    sources = gr.HTML(EMPTY_SOURCES)
            with gr.Accordion(f"Papers in the collection ({len(papers)})", open=False):
                gr.HTML(papers_html(papers))
            gr.HTML(
                '<p class="app-footer">Answers are generated from the paper excerpts only (Gemini by default). '
                "Check the cited pages before relying on a claim.</p>"
            )

        # Locked while a query runs, so Enter can't start a second one.
        controls = [ask_btn, search_btn, question]
        outputs = [status, answer, sources, download]

        def lock():
            return [gr.update(interactive=False)] * len(controls)

        def unlock():
            return [gr.update(interactive=True)] * len(controls)

        for trigger, generate in ((ask_btn.click, True), (question.submit, True), (search_btn.click, False)):
            (
                trigger(lock, outputs=controls, queue=False)
                .then(
                    submit,
                    inputs=[question, k, rerank, gr.State(generate)],
                    outputs=outputs,
                    show_progress="hidden",
                    concurrency_id="query",  # all three triggers share one slot:
                    concurrency_limit=1,  # queries run one at a time, never interleaved
                )
                .then(unlock, outputs=controls, queue=False)
            )
    return app
