"""Gradio front end for the RAG pipeline (November, Task 6).

A chat-style page: each question becomes a turn with the retrieval steps,
the cited answer, and the source excerpts it was drawn from. "Search only"
shows what the retriever finds without calling the LLM, so exploring the
papers doesn't use the daily API quota. Each question is answered on its own;
earlier turns are not sent to the model.

The query flow (`run_turn`) and the HTML rendering are plain Python, so they
are unit-tested without a browser; `build_app` only wires them to Gradio.
Launch with `python scripts/app.py`.
"""

import html
import inspect
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

from markdown_it import MarkdownIt

from src.rag import (
    CITATION_GROUP,
    NO_CONTEXT_ANSWER,
    MissingConfigError,
    _http_status,
    answer_from_docs,
    get_llm,
    is_daily_quota_error,
    page_label,
    source_label,
    sources_from,
)
from src.retriever import DEFAULT_K, DEFAULT_RERANK_POOL

log = logging.getLogger(__name__)

APP_TITLE = "Zayi Research Assistant"
ASSETS = Path(__file__).resolve().parent.parent / "assets"
LOGO = ASSETS / "logo.svg"  # also the tab icon
# Green aurora over Snæfellsnes, Iceland — photo by Jack Anstey on Unsplash
# (Unsplash License: free to use), resized to 2560 px WebP.
BACKGROUND = ASSETS / "background.webp"
PREVIEW_CHARS = 260
MODE_ANSWER, MODE_SEARCH = "Answer", "Search only"

# (chip label, full question, lucide icon name)
EXAMPLES = [
    ("LoRA parameters", "How does LoRA reduce the number of trainable parameters?", "layers"),
    ("Self-RAG tokens", "What are reflection tokens in Self-RAG used for?", "tag"),
    ("FlashAttention memory", "How does FlashAttention reduce reads and writes to GPU memory?", "zap"),
    ("Lost in the middle", "Where in a long context do language models use information best?", "align"),
    ("CRAG trust", "How does CRAG decide whether retrieved documents can be trusted?", "shield"),
]

# Lucide icons (ISC license), drawn as CSS masks on the example chips.
ICONS = {
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "tag": '<path d="M12.59 2.59A2 2 0 0 0 11.17 2H4a2 2 0 0 0-2 2v7.17a2 2 0 0 0 .59 1.42l8.7 8.7a2.43 2.43 0 0 0 3.42 0l6.58-6.58a2.43 2.43 0 0 0 0-3.42z"/><circle cx="7.5" cy="7.5" r="1"/>',
    "zap": '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>',
    "align": '<path d="M21 6H3"/><path d="M17 12H7"/><path d="M19 18H5"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-4"/>',
    "send": '<path d="M12 19V5"/><path d="m5 12 7-7 7 7"/>',
    "plus": '<path d="M12 5v14"/><path d="M5 12h14"/>',
    "chev": '<path d="m6 9 6 6 6-6"/>',
    "external": '<path d="M15 3h6v6"/><path d="M10 14 21 3"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/>',
    "download": '<path d="M12 15V3"/><path d="m7 10 5 5 5-5"/><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>',
}


def icon_svg(name, cls="ico"):
    return (
        f'<svg class="{cls}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


def _icon_mask(name):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>'
    return f'url("data:image/svg+xml,{quote(svg)}")'


# ------------------------------------------------------------ answer text


def pdf_page_url(paper_id, page):
    return f"https://arxiv.org/pdf/{paper_id}#page={page}"


def link_citations(answer, n_sources, anchor="src-"):
    """Turn `[S2]`, `[S1, S3]`, `[S1-S3]` into Markdown links to the source
    cards (`#{anchor}2`). Labels outside 1..n_sources stay plain text, so a
    citation that points nowhere never looks clickable."""

    def link(token):
        n = int(token.group(1))
        label = source_label(n)
        return f"[{label}](#{anchor}{n})" if 1 <= n <= n_sources else f"\\[{label}\\]"

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


# Math as Gemini writes it. Inline $…$ follows Pandoc's rule so prices stay
# text: no space just inside the dollars, no digit right after the closing
# one ("$5 and $10" is not math).
_MATH = re.compile(
    r"\$\$(?P<d1>.+?)\$\$"
    r"|\\\[(?P<d2>.+?)\\\]"
    r"|\\\((?P<i1>.+?)\\\)"
    r"|(?<![\\$])\$(?![\s$])(?P<i2>[^$\n]+?)(?<!\s)\$(?![\d$])",
    re.S,
)
_MATH_TOKEN = "@@MATH{}@@"


def protect_math(text):
    """Swap math spans for placeholders so Markdown leaves the TeX alone.
    Returns (text, [(tex, display), ...])."""
    found = []

    def keep(m):
        display = m.group("d1") is not None or m.group("d2") is not None
        tex = next(g for g in m.group("d1", "d2", "i1", "i2") if g is not None)
        found.append((tex.strip(), display))
        return _MATH_TOKEN.format(len(found) - 1)

    return _MATH.sub(keep, text), found


_markdown = MarkdownIt("commonmark", {"html": False}).enable(["table", "strikethrough"])


def answer_html(answer, n_sources, anchor="src-"):
    """The model's Markdown answer as safe HTML: raw HTML is escaped,
    citations become chips linking to the source cards, and math becomes
    `.math` spans that the page typesets with KaTeX (plain TeX if offline)."""
    text, maths = protect_math(answer.strip())
    out = _markdown.render(link_citations(text, n_sources, anchor))
    for i, (tex, display) in enumerate(maths):
        cls = "math display" if display else "math"
        out = out.replace(_MATH_TOKEN.format(i), f'<span class="{cls}">{html.escape(tex)}</span>')
    return out


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


def describe_error(err, stage):
    """Short, actionable message for the chat (full trace goes to the log)."""
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


def model_name(llm):
    name = getattr(llm, "model", None) or getattr(llm, "model_name", None) or ""
    return str(name).removeprefix("models/")


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
        llm = self._make_llm() if docs else None
        return {**answer_from_docs(question, docs, llm), "model": model_name(llm)}


@dataclass
class Turn:
    """One question and everything shown for it. `run_turn` fills it in."""

    n: int  # 1-based position; prefixes the element ids in this turn
    question: str
    generate: bool = True  # False = search only
    k: int = DEFAULT_K
    rerank: bool = True
    phase: str = "search"  # search -> write -> done, or error
    docs: list = field(default_factory=list)
    search_s: float | None = None
    answer_s: float | None = None
    answer: str = ""  # rendered HTML
    cited: set | None = None
    invalid: list = field(default_factory=list)
    model: str = ""
    memo: str = ""
    error: str = ""

    @property
    def anchor(self):
        return f"t{self.n}-s"


def run_turn(backend, turn):
    """Search, then (unless search only) answer — yielding the turn after
    each step so the page shows progress and the sources before the answer."""
    yield turn
    start = time.perf_counter()
    try:
        turn.docs = backend.search(turn.question, turn.k, turn.rerank)
    except Exception as err:
        log.exception("search failed")
        turn.phase, turn.error = "error", describe_error(err, "search")
        yield turn
        return
    turn.search_s = time.perf_counter() - start

    if not turn.generate or not turn.docs:
        if turn.generate:  # nothing to ground an answer in: say so, no LLM call
            turn.answer = answer_html(NO_CONTEXT_ANSWER, 0)
        turn.phase = "done"
        yield turn
        return

    turn.phase = "write"
    yield turn
    start = time.perf_counter()
    try:
        result = backend.answer(turn.question, turn.docs)
    except Exception as err:
        log.exception("answer failed")
        turn.phase, turn.error = "error", describe_error(err, "answer")
        yield turn
        return
    turn.answer_s = time.perf_counter() - start
    citations = result["citations"]
    turn.cited = set(citations["cited_sources"])
    turn.invalid = citations["invalid_sources"]
    turn.answer = answer_html(result["answer"], len(turn.docs), turn.anchor)
    turn.model = result.get("model", "")
    turn.memo = memo_markdown(result)
    turn.phase = "done"
    yield turn


# ---------------------------------------------------------------- rendering


def _plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def render_sources(docs, cited=None, anchor="src-"):
    """Source cards, numbered like the prompt (S1 = docs[0]). `cited`: the
    source numbers the answer used, to dim the rest; None = search only."""
    cards = []
    for source, doc in zip(sources_from(docs), docs):
        n = source["n"]
        text = " ".join(doc.page_content.split())
        preview = text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS].rsplit(" ", 1)[0] + "…"
        state = "" if cited is None else (" cited" if n in cited else " uncited")
        badge = '<span class="src-badge">Not cited</span>' if cited is not None and n not in cited else ""
        full = (
            f'<details class="src-full" id="{anchor}{n}-full"><summary>Full excerpt</summary>'
            f"<p>{html.escape(text)}</p></details>"
            if len(text) > len(preview)
            else ""
        )
        url = html.escape(pdf_page_url(source["paper_id"], source["page_start"]))
        cards.append(
            f'<article class="src-card{state}" id="{anchor}{n}">'
            f'<div class="src-head"><span class="src-tag">{source_label(n)}</span>'
            f'<span class="src-title">{html.escape(source["paper_title"])}</span>{badge}</div>'
            f'<p class="src-preview">{html.escape(preview)}</p>{full}'
            f'<div class="src-foot"><span>arXiv:{html.escape(source["paper_id"])} · {html.escape(page_label(source))}</span>'
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">Open in PDF {icon_svg("external")}</a></div>'
            f"</article>"
        )
    return '<div class="src-grid">' + "".join(cards) + "</div>"


def _steps(turn):
    """The retrieval/answer timeline: (state, text, detail) rows + summary label."""
    how = (
        f"embedding search, top {max(DEFAULT_RERANK_POOL, turn.k)} reranked to {turn.k}"
        if turn.rerank
        else f"embedding search, top {turn.k}"
    )
    rows = []
    n = len(turn.docs)
    papers = len({d.metadata.get("paper_id") for d in turn.docs})
    if turn.phase == "search":
        rows.append(("pending", "Searching the papers", how))
        return rows, "Searching the papers…"
    if turn.search_s is None:  # search failed
        rows.append(("failed", "Search failed", how))
        return rows, "Search failed"
    found = f"Found {_plural(n, 'excerpt')} from {_plural(papers, 'paper')}" if n else "Found no matching excerpts"
    rows.append(("done", found, f"{how} · {turn.search_s:.1f} s"))
    if not turn.generate or not n:
        return rows, found
    if turn.phase == "write":
        rows.append(("pending", "Writing the answer", "from these excerpts only"))
        return rows, "Writing the answer…"
    if turn.answer_s is None:
        rows.append(("failed", "Answering failed", ""))
        return rows, found
    rows.append(("done", "Wrote the answer", " · ".join(filter(None, [turn.model, f"{turn.answer_s:.1f} s"]))))
    if turn.invalid:
        rows.append(("failed", "Checked the citations", f"{', '.join(source_label(i) for i in turn.invalid)} not in the excerpts"))
    else:
        cited = len(turn.cited or ())
        detail = f"{_plural(cited, 'excerpt')} cited, all retrieved" if cited else "the answer cites no excerpts"
        rows.append(("done" if cited else "failed", "Checked the citations", detail))
    return rows, f"Used {_plural(n, 'excerpt')} from {_plural(papers, 'paper')}"


def _memo_link(turn):
    words = re.findall(r"[a-z0-9]+", turn.question.lower())[:6]
    name = "zayi-" + ("-".join(words) or f"answer-{turn.n}") + ".md"
    return (
        f'<a class="dl" download="{html.escape(name)}" '
        f'href="data:text/markdown;charset=utf-8,{quote(turn.memo)}">{icon_svg("download")}Download .md</a>'
    )


def render_turn(turn):
    rows, label = _steps(turn)
    busy = turn.phase in ("search", "write")
    steps = "".join(
        f'<div class="step {state}">{html.escape(text)}'
        f'{f" <span class=io>{html.escape(detail)}</span>" if detail else ""}</div>'
        for state, text, detail in rows
    )
    think = (
        f'<details class="think{"" if busy else " done"}" id="t{turn.n}-steps"{" open" if busy else ""}>'
        f'<summary><span class="pulse"></span><span>{html.escape(label)}</span>{icon_svg("chev", "chev")}</summary>'
        f'<div class="track">{steps}</div></details>'
    )
    body = []
    if turn.phase == "write":
        body.append('<div class="dots" aria-label="Writing"><span></span><span></span><span></span></div>')
    if turn.answer:
        body.append(f'<div class="prose">{turn.answer}</div>')
    if turn.invalid:
        problem = not_in_sources(turn.invalid, "the retrieved excerpts")
        body.append(f'<div class="note warn">{html.escape(problem[0].upper() + problem[1:])}. Treat that claim with caution.</div>')
    if turn.error:
        body.append(f'<div class="note error">{html.escape(turn.error)}</div>')
    if turn.docs:
        heading = "Sources" if turn.generate else "Closest excerpts"
        body.append(
            f'<section class="sources"><h3>{heading}<span>{len(turn.docs)}</span></h3>'
            f"{render_sources(turn.docs, turn.cited, turn.anchor)}</section>"
        )
    meta = []
    if turn.phase == "done":
        if not turn.generate:
            meta.append(f"Search only · no API call · {turn.search_s:.1f} s")
        elif turn.answer_s is not None:
            meta.append(" · ".join(filter(None, [turn.model, f"{turn.search_s + turn.answer_s:.1f} s"])))
        if turn.memo:
            meta.append(_memo_link(turn))
    if meta:
        body.append(f'<div class="meta">{" ".join(meta)}</div>')
    return (
        f'<section class="turn" id="t{turn.n}">'
        f'<div class="msg user"><div class="bubble">{html.escape(turn.question)}</div></div>'
        f'<div class="msg bot"><div class="avatar" aria-hidden="true"></div>'
        f'<div class="content">{think}{"".join(body)}</div></div></section>'
    )


def render_hero(papers):
    items = "".join(
        f'<li><a href="https://arxiv.org/abs/{html.escape(p["id"])}" target="_blank" rel="noopener noreferrer">'
        f'{html.escape(p["title"])}</a><span>{html.escape(p.get("category", ""))}</span></li>'
        for p in papers
    )
    return (
        '<div class="hero"><div class="hero-icon" aria-hidden="true"></div><h1>Zayi</h1>'
        f"<p>Ask anything about the <b>{_plural(len(papers), 'research paper')}</b> in the collection. "
        "Every answer cites the excerpts it came from, so each claim can be checked.</p>"
        f'<details class="papers" id="papers"><summary>See the papers</summary><ol>{items}</ol></details></div>'
    )


def render_thread(turns, papers):
    if not turns:
        return render_hero(papers)
    return "".join(render_turn(t) for t in turns)


# ------------------------------------------------------------------- styles

FONT = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'

CSS = (
    """
/* Gradio adds a copy of each rule prefixed with ".gradio-container… .contain"
   (inside @media it only keeps the prefixed one). So rules that switch on
   "a chat has started" use body:has(.turn) with !important at top level, and
   #thread:has(.turn) ~ #dock inside media queries. */
:root {
  --z-bg: #060d0b; --z-surface: rgba(255,255,255,.03); --z-surface-2: rgba(255,255,255,.055);
  --z-panel: 7,17,14; /* rgb of the glass panels */
  --z-border: rgba(255,255,255,.08); --z-border-strong: rgba(255,255,255,.14);
  --z-text: #e7efeb; --z-soft: #b7c7c0; --z-muted: #8a9c94; --z-faint: #5b6d65;
  --z-accent: #4ee6a0; --z-accent-rgb: 78,230,160; --z-accent-soft: rgba(78,230,160,.12); --z-accent-line: rgba(78,230,160,.42);
  --z-logo: LOGO_URL;
  --z-ok: #9ece6a; --z-warn: #e0af68; --z-danger: #f7768e;
  --z-mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, monospace;
  --z-col: 760px;
}
html { background: var(--z-bg) !important; color-scheme: dark; }
body, gradio-app { background: transparent !important; }
body { -webkit-font-smoothing: antialiased; }
.gradio-container { max-width: none !important; width: 100% !important; padding: 0 !important; margin: 0 !important;
  background: transparent !important; overflow: clip !important; color: var(--z-text); }
.gradio-container > .main, .gradio-container .wrap.app, .gradio-container .contain { gap: 0 !important; padding: 0 !important; max-width: none !important; }
.gradio-container .html-container { padding: 0 !important; }
footer { display: none !important; }

/* background: aurora photo under a dark scrim (darker once chatting), fine grain on top */
#bg { position: fixed; inset: 0; z-index: -1; pointer-events: none; overflow: hidden; background: var(--z-bg); }
#bg .photo { position: absolute; inset: 0; background: BG_URL 62% 30% / cover no-repeat; animation: fadein 1.2s ease; }
#bg .scrim { position: absolute; inset: 0; transition: background .5s ease; background:
  radial-gradient(120% 90% at 50% 45%, rgba(4,10,8,.15), rgba(4,10,8,.65)),
  linear-gradient(rgba(4,10,8,.45), rgba(4,10,8,.2) 40%, rgba(4,10,8,.7)); }
body:has(.turn) #bg .scrim { background: rgba(4,10,8,.8) !important; }
#bg .grain { position: absolute; inset: 0; opacity: .55; background: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3CfeColorMatrix values='0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0.035 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }

/* top bar */
#topbar { position: sticky; top: 0; z-index: 30; align-items: center !important; gap: 10px !important; flex-wrap: nowrap !important;
  padding: 12px 22px !important; border-bottom: 1px solid var(--z-border);
  background: rgba(var(--z-panel),.6); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }
#topbar > * { min-width: 0 !important; }
.brand { display: flex; align-items: center; gap: 10px; }
.brand .logo { width: 26px; height: 26px; background: var(--z-logo) center / contain no-repeat; }
.brand b { font-size: 15px; font-weight: 700; letter-spacing: -.01em; color: var(--z-text); }
.brand .tag { font-size: 10.5px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--z-accent);
  background: var(--z-accent-soft); border: 1px solid rgba(var(--z-accent-rgb),.25); border-radius: 999px; padding: 2px 8px; white-space: nowrap; }
button#new-chat { flex: none !important; width: auto !important; min-width: 0 !important; padding: 6px 12px 6px 10px !important;
  gap: 6px; font-size: 12.5px !important; font-weight: 600; color: var(--z-muted) !important;
  background: rgba(255,255,255,.05) !important; border: 1px solid var(--z-border) !important; border-radius: 9px !important; box-shadow: none !important; }
button#new-chat::before { content: ""; width: 14px; height: 14px; background: currentColor; -webkit-mask: ICON_PLUS center / contain no-repeat; mask: ICON_PLUS center / contain no-repeat; }
button#new-chat:hover:not(:disabled) { color: var(--z-text) !important; border-color: var(--z-border-strong) !important; }
button#new-chat:disabled { opacity: .45; }
body:not(:has(.turn)) button#new-chat { visibility: hidden !important; }

/* thread column */
#thread { width: 100%; max-width: calc(var(--z-col) + 44px); margin: 0 auto; padding: 26px 22px 8px; font-family: FONT; }
#thread > div, #thread .html-container > div { min-width: 0; }

.hero { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center;
  min-height: max(260px, calc(100dvh - 400px)); padding: 20px 0; animation: rise .4s ease; }
.hero-icon { width: 76px; height: 76px; margin-bottom: 22px; background: var(--z-logo) center / contain no-repeat;
  filter: drop-shadow(0 6px 28px rgba(var(--z-accent-rgb),.38)); }
.hero h1 { font-size: 42px; font-weight: 700; letter-spacing: -.03em; color: #fff; margin: 0;
  text-shadow: 0 2px 30px rgba(20,120,90,.55); }
.hero p { color: var(--z-soft); font-size: 15px; line-height: 1.55; margin: 10px 0 0; max-width: 520px; text-shadow: 0 1px 12px rgba(4,10,8,.8); }
.hero p b { color: #fff; font-weight: 600; }
.papers { margin-top: 14px; font-size: 13px; }
.papers summary { cursor: pointer; color: var(--z-muted); list-style: none; display: inline-block; padding: 4px 10px;
  border-radius: 999px; transition: color .15s ease; }
.papers summary::-webkit-details-marker { display: none; }
.papers summary:hover, .papers[open] summary { color: var(--z-text); }
.papers ol { text-align: left; margin: 10px auto 0; padding: 12px 16px 12px 34px; max-width: 560px; color: var(--z-faint);
  background: rgba(var(--z-panel),.6); border: 1px solid var(--z-border); border-radius: 12px;
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }
.papers li { padding: 3px 0; line-height: 1.45; }
.papers li a { color: var(--z-soft); text-decoration: none; }
.papers li a:hover { color: #fff; text-decoration: underline; }
.papers li span { color: var(--z-faint); margin-left: 8px; font-size: 12px; }

.turn { padding-bottom: 6px; }
.turn:last-child { min-height: calc(100dvh - 250px); }
.msg { display: flex; margin-bottom: 22px; animation: rise .3s ease; }
.msg.user { justify-content: flex-end; }
.bubble { max-width: 82%; background: var(--z-surface-2); border: 1px solid var(--z-border); border-radius: 18px 18px 4px 18px;
  padding: 11px 16px; font-size: 15px; line-height: 1.55; white-space: pre-wrap; word-break: break-word; color: var(--z-text); }
.msg.bot { gap: 12px; align-items: flex-start; }
.avatar { flex: none; width: 30px; height: 30px; margin-top: 1px; background: var(--z-logo) center / contain no-repeat; }
.content { min-width: 0; flex: 1; padding-top: 3px; }

/* steps timeline */
.think { margin-bottom: 12px; }
.think summary { display: inline-flex; align-items: center; gap: 7px; cursor: pointer; list-style: none;
  font-size: 12.5px; font-weight: 500; color: var(--z-muted); padding: 2px 0; transition: color .15s ease; }
.think summary::-webkit-details-marker { display: none; }
.think summary:hover { color: var(--z-text); }
.think .pulse { width: 7px; height: 7px; border-radius: 50%; background: var(--z-accent); animation: dotpulse 1.1s ease-in-out infinite; }
.think.done .pulse { display: none; }
.think .chev { width: 13px; height: 13px; transition: transform .25s ease; }
.think[open] .chev { transform: rotate(180deg); }
.track { padding: 6px 0 2px 2px; }
.step { position: relative; padding: 3px 0 3px 18px; font-size: 12.5px; line-height: 1.55; color: var(--z-muted); animation: rise .25s ease; }
.step::before { content: ""; position: absolute; left: 0; top: 8px; width: 7px; height: 7px; border-radius: 50%; background: var(--z-faint); }
.step.pending::before { background: var(--z-accent); animation: dotpulse 1.1s ease-in-out infinite; }
.step.done::before { background: var(--z-ok); }
.step.failed::before { background: var(--z-danger); }
.step::after { content: ""; position: absolute; left: 3px; top: 17px; bottom: -4px; width: 1px; background: var(--z-border-strong); }
.step:last-child::after { background: linear-gradient(var(--z-border-strong), transparent); }
.step .io { font-family: var(--z-mono); font-size: 11px; color: var(--z-faint); margin-left: 4px; }

.dots { display: inline-flex; gap: 5px; padding: 6px 2px 14px; }
.dots span { width: 7px; height: 7px; border-radius: 50%; background: var(--z-muted); animation: bounce 1.2s ease-in-out infinite; }
.dots span:nth-child(2) { animation-delay: .15s; } .dots span:nth-child(3) { animation-delay: .3s; }

/* answer */
.prose { font-size: 15px; line-height: 1.68; color: var(--z-text); word-break: break-word; }
.prose > :first-child { margin-top: 0; } .prose > :last-child { margin-bottom: 0; }
.prose p, .prose ul, .prose ol { margin: 0 0 12px; }
.prose ul, .prose ol { padding-left: 22px; } .prose li { margin: 4px 0; }
.prose h1, .prose h2, .prose h3, .prose h4 { font-size: 15.5px; font-weight: 650; margin: 18px 0 8px; color: #fff; }
.prose strong { color: #fff; font-weight: 600; }
.prose code { font-family: var(--z-mono); font-size: 13px; background: var(--z-surface-2); border-radius: 5px; padding: 1px 5px; }
.prose pre { background: rgba(0,0,0,.25); border: 1px solid var(--z-border); border-radius: 10px; padding: 12px 14px; overflow-x: auto; }
.prose pre code { background: none; padding: 0; }
.prose table { border-collapse: collapse; margin: 0 0 12px; font-size: 13.5px; display: block; overflow-x: auto; }
.prose th, .prose td { border: 1px solid var(--z-border-strong); padding: 6px 10px; text-align: left; }
.prose blockquote { margin: 0 0 12px; padding-left: 12px; border-left: 2px solid var(--z-border-strong); color: var(--z-soft); }
.prose a { color: var(--z-accent); }
.prose a[href^="#t"] { display: inline-block; font-family: var(--z-mono); font-size: 10.5px; font-weight: 600; line-height: 1;
  padding: 3px 5px; margin: 0 1px; border-radius: 5px; vertical-align: 2px; text-decoration: none;
  color: var(--z-accent); background: var(--z-accent-soft); border: 1px solid rgba(var(--z-accent-rgb),.22); transition: background .15s ease; }
.prose a[href^="#t"]:hover { background: rgba(var(--z-accent-rgb),.24); }
.math.display { display: block; margin: 10px 0; overflow-x: auto; text-align: center; }

.note { margin-top: 12px; border-radius: 10px; padding: 10px 14px; font-size: 13.5px; line-height: 1.5; }
.note.warn { border: 1px solid rgba(224,175,104,.3); border-left: 3px solid var(--z-warn); background: rgba(224,175,104,.07); color: #f0d3a8; }
.note.error { border: 1px solid rgba(247,118,142,.35); border-left: 3px solid var(--z-danger); background: rgba(247,118,142,.07); color: #f5c1cb; }

/* sources */
.sources { margin-top: 18px; }
.sources h3 { display: flex; align-items: center; gap: 8px; margin: 0 0 10px; font-size: 11px; font-weight: 600;
  letter-spacing: .07em; text-transform: uppercase; color: var(--z-muted); }
.sources h3 span { font-family: var(--z-mono); letter-spacing: 0; color: var(--z-faint); font-weight: 400; }
.src-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }
.src-card { display: flex; flex-direction: column; min-width: 0; border: 1px solid var(--z-border); border-radius: 12px;
  padding: 12px 14px; background: rgba(var(--z-panel),.55); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  scroll-margin: 80px 0 200px; transition: border-color .2s ease, box-shadow .2s ease, opacity .2s ease; }
.src-card:target { border-color: var(--z-accent-line); box-shadow: 0 0 0 3px rgba(var(--z-accent-rgb),.1); opacity: 1; }
.src-card.uncited { opacity: .55; } .src-card.uncited:hover { opacity: .85; }
.src-head { display: flex; align-items: flex-start; gap: 8px; }
.src-tag { flex: none; font-family: var(--z-mono); font-size: 10.5px; font-weight: 600; padding: 3px 5px; border-radius: 5px; line-height: 1;
  margin-top: 1px; color: var(--z-accent); background: var(--z-accent-soft); }
.src-title { flex: 1; min-width: 0; font-size: 13px; font-weight: 600; line-height: 1.35; color: var(--z-text);
  display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
.src-badge { flex: none; font-size: 10.5px; color: var(--z-faint); border: 1px solid var(--z-border-strong); border-radius: 999px; padding: 1px 7px; }
.src-preview, .src-full p { margin: 8px 0 0; font-size: 12.5px; line-height: 1.55; color: var(--z-soft); }
.src-card:has(.src-full[open]) .src-preview { display: none; }
.src-full summary { cursor: pointer; margin-top: 6px; font-size: 12px; color: var(--z-muted); }
.src-full summary:hover { color: var(--z-text); }
.src-foot { margin-top: auto; padding-top: 10px; display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap;
  font-family: var(--z-mono); font-size: 11px; color: var(--z-faint); }
.src-foot a { display: inline-flex; align-items: center; gap: 4px; color: var(--z-muted); text-decoration: none; font-family: FONT; font-size: 12px; }
.src-foot a:hover { color: var(--z-accent); }
.ico { width: 12px; height: 12px; }

.meta { display: flex; align-items: center; flex-wrap: wrap; gap: 6px 14px; margin-top: 14px;
  font-family: var(--z-mono); font-size: 11px; color: var(--z-faint); }
.meta .dl { display: inline-flex; align-items: center; gap: 5px; color: var(--z-muted); text-decoration: none; font-family: FONT; font-size: 12px; }
.meta .dl:hover { color: var(--z-accent); }

/* composer dock */
#dock { position: sticky; bottom: 0; z-index: 20; width: 100%; max-width: calc(var(--z-col) + 44px); margin: 0 auto;
  padding: 0 22px calc(14px + env(safe-area-inset-bottom)) !important; gap: 0 !important; transition: padding .45s ease; }
body:not(:has(.turn)) #dock { padding-bottom: calc(9vh + env(safe-area-inset-bottom)) !important; }
body:has(.turn) #dock::before { content: ""; position: absolute; top: -40px; bottom: 0; left: 50%; width: 100vw; transform: translateX(-50%);
  z-index: -1; pointer-events: none; background: linear-gradient(rgba(5,11,9,0), rgba(5,11,9,.94) 46px); }
#chips { order: 2; display: flex !important; flex-wrap: wrap; justify-content: center; gap: 9px !important; padding: 16px 2px 2px; }
body:has(.turn) #dock #chips { order: 0; flex-wrap: nowrap !important; justify-content: flex-start !important; overflow-x: auto; padding: 10px 2px !important; scrollbar-width: none; }
body:has(.turn) #dock #chips::-webkit-scrollbar { display: none; }
#composer { order: 1; }
#hint { order: 3; }
button.chip { flex: none !important; width: auto !important; min-width: 0 !important; gap: 7px; display: inline-flex; align-items: center;
  border: 1px solid var(--z-border-strong) !important; background: rgba(var(--z-panel),.55) !important; color: var(--z-soft) !important;
  font-size: 12.5px !important; font-weight: 400 !important; border-radius: 999px !important; padding: 7px 14px !important; box-shadow: none !important;
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); transition: color .15s ease, border-color .15s ease, transform .15s ease; }
button.chip::before { content: ""; width: 14px; height: 14px; flex: none; background: currentColor; }
button.chip:hover:not(:disabled) { color: #fff !important; border-color: var(--z-accent-line) !important; transform: translateY(-1px); }
button.chip:focus:not(:focus-visible) { border-color: var(--z-border-strong) !important; outline: none; }
body:has(.turn) #dock #chips { -webkit-mask: linear-gradient(90deg, #000 calc(100% - 48px), transparent); mask: linear-gradient(90deg, #000 calc(100% - 48px), transparent); }
button.chip:disabled { opacity: .5; }
CHIP_ICONS

#composer { background: rgba(10,24,19,.8) !important; border: 1px solid var(--z-border-strong) !important; border-radius: 18px !important;
  padding: 6px 6px 6px 16px !important; gap: 0 !important; box-shadow: 0 12px 40px rgba(0,0,0,.35);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); transition: border-color .18s ease, box-shadow .18s ease; }
#composer:focus-within { border-color: var(--z-accent-line) !important; box-shadow: 0 12px 40px rgba(0,0,0,.35), 0 0 0 3px rgba(var(--z-accent-rgb),.08); }
#composer .block, #composer .form { background: none !important; border: 0 !important; box-shadow: none !important; padding: 0 !important; }
#q textarea { background: none !important; border: 0 !important; box-shadow: none !important; outline: none !important; resize: none;
  color: var(--z-text) !important; font-family: FONT !important; font-size: 15px !important; line-height: 1.5 !important; padding: 9px 8px 4px 0 !important; }
#q textarea::placeholder { color: var(--z-faint); }
#q textarea:disabled { opacity: .55; }
#bar { align-items: center !important; gap: 8px !important; flex-wrap: nowrap !important; padding: 4px 0 2px; }
#bar > * { min-width: 0 !important; flex: none !important; width: auto !important; margin: 0 !important; }
#bar .spacer { flex: 1 !important; }

/* Answer / Search only: segmented control */
#mode .wrap { display: flex !important; gap: 0 !important; background: rgba(255,255,255,.05); border: 1px solid var(--z-border);
  border-radius: 9px; padding: 2px; }
#mode label { margin: 0 !important; padding: 5px 12px !important; border: 0 !important; border-radius: 7px !important; cursor: pointer;
  background: none !important; box-shadow: none !important; color: var(--z-muted) !important; font-size: 12px !important; font-weight: 600;
  transition: color .18s ease, background .18s ease; outline: 1px solid transparent; }
#mode label input { display: none !important; }
#mode label span { margin: 0 !important; padding: 0 !important; }
#mode label.selected, #mode label:has(input:checked) { color: var(--z-text) !important; background: rgba(255,255,255,.09) !important; outline-color: var(--z-border-strong); }
#mode label:hover { color: var(--z-text) !important; }

/* settings: excerpt count + rerank */
#k { width: 124px !important; }
#k .wrap { background: rgba(255,255,255,.05) !important; border: 1px solid var(--z-border) !important; border-radius: 9px !important;
  box-shadow: none !important; padding: 0 !important; min-height: 0 !important; }
#k .wrap:hover { border-color: var(--z-border-strong) !important; }
#k .wrap-inner, #k .secondary-wrap { background: none !important; border: 0 !important; box-shadow: none !important; padding: 0 !important; min-height: 0 !important; }
#k input { color: var(--z-muted) !important; font-size: 12px !important; font-weight: 600 !important; padding: 5px 0 5px 10px !important;
  height: auto !important; margin: 0 !important; cursor: pointer; background: none !important; }
#k .wrap:hover input { color: var(--z-text) !important; }
#k .icon-wrap { color: var(--z-muted); width: 12px !important; margin: 0 8px 0 2px !important; right: auto !important; position: static !important; }
ul.options { background: #0c1d17 !important; border: 1px solid var(--z-border-strong) !important; border-radius: 10px !important; }
ul.options li { color: var(--z-soft) !important; font-size: 12.5px !important; }
ul.options li.selected, ul.options li:hover { background: rgba(255,255,255,.07) !important; color: #fff !important; }
#rerank label { display: inline-flex; align-items: center; gap: 6px; margin: 0; padding: 5px 10px; border-radius: 9px; cursor: pointer;
  background: rgba(255,255,255,.05) !important; border: 1px solid var(--z-border) !important; color: var(--z-muted) !important;
  font-size: 12px !important; font-weight: 600; }
#rerank label:hover { color: var(--z-text) !important; }
#rerank label > span, #rerank label span { padding: 0 !important; margin: 0 !important; color: inherit !important; font-size: 12px !important; font-weight: 600 !important; }
#rerank input { width: 13px !important; height: 13px !important; margin: 0 !important; accent-color: var(--z-accent);
  background-color: transparent !important; border-color: var(--z-border-strong) !important; }
#rerank input:checked { background-color: var(--z-accent) !important; border-color: var(--z-accent) !important; }

button#send { flex: none !important; width: 34px !important; min-width: 34px !important; height: 34px; padding: 0 !important;
  border: 0 !important; border-radius: 50% !important; color: transparent !important; font-size: 0 !important; overflow: hidden; position: relative;
  background: linear-gradient(180deg, #a6f8cf, var(--z-accent)) !important; box-shadow: none !important;
  transition: transform .15s ease, box-shadow .2s ease, opacity .15s ease; }
button#send::after { content: ""; position: absolute; inset: 0; margin: auto; width: 16px; height: 16px; background: #04261a;
  -webkit-mask: ICON_SEND center / contain no-repeat; mask: ICON_SEND center / contain no-repeat; }
button#send:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(var(--z-accent-rgb),.3) !important; }
button#send:active:not(:disabled) { transform: scale(.94); }
button#send:disabled { opacity: 1; cursor: default; animation: throb 1.4s ease-in-out infinite; }

.hint { text-align: center; color: var(--z-faint); font-size: 11.5px; padding-top: 9px; margin: 0; }

@media (max-width: 600px) {
  #topbar { padding: 10px 16px !important; }
  .brand .tag { display: none; }
  #thread { padding: 18px 16px 8px; }
  #dock { padding-left: 16px !important; padding-right: 16px !important; }
  .hero h1 { font-size: 34px; }
  .bubble { max-width: 92%; }
  .msg.bot { gap: 9px; }
  .avatar { width: 26px; height: 26px; }
  #bar { flex-wrap: wrap !important; justify-content: flex-start !important; row-gap: 6px !important; }
  #bar .spacer { display: none !important; }
  button#send { margin-left: auto !important; }
  #k { width: 112px !important; }
  #mode label { padding: 5px 10px !important; }
  #thread:has(.turn) ~ #dock #chips { display: none !important; }
  #hint { display: none !important; }
}

@keyframes fadein { from { opacity: 0; } to { opacity: 1; } }
@keyframes rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes dotpulse { 0%, 100% { opacity: .4; transform: scale(.85); } 50% { opacity: 1; transform: none; } }
@keyframes bounce { 0%, 60%, 100% { transform: none; opacity: .5; } 30% { transform: translateY(-5px); opacity: 1; } }
@keyframes throb { 0%, 100% { box-shadow: 0 0 0 0 rgba(var(--z-accent-rgb),.35); } 50% { box-shadow: 0 0 0 8px rgba(var(--z-accent-rgb),0); } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
"""
    .replace("FONT", FONT)
    .replace("LOGO_URL", f'url("data:image/svg+xml,{quote(LOGO.read_text(encoding="utf-8"))}")')
    # served by Gradio from the static path set in build_app
    .replace("BG_URL", f'url("/gradio_api/file={quote(str(BACKGROUND))}")')
    .replace("ICON_PLUS", _icon_mask("plus"))
    .replace("ICON_SEND", _icon_mask("send"))
    .replace(
        "CHIP_ICONS",
        "\n".join(
            f"button.chip-{i}::before {{ -webkit-mask: {_icon_mask(icon)} center / contain no-repeat; "
            f"mask: {_icon_mask(icon)} center / contain no-repeat; }}"
            for i, (_, _, icon) in enumerate(EXAMPLES)
        ),
    )
)

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/"

# Runs once when the thread mounts. Gradio re-renders the thread by diffing
# the DOM, which resets <details> to what the server sent, so the reader's own
# open/close choices are re-applied after each update. Also typesets math and
# scrolls a new question to the top of the view.
THREAD_JS = """
const chosen = new Map();
let shown = 0;
element.addEventListener('click', (e) => {
  const summary = e.target.closest('summary');
  if (summary && summary.parentElement.id) chosen.set(summary.parentElement.id, !summary.parentElement.open);
}, true);
function typeset() {
  if (!window.katex) return;
  for (const el of element.querySelectorAll('.math')) {
    if (el.querySelector('.katex')) continue;
    try { katex.render(el.textContent, el, { displayMode: el.classList.contains('display'), throwOnError: false }); } catch (_) {}
  }
}
function sync() {
  const turns = element.querySelectorAll('.turn');
  if (!turns.length) chosen.clear();
  for (const [id, open] of chosen) {
    const d = element.querySelector('#' + CSS.escape(id));
    if (d) d.open = open;
  }
  typeset();
  if (turns.length > shown) turns[turns.length - 1].scrollIntoView({ behavior: 'smooth', block: 'start' });
  shown = turns.length;
}
if (!window.katex && !document.getElementById('katex-js')) {
  const css = document.createElement('link');
  css.rel = 'stylesheet'; css.href = 'KATEXkatex.min.css';
  const js = document.createElement('script');
  js.id = 'katex-js'; js.src = 'KATEXkatex.min.js'; js.onload = typeset;
  document.head.append(css, js);
}
watch('value', sync);
sync();
""".replace("KATEX", KATEX)

# The tab icon (served by Gradio from favicon_path), and new questions scroll
# to just under the sticky top bar instead of behind it.
HEAD = (
    '<link rel="icon" type="image/svg+xml" href="favicon.ico">'
    "<style>html { scroll-padding-top: 64px; }</style>"
)


def make_theme():
    """Dark in both browser modes, so Gradio's own controls match the page."""
    import gradio as gr

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=[f.strip().strip('"') for f in FONT.split(",")],
        font_mono=["ui-monospace", "SF Mono", "Menlo", "Consolas", "monospace"],
    )
    values = {
        "body_background_fill": "#060d0b",
        "body_text_color": "#e7efeb",
        "body_text_color_subdued": "#8a9c94",
        "background_fill_primary": "transparent",
        "background_fill_secondary": "rgba(255,255,255,.04)",
        "block_background_fill": "transparent",
        "block_border_width": "0px",
        "block_shadow": "none",
        "border_color_primary": "rgba(255,255,255,.08)",
        "input_background_fill": "transparent",
        "input_border_color": "rgba(255,255,255,.08)",
        "input_shadow": "none",
        "input_shadow_focus": "none",
        "color_accent": "#4ee6a0",
        "color_accent_soft": "rgba(78,230,160,.12)",
        "checkbox_label_background_fill": "transparent",
        "checkbox_label_background_fill_selected": "transparent",
        "checkbox_label_text_color": "#8a9c94",
        "checkbox_label_text_color_selected": "#e7efeb",
        "layout_gap": "0px",
    }
    accepted = inspect.signature(theme.set).parameters
    dark = {f"{k}_dark": v for k, v in values.items() if f"{k}_dark" in accepted}
    return theme.set(**values, **dark)


# ------------------------------------------------------------------- layout


def build_app(backend=None, papers=None):
    """The Gradio Blocks app. Pass theme/css/head to `launch` (Gradio 6):
    `build_app().launch(theme=make_theme(), css=CSS, head=HEAD)`."""
    import gradio as gr

    backend = backend or Backend()
    if papers is None:
        from src.data_io import load_papers

        papers = load_papers()

    def respond(question, mode, k, rerank, turns):
        question = (question or "").strip()
        if not question:
            yield gr.skip(), gr.skip(), gr.skip()
            return
        turn = Turn(n=len(turns) + 1, question=question, generate=mode != MODE_SEARCH, k=int(k), rerank=bool(rerank))
        turns = [*turns, turn]
        for _ in run_turn(backend, turn):
            yield render_thread(turns, papers), turns, ""

    gr.set_static_paths([ASSETS])  # the background photo
    with gr.Blocks(title=APP_TITLE, analytics_enabled=False) as app:
        gr.HTML('<div class="photo"></div><div class="scrim"></div><div class="grain"></div>', elem_id="bg", apply_default_css=False)
        with gr.Row(elem_id="topbar"):
            gr.HTML(
                '<div class="brand"><span class="logo" aria-hidden="true"></span><b>Zayi</b>'
                '<span class="tag">Research assistant</span></div>',
                apply_default_css=False,
            )
            new_chat = gr.Button("New chat", elem_id="new-chat", scale=0)
        turns = gr.State([])
        thread = gr.HTML(render_thread([], papers), elem_id="thread", js_on_load=THREAD_JS, apply_default_css=False)
        with gr.Column(elem_id="dock"):
            with gr.Row(elem_id="chips"):
                chips = []
                for i, (label, text, _) in enumerate(EXAMPLES):
                    chip = gr.Button(label, elem_classes=["chip", f"chip-{i}"], scale=0)
                    # fill the box in the browser; no server round trip
                    chip.click(None, outputs=None, js=f"() => {{ const q = document.querySelector('#q textarea'); "
                               f"q.value = {json.dumps(text)}; q.dispatchEvent(new Event('input', {{ bubbles: true }})); q.focus(); }}")
                    chips.append(chip)
            with gr.Column(elem_id="composer"):
                question = gr.Textbox(
                    show_label=False,
                    placeholder="Ask about the papers…",
                    lines=1,  # Enter sends, Shift+Enter adds a line
                    max_lines=6,
                    autofocus=True,
                    container=False,
                    elem_id="q",
                )
                with gr.Row(elem_id="bar"):
                    mode = gr.Radio([MODE_ANSWER, MODE_SEARCH], value=MODE_ANSWER, show_label=False, container=False, elem_id="mode")
                    k = gr.Dropdown(
                        [(f"{n} excerpts", n) for n in range(3, 11)], value=DEFAULT_K,
                        show_label=False, container=False, elem_id="k", filterable=False,
                    )
                    rerank = gr.Checkbox(value=True, label="Rerank", container=False, elem_id="rerank")
                    gr.HTML("", elem_classes="spacer", apply_default_css=False)
                    send = gr.Button("Send", elem_id="send", scale=0)
            gr.HTML(
                '<p class="hint">Enter to send · Shift+Enter for a new line · Search only skips the API call</p>',
                elem_id="hint",
                apply_default_css=False,
            )

        # Locked while a question runs, so Enter can't start a second one.
        controls = [question, send, new_chat, *chips]

        def lock():
            return [gr.update(interactive=False)] * len(controls)

        def unlock():
            return [gr.update(interactive=True)] * len(controls)

        for trigger in (send.click, question.submit):
            (
                trigger(lock, outputs=controls, queue=False, show_progress="hidden")
                .then(
                    respond,
                    inputs=[question, mode, k, rerank, turns],
                    outputs=[thread, turns, question],
                    show_progress="hidden",
                    concurrency_id="query",  # both triggers share one slot:
                    concurrency_limit=1,  # questions run one at a time, never interleaved
                )
                .then(unlock, outputs=controls, queue=False, show_progress="hidden")
            )
        new_chat.click(lambda: ([], render_thread([], papers)), outputs=[turns, thread], queue=False, show_progress="hidden")
    return app
