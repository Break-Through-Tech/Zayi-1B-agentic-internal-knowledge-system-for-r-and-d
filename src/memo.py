"""Structured, cited triage memos from the paper excerpts (November, Task 6).

The project brief's end product is a "Draft Triage Memo": a structured
technical memo whose every claim cites the excerpts it came from, checked by
a reviewing engineer before it is used. This module drafts that memo with the
same retrieval, [S#] labels, retry and citation check as `src.rag`, then runs
automatic checks (all sections present, every citation points at a retrieved
excerpt, every section cited or marked as not covered) so the reviewer sees
problems before reading. The human review step lives in the Gradio app.

One LLM call per memo, like a normal answer.
"""

import re
from datetime import datetime

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.rag import (
    CITATION_GROUP,
    _check_question,
    cited_numbers,
    format_context,
    invoke_with_retry,
    page_label,
    source_label,
    sources_from,
    validate_citations,
)

# Fixed order. The model is told to use exactly these headings.
MEMO_SECTIONS = (
    "Bottom line",
    "Key findings",
    "How it works",
    "Evidence and results",
    "Limitations and open questions",
)

# What a section must say when the excerpts don't cover it (so a reviewer
# can tell "nothing to report" from "the model skipped it").
NOT_COVERED = "Not covered in the retrieved excerpts."

NO_CONTEXT_MEMO = (
    "No relevant excerpts were found in the paper collection for this topic, "
    "so no memo was drafted."
)

MEMO_SYSTEM_PROMPT = f"""\
You are a research assistant for Zayi's R&D team. Draft a triage memo that \
helps an engineer decide whether the topic below matters for their work, \
using ONLY the numbered paper excerpts provided.

Format (Markdown):
- Use exactly these five headings, in this order, each as a level-2 heading \
("## "): {", ".join(MEMO_SECTIONS)}.
- Bottom line: two or three sentences.
- Key findings: three to five bullet points.
- How it works: the method or architecture, in short paragraphs or bullets.
- Evidence and results: concrete numbers, datasets and comparisons.
- Limitations and open questions: what the papers say is limited or unknown.
- Do not add a title, an introduction, or a list of sources.

Rules:
- Cite the excerpts that support each claim inline with their source labels, \
e.g. [S1] or [S2][S3]. Every bullet and every factual sentence needs one.
- Only cite the [S<number>] labels of the excerpts provided. Bracketed \
numbers inside the excerpt text (like [12]) are the papers' own references \
— never cite those.
- If the excerpts do not cover a section, write exactly: {NOT_COVERED}
- Do not fill gaps with outside knowledge. Be concise and precise."""

MEMO_HUMAN_PROMPT = """\
Excerpts:

{context}

Memo topic: {question}"""

MEMO_PROMPT = ChatPromptTemplate.from_messages(
    [("system", MEMO_SYSTEM_PROMPT), ("human", MEMO_HUMAN_PROMPT)]
)

_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")


def pdf_page_url(paper_id, page):
    """Link to a page of the arXiv PDF (browsers' PDF viewers honour #page=)."""
    return f"https://arxiv.org/pdf/{paper_id}#page={page}"


def _normalize(title):
    return re.sub(r"[^a-z ]", "", title.lower().replace("*", "")).strip()


_SECTION_KEYS = {_normalize(s): s for s in MEMO_SECTIONS}


def split_sections(memo):
    """Map each expected section to its text. Headings are matched loosely
    (any level, case, bold, trailing colon); text under an unknown heading
    stays with the section before it, and text before the first heading is
    returned under the key None."""
    sections, current, lines = {}, None, []
    for line in memo.splitlines():
        match = _HEADING.match(line)
        key = _SECTION_KEYS.get(_normalize(match.group(1))) if match else None
        if key is None:
            lines.append(line)
            continue
        sections[current] = "\n".join(lines).strip()
        current, lines = key, []
    sections[current] = "\n".join(lines).strip()
    if not sections.get(None):
        sections.pop(None, None)
    return sections


def _not_covered(text):
    return _normalize(text).startswith(_normalize(NOT_COVERED))


def check_memo(memo, n_sources):
    """Automatic pre-review checks. Each is {"name", "ok", "detail"}."""
    sections = split_sections(memo)
    missing = [s for s in MEMO_SECTIONS if not sections.get(s)]
    cited = set().union(*map(cited_numbers, CITATION_GROUP.findall(memo)))
    invalid = sorted(n for n in cited if not 1 <= n <= n_sources)
    uncited = [
        s
        for s in MEMO_SECTIONS
        if sections.get(s) and not _not_covered(sections[s]) and not CITATION_GROUP.search(sections[s])
    ]
    not_covered = [s for s in MEMO_SECTIONS if sections.get(s) and _not_covered(sections[s])]
    present = [s for s in MEMO_SECTIONS if sections.get(s)]
    return [
        {
            "name": "All sections present",
            "ok": not missing,
            "detail": f"missing: {', '.join(missing)}" if missing else f"{len(MEMO_SECTIONS)} of {len(MEMO_SECTIONS)}",
        },
        {
            "name": "Citations point at retrieved excerpts",
            "ok": bool(cited) and not invalid,
            "detail": (
                f"{', '.join(source_label(n) for n in invalid)} not among the {n_sources} excerpts"
                if invalid
                else f"{len(cited)} of {n_sources} excerpts cited"
                if cited
                else "no citations"
            ),
        },
        {
            "name": "Every section cited or marked not covered",
            "ok": bool(present) and not uncited,
            "detail": (
                "no sections found"
                if not present
                else f"uncited: {', '.join(uncited)}"
                if uncited
                else f"not covered: {', '.join(not_covered)}"
                if not_covered
                else "all sections cited"
            ),
        },
    ]


def draft_memo(topic, docs, llm, config=None):
    """Draft a triage memo from already-retrieved excerpts.

    Returns {"topic", "memo", "sources", "citations", "checks", "passed"}.
    `memo` is the model's Markdown (the five sections); `checks` are the
    automatic pre-review checks and `passed` is True when all of them pass.
    With no excerpts the LLM is not called and `memo` is NO_CONTEXT_MEMO."""
    topic, docs = _check_question(topic), list(docs)
    if not docs:
        return {
            "topic": topic,
            "memo": NO_CONTEXT_MEMO,
            "sources": [],
            "citations": validate_citations(NO_CONTEXT_MEMO, []),
            "checks": [],
            "passed": False,
        }
    generate = MEMO_PROMPT | llm | StrOutputParser()
    memo = invoke_with_retry(generate, {"question": topic, "context": format_context(docs)}, config).strip()
    sources = sources_from(docs)
    checks = check_memo(memo, len(sources))
    return {
        "topic": topic,
        "memo": memo,
        "sources": sources,
        "citations": validate_citations(memo, sources),
        "checks": checks,
        "passed": all(c["ok"] for c in checks),
    }


def build_memo(topic, retriever=None, llm=None, k=5):
    """One-call convenience: retrieve with the default retriever, then draft."""
    if llm is None:  # first, so a missing key fails before any model loads
        from src.rag import get_llm

        llm = get_llm()
    if retriever is None:
        from src.retriever import get_retriever

        retriever = get_retriever(k=k)
    topic = _check_question(topic)
    return draft_memo(topic, retriever.invoke(topic), llm)


def memo_document(result, when=None, app="Zayi 1B Research Assistant"):
    """The memo as a standalone Markdown file: title, draft notice with the
    check results, the memo, and the sources with links to the PDF pages."""
    when = when or datetime.now()
    topic = " ".join(result["topic"].split())
    n = len(result["sources"])
    failed = [c["name"].lower() for c in result["checks"] if not c["ok"]]
    status = (
        "Automatic checks: all passed."
        if result["checks"] and not failed
        else f"Automatic checks failed: {'; '.join(failed)}."
        if failed
        else ""
    )
    lines = [
        f"# Triage memo: {topic}",
        "",
        "*" + " ".join(filter(None, [
            f"Draft by the {app} on {when:%Y-%m-%d %H:%M} from {n} paper excerpt{'s' if n != 1 else ''}.",
            "Not yet reviewed.",
            status,
        ])) + "*",
        "",
        result["memo"].strip(),
        "",
    ]
    if result["sources"]:
        lines += ["## Sources", ""]
        lines += [
            f"- **[{source_label(s['n'])}]** {s['paper_title']} (arXiv:{s['paper_id']}), "
            f"{page_label(s)}: {pdf_page_url(s['paper_id'], s['page_start'])}"
            for s in result["sources"]
        ]
    return "\n".join(lines) + "\n"
