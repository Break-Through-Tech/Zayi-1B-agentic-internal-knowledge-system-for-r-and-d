<<<<<<< HEAD
"""Baseline retrieval-augmented generation over the paper knowledge base.

The module keeps retrieval and generation separate so the Gemini client can be
replaced with a fake model in tests or with another LangChain chat model later.
The prompt requires the model to cite retrieved excerpts and to say when the
context does not contain an answer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable

DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_RAG_PROMPT = """You are a research assistant. Answer the user's question using
only the retrieved research-paper context below. Do not use outside knowledge
to fill gaps. If the context is insufficient, say so clearly and identify what
is missing.

Grounding requirements:
- Cite every substantive claim with the citation label included in the context,
  using the form [Paper title, p. N] or [Paper title, pp. N-M].
- Do not invent citations, page numbers, results, or claims.
- Prefer a concise, technically precise answer. Use bullets when they improve
  readability.

Retrieved context:
{context}

User question:
{question}
"""


@dataclass(frozen=True)
class RAGResult:
    """Generated answer together with the exact documents used as context."""

    answer: str
    source_documents: tuple[Document, ...]


def _page_label(document: Document) -> str:
    title = document.metadata.get("paper_title", "Unknown paper")
    start = document.metadata.get("page_start")
    end = document.metadata.get("page_end")
    if start is None:
        return str(title)
    if end is None or end == start:
        return f"{title}, p. {start}"
    return f"{title}, pp. {start}-{end}"


def format_context(documents: Iterable[Document]) -> str:
    """Format retrieved documents with stable, model-visible citation labels."""
    formatted = []
    for index, document in enumerate(documents, start=1):
        formatted.append(
            f"[{index}] {_page_label(document)}\n{document.page_content.strip()}"
        )
    return "\n\n".join(formatted)


def _build_gemini(
    model_name: str = DEFAULT_GEMINI_MODEL,
    *,
    temperature: float = 0.0,
    api_key: str | None = None,
) -> BaseChatModel:
    """Construct Gemini lazily so importing this module never requires a key."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "Gemini support requires langchain-google-genai; install the project "
            "requirements before creating a Gemini pipeline."
        ) from exc

    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise ValueError(
            "A Gemini API key is required. Set GOOGLE_API_KEY or pass api_key."
        )
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=key,
    )


class RAGPipeline:
    """Retriever plus a grounded LangChain generation chain."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel | Runnable | None = None,
        *,
        model_name: str = DEFAULT_GEMINI_MODEL,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm or _build_gemini(
            model_name, temperature=temperature, api_key=api_key
        )
        self.prompt = ChatPromptTemplate.from_template(_RAG_PROMPT)
        self.chain = (
            {
                "context": self._retrieve_and_format,
                "question": lambda question: question,
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _retrieve_and_format(self, question: str) -> str:
        return format_context(self.retriever.invoke(question))

    def invoke(self, question: str) -> str:
        """Retrieve context and return one grounded answer."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        return self.chain.invoke(question)

    def answer(self, question: str) -> RAGResult:
        """Return the answer and the source Documents used to produce it."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        documents = tuple(self.retriever.invoke(question))
        answer_chain = self.prompt | self.llm | StrOutputParser()
        return RAGResult(
            answer=answer_chain.invoke(
                {"context": format_context(documents), "question": question}
            ),
            source_documents=documents,
        )

    def summarize(self, topic: str) -> str:
        """Generate a concise, citation-grounded summary for a topic or paper."""
        if not topic or not topic.strip():
            raise ValueError("topic must not be empty")
        return self.invoke(
            f"Summarize the research paper or topic '{topic}'. Cover its "
            "motivation, method, and key findings only when supported by the "
            "retrieved context."
        )


def create_rag_pipeline(
    retriever: BaseRetriever | None = None,
    llm: BaseChatModel | Runnable | None = None,
    *,
    model_name: str = DEFAULT_GEMINI_MODEL,
    temperature: float = 0.0,
    api_key: str | None = None,
    retriever_kwargs: dict[str, Any] | None = None,
) -> RAGPipeline:
    """Create the baseline pipeline, opening the default paper retriever."""
    if retriever is None:
        from src.retriever import get_retriever

        retriever = get_retriever(**(retriever_kwargs or {}))
    return RAGPipeline(
        retriever,
        llm,
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
    )


def create_rag_chain(*args: Any, **kwargs: Any) -> Runnable:
    """Compatibility helper returning a plain LangChain answer chain."""
    return create_rag_pipeline(*args, **kwargs).chain
=======
"""Baseline RAG pipeline: retrieve paper excerpts, answer with an LLM (Task 13).

Linear on purpose — question -> retriever -> numbered context -> grounded
prompt -> LLM -> answer + sources. The context blocks and the returned
`sources` list are numbered by the SAME function order, so a citation
like [S2] in the answer always points at sources[1]. Labels are `S`-prefixed
because the excerpts themselves are full of the papers' own bracketed
reference numbers ("BART [32]") — plain [n] labels would collide with them.

The chain accepts any LangChain `BaseRetriever` whose Documents carry the
metadata `src.retriever.PaperRetriever` produces (paper_id, paper_title,
page_start, page_end, chunk_id), and any chat model — Gemini by default,
OpenRouter as a fallback provider (see `get_llm`).
"""

import os
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Repo-root .env, resolved from this file so it works from any working dir.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

PROVIDERS = ("gemini", "openrouter")
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Returned without calling the LLM when retrieval finds nothing — there is
# nothing to ground an answer in, so the model is never asked to guess.
NO_CONTEXT_ANSWER = (
    "No relevant excerpts were found in the paper collection for this "
    "question, so it can't be answered from the papers."
)

SYSTEM_PROMPT = """\
You are a research assistant for Zayi's R&D team. Answer the question using \
ONLY the numbered paper excerpts provided.

Rules:
- Cite the excerpts that support each claim inline with their source \
labels, e.g. [S1] or [S2][S3].
- Only cite the [S<number>] labels of the excerpts provided. Bracketed \
numbers inside the excerpt text (like [12]) are the papers' own references \
— never cite those.
- If the excerpts do not contain the answer, say so plainly. Do not fill \
gaps with outside knowledge.
- Be concise and precise."""

HUMAN_PROMPT = """\
Excerpts:

{context}

Question: {question}"""

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
)


def _sampling(temperature):
    # omitted entirely when None, so the client keeps its own default
    return {} if temperature is None else {"temperature": temperature}


# Values this module last copied from .env into os.environ, so a later
# _load_env() can tell "set by us from .env" (refresh) from "set by the
# shell" (leave alone).
_from_env_file = {}


def _load_env():
    """Apply the repo-root .env on every call, so editing it takes effect
    without restarting a notebook kernel. Variables set outside .env (e.g.
    exported in the shell) always win."""
    values = {k: v for k, v in dotenv_values(_ENV_PATH).items() if v is not None}
    for key, previous in list(_from_env_file.items()):
        if key not in values and os.environ.get(key) == previous:
            del os.environ[key]  # removed from .env since we loaded it
            del _from_env_file[key]
    for key, value in values.items():
        if key not in os.environ or os.environ[key] == _from_env_file.get(key):
            os.environ[key] = value
            _from_env_file[key] = value


class MissingConfigError(RuntimeError):
    """An API key or model setting the chosen provider needs is not set."""


def get_llm(provider=None, model=None, temperature=None):
    """Chat model for the configured provider.

    Keys and defaults come from the environment, with a repo-root `.env`
    loaded first (already-set variables win). Provider packages are imported
    lazily so only the one in use needs to be installed. Retries on rate
    limits/transient errors are left to the client libraries' defaults.

    `temperature=None` keeps the provider's default — Google recommends
    leaving Gemini 3 models at theirs, and langchain-google-genai only does
    so when no temperature is passed."""
    _load_env()
    provider = (provider or os.getenv("LLM_PROVIDER") or "gemini").lower()

    if provider == "gemini":
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            raise MissingConfigError(
                "No Gemini API key found. Get one from Google AI Studio "
                "(https://aistudio.google.com/apikey) and add the line "
                "GOOGLE_API_KEY=... to a .env file at the repo root."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or os.getenv("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
            **_sampling(temperature),
        )

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise MissingConfigError(
                "No OpenRouter API key found. Add OPENROUTER_API_KEY=... to a "
                ".env file at the repo root."
            )
        # No default model: OpenRouter's free-model slugs change often.
        model = model or os.getenv("OPENROUTER_MODEL")
        if not model:
            raise MissingConfigError(
                "No OpenRouter model set. Pass a model name or add "
                "OPENROUTER_MODEL=<model slug from openrouter.ai/models> to .env."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
            model=model,
            **_sampling(temperature),
        )

    raise ValueError(
        f"unknown LLM provider {provider!r}; choose one of {', '.join(PROVIDERS)}"
    )


def source_label(n):
    """Citation label for the n-th excerpt: `S1`, `S2`, ..."""
    return f"S{n}"


def page_label(meta):
    """`p. 4` for a single page, `pp. 4–5` for a range."""
    start, end = meta["page_start"], meta["page_end"]
    return f"p. {start}" if start == end else f"pp. {start}–{end}"


def format_context(docs):
    """Numbered excerpt blocks for the prompt, e.g.
    `[S1] <title> (arXiv:<id>), p. 4` then the chunk text."""
    return "\n\n".join(
        f"[{source_label(n)}] {doc.metadata['paper_title']} (arXiv:{doc.metadata['paper_id']}), "
        f"{page_label(doc.metadata)}\n{doc.page_content}"
        for n, doc in enumerate(docs, start=1)
    )


def sources_from(docs):
    """Source list numbered exactly like `format_context` (n = bracket number)."""
    return [
        {
            "n": n,
            "paper_title": doc.metadata["paper_title"],
            "paper_id": doc.metadata["paper_id"],
            "page_start": doc.metadata["page_start"],
            "page_end": doc.metadata["page_end"],
            "chunk_id": doc.metadata.get("chunk_id", doc.id),
        }
        for n, doc in enumerate(docs, start=1)
    ]


def _check_question(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return question.strip()


def build_rag_chain(retriever, llm):
    """LCEL chain: question str -> {"question", "answer", "sources"}.

    Retrieves once per question; if nothing is retrieved, returns
    NO_CONTEXT_ANSWER without calling the LLM."""
    generate = RAG_PROMPT | llm | StrOutputParser()

    def _answer(inputs, config):
        question, docs = inputs["question"], inputs["docs"]
        if not docs:
            return {"question": question, "answer": NO_CONTEXT_ANSWER, "sources": []}
        answer = generate.invoke(
            {"question": question, "context": format_context(docs)}, config
        )
        return {"question": question, "answer": answer, "sources": sources_from(docs)}

    chain = (
        RunnableLambda(_check_question)
        | {"question": RunnablePassthrough(), "docs": retriever}
        | RunnableLambda(_answer)
    )
    return chain.with_config(run_name="rag_chain")


def answer_question(question, retriever=None, llm=None):
    """One-call convenience: answer with the default retriever and LLM."""
    if llm is None:  # first, so a missing key fails before any model loads
        llm = get_llm()
    if retriever is None:
        # Lazy: loads the knowledge base and embedding model only when needed.
        from src.retriever import get_retriever

        retriever = get_retriever()
    return build_rag_chain(retriever, llm).invoke(question)
>>>>>>> 32b3abe4529a00117183e00a218f28176b2f7cca
