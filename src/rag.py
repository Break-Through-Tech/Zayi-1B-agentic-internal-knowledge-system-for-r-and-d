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
import re
from pathlib import Path

from dotenv import dotenv_values
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from google.genai.errors import ServerError
from langchain_google_genai.chat_models import GoogleModelNotFoundError

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

def is_retryable_error(error):
    """Retry temporary API errors, not permanent configuration errors."""
    if isinstance(error, GoogleModelNotFoundError):
        return False

    status = getattr(error, "status_code", None)

    if status is None:
        status = getattr(error, "code", None)

    return status in (429, 500, 502, 503, 504)


@retry(
    retry=retry_if_exception(is_retryable_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    reraise=True,
)
def invoke_with_retry(generate, inputs, config):
    """Invoke Gemini with exponential backoff for temporary failures."""
    return generate.invoke(inputs, config=config)

def validate_citations(answer, sources):
    """Check that cited source labels exist in the retrieved sources."""
    cited_numbers = {
        int(number)
        for number in re.findall(r"\[S(\d+)\]", answer)
    }

    valid_numbers = {source["n"] for source in sources}
    invalid_numbers = sorted(cited_numbers - valid_numbers)

    return {
        "valid": len(invalid_numbers) == 0,
        "cited_sources": sorted(cited_numbers),
        "invalid_sources": invalid_numbers,
    }


def build_rag_chain(retriever, llm):
    """LCEL chain: question str -> {"question", "answer", "sources"}.

    Retrieves once per question; if nothing is retrieved, returns
    NO_CONTEXT_ANSWER without calling the LLM."""
    generate = RAG_PROMPT | llm | StrOutputParser()

    def _answer(inputs, config):
        question, docs = inputs["question"], inputs["docs"]
        if not docs:
            return {"question": question, "answer": NO_CONTEXT_ANSWER, "sources": []}
        answer = invoke_with_retry(
                generate,
               {"question": question, "context": format_context(docs)},
               config,
)
        sources = sources_from(docs)
        citation_check = validate_citations(answer, sources)

        if not citation_check["valid"]:
            raise ValueError(
                f"Invalid citation references: {citation_check['invalid_sources']}"
            )

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

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
