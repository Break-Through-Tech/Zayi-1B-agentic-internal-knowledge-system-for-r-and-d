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
