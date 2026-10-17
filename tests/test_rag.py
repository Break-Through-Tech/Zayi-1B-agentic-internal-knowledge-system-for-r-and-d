import os
import socket
from unittest import mock

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from src import rag
from src.rag import (
    NO_CONTEXT_ANSWER,
    build_rag_chain,
    format_context,
    get_llm,
    sources_from,
)


def _doc(chunk_id, paper_id, title, start, end, text):
    return Document(
        page_content=text,
        id=chunk_id,
        metadata={
            "chunk_id": chunk_id,
            "paper_id": paper_id,
            "paper_title": title,
            "page_start": start,
            "page_end": end,
            "similarity": 0.8,
        },
    )


DOCS = [
    _doc("lora-p4-0", "2106.09685", "LoRA", 4, 4, "LoRA freezes the pretrained weights."),
    _doc("rag-p2-1", "2005.11401", "RAG", 2, 3, "RAG combines a retriever with a generator."),
]


class FakeRetriever(BaseRetriever):
    docs: list = []
    calls: list = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ):
        self.calls.append(query)
        return list(self.docs)


class RecordingLLM:
    """Chat-model stand-in that records the prompt messages it receives."""

    def __init__(self, reply="LoRA freezes weights [S1]."):
        self.prompts = []
        self.runnable = RunnableLambda(self._respond)
        self.reply = reply

    def _respond(self, prompt_value):
        self.prompts.append(prompt_value.to_messages())
        return AIMessage(content=self.reply)


def _chain(docs=DOCS, reply="LoRA freezes weights [S1]."):
    retriever = FakeRetriever(docs=docs, calls=[])
    llm = RecordingLLM(reply)
    return build_rag_chain(retriever, llm.runnable), retriever, llm


def test_context_numbering_and_page_labels():
    context = format_context(DOCS)
    assert context == (
        "[S1] LoRA (arXiv:2106.09685), p. 4\nLoRA freezes the pretrained weights.\n\n"
        "[S2] RAG (arXiv:2005.11401), pp. 2–3\nRAG combines a retriever with a generator."
    )


def test_sources_numbered_like_context():
    sources = sources_from(DOCS)
    assert [s["n"] for s in sources] == [1, 2]
    assert sources[1] == {
        "n": 2,
        "paper_title": "RAG",
        "paper_id": "2005.11401",
        "page_start": 2,
        "page_end": 3,
        "chunk_id": "rag-p2-1",
    }
    context = format_context(DOCS)
    for s in sources:
        assert f"[S{s['n']}] {s['paper_title']} (arXiv:{s['paper_id']})" in context


def test_chain_output_shape_and_prompt_contents():
    chain, retriever, llm = _chain()
    result = chain.invoke("  What does LoRA freeze?  ")

    assert result == {
        "question": "What does LoRA freeze?",
        "answer": "LoRA freezes weights [S1].",
        "sources": sources_from(DOCS),
    }
    assert retriever.calls == ["What does LoRA freeze?"]  # retrieved exactly once
    assert len(llm.prompts) == 1
    system, human = llm.prompts[0]
    assert system.type == "system" and "ONLY" in system.content
    assert "What does LoRA freeze?" in human.content
    assert format_context(DOCS) in human.content


def test_works_with_a_real_chat_model_class():
    retriever = FakeRetriever(docs=DOCS, calls=[])
    llm = GenericFakeChatModel(messages=iter(["Answer [S2]."]))
    result = build_rag_chain(retriever, llm).invoke("What is RAG?")
    assert result["answer"] == "Answer [S2]."
    assert len(result["sources"]) == 2


def test_no_docs_skips_llm():
    chain, retriever, llm = _chain(docs=[])
    result = chain.invoke("Something not in the papers?")
    assert result == {
        "question": "Something not in the papers?",
        "answer": NO_CONTEXT_ANSWER,
        "sources": [],
    }
    assert llm.prompts == []
    assert len(retriever.calls) == 1


@pytest.mark.parametrize("question", ["", "   \n", None])
def test_empty_question_rejected(question):
    chain, retriever, llm = _chain()
    with pytest.raises(ValueError, match="non-empty"):
        chain.invoke(question)
    assert retriever.calls == [] and llm.prompts == []


@pytest.fixture()
def clean_env(monkeypatch, tmp_path):
    """No provider env vars, and the repo .env redirected to an empty tmp dir
    so a developer's real keys can't leak into these tests. os.environ is
    restored wholesale afterwards — get_llm writes .env values into it."""
    env_patch = mock.patch.dict(os.environ)
    env_patch.start()
    monkeypatch.setattr(rag, "_from_env_file", {})
    for var in (
        "LLM_PROVIDER",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(rag, "_ENV_PATH", tmp_path / ".env")
    yield tmp_path
    env_patch.stop()


@pytest.fixture()
def no_network(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket, "getaddrinfo", _blocked)


def test_gemini_requires_key(clean_env):
    with pytest.raises(rag.MissingConfigError, match="GOOGLE_API_KEY=.*\\.env"):
        get_llm("gemini")


def test_gemini_builds_offline_with_default_model(clean_env, no_network, monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI

    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-key")
    llm = get_llm()
    assert isinstance(llm, ChatGoogleGenerativeAI)
    assert llm.model == "gemini-3.5-flash"
    # Gemini 3 keeps Google's recommended default unless one is passed
    assert llm.temperature is None
    assert get_llm(temperature=0.3).temperature == 0.3

    assert get_llm("gemini", model="gemini-other").model == "gemini-other"


def test_env_file_is_loaded(clean_env, no_network, monkeypatch):
    (clean_env / ".env").write_text("GEMINI_API_KEY=from-file\nGEMINI_MODEL=gemini-x\n")
    llm = get_llm()
    assert llm.model == "gemini-x"


def test_env_file_edits_apply_without_restart_but_shell_wins(clean_env, no_network, monkeypatch):
    env_file = clean_env / ".env"
    env_file.write_text("GEMINI_API_KEY=typo\nGEMINI_MODEL=gemini-old\n")
    assert get_llm().model == "gemini-old"
    env_file.write_text("GEMINI_API_KEY=fixed\nGEMINI_MODEL=gemini-new\n")
    assert get_llm().model == "gemini-new"
    assert os.environ["GEMINI_API_KEY"] == "fixed"

    env_file.write_text("GEMINI_API_KEY=fixed\n")  # model line removed
    assert get_llm().model == "gemini-3.5-flash"

    monkeypatch.setenv("GEMINI_MODEL", "from-shell")
    env_file.write_text("GEMINI_API_KEY=fixed\nGEMINI_MODEL=from-file\n")
    assert get_llm().model == "from-shell"


def test_env_does_not_leak_between_tests():
    # runs after the .env tests above; their values must be gone
    assert os.environ.get("GEMINI_MODEL") not in {"gemini-x", "gemini-old", "gemini-new", "from-file"}


def test_openrouter_requires_key_and_model(clean_env, no_network, monkeypatch):
    with pytest.raises(rag.MissingConfigError, match="OPENROUTER_API_KEY"):
        get_llm("openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy-key")
    with pytest.raises(rag.MissingConfigError, match="OPENROUTER_MODEL"):
        get_llm("openrouter")

    llm = get_llm("openrouter", model="some/model:free")
    assert llm.model_name == "some/model:free"
    assert str(llm.openai_api_base).startswith("https://openrouter.ai/api/v1")


def test_unknown_provider_rejected(clean_env):
    with pytest.raises(ValueError, match="gemini, openrouter"):
        get_llm("nope")
