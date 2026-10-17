from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.rag import RAGResult, RAGPipeline, create_rag_chain, format_context


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def invoke(self, query):
        self.queries.append(query)
        return self.documents


def _llm(seen):
    def respond(prompt):
        seen.append(prompt.to_string())
        return AIMessage(content="Grounded answer [LoRA, p. 4]")

    return RunnableLambda(respond)


def test_format_context_includes_page_citation_labels():
    documents = [
        Document(
            page_content="Low-rank updates reduce trainable parameters.",
            metadata={"paper_title": "LoRA", "page_start": 4, "page_end": 5},
        )
    ]

    context = format_context(documents)

    assert "[1] LoRA, pp. 4-5" in context
    assert "Low-rank updates" in context


def test_pipeline_retrieves_before_generation_and_returns_sources():
    retriever = FakeRetriever(
        [Document(page_content="retrieved fact", metadata={"paper_title": "Paper", "page_start": 2})]
    )
    prompts = []
    pipeline = RAGPipeline(retriever, _llm(prompts))

    result = pipeline.answer("What does the paper find?")

    assert isinstance(result, RAGResult)
    assert result.answer == "Grounded answer [LoRA, p. 4]"
    assert result.source_documents[0].page_content == "retrieved fact"
    assert retriever.queries == ["What does the paper find?"]
    assert "retrieved fact" in prompts[0]
    assert "only the retrieved research-paper context" in prompts[0]


def test_chain_returns_plain_answer_and_summary_uses_retrieval():
    retriever = FakeRetriever(
        [Document(page_content="summary evidence", metadata={"paper_title": "Paper"})]
    )
    prompts = []
    pipeline = RAGPipeline(retriever, _llm(prompts))

    assert pipeline.invoke("question") == "Grounded answer [LoRA, p. 4]"
    assert pipeline.summarize("Paper") == "Grounded answer [LoRA, p. 4]"
    assert retriever.queries[1].startswith("Summarize the research paper")
    chain = create_rag_chain(retriever=retriever, llm=_llm([]))
    assert chain.invoke("question") == "Grounded answer [LoRA, p. 4]"


def test_pipeline_rejects_blank_questions():
    pipeline = RAGPipeline(FakeRetriever([]), _llm([]))

    for method in (pipeline.invoke, pipeline.answer):
        try:
            method(" ")
        except ValueError as exc:
            assert "empty" in str(exc)
        else:
            raise AssertionError("blank questions must be rejected")
