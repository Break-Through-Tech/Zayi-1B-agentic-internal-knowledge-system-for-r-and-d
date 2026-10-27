from langchain_core.documents import Document

from src.rag import format_context, sources_from, validate_citations

def test_valid_answer_citations():
    sources = [{"n": 1}, {"n": 2}]

    result = validate_citations(
        "The model uses retrieval [S1] and grounding [S2].",
        sources,
    )

    assert result["valid"] is True
    assert result["cited_sources"] == [1, 2]
    assert result["invalid_sources"] == []


def test_invalid_answer_citations():
    sources = [{"n": 1}, {"n": 2}]

    result = validate_citations(
        "The finding is supported by [S1] and [S5].",
        sources,
    )

    assert result["valid"] is False
    assert result["invalid_sources"] == [5]

def make_doc(title, paper_id, page_start, page_end, chunk_id, text):
    return Document(
        page_content=text,
        metadata={
            "paper_title": title,
            "paper_id": paper_id,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_id": chunk_id,
        },
    )


def test_citation_numbers_match_sources():
    docs = [
        make_doc("Paper A", "2401.00001", 2, 3, "chunk-1", "First passage."),
        make_doc("Paper B", "2401.00002", 5, 6, "chunk-2", "Second passage."),
    ]

    context = format_context(docs)
    sources = sources_from(docs)

    assert "[S1]" in context
    assert "[S2]" in context
    assert len(sources) == 2
    assert [source["n"] for source in sources] == [1, 2]
    assert sources[0]["paper_title"] == "Paper A"
    assert sources[1]["paper_title"] == "Paper B"


def test_source_metadata_is_preserved():
    docs = [
        make_doc(
            "Research Paper",
            "2401.12345",
            4,
            7,
            "chunk-42",
            "Important research finding.",
        )
    ]

    sources = sources_from(docs)

    assert sources[0]["paper_id"] == "2401.12345"
    assert sources[0]["page_start"] == 4
    assert sources[0]["page_end"] == 7
    assert sources[0]["chunk_id"] == "chunk-42"


def test_empty_documents_return_empty_sources():
    assert sources_from([]) == []
    assert format_context([]) == ""