from langchain_core.documents import Document

from src.rag import (
    format_context,
    sources_from,
    validate_citations,
    passage_contains_evidence,
)

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


def test_grouped_citations_are_checked():
    # regression: "[S1, S9]" matched nothing, so a bad label passed as valid
    sources = [{"n": 1}, {"n": 2}]
    for answer in ("Claim [S1, S9].", "Claim [S1; S9].", "Claim [S1,9]."):
        result = validate_citations(answer, sources)
        assert result["valid"] is False, answer
        assert result["invalid_sources"] == [9], answer
    assert validate_citations("Claim [S1, S2].", sources)["cited_sources"] == [1, 2]


def test_citation_ranges_include_every_number():
    # regression: [S1-S3] counted only 1 and 3, so S2 looked uncited and the
    # middle of an out-of-range span was never checked
    sources = [{"n": 1}, {"n": 2}, {"n": 3}]
    assert validate_citations("Claim [S1-S3].", sources) == {
        "valid": True, "cited_sources": [1, 2, 3], "invalid_sources": []
    }
    assert validate_citations("Claim [S2–5].", sources)["invalid_sources"] == [4, 5]
    assert validate_citations("Claim [S1, S2-S3; S1].", sources)["cited_sources"] == [1, 2, 3]


def test_papers_own_reference_numbers_are_not_citations():
    result = validate_citations("BART [32] and DPR [12, 13] as cited in [S2].", [{"n": 1}, {"n": 2}])
    assert result == {"valid": True, "cited_sources": [2], "invalid_sources": []}

def test_uncited_answer_is_flagged_when_sources_exist():
    """A factual answer without citations should not pass validation."""
    sources = [{"n": 1}, {"n": 2}]

    result = validate_citations(
        "The system uses retrieval to improve answer accuracy.",
        sources,
    )

    assert result["valid"] is False
    assert result["cited_sources"] == []

def test_no_context_answer_without_citations_is_valid():
    """No retrieved sources means no citation is required."""
    result = validate_citations(
        "I couldn't find enough information to answer the question.",
        [],
    )

    assert result["valid"] is True
    assert result["cited_sources"] == []
    assert result["invalid_sources"] == []

def test_abstention_with_sources_does_not_require_citations():
    """An explicit abstention should not need a source citation."""
    sources = [{"n": 1}, {"n": 2}]

    result = validate_citations(
        "I don't have enough information to answer this question.",
        sources,
    )

    assert result["valid"] is True
    assert result["cited_sources"] == []
    assert result["invalid_sources"] == []

def test_claim_is_supported_by_cited_passage():
    """Check that evidence terms appear in the cited passage."""
    docs = [
        make_doc(
            "LoRA", "2106.09685", 4, 4, "lora-1",
            "LoRA freezes the pretrained weights."
        ),
        make_doc(
            "RAG", "2005.11401", 2, 3, "rag-1",
            "RAG combines a retriever with a generator."
        ),
    ]

    required_terms = ["lora", "freezes", "pretrained weights"]

    # Correct source contains the expected evidence.
    assert passage_contains_evidence(docs, 1, required_terms)

    # Wrong source does not contain the expected evidence.
    assert not passage_contains_evidence(docs, 2, required_terms)

    # Invalid source numbers should be rejected.
    assert not passage_contains_evidence(docs, 99, required_terms)

    # Empty evidence requirements should be rejected.
    assert not passage_contains_evidence(docs, 1, [])