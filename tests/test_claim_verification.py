
from src.rag import extract_claims


def test_extracts_claims_with_individual_citations():
    answer = (
        "LoRA freezes pretrained weights [S1]. "
        "RAG combines retrieval and generation [S2]."
    )

    claims = extract_claims(answer)

    assert len(claims) == 2
    assert claims[0]["source_numbers"] == [1]
    assert claims[1]["source_numbers"] == [2]
    assert "LoRA freezes pretrained weights" in claims[0]["claim"]
    assert "RAG combines retrieval and generation" in claims[1]["claim"]


def test_extracts_grouped_citations():
    answer = "The approach uses multiple research sources [S1, S2-S3]."

    claims = extract_claims(answer)

    assert len(claims) == 1
    assert claims[0]["source_numbers"] == [1, 2, 3]


def test_preserves_uncited_claims():
    answer = "LoRA freezes pretrained weights."

    claims = extract_claims(answer)

    assert len(claims) == 1
    assert claims[0]["source_numbers"] == []


def test_preserves_invalid_source_numbers():
    answer = "This claim references an unavailable source [S99]."

    claims = extract_claims(answer)

    assert len(claims) == 1
    assert claims[0]["source_numbers"] == [99]


def test_empty_answer_returns_no_claims():
    assert extract_claims("") == []
    assert extract_claims("   ") == []


import pytest
from unittest.mock import Mock

from src.rag import verify_claim


@pytest.mark.parametrize(
    "model_response, expected",
    [
        ("supported", "supported"),
        ("contradicted", "contradicted"),
        ("insufficient_evidence", "insufficient_evidence"),
        ("SUPPORTED", "supported"),
        ("uncertain", "insufficient_evidence"),
    ],
)
def test_verify_claim_classifies_model_responses(model_response, expected):
    """Verify that model responses are normalized and validated."""
    from langchain_core.messages import AIMessage

    llm = Mock()
    llm.__or__ = None

    from langchain_core.runnables import RunnableLambda

    fake_llm = RunnableLambda(
        lambda prompt: AIMessage(content=model_response)
    )

    result = verify_claim(
        "LoRA freezes pretrained weights.",
        "LoRA freezes the pretrained model weights.",
        fake_llm,
    )

    assert result == expected


def test_verify_claim_handles_model_failure():
    """Verification failures must not be treated as supported claims."""
    from langchain_core.runnables import RunnableLambda

    def failing_model(prompt):
        raise RuntimeError("Verification service unavailable")

    fake_llm = RunnableLambda(failing_model)

    result = verify_claim(
        "LoRA freezes pretrained weights.",
        "LoRA freezes the pretrained model weights.",
        fake_llm,
    )

    assert result == "insufficient_evidence"


from unittest.mock import patch
from langchain_core.documents import Document

from src.rag import verify_answer_claims


def test_verify_answer_with_supported_claim():
    docs = [
        Document(page_content="LoRA freezes pretrained model weights.")
    ]

    with patch("src.rag.verify_claim", return_value="supported") as mock_verify:
        result = verify_answer_claims(
            "LoRA freezes pretrained weights [S1].",
            docs,
            llm=None,
        )

    assert result["valid"] is True
    assert result["claims"][0]["status"] == "supported"
    mock_verify.assert_called_once()


def test_verify_answer_with_contradicted_claim():
    docs = [
        Document(page_content="LoRA freezes pretrained model weights.")
    ]

    with patch("src.rag.verify_claim", return_value="contradicted"):
        result = verify_answer_claims(
            "LoRA updates every pretrained weight [S1].",
            docs,
            llm=None,
        )

    assert result["valid"] is False
    assert result["claims"][0]["status"] == "contradicted"


def test_verify_answer_with_missing_citation():
    docs = [
        Document(page_content="LoRA freezes pretrained model weights.")
    ]

    with patch("src.rag.verify_claim") as mock_verify:
        result = verify_answer_claims(
            "LoRA freezes pretrained weights.",
            docs,
            llm=None,
        )

    assert result["valid"] is False
    assert result["claims"][0]["status"] == "missing_citation"
    mock_verify.assert_not_called()


def test_verify_answer_with_invalid_citation():
    docs = [
        Document(page_content="LoRA freezes pretrained model weights.")
    ]

    with patch("src.rag.verify_claim") as mock_verify:
        result = verify_answer_claims(
            "LoRA freezes pretrained weights [S99].",
            docs,
            llm=None,
        )

    assert result["valid"] is False
    assert result["claims"][0]["status"] == "invalid_citation"
    mock_verify.assert_not_called()


def test_verify_answer_with_multiple_sources():
    docs = [
        Document(page_content="LoRA freezes pretrained weights."),
        Document(page_content="LoRA trains low-rank adaptation matrices."),
    ]

    with patch("src.rag.verify_claim", return_value="supported") as mock_verify:
        result = verify_answer_claims(
            "LoRA freezes weights and trains adaptation matrices [S1][S2].",
            docs,
            llm=None,
        )

    assert result["valid"] is True
    assert result["claims"][0]["source_numbers"] == [1, 2]

    passage = mock_verify.call_args.args[1]
    assert "freezes pretrained weights" in passage
    assert "low-rank adaptation matrices" in passage


def test_answer_from_docs_runs_optional_verification():
    """Verify that the RAG entry point triggers claim verification."""
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnableLambda
    from src.rag import answer_from_docs

    # Create a test document with all required metadata.
    docs = [
        Document(
            page_content="LoRA freezes pretrained model weights.",
            metadata={
                "paper_title": "LoRA: Low-Rank Adaptation of Large Language Models",
                "paper_id": "2106.09685",
                "page_start": 1,
                "page_end": 1,
            },
        )
    ]

    # Track how many times the simulated LLM is called.
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)

        # First call: Generate the RAG answer.
        if len(calls) == 1:
            return "LoRA freezes pretrained weights [S1]."

        # Second call: Verify the generated claim.
        return "supported"

    llm = RunnableLambda(fake_llm)

    # Generate an answer with automatic verification enabled.
    result = answer_from_docs(
        "What does LoRA freeze?",
        docs,
        llm,
        verify=True,
    )

    # Check that the answer was generated correctly.
    assert result["answer"] == "LoRA freezes pretrained weights [S1]."

    # Check that the claim was verified successfully.
    assert result["verification"]["valid"] is True
    assert result["verification"]["claims"][0]["status"] == "supported"

    # Confirm one generation call and one verification call.
    assert len(calls) == 2
