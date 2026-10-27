from unittest.mock import Mock, patch

from google.genai.errors import ClientError
from src.rag import invoke_with_retry


def test_retries_after_rate_limit():
    """Verify that a 429 error triggers a retry."""

    rate_limit_error = ClientError(
        429,
        {"error": {"message": "Rate limit exceeded"}},
    )

    fake_generate = Mock()
    fake_generate.invoke.side_effect = [
        rate_limit_error,
        "Connection successful",
    ]

    # Skip waiting during the test.
    with patch.object(invoke_with_retry.retry, "sleep", return_value=None):
        result = invoke_with_retry(
            fake_generate,
            {"question": "Test question"},
            None,
        )

    assert result == "Connection successful"
    assert fake_generate.invoke.call_count == 2


def test_no_retry_for_permanent_error():
    """Verify that a 404 error does not trigger retries."""

    not_found_error = ClientError(
        404,
        {"error": {"message": "Model not found"}},
    )

    fake_generate = Mock()
    fake_generate.invoke.side_effect = not_found_error

    try:
        invoke_with_retry(
            fake_generate,
            {"question": "Test question"},
            None,
        )
    except ClientError:
        pass
    else:
        raise AssertionError("Expected a ClientError")

    assert fake_generate.invoke.call_count == 1