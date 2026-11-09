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


def _wrapped(code):
    """The exception langchain-google-genai actually raises for a ClientError."""
    from langchain_google_genai.chat_models import _handle_client_error

    try:
        _handle_client_error(ClientError(code, {"error": {"message": "x"}}), {"model": "m"})
    except Exception as wrapped:
        return wrapped
    raise AssertionError("expected the client error to be re-raised")


def test_real_gemini_rate_limit_is_retried():
    # regression: Gemini's 429 reaches us as GoogleRateLimitError, which has
    # no status of its own, so it was never retried
    from src.rag import is_retryable_error

    rate_limited, not_found = _wrapped(429), _wrapped(404)
    assert type(rate_limited).__name__ == "GoogleRateLimitError"
    assert is_retryable_error(rate_limited)
    assert not is_retryable_error(not_found)

    fake_generate = Mock()
    fake_generate.invoke.side_effect = [rate_limited, "ok"]
    with patch.object(invoke_with_retry.retry, "sleep", return_value=None):
        assert invoke_with_retry(fake_generate, {}, None) == "ok"
    assert fake_generate.invoke.call_count == 2


def test_openai_style_status_code_and_plain_errors():
    from src.rag import is_retryable_error

    class APIStatusError(Exception):
        def __init__(self, status_code):
            self.status_code = status_code

    assert is_retryable_error(APIStatusError(503))
    assert not is_retryable_error(APIStatusError(401))
    assert not is_retryable_error(ValueError("bad input"))
