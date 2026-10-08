import pytest

from auditix.llm import retry
from auditix.llm.retry import call_groq_with_retry, parse_wait_seconds


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Rate limit reached. Please try again in 2.5s.", 3.5),
        ("Please try again in 800ms.", 1.8),
        ("no hint here", 11.0),  # default 10s + 1s padding
    ],
)
def test_parse_wait_seconds(message, expected):
    assert parse_wait_seconds(message) == pytest.approx(expected)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


def test_success_returns_immediately(monkeypatch):
    calls = []
    monkeypatch.setattr(retry.requests, "post", lambda *a, **k: calls.append(1) or FakeResponse({"choices": [1]}))
    sleeps = []
    result = call_groq_with_retry({"model": "x"}, api_key="k", sleep=sleeps.append)
    assert result == {"choices": [1]}
    assert len(calls) == 1 and sleeps == []


def test_rate_limit_waits_then_succeeds(monkeypatch):
    responses = iter([
        FakeResponse({"error": {"code": "rate_limit_exceeded", "message": "try again in 1.2s."}}),
        FakeResponse({"choices": ["ok"]}),
    ])
    monkeypatch.setattr(retry.requests, "post", lambda *a, **k: next(responses))
    sleeps = []
    result = call_groq_with_retry({}, api_key="k", sleep=sleeps.append)
    assert result == {"choices": ["ok"]}
    assert sleeps == [pytest.approx(2.2)]


def test_non_rate_limit_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(retry.requests, "post",
                        lambda *a, **k: FakeResponse({"error": {"code": "invalid_api_key", "message": "bad"}}))
    sleeps = []
    result = call_groq_with_retry({}, api_key="k", sleep=sleeps.append)
    assert result["error"]["code"] == "invalid_api_key"
    assert sleeps == []


def test_missing_key_returns_error_without_calling_api(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GROK_API_KEY", raising=False)
    monkeypatch.setattr(retry.requests, "post", lambda *a, **k: pytest.fail("should not call API"))
    result = call_groq_with_retry({})
    assert "error" in result
