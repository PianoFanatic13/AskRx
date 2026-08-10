import re
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.agent.graph import AgentAnswer, Citation
from backend.api.main import app

_API_MODULE = "backend.api.main"
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

client = TestClient(app)


def test_thread_id_generated_when_omitted():
    final_answer = AgentAnswer(answer="An answer.", citations=[], high_risk=False)

    with patch(f"{_API_MODULE}.ask", return_value=final_answer) as mock_ask:
        response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert _UUID_RE.match(body["thread_id"])
    mock_ask.assert_called_once_with("hello", body["thread_id"])


def test_thread_id_passthrough_when_provided():
    final_answer = AgentAnswer(answer="An answer.", citations=[], high_risk=False)

    with patch(f"{_API_MODULE}.ask", return_value=final_answer) as mock_ask:
        response = client.post("/chat", json={"message": "hello", "thread_id": "my-thread"})

    assert response.status_code == 200
    assert response.json()["thread_id"] == "my-thread"
    mock_ask.assert_called_once_with("hello", "my-thread")


def test_response_shape_matches_chat_response():
    final_answer = AgentAnswer(answer="Metformin can cause nausea.", citations=[], high_risk=False)

    with patch(f"{_API_MODULE}.ask", return_value=final_answer):
        response = client.post("/chat", json={"message": "side effects of metformin"})

    body = response.json()
    assert set(body.keys()) == {"answer", "citations", "high_risk", "thread_id"}
    assert body["answer"] == "Metformin can cause nausea."
    assert body["citations"] == []
    assert body["high_risk"] is False


def test_citations_serialize_correctly():
    citation = Citation(marker=1, setid="abc-123", loinc_code="34084-4", section_title_path="ADVERSE REACTIONS")
    final_answer = AgentAnswer(answer="Some facts. [1]", citations=[citation], high_risk=False)

    with patch(f"{_API_MODULE}.ask", return_value=final_answer):
        response = client.post("/chat", json={"message": "hello"})

    body = response.json()
    assert body["citations"] == [
        {
            "marker": 1,
            "setid": "abc-123",
            "loinc_code": "34084-4",
            "section_title_path": "ADVERSE REACTIONS",
        }
    ]


def test_error_path_returns_503_with_error_body():
    with patch(f"{_API_MODULE}.ask", side_effect=RuntimeError("boom")):
        response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 503
    assert "error" in response.json()


def test_high_risk_true_comes_through():
    final_answer = AgentAnswer(answer="Consult your pharmacist.", citations=[], high_risk=True)

    with patch(f"{_API_MODULE}.ask", return_value=final_answer):
        response = client.post("/chat", json={"message": "hello"})

    assert response.json()["high_risk"] is True


def test_high_risk_false_comes_through():
    final_answer = AgentAnswer(answer="Some facts.", citations=[], high_risk=False)

    with patch(f"{_API_MODULE}.ask", return_value=final_answer):
        response = client.post("/chat", json={"message": "hello"})

    assert response.json()["high_risk"] is False


@pytest.mark.integration
class TestChatEndpointLive:
    def test_real_chat_call(self):
        response = client.post("/chat", json={"message": "What is metformin used for?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert _UUID_RE.match(body["thread_id"])
        assert isinstance(body["citations"], list)
        for citation in body["citations"]:
            assert citation["marker"]
            assert citation["setid"]
            assert citation["section_title_path"]
        assert isinstance(body["high_risk"], bool)
