from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.graph import AgentAnswer, build_graph

_GRAPH_MODULE = "backend.agent.graph"
_TOOLS_MODULE = "backend.agent.tools"


def _mock_llm(tool_llm_responses: list, final_answer: AgentAnswer) -> MagicMock:
    """Build a get_llm() replacement wired the same way build_graph() chains it.

    tool_llm_responses are returned in order across successive agent_node
    calls (one AIMessage per loop iteration); final_answer is what the
    generate_structured_answer step returns, regardless of how many tool
    loops happened first.
    """
    mock_llm = MagicMock()

    tool_llm = MagicMock()
    tool_llm.invoke.side_effect = tool_llm_responses
    mock_llm.bind_tools.return_value.with_retry.return_value = tool_llm

    structured_llm = MagicMock()
    structured_llm.invoke.return_value = final_answer
    mock_llm.with_structured_output.return_value.with_retry.return_value = structured_llm

    return mock_llm


def test_no_tool_call_routes_straight_to_final_answer():
    final_msg = AIMessage(content="No tool needed.")
    final_answer = AgentAnswer(answer="A plain answer.", citations=[], high_risk=False)

    with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg], final_answer)):
        graph = build_graph()
        result = graph.invoke({"messages": [HumanMessage("hello")]})

    assert result["structured_response"].answer.startswith("A plain answer.")


def test_tool_call_loops_back_before_final_answer():
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "resolve_drug_name", "args": {"name": "metformin"}, "id": "1", "type": "tool_call"}],
    )
    final_msg = AIMessage(content="Done reasoning.")
    final_answer = AgentAnswer(answer="Resolved and answered.", citations=[], high_risk=False)
    resolution = {"rxcui": "6809", "match_type": "exact", "candidates": []}

    with (
        patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([tool_call_msg, final_msg], final_answer)),
        patch(f"{_TOOLS_MODULE}.resolve_query_drug", return_value=resolution) as mock_resolve,
    ):
        graph = build_graph()
        result = graph.invoke({"messages": [HumanMessage("what is metformin's rxcui?")]})

    mock_resolve.assert_called_once_with("metformin")
    assert result["structured_response"].answer.startswith("Resolved and answered.")
    # tool call, its result, and both agent responses should all be in the trace
    assert any(getattr(m, "tool_calls", None) for m in result["messages"])
    assert any(type(m).__name__ == "ToolMessage" for m in result["messages"])


class TestPostProcessing:
    def test_disclaimer_always_appended(self):
        final_msg = AIMessage(content="done")
        final_answer = AgentAnswer(answer="Some facts.", citations=[], high_risk=False)

        with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg], final_answer)):
            graph = build_graph()
            result = graph.invoke({"messages": [HumanMessage("hello")]})

        assert "informational purposes only" in result["structured_response"].answer

    def test_pharmacist_routing_added_when_high_risk(self):
        final_msg = AIMessage(content="done")
        final_answer = AgentAnswer(answer="Some facts.", citations=[], high_risk=True)

        with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg], final_answer)):
            graph = build_graph()
            result = graph.invoke({"messages": [HumanMessage("hello")]})

        assert "consult your pharmacist or doctor" in result["structured_response"].answer.lower()

    def test_pharmacist_routing_absent_when_not_high_risk(self):
        final_msg = AIMessage(content="done")
        final_answer = AgentAnswer(answer="Some facts.", citations=[], high_risk=False)

        with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg], final_answer)):
            graph = build_graph()
            result = graph.invoke({"messages": [HumanMessage("hello")]})

        assert "high-risk topic" not in result["structured_response"].answer.lower()


class TestAsk:
    def test_builds_graph_once_and_reuses_it(self, monkeypatch):
        import backend.agent.graph as graph_module

        monkeypatch.setattr(graph_module, "_graph", None)
        fake_graph = MagicMock()
        fake_graph.invoke.return_value = {
            "structured_response": AgentAnswer(answer="hi", citations=[], high_risk=False)
        }

        with patch(f"{_GRAPH_MODULE}.build_graph", return_value=fake_graph) as mock_build:
            graph_module.ask("first query")
            graph_module.ask("second query")

        mock_build.assert_called_once()
        assert fake_graph.invoke.call_count == 2
