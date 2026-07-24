from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from backend.agent.graph import AgentAnswer, build_graph

_GRAPH_MODULE = "backend.agent.graph"
_TOOLS_MODULE = "backend.agent.tools"
_THREAD = "test-thread"


def _mock_llm(tool_llm_responses: list, final_answer: AgentAnswer | list[AgentAnswer]) -> MagicMock:
    """Build a get_llm() replacement wired the same way build_graph() chains it.

    tool_llm_responses are returned in order across successive agent_node
    calls (one AIMessage per loop iteration). final_answer is what the
    generate_structured_answer step returns; pass a list to give successive
    graph.invoke() calls (e.g. separate conversation turns) different final
    answers, or a single AgentAnswer to return the same one every time.
    """
    mock_llm = MagicMock()

    tool_llm = MagicMock()
    tool_llm.invoke.side_effect = tool_llm_responses
    mock_llm.bind_tools.return_value.with_retry.return_value = tool_llm

    structured_llm = MagicMock()
    if isinstance(final_answer, list):
        structured_llm.invoke.side_effect = final_answer
    else:
        structured_llm.invoke.return_value = final_answer
    mock_llm.with_structured_output.return_value.with_retry.return_value = structured_llm

    return mock_llm


def test_no_tool_call_routes_straight_to_final_answer():
    final_msg = AIMessage(content="No tool needed.")
    final_answer = AgentAnswer(answer="A plain answer.", citations=[], high_risk=False)

    with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg], final_answer)):
        graph = build_graph()
        result = graph.invoke(
            {"messages": [HumanMessage("hello")]}, {"configurable": {"thread_id": _THREAD}}
        )

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
        result = graph.invoke(
            {"messages": [HumanMessage("what is metformin's rxcui?")]},
            {"configurable": {"thread_id": _THREAD}},
        )

    mock_resolve.assert_called_once_with("metformin")
    assert result["structured_response"].answer.startswith("Resolved and answered.")
    # tool call, its result, and both agent responses should all be in the trace
    assert any(getattr(m, "tool_calls", None) for m in result["messages"])
    assert any(type(m).__name__ == "ToolMessage" for m in result["messages"])


class TestConversationMemory:
    def test_second_call_with_same_thread_id_sees_prior_turn(self):
        final_msg_1 = AIMessage(content="first response")
        final_msg_2 = AIMessage(content="second response")
        final_answer = AgentAnswer(answer="answer", citations=[], high_risk=False)
        config = {"configurable": {"thread_id": _THREAD}}

        with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg_1, final_msg_2], final_answer)):
            graph = build_graph()
            graph.invoke({"messages": [HumanMessage("first turn")]}, config)
            result = graph.invoke({"messages": [HumanMessage("second turn")]}, config)

        contents = [m.content for m in result["messages"]]
        assert "first turn" in contents
        assert "second turn" in contents

    def test_different_thread_ids_are_isolated(self):
        final_msg = AIMessage(content="response")
        final_answer = AgentAnswer(answer="answer", citations=[], high_risk=False)

        with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg, final_msg], final_answer)):
            graph = build_graph()
            graph.invoke(
                {"messages": [HumanMessage("thread one message")]},
                {"configurable": {"thread_id": "thread-1"}},
            )
            result = graph.invoke(
                {"messages": [HumanMessage("thread two message")]},
                {"configurable": {"thread_id": "thread-2"}},
            )

        contents = [m.content for m in result["messages"]]
        assert "thread one message" not in contents


class TestAmbiguousDrugClarification:
    def test_ambiguous_match_asks_then_resolves_on_next_turn(self):
        ambiguous_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_drug_info",
                    "args": {"query_text": "side effects", "drug_name": "metfromin"},
                    "id": "1",
                    "type": "tool_call",
                }
            ],
        )
        clarify_msg = AIMessage(content="Did you mean metformin or merbromin?")
        resolved_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_drug_info",
                    "args": {"query_text": "side effects", "drug_name": "metformin"},
                    "id": "2",
                    "type": "tool_call",
                }
            ],
        )
        final_msg = AIMessage(content="Done reasoning.")

        clarify_answer = AgentAnswer(answer="Did you mean metformin or merbromin?", citations=[], high_risk=False)
        resolved_answer = AgentAnswer(answer="Metformin can cause nausea.", citations=[], high_risk=False)

        ambiguous_resolution = {
            "rxcui": None,
            "match_type": "ambiguous",
            "candidates": [{"name": "metformin", "rxcui": "6809"}, {"name": "merbromin", "rxcui": "9999"}],
        }
        exact_resolution = {"rxcui": "6809", "match_type": "exact", "candidates": []}
        chunk = {
            "chunk_text": "Common side effects include nausea.",
            "setid": "abc-123",
            "loinc_code": "34084-4",
            "section_title_path": "ADVERSE REACTIONS",
            "drug_name": "metformin",
            "section_type": "adverse_reactions",
        }

        config = {"configurable": {"thread_id": _THREAD}}

        with (
            patch(
                f"{_GRAPH_MODULE}.get_llm",
                return_value=_mock_llm(
                    [ambiguous_call, clarify_msg, resolved_call, final_msg],
                    [clarify_answer, resolved_answer],
                ),
            ),
            patch(
                f"{_TOOLS_MODULE}.resolve_query_drug",
                side_effect=[ambiguous_resolution, exact_resolution],
            ) as mock_resolve,
            patch(f"{_TOOLS_MODULE}.hybrid_search", return_value=[chunk]) as mock_search,
        ):
            graph = build_graph()
            turn1 = graph.invoke({"messages": [HumanMessage("side effects of metfromin?")]}, config)
            turn2 = graph.invoke({"messages": [HumanMessage("I meant metformin")]}, config)

        assert "metformin or merbromin" in turn1["structured_response"].answer.lower()
        assert turn2["structured_response"].answer.startswith("Metformin can cause nausea.")
        assert mock_resolve.call_args_list == [(("metfromin",),), (("metformin",),)]
        mock_search.assert_called_once()

        # turn 2's agent call should still see turn 1's clarifying exchange in history
        turn2_human_contents = [m.content for m in turn2["messages"] if isinstance(m, HumanMessage)]
        assert "side effects of metfromin?" in turn2_human_contents
        assert "I meant metformin" in turn2_human_contents


class TestPostProcessing:
    def test_disclaimer_always_appended(self):
        final_msg = AIMessage(content="done")
        final_answer = AgentAnswer(answer="Some facts.", citations=[], high_risk=False)

        with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg], final_answer)):
            graph = build_graph()
            result = graph.invoke(
                {"messages": [HumanMessage("hello")]}, {"configurable": {"thread_id": _THREAD}}
            )

        assert "informational purposes only" in result["structured_response"].answer

    def test_pharmacist_routing_added_when_high_risk(self):
        final_msg = AIMessage(content="done")
        final_answer = AgentAnswer(answer="Some facts.", citations=[], high_risk=True)

        with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg], final_answer)):
            graph = build_graph()
            result = graph.invoke(
                {"messages": [HumanMessage("hello")]}, {"configurable": {"thread_id": _THREAD}}
            )

        assert "consult your pharmacist or doctor" in result["structured_response"].answer.lower()

    def test_pharmacist_routing_absent_when_not_high_risk(self):
        final_msg = AIMessage(content="done")
        final_answer = AgentAnswer(answer="Some facts.", citations=[], high_risk=False)

        with patch(f"{_GRAPH_MODULE}.get_llm", return_value=_mock_llm([final_msg], final_answer)):
            graph = build_graph()
            result = graph.invoke(
                {"messages": [HumanMessage("hello")]}, {"configurable": {"thread_id": _THREAD}}
            )

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
            graph_module.ask("first query", _THREAD)
            graph_module.ask("second query", _THREAD)

        mock_build.assert_called_once()
        assert fake_graph.invoke.call_count == 2
        for call in fake_graph.invoke.call_args_list:
            assert call.args[1] == {"configurable": {"thread_id": _THREAD}}

    def test_delete_thread_resets_conversation(self, monkeypatch):
        import backend.agent.graph as graph_module

        monkeypatch.setattr(graph_module, "_graph", None)
        final_msg = AIMessage(content="response")
        final_answer = AgentAnswer(answer="answer", citations=[], high_risk=False)
        mock_llm = _mock_llm([final_msg, final_msg], final_answer)
        tool_llm = mock_llm.bind_tools.return_value.with_retry.return_value

        with patch(f"{_GRAPH_MODULE}.get_llm", return_value=mock_llm):
            graph_module.ask("before delete", _THREAD)
            graph_module.delete_thread(_THREAD)
            graph_module.ask("after delete", _THREAD)

        second_call_messages = tool_llm.invoke.call_args_list[1].args[0]
        contents = [m.content for m in second_call_messages]
        assert "before delete" not in contents
        assert "after delete" in contents
