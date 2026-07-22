import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

from backend.agent.prompts import DISCLAIMER_TEXT, PHARMACIST_ROUTING_TEXT, SYSTEM_PROMPT
from backend.agent.tools import resolve_drug_name, retrieve_drug_info, retrieve_interactions

load_dotenv()

_DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
_DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
_DEFAULT_OLLAMA_MODEL = "llama3.1"

_TOOLS = [resolve_drug_name, retrieve_drug_info, retrieve_interactions]


def get_llm() -> BaseChatModel:
    """Return a chat model instance, backend selected via LLM_BACKEND (gemini|groq|ollama)."""
    backend = os.getenv("LLM_BACKEND", "gemini")

    if backend == "gemini":
        return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL))
    if backend == "groq":
        return ChatGroq(model=os.getenv("GROQ_MODEL", _DEFAULT_GROQ_MODEL))
    if backend == "ollama":
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL))

    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (expected 'gemini', 'groq', or 'ollama')")


class Citation(BaseModel):
    marker: int
    setid: str
    loinc_code: Optional[str]
    section_title_path: str


class AgentAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    high_risk: bool


class AgentState(MessagesState):
    structured_response: Optional[AgentAnswer]


def build_graph() -> CompiledStateGraph:
    """Assemble the ReAct loop: reason (agent) -> act (tools) -> ... -> structured answer.

    LLM binding and node closures live inside this function (not at module
    level) so tests can patch get_llm and call build_graph() fresh to get a
    graph wired to the mock, instead of fighting import-time state.
    """
    llm_with_tools = get_llm().bind_tools(_TOOLS).with_retry(stop_after_attempt=4)
    structured_llm = get_llm().with_structured_output(AgentAnswer).with_retry(stop_after_attempt=4)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"messages": [response]}

    def generate_structured_answer_node(state: AgentState) -> dict:
        result = structured_llm.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        return {"structured_response": result}

    def post_process_node(state: AgentState) -> dict:
        answer = state["structured_response"]
        answer.answer += "\n\n" + DISCLAIMER_TEXT
        if answer.high_risk:
            answer.answer += "\n\n" + PHARMACIST_ROUTING_TEXT
        return {"structured_response": answer}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(_TOOLS))
    builder.add_node("generate_structured_answer", generate_structured_answer_node)
    builder.add_node("post_process", post_process_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
        {"tools": "tools", "__end__": "generate_structured_answer"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("generate_structured_answer", "post_process")
    builder.add_edge("post_process", END)

    return builder.compile()


_graph: Optional[CompiledStateGraph] = None


def ask(query: str) -> AgentAnswer:
    """Run a single-turn query through the agent, building the graph once and reusing it after."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    result = _graph.invoke({"messages": [HumanMessage(query)]})
    return result["structured_response"]
