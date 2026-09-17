import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from backend.agent.prompts import (
    AGENT_LOOP_SYSTEM_PROMPT,
    DISCLAIMER_TEXT,
    PHARMACIST_ROUTING_TEXT,
    SYSTEM_PROMPT,
)
from backend.agent.tools import (
    resolve_drug_name,
    retrieve_drug_info,
    retrieve_interactions,
)

load_dotenv()

_DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
_DEFAULT_OLLAMA_MODEL = "llama3.1"
_DEFAULT_DSN = "postgresql://postgres:postgres@localhost:5432/asrx"

_TOOLS = [resolve_drug_name, retrieve_drug_info, retrieve_interactions]

_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    """Module-level connection pool for the Postgres-backed checkpointer.

    Small pool size (Render's free tier runs one instance, one worker) -
    kept modest so a micro-sized RDS instance's connection limit isn't
    threatened by the checkpointer alongside every other module's own
    psycopg.connect() calls.
    """
    global _pool
    if _pool is None:
        dsn = os.getenv("DATABASE_URL", _DEFAULT_DSN)
        _pool = ConnectionPool(
            conninfo=dsn,
            min_size=1,
            max_size=5,
            kwargs={"autocommit": True, "row_factory": dict_row},
            open=True,
        )
    return _pool


_LLM_TIMEOUT_SECONDS = 30.0


def get_llm() -> BaseChatModel:
    """Return a chat model instance, backend selected via LLM_BACKEND (gemini|ollama).

    Gemini gets an explicit timeout (defaults to none otherwise) - discovered
    live on Render, where a hung call with no timeout sat for 13+ minutes
    with no error, no traceback, and no completed-request log, until the
    process was eventually killed. A bounded timeout turns that into a
    fast, visible failure (caught by the API's error handler) instead of
    an indefinite silent hang. ChatOllama has no timeout field to set, but
    it's local-dev-only - never the production backend.
    """
    backend = os.getenv("LLM_BACKEND", "gemini")

    if backend == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", _DEFAULT_GEMINI_MODEL),
            timeout=_LLM_TIMEOUT_SECONDS,
        )
    if backend == "ollama":
        return ChatOllama(model=os.getenv("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL))

    raise ValueError(f"Unknown LLM_BACKEND: {backend!r} (expected 'gemini' or 'ollama')")


class Citation(BaseModel):
    marker: int
    setid: str
    loinc_code: str | None
    section_title_path: str


class AgentAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    high_risk: bool


class AgentState(MessagesState):
    structured_response: AgentAnswer | None


def build_graph(*, use_postgres: bool = True) -> CompiledStateGraph:
    """Assemble the ReAct loop: reason (agent) -> act (tools) -> ... -> structured answer.

    LLM binding and node closures live inside this function (not at module
    level) so tests can patch get_llm and call build_graph() fresh to get a
    graph wired to the mock, instead of fighting import-time state.

    use_postgres=True (production default) persists conversation state to
    Postgres (DATABASE_URL) via PostgresSaver, so it survives process
    restarts - needed on Render's free tier, which spins down after 15
    minutes idle and would otherwise silently wipe every conversation on
    each cold start. Tests pass use_postgres=False for the original
    in-process MemorySaver behavior, keeping the mocked test suite free of
    any real Postgres dependency.
    """
    llm_with_tools = get_llm().bind_tools(_TOOLS).with_retry(stop_after_attempt=4)
    structured_llm = get_llm().with_structured_output(AgentAnswer).with_retry(stop_after_attempt=4)

    def agent_node(state: AgentState) -> dict:
        response = llm_with_tools.invoke([SystemMessage(AGENT_LOOP_SYSTEM_PROMPT)] + state["messages"])
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

    serde = JsonPlusSerializer(allowed_msgpack_modules=[AgentAnswer, Citation])
    if use_postgres:
        checkpointer = PostgresSaver(_get_pool(), serde=serde)
        checkpointer.setup()
    else:
        checkpointer = MemorySaver(serde=serde)
    return builder.compile(checkpointer=checkpointer)


_graph: CompiledStateGraph | None = None


def ask(query: str, thread_id: str) -> AgentAnswer:
    """Run a query through the agent, building the graph once and reusing it after.

    thread_id scopes conversation memory (see build_graph's checkpointer) -
    reuse the same value across calls to continue a conversation, or pass a
    fresh one for an unrelated query.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    result = _graph.invoke({"messages": [HumanMessage(query)]}, config)
    return result["structured_response"]


def ask_with_trace(query: str, thread_id: str) -> dict:
    """Like ask(), but also returns the raw message list and a run_id.

    For eval tooling that needs the full ReAct trace (tool calls, not just
    the final answer) and a LangSmith run to link to - not used by the
    CLI/API, which only need the final AgentAnswer.
    """
    global _graph
    if _graph is None:
        _graph = build_graph()
    run_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}, "run_id": run_id}
    result = _graph.invoke({"messages": [HumanMessage(query)]}, config)
    return {"answer": result["structured_response"], "messages": result["messages"], "run_id": run_id}


def delete_thread(thread_id: str) -> None:
    """Delete a thread's checkpointed state, mainly for test cleanup."""
    if _graph is not None:
        _graph.checkpointer.delete_thread(thread_id)
