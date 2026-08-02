import argparse
import json
import warnings
from pathlib import Path
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langsmith import Client

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

import ragas.messages as ragas_messages  # noqa: E402
from ragas import evaluate  # noqa: E402
from ragas.dataset_schema import EvaluationDataset, MultiTurnSample, SingleTurnSample  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.messages import ToolCall  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402
from ragas.metrics import (  # noqa: E402
    AgentGoalAccuracyWithReference,
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ToolCallAccuracy,
)

from backend.agent.graph import ask_with_trace

# Judge LLM/embeddings are deliberately not the agent's own Gemini/BGE stack -
# grading a model with itself (or the same embedding space its retrieval was
# built on) risks self-preference bias and shared blind spots.
_JUDGE_LLM = LangchainLLMWrapper(ChatOllama(model="llama3.1"))
_JUDGE_EMBEDDINGS = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"))

# Ollama serves one local model, not built for the default 16-way concurrency;
# local CPU inference also needs more time per call than a hosted API.
_RUN_CONFIG = RunConfig(timeout=600, max_workers=2)


def _extract_contexts(messages: list) -> list[str]:
    contexts = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        try:
            payload = json.loads(m.content)
        except (json.JSONDecodeError, TypeError):
            continue
        for result in payload.get("results", []):
            text = result.get("chunk_text")
            if text:
                contexts.append(text)
    return contexts


def _content_to_str(content) -> str:
    """Flatten Gemini's list-of-content-blocks message content to a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return str(content)


def _to_ragas_messages(messages: list) -> list:
    """Convert LangChain messages to ragas's message format.

    Not using ragas.integrations.langgraph.convert_to_ragas_messages because
    it reads tool calls from additional_kwargs["tool_calls"] (OpenAI/Groq's
    raw format) - Gemini's tool calls live in the standardized .tool_calls
    attribute instead, so that helper silently produces empty tool-call lists
    for our agent.
    """
    out = []
    for m in messages:
        if isinstance(m, HumanMessage):
            out.append(ragas_messages.HumanMessage(content=_content_to_str(m.content)))
        elif isinstance(m, AIMessage):
            tool_calls = [ragas_messages.ToolCall(name=tc["name"], args=tc["args"]) for tc in (m.tool_calls or [])]
            out.append(ragas_messages.AIMessage(content=_content_to_str(m.content), tool_calls=tool_calls or None))
        elif isinstance(m, ToolMessage):
            out.append(ragas_messages.ToolMessage(content=_content_to_str(m.content)))
    return out


def _trace_url(run_id: str) -> str | None:
    try:
        client = Client()
        return client.get_run_url(run=client.read_run(run_id))
    except Exception:
        return None


def run_single_query(entry: dict) -> dict:
    """Run one dataset entry through the agent and build both RAGAS sample tracks."""
    trace = ask_with_trace(entry["query"], thread_id=str(uuid4()))
    contexts = _extract_contexts(trace["messages"])

    single_turn = SingleTurnSample(
        user_input=entry["query"],
        response=trace["answer"].answer,
        retrieved_contexts=contexts,
        reference=entry["reference_answer"],
    )
    multi_turn = MultiTurnSample(
        user_input=_to_ragas_messages(trace["messages"]),
        reference=entry["reference_answer"],
        reference_tool_calls=[
            ToolCall(name=c["tool"], args=c["args"]) for c in entry["reference_tool_calls"]
        ],
    )
    return {
        "id": entry["id"],
        "query": entry["query"],
        "single_turn_sample": single_turn,
        "multi_turn_sample": multi_turn,
        "trace_url": _trace_url(trace["run_id"]),
    }


def run_eval(dataset_path: Path, report_path: Path) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    per_query = [run_single_query(entry) for entry in dataset]

    single_ds = EvaluationDataset(samples=[q["single_turn_sample"] for q in per_query])
    multi_ds = EvaluationDataset(samples=[q["multi_turn_sample"] for q in per_query])

    single_result = evaluate(
        single_ds,
        metrics=[Faithfulness(), AnswerRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()],
        llm=_JUDGE_LLM,
        embeddings=_JUDGE_EMBEDDINGS,
        run_config=_RUN_CONFIG,
    )
    multi_result = evaluate(
        multi_ds,
        metrics=[ToolCallAccuracy(), AgentGoalAccuracyWithReference()],
        llm=_JUDGE_LLM,
        run_config=_RUN_CONFIG,
    )

    single_scores = single_result.to_pandas().to_dict(orient="records")
    multi_scores = multi_result.to_pandas().to_dict(orient="records")

    report = {
        "queries": [
            {
                "id": q["id"],
                "query": q["query"],
                "trace_url": q["trace_url"],
                "single_turn_scores": single_scores[i],
                "multi_turn_scores": multi_scores[i],
            }
            for i, q in enumerate(per_query)
        ],
        "single_turn_summary": single_result._repr_dict if hasattr(single_result, "_repr_dict") else dict(single_result),
        "multi_turn_summary": multi_result._repr_dict if hasattr(multi_result, "_repr_dict") else dict(multi_result),
    }
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("tests/eval/dataset.json"))
    parser.add_argument("--report", type=Path, default=Path("backend/eval/report.json"))
    args = parser.parse_args()

    run_eval(args.dataset, args.report)
    print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()
