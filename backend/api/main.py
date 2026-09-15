from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.agent.graph import Citation, ask

# Run with `uvicorn backend.api.main:app --workers 1` - conversation state is
# now Postgres-backed (PostgresSaver, see graph.py) and shared across workers,
# so this is no longer a correctness requirement. Kept as the deploy config
# anyway since Render's free tier only ever runs one instance.
app = FastAPI(title="AskRx API")

# permissive dev CORS; restrict to the real frontend origin once Phase 5 picks one
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    high_risk: bool
    thread_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    thread_id = request.thread_id or str(uuid4())
    try:
        result = ask(request.message, thread_id)
    except Exception as e:  # noqa: BLE001 - API boundary: any agent failure becomes a clean 503, never a raw traceback
        return JSONResponse(
            status_code=503,
            content={"error": f"agent failed: {e}"},
            headers={"Retry-After": "10"},
        )
    return ChatResponse(
        answer=result.answer,
        citations=result.citations,
        high_risk=result.high_risk,
        thread_id=thread_id,
    )
