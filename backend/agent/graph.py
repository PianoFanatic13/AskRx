import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

_DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"
_DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
_DEFAULT_OLLAMA_MODEL = "llama3.1"


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
