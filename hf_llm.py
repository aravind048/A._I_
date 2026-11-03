
import logging
import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

logger = logging.getLogger(__name__)

# Local Ollama configuration
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

logger.info(
    "Initializing local LLM: model=%s, base_url=%s",
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)

# LangChain-compatible local chat model
hf_llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    num_predict=700,
)
