from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    llm_mode: str = os.getenv("LLM_MODE", "MOCK")
    
    # data
    data_dir: str = os.getenv("DATA_DIR", "./data")
    index_dir: str = os.getenv("INDEX_DIR", "./data/index")

    # retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # LLM (NEW)
    hf_router_url: str = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")
    hf_api_key: str = os.getenv("HF_API_KEY", "")
    hf_model: str = os.getenv("HF_MODEL", "sentence-transformers/all-mpnet-base-v2")

    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "900"))
    llm_retries: int = int(os.getenv("LLM_RETRIES", "2"))

    min_citation_coverage: float = float(os.getenv("MIN_CITATION_COVERAGE", "1.0"))
    require_per_field_citations: bool = os.getenv("REQUIRE_PER_FIELD_CITATIONS", "true").lower() == "true"



settings = Settings()
