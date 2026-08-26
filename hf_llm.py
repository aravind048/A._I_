import logging
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda

load_dotenv()

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")

if not HF_TOKEN:
    logger.warning("HF_TOKEN environment variable is not set. Hugging Face API calls will fail.")

client = InferenceClient(
    model=HF_MODEL,
    token=HF_TOKEN,
)


def hf_generate(prompt) -> str:
    """Generate a response using the configured Hugging Face chat model.

    Input:
        prompt: str or LangChain ChatPromptValue

    Output:
        Generated response as a string
    """
    if isinstance(prompt, ChatPromptValue):
        prompt = prompt.to_string()

    messages = [{"role": "user", "content": prompt}]

    try:
        logger.info("Calling Hugging Face model: %s", HF_MODEL)
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=700,
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception:
        logger.exception("Hugging Face API request failed")
        raise


hf_llm = RunnableLambda(hf_generate)
