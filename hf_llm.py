from huggingface_hub import InferenceClient
from langchain_core.runnables import RunnableLambda
import os
from langchain_core.prompt_values import ChatPromptValue
import logging

logger = logging.getLogger(__name__)

# Check if HF_TOKEN is set
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    logger.warning("HF_TOKEN environment variable is not set. API calls will fail.")

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=hf_token
)


def hf_generate(prompt) -> str:
    import logging
    logger = logging.getLogger(__name__)

    # 🔑 CRITICAL FIX: unwrap LangChain object
    if isinstance(prompt, ChatPromptValue):
        prompt = prompt.to_string()
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        logger.debug(f"Calling HF API with prompt: {prompt[:100]}...")
        response = client.chat.completions.create(
            messages=messages,
            max_tokens=512,
            temperature=0.7
        )
        logger.debug(f"HF API response: {response}")
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"HF API error: {str(e)}", exc_info=True)
        raise


hf_llm = RunnableLambda(hf_generate)
