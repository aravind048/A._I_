from huggingface_hub import InferenceClient
from langchain_core.runnables import RunnableLambda
import os
from langchain_core.prompt_values import ChatPromptValue


client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-beta",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)


def hf_generate(prompt) -> str:

     # 🔑 CRITICAL FIX: unwrap LangChain object
    if isinstance(prompt, ChatPromptValue):
        prompt = prompt.to_string()
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    response = client.chat.completions.create(
        messages=messages,
        max_tokens=512,
        temperature=0.7
    )
    return response.choices[0].message.content


hf_llm = RunnableLambda(hf_generate)
