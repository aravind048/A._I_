import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}


def get_stem_answer(question: str) -> str:
    payload = {
        "model": "HuggingFaceH4/zephyr-7b-beta:featherless-ai",
        "max_tokens": 100,  # ⬅️ Add this
        "messages": [
            {"role": "system", "content": "You are a helpful STEM tutor who explains math, physics, and computer science clearly."},
            {"role": "user", "content": question}
        ]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Model error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception occurred: {str(e)}"
