import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_intervention(prompt):

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False # receive the complete response at once rather than receiving it token-by-token
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]


if __name__ == "__main__":

    test_prompt = """
You are a Python programming tutor.

Explain why the following code prints
[1, 2, 3, 4]:

x = [1, 2, 3]
y = x
y.append(4)

Use a simple worked example for a beginner.
"""

    intervention = generate_intervention(
        test_prompt
    )

    print("Generated Intervention:")
    print(intervention)