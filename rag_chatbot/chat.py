import os
import requests

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "microsoft/Phi-4-mini-instruct"

def build_prompt(context: str, question: str, chat_history: list) -> str:
    history_str = "\n".join([f"{role.capitalize()}: {msg}" for role, msg in chat_history])
    return (
        f"Context:\n{context}\n\n"
        f"{history_str}\n\n"
        f"User: {question}\nAssistant:"
    )

def query_llm(prompt: str) -> str:
    if not HF_TOKEN:
        return "(Mock) HF_TOKEN is missing. Set it in your Render Environment."

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 200,
                    "do_sample": False,
                    "temperature": 0.3
                }
            },
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        return "(Mock) Unexpected response format."

    except Exception as e:
        return f"(Mock) HF API call failed: {str(e)}"
