import os
import requests
from dotenv import load_dotenv

load_dotenv()  # Load HF_TOKEN from .env file (local dev)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "microsoft/Phi-4-mini-instruct"

def build_prompt(context: str, question: str, chat_history: list) -> str:
    """
    Constructs a prompt for the LLM using chat history and context.
    """
    history_str = "\n".join([f"{role.capitalize()}: {msg}" for role, msg in chat_history])
    return (
        f"Context:\n{context}\n\n"
        f"{history_str}\n\n"
        f"User: {question}\nAssistant:"
    )

def query_llm(prompt: str) -> str:
    """
    Queries Hugging Face Inference API for response generation.
    """
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "do_sample": False,
            "temperature": 0.3
        }
    }

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        json=payload,
    )

    try:
        return response.json()[0]["generated_text"]
    except Exception as e:
        return f"Error from HF API: {str(e)}"

