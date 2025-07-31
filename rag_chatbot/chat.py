import os
import requests
from dotenv import load_dotenv

# ✅ Load environment variables from .env if running locally
load_dotenv()

# ✅ Read Hugging Face token and model name
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
    if not HF_TOKEN:
        return "❌ HF_TOKEN is not set in the environment."

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

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        # HF might return a list or a dict
        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "error" in result:
            return f"❌ Hugging Face API Error: {result['error']}"
        else:
            return str(result)

    except requests.exceptions.Timeout:
        return "❌ Request timed out. Try again later."
    except requests.exceptions.RequestException as e:
        return f"❌ HF API request failed: {str(e)}"


