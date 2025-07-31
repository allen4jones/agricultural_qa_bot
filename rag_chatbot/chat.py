import os
import requests
import logging
from dotenv import load_dotenv

# Load .env file (for local dev)
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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
    Queries Hugging Face Inference API, with fallback to mock.
    """
    if not HF_TOKEN:
        logger.warning("⚠️ HF_TOKEN is not set. Using mock response.")
        return f"(Mock Response) {prompt[:100]}..."

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
        logger.info(f"HF API response code: {response.status_code}")
        logger.debug(f"HF API raw response: {response.text[:500]}")

        response.raise_for_status()
        result = response.json()

        if isinstance(result, list) and "generated_text" in result[0]:
            return result[0]["generated_text"]
        elif isinstance(result, dict) and "error" in result:
            return f"❌ Hugging Face API Error: {result['error']}"
        else:
            return f"⚠️ Unexpected response format: {str(result)}"

    except requests.exceptions.Timeout:
        logger.error("❌ Request to HF API timed out.")
        return "❌ Request to model timed out."
    except Exception as e:
        logger.exception("❌ HF API call failed.")
        return f"(Mock) Error fallback response: {str(e)}"



