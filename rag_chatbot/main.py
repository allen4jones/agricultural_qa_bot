from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

from chat import build_prompt, query_llm
from src.best_chunk import get_top_chunks

# Setup FastAPI
app = FastAPI()

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request schema
class ChatRequest(BaseModel):
    question: str
    country: str = "albania"

# Health check for Render
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def health_check():
    return JSONResponse(content={"status": "ok"})

# POST endpoint for chatbot
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        chunks = get_top_chunks(
            country=req.country,
            query=req.question,
            chroma_path="processed/chroma_db",
            top_k=5
        )
        context = "\n\n".join([c["text"] for c in chunks])

        prompt = build_prompt(context, req.question, chat_history=[])
        result = query_llm(prompt)

        return {"answer": result.strip()}

    except Exception as e:
        logger.exception("🔥 Exception in /chat route")
        return {"answer": f"🔥 Backend error: {str(e)}"}
