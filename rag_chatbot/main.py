from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from src.best_chunk import get_top_chunks  # ✅ your chunking logic
from rag_chatbot.chat import build_prompt, READER_LLM  # ✅ your LLM + prompt logic

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    country: str = "albania"

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    chunks = get_top_chunks(req.question, req.country, db_path="processed/chroma_db")
    context = "\n\n".join([c["text"] for c in chunks])
    prompt = build_prompt(context, req.question, chat_history=[])
    result = READER_LLM(prompt)[0]["generated_text"]
    return {"answer": result.strip()}

