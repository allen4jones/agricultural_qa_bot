from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from src.best_chunk import get_top_chunks
from chat import build_prompt, query_llm

app = FastAPI()

# Allow frontend access (Streamlit)
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
    # Retrieve top relevant chunks for the country
    chunks = get_top_chunks(req.question, req.country, db_path="processed/chroma_db")
    context = "\n\n".join([c["text"] for c in chunks])

    # Build prompt and call HF API
    prompt = build_prompt(context, req.question, chat_history=[])
    result = query_llm(prompt)
    
    return {"answer": result.strip()}

