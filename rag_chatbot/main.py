from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# ✅ Add this block to handle both GET and HEAD at "/"
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def health_check():
    return JSONResponse(content={"status": "ok"})

@app.post("/chat")
def chat(req: ChatRequest):
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


