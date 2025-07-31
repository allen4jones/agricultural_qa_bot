from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from chat import build_prompt, query_llm
from src.best_chunk import get_top_chunks

app = FastAPI()

# ✅ CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Request body format
class ChatRequest(BaseModel):
    question: str
    country: str = "albania"

# ✅ Health check route (supports GET + HEAD for Render)
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def health_check():
    return JSONResponse(content={"status": "ok"})

# ✅ Debug-safe chatbot route
@app.post("/chat")
def chat(req: ChatRequest):
    try:
        # Retrieve top relevant chunks
        chunks = get_top_chunks(
            country=req.country,
            query=req.question,
            chroma_path="processed/chroma_db",
            top_k=5
        )

        context = "\n\n".join([c["text"] for c in chunks])

        # Build prompt and query the LLM
        prompt = build_prompt(context, req.question, chat_history=[])
        result = query_llm(prompt)

        return {"answer": result.strip()}

    except Exception as e:
        # 🔥 Return clear error for frontend display
        return {"answer": f"🔥 Error in backend: {str(e)}"}



