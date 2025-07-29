from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def chat(req: ChatRequest):
    answer = get_answer(req.question)
    return {"answer": answer}
