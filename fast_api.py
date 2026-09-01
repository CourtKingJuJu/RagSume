from fastapi import FastAPI
from ragsume import RAG

app = FastAPI()
rag = RAG()

@app.post("/chat")
def ragsume(prompt: str):
    answer = rag.rag_pipeline(prompt)
    return {"answer": answer}