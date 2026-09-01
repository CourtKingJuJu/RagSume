from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ragsume import RAG

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAG()


@app.post("/chat")
def ragsume(question: str):
    answer = rag.rag_pipeline(question)

    return {"answer": answer}