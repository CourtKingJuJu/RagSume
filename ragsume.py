import numpy as np
import faiss
import pickle

from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
DATA = Path("data")
PROMPT_T = """
You are RagSume, a career assistant for Julian McCarty.

Answer questions about Julian using the provided context.

Rules:
- Use the provided context as the source of truth.
- Do not invent or assume information about Julian.
- If the context doesn't contain enough information, say you don't know.
- If the question isn't about Julian, respond appropriately but with a witty answer.

Context:
----------------
{context}
----------------

Question:
{question}
"""

class RAG():
    
    def __init__(self) -> None:
        load_dotenv()
        
        self.documents = self._load_documents()
        self.index = faiss.read_index(
            str(DATA / "index.faiss")
        )
        self.client = OpenAI()
      
    def _load_documents(self):
        with open(DATA / "documents.pkl", "rb") as f:
            return pickle.load(f)

    def embed_question(self, question):
        response = self.client.embeddings.create(
            model=EMBEDDING_MODEL_NAME,
            input=question
        )

        return np.array(
            response.data[0].embedding,
            dtype="float32"
        )
        
    def retriever(self, question, k):
        question_embedding = self.embed_question(question)
        _, indices = self.index.search(np.array([question_embedding]).astype('float32'), k)
        
        return [self.documents[i] for i in indices[0]]

    def build_prompt(self, question, documents, prompt_t):
        context = "\n\n".join(
            doc.page_content for doc in documents
        )
        
        return prompt_t.format(
            context=context,
            question=question
        )

    def generate(self, prompt):
        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )
        
        return response.output_text


    def rag_pipeline(self, question):
        documents = self.retriever(question, 3)
        prompt = self.build_prompt(question, documents, PROMPT_T)
        answer = self.generate(prompt)
        
        return answer