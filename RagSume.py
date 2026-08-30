import numpy as np
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# Generator imports
from openai import OpenAI
import os
from dotenv import load_dotenv



KB = Path('knowledge_base')
# Generator Prompt
prompt_t = """
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

# Global test question
question = "Show me Julian's experience in machine learning"


class RAG():
    
    def __init__(self) -> None:
        self.documents = []
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.index = None
        
        load_dotenv()
        self.client = OpenAI()
      
    def split_kb(self):
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)

        for item in KB.iterdir():
            if Path(item).is_dir():
                for file in item.iterdir():
                    if file.suffix == '.md':
                        with open(file, "r", encoding="utf-8") as f:
                            chunks = markdown_splitter.split_text(f.read())
                            for chunk in chunks:
                                chunk.metadata["source"] = str(file)
                                self.documents.append(chunk)


    def embedding_model(self):
        
        embedding_model = SentenceTransformer(self.embedding_model_name)
        chunk_content = [doc.page_content for doc in self.documents]
        embeddings = embedding_model.encode(chunk_content, show_progress_bar=True, convert_to_numpy=True)

        # Create Index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)


    # RAG Pipeline
    def retriever(self, question, k):
        """
        Retrieves k chunks based off of question string

        Args:
            question str: the query to embed and match
            k int: number of chunks to return
        """
        question_embedding = self.embedding_model.encode(question)
        _, indices = self.index.search(np.array([question_embedding]).astype('float32'), k)
        
        return [self.documents[i] for i in indices[0]]

        

    # Generator
    def generate(self, prompt):
        
        response = self.client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )
        
        return response.output_text

    def build_prompt(self, question, documents, prompt_t):
         
        context = "\n\n".join(
            doc.page_content for doc in documents
        )
        
        return prompt_t.format(
            context=context,
            question=question
        )

    def rag_pipeline(self, question, k, prompt_t):
        documents = self.retriever(question, k)
        prompt = self.build_prompt(question, documents, prompt_t)
        print(prompt)
        answer = self.generate(prompt)
        
        return answer