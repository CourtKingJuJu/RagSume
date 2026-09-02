"""
This file is run whenever the KB is updated. 
It saves the Index so ragsume can use it to retrieve chunks
"""
import pickle
from pathlib import Path

import faiss
from langchain_text_splitters import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer

KB = Path("knowledge_base")
DATA = Path("data")
DATA.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_split_kb():
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    documents = []
    
    for directory in KB.iterdir():
        if not directory.is_dir():
            continue
        
        for file in directory.iterdir():
            if file.suffix != ".md":
                continue
                
            with open(file, "r", encoding="utf-8") as f:
                chunks = splitter.split_text(f.read())
                
            for chunk in chunks:
                chunk.metadata["source"] = str(file)
                documents.append(chunk)
            
    return documents

def build_index(documents):
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu"
    )
    chunk_content = [doc.page_content for doc in documents]
    embeddings = embedding_model.encode(
        chunk_content,
        convert_to_numpy=True
    )
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index

documents = load_and_split_kb()
index = build_index(documents)

faiss.write_index(index, str(DATA/ "index.faiss"))

with open(DATA / "documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"Saved {index.ntotal} embeddings")
print(f"Saved {len(documents)} documents")
    
