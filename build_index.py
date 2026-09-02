"""
This file is run whenever the KB is updated.

It creates embeddings for the knowledge base and saves the
FAISS index so RagSume can use it for retrieval.
"""

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import MarkdownHeaderTextSplitter
from openai import OpenAI


KB = Path("knowledge_base")
DATA = Path("data")

DATA.mkdir(exist_ok=True)

EMBEDDING_MODEL_NAME = "text-embedding-3-small"


def load_and_split_kb():
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

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
    client = OpenAI()

    chunk_content = [
        doc.page_content
        for doc in documents
    ]

    response = client.embeddings.create(
        model=EMBEDDING_MODEL_NAME,
        input=chunk_content
    )

    embeddings = np.array(
        [item.embedding for item in response.data],
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


load_dotenv()

documents = load_and_split_kb()
index = build_index(documents)

faiss.write_index(
    index,
    str(DATA / "index.faiss")
)

with open(DATA / "documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"Saved {index.ntotal} embeddings")
print(f"Saved {len(documents)} documents")