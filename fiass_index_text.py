"""
Build a FAISS index for narrative PDF text chunks extracted into:
  data/processed/text_chunks.jsonl

Output:
  data/index/faiss_text/
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TEXT_CHUNKS_FILE = Path("data/processed/text_chunks.jsonl")
FAISS_TEXT_DIR = Path("index/faiss_text")


def load_text_documents(jsonl_path: Path) -> List[Document]:
    """Load text chunks JSONL into LangChain Document objects with metadata."""
    documents: List[Document] = []

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing input file: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record: Dict[str, Any] = json.loads(line)

            text = (record.get("text") or "").strip()
            if not text:
                continue

            metadata = {
                "doc_type": "pdf_text",
                "company": record.get("company"),
                "report_year": int(record["report_year"]) if str(record.get("report_year", "")).isdigit() else record.get("report_year"),
                "page": record.get("page"),
                "source_file": record.get("source_file"),
            }

            documents.append(Document(page_content=text, metadata=metadata))

    return documents


def main() -> None:
    FAISS_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading text chunks from: {TEXT_CHUNKS_FILE}")
    text_documents = load_text_documents(TEXT_CHUNKS_FILE)
    print(f"Loaded {len(text_documents)} text documents")

    print(f"Initializing embeddings: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Building FAISS index (text)...")
    vector_store = FAISS.from_documents(text_documents, embeddings)

    print(f"Saving FAISS text index to: {FAISS_TEXT_DIR}")
    vector_store.save_local(str(FAISS_TEXT_DIR))

    print("Done. Text FAISS index created successfully.")


if __name__ == "__main__":
    main()
