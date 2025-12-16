

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TABLE_INDEX_FILE = Path("/kaggle/input/consigli/data/processed/table_index.jsonl")
FAISS_TABLES_DIR = Path("index/faiss_tables")


def format_preview(preview: List[List[str]]) -> str:
    """Pretty-format top-left preview for embedding."""
    lines = []
    for row in preview[:3]:
        row = [str(cell).strip() for cell in row[:5]]
        row = [cell for cell in row if cell]
        if row:
            lines.append(" | ".join(row))
    return "\n".join(lines) if lines else "(no preview)"


def build_table_card(record: Dict[str, Any]) -> str:
    """
    Convert a table_index.jsonl record into an embedding-friendly "table card".
    This text is what we embed into FAISS so the retriever can find the right table.
    """
    table_id = record.get("table_id", "")
    company = record.get("company", "")
    report_year = record.get("report_year", "")
    source_pdf = record.get("source_pdf", "")
    page_start = record.get("page_start", "")
    page_end = record.get("page_end", "")
    csv_path = record.get("csv_path", "")

    table_type_guess = record.get("table_type_guess", "unknown")
    currency_hint = record.get("currency_hint") or "unknown"
    units_hint = record.get("units_hint") or "unknown"
    header_quality = record.get("header_quality") or "unknown"

    preview = record.get("top_left_preview") or []
    preview_text = format_preview(preview)

    keywords = record.get("keywords") or []
    keywords_text = ", ".join(keywords[:25]) if keywords else "(none)"

    # Keep this stable and explicit; it will be embedded.
    table_card = f"""TABLE_INDEX_ENTRY
                    TableId: {table_id}
                    Company: {company}
                    ReportYear: {report_year}
                    SourcePDF: {source_pdf}
                    SourcePages: {page_start}-{page_end}
                    CSVFile: {csv_path}

                    TableTypeGuess: {table_type_guess}
                    CurrencyHint: {currency_hint}
                    UnitsHint: {units_hint}
                    HeaderQuality: {header_quality}

                    PreviewTopLeft:
                    {preview_text}

                    KeyTerms: {keywords_text}
                """
    return table_card.strip()


def load_table_documents(jsonl_path: Path) -> List[Document]:
    """Load table index JSONL into Documents containing table-card text + metadata."""
    documents: List[Document] = []

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Missing input file: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            record: Dict[str, Any] = json.loads(line)
            table_card_text = build_table_card(record)

            # Store metadata needed later to open CSV and cite pages.
            metadata = {
                "doc_type": "table_index",
                "table_id": record.get("table_id"),
                "company": record.get("company"),
                "report_year": record.get("report_year"),
                "source_pdf": record.get("source_pdf"),
                "page_start": record.get("page_start"),
                "page_end": record.get("page_end"),
                "csv_path": record.get("csv_path"),
                "table_type_guess": record.get("table_type_guess"),
                "extraction_mode": record.get("extraction_mode"),
                "header_quality": record.get("header_quality"),
            }

            documents.append(Document(page_content=table_card_text, metadata=metadata))

    return documents


def main() -> None:
    FAISS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading table index from: {TABLE_INDEX_FILE}")
    table_documents = load_table_documents(TABLE_INDEX_FILE)
    print(f"Loaded {len(table_documents)} table index documents")

    print(f"Initializing embeddings: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Building FAISS index (tables)...")
    vector_store = FAISS.from_documents(table_documents, embeddings)

    print(f"Saving FAISS table index to: {FAISS_TABLES_DIR}")
    vector_store.save_local(str(FAISS_TABLES_DIR))

    print("Done. Table FAISS index created successfully.")


if __name__ == "__main__":
    main()

