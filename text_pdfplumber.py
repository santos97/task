import os
import json
import pdfplumber
from pathlib import Path

RAW_DATA_DIR = Path("./data/raw")

OUTPUT_DIR = Path("./data/processed/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "text_chunks.jsonl"


def extract_text_from_pdf(pdf_path: Path, company: str, report_year: str):
    """
    Extracts text from each page of a PDF and returns a list of records
    with metadata required for downstream RAG usage."""
    extracted_records = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text()

            if not page_text:
                continue
            record = {
                "company": company,
                "report_year": report_year,
                "page": page_index,
                "source_file": pdf_path.name,
                "text": page_text.strip()
            }

            extracted_records.append(record)
    return extracted_records



all_records = []

for company_dir in RAW_DATA_DIR.iterdir():
    if not company_dir.is_dir():
        continue

    company_name = company_dir.name

    for pdf_file in company_dir.glob("*.pdf"):
        try:
            report_year = pdf_file.stem.split("_")[-1]
        except IndexError:
            print(f"Skipping file with unexpected name format: {pdf_file.name}")
            continue

        print(f"Extracting text from {pdf_file.name}")

        records = extract_text_from_pdf(
            pdf_path=pdf_file,
            company=company_name,
            report_year=report_year
        )

        all_records.extend(records)

# Write all extracted text chunks to JSONL
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for record in all_records:
        f.write(json.dumps(record) + "\n")

print(f"Text extraction complete. Saved to {OUTPUT_FILE}")


