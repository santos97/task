import os
import camelot
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")

OUTPUT_DIR = Path("data/processed/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_tables_from_pdf(pdf_path: Path, company: str, report_year: str):
    """
    Extracts tables from a PDF using Camelot and saves each table as a CSV file.
    """
    print(f"Extracting tables from {pdf_path.name}")

    try:
        tables = camelot.read_pdf(
            str(pdf_path),
            pages="all",
            flavor="lattice"  
        )
    except Exception as e:
        print(f"Failed to extract tables from {pdf_path.name}: {e}")
        return

    if tables.n == 0:
        print(f"No tables found in {pdf_path.name}")
        return

    for table_index, table in enumerate(tables):
        output_file = OUTPUT_DIR / f"{company}_{report_year}_table_{table_index + 1}.csv"

        table.df.to_csv(output_file, index=False)

        print(f"Saved table to {output_file}")



# Iterate through company folders
for company_dir in RAW_DATA_DIR.iterdir():
    if not company_dir.is_dir():
        continue

    company_name = company_dir.name

    # Iterate through PDF files
    for pdf_file in company_dir.glob("*.pdf"):
        try:
            report_year = pdf_file.stem.split("_")[-1]
        except IndexError:
            print(f"Skipping file with unexpected name format: {pdf_file.name}")
            continue

        extract_tables_from_pdf(
            pdf_path=pdf_file,
            company=company_name,
            report_year=report_year
        )

