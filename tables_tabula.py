from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pdfplumber
import tabula


RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TABLES_DIR = PROCESSED_DIR / "tables"
INDEX_FILE = PROCESSED_DIR / "table_index.jsonl"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


YEAR_REGEX = re.compile(r"\b(20\d{2})\b")
CURRENCY_HINTS = {
    "USD": ["USD", "$", "US$"],
    "EUR": ["EUR", "€"],
    "GBP": ["GBP", "£"],
}


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return normalize_whitespace(str(x))


def top_left_preview(df: pd.DataFrame, max_rows: int = 3, max_cols: int = 5) -> List[List[str]]:
    """
    Return a small preview block from the top-left of the table for indexing and retrieval anchoring.
    """
    preview = []
    r = min(max_rows, len(df))
    c = min(max_cols, len(df.columns))
    for i in range(r):
        row = []
        for j in range(c):
            row.append(safe_str(df.iat[i, j]))
        preview.append(row)
    return preview


def infer_currency_hint(preview: List[List[str]]) -> Optional[str]:
    text = " ".join(cell for row in preview for cell in row if cell)
    for currency, markers in CURRENCY_HINTS.items():
        for m in markers:
            if m in text:
                return currency
    return None


def infer_units_hint(preview: List[List[str]]) -> Optional[str]:
    text = " ".join(cell.lower() for row in preview for cell in row if cell)
    if "in million" in text or "in millions" in text or "(in millions)" in text:
        return "in millions"
    if "in thousand" in text or "in thousands" in text or "(in thousands)" in text:
        return "in thousands"
    if "in billion" in text or "in billions" in text or "(in billions)" in text:
        return "in billions"
    return None


def infer_table_type(preview: List[List[str]]) -> str:
    """
    Classification to help retrieval, indicates what type of data
    """
    text = " ".join(cell.lower() for row in preview for cell in row if cell)
    if "cash flow" in text or "cashflows" in text:
        return "cash_flow"
    if "balance sheet" in text or "financial position" in text:
        return "balance_sheet"
    if "statements of operations" in text or "income statement" in text or "profit and loss" in text:
        return "income_statement"
    if "key figures" in text or "highlights" in text or "key metrics" in text:
        return "key_metrics"
    if "segment" in text:
        return "segment_results"
    if "note" in text or "notes" in text:
        return "notes"
    return "unknown"


def extract_keywords(preview: List[List[str]]) -> List[str]:
    """
    Generate keywords from preview, plus common finance terms if present.
    Helps FAISS retrieval over table index entries.
    """
    text = " ".join(cell.lower() for row in preview for cell in row if cell)
    candidates = set(re.findall(r"[a-z]{4,}", text))
    finance_terms = [
        "revenue", "sales", "net", "income", "profit", "loss",
        "ebit", "ebitda", "operating", "margin", "cash", "assets",
        "liabilities", "equity", "dividend", "expenses", "cost"
        ]
    for term in finance_terms:
        if term in text:
            candidates.add(term)

    keywords = sorted(candidates)
    return keywords[:25]  


def header_quality_label(df: pd.DataFrame, preview: List[List[str]]) -> str:
    """
    we can keep heuristic as:
    - high: years visible in first 2 rows, AND table has >2 columns
    - medium: years visible in first 3 rows
    - low: no years visible in preview
    """
    preview_text = " ".join(cell for row in preview for cell in row if cell)
    years = YEAR_REGEX.findall(preview_text)
    if len(df.columns) > 2 and any(int(y) >= 2000 for y in years) and len(years) >= 2:
        return "high"
    if years:
        return "medium"
    return "low"


def clean_table_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    cleanup: remove fully empty rows or columns and normalize whitespace.
    """
    df = df.copy()
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")

    for col in df.columns:
        df[col] = df[col].apply(safe_str)
    mask_all_empty = df.apply(lambda row: all(cell == "" for cell in row), axis=1)
    df = df.loc[~mask_all_empty].reset_index(drop=True)

    return df






def get_pdf_page_count(pdf_path: Path) -> int:
    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def parse_company_and_year_from_filename(pdf_path: Path) -> Tuple[str, int]:
    """
    Company is taken from parent folder name.
    Year is taken from final underscore chunk.
    """
    company = pdf_path.parent.name
    parts = pdf_path.stem.split("_")
    year_str = parts[-1]
    if not year_str.isdigit():
        raise ValueError(f"Could not parse year from filename: {pdf_path.name}")
    return company, int(year_str)


def tabula_extract_tables_for_page(pdf_path: Path, page_number: int) -> List[Tuple[str, pd.DataFrame]]:
    """
    Try lattice then stream for a single page.
    Returns list of (mode, dataframe).
    """
    results: List[Tuple[str, pd.DataFrame]] = []
    attempts = [
        ("lattice", {"lattice": True, "stream": False}),
        ("stream", {"lattice": False, "stream": True}),
    ]

    for mode_name, mode_kwargs in attempts:
        try:
            dfs = tabula.read_pdf(
                input_path=str(pdf_path),
                pages=str(page_number),
                multiple_tables=True,
                guess=True,
                **mode_kwargs,
            )
            if not dfs:
                continue
            for df in dfs:
                if df is None or df.empty:
                    continue
                results.append((mode_name, df))
        except Exception:
            continue

    return results


def write_jsonl_record(fp, record: Dict[str, Any]) -> None:
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    total_tables_saved = 0
    total_pdfs_processed = 0

    with INDEX_FILE.open("w", encoding="utf-8") as index_fp:
        for company_dir in RAW_DATA_DIR.iterdir():
            if not company_dir.is_dir():
                continue

            for pdf_path in sorted(company_dir.glob("*.pdf")):
                total_pdfs_processed += 1

                try:
                    company, report_year = parse_company_and_year_from_filename(pdf_path)
                except ValueError as e:
                    print(f"[SKIP] {e}")
                    continue

                print(f"\nProcessing: {pdf_path} (company={company}, year={report_year})")

                try:
                    page_count = get_pdf_page_count(pdf_path)
                except Exception as e:
                    print(f"[ERROR] Failed to open PDF for page count: {pdf_path.name} -> {e}")
                    continue

                for page_number in range(1, page_count + 1):
                    extracted = tabula_extract_tables_for_page(pdf_path, page_number)
                    if not extracted:
                        continue
                    table_idx_on_page = 0

                    for extraction_mode, raw_df in extracted:
                        cleaned_df = clean_table_df(raw_df)
                        if cleaned_df.empty:
                            continue

                        table_idx_on_page += 1
                        total_tables_saved += 1

                        table_id = f"{company}_{report_year}_p{page_number:03d}_t{table_idx_on_page:02d}"
                        csv_filename = f"{table_id}.csv"
                        csv_path = TABLES_DIR / csv_filename

                        cleaned_df.to_csv(csv_path, index=False)

                        preview = top_left_preview(cleaned_df, max_rows=3, max_cols=5)
                        currency_hint = infer_currency_hint(preview)
                        units_hint = infer_units_hint(preview)
                        table_type_guess = infer_table_type(preview)
                        keywords = extract_keywords(preview)
                        header_quality = header_quality_label(cleaned_df, preview)

                        record: Dict[str, Any] = {
                            "table_id": table_id,
                            "company": company,
                            "report_year": report_year,
                            "source_pdf": pdf_path.name,
                            "source_pdf_path": str(pdf_path.as_posix()),
                            "extraction_tool": "tabula",
                            "extraction_mode": extraction_mode,
                            "page_start": page_number,
                            "page_end": page_number,
                            "csv_filename": csv_filename,
                            "csv_path": str(csv_path.as_posix()),
                            "table_index_on_page": table_idx_on_page,
                            "table_type_guess": table_type_guess,
                            "units_hint": units_hint,
                            "currency_hint": currency_hint,
                            "header_quality": header_quality,
                            "row_count": int(cleaned_df.shape[0]),
                            "col_count": int(cleaned_df.shape[1]),
                            "top_left_preview": preview,
                            "keywords": keywords,
                            "notes": (
                                "Auto-generated from Tabula output. "
                                "Use preview + keywords to locate correct header row and year columns if needed."
                            ),
                        }

                        write_jsonl_record(index_fp, record)

                        print(
                            f"  [OK] p{page_number:03d} t{table_idx_on_page:02d} "
                            f"mode={extraction_mode} rows={record['row_count']} cols={record['col_count']} -> {csv_filename}"
                        )

    print("\n==============================")
    print("Table index build summary")
    print(f"PDF files processed:{total_pdfs_processed}")
    print(f"Tables saved:{total_tables_saved}")
    print(f"Index file:{INDEX_FILE.resolve()}")
    print(f"Tables directory:{TABLES_DIR.resolve()}")


if __name__ == "__main__":
    main()
