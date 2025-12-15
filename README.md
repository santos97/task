# Financial RAG System for Automotive Companies

## Task Overview

The goal of this task was to build a **Retrieval-Augmented Generation (RAG)** system using the provided data.

The system is designed to:
- Extract key financial metrics such as **revenue, EBITDA, and growth figures**
- Answer analytical questions across **multiple car companies**
- Support **comparisons across years, companies, and sectors**
- Provide answers strictly grounded in the available data

---

## Approach

### PDF Text Data Extraction
Annual report PDFs were processed to extract financial narrative text such as summaries and performance discussions.  
The text was cleaned, split into chunks, and stored with metadata like company name, report year, page number, file name to support accurate retrieval.

### PDF Table Data Extraction
Financial tables embedded in PDFs were extracted separately and converted into structured **CSV files**.  
Each extracted table was linked back to its source PDF pages and report year to ensure traceability between numeric data and the original document.

Table metadata for reliable table linking: 
Since table extraction from PDFs is not always fully accurate, additional structured metadata was maintained in the form of metadata.  
The metadata capture how and where each table appears in the PDF and help the RAG system reason about tables more reliably.

Each metadata includes details such as:
- A stable table ID that uniquely identifies the table and matches the CSV filename
- Company name and report year
- Source PDF and page range where the table appears
- Extraction method and tool used

#### Example metadata
```json
{
  "table_id": "Ford_2022_p085_t02",
  "company": "Ford",
  "report_year": 2022,
  "source_pdf": "Ford_Annual_Report_2022.pdf",
  "source_pdf_path": "data/raw/Ford/Ford_Annual_Report_2022.pdf",
  "extraction_tool": "tabula",
  "extraction_mode": "lattice",
  "page_start": 85,
  "page_end": 85,
  "csv_filename": "Ford_2022_p085_t02.csv",
  "csv_path": "data/processed/tables/Ford_2022_p085_t02.csv",
  "table_index_on_page": 2,
  "table_type_guess": "income_statement",
  "units_hint": "USD in millions",
  "currency_hint": "USD",
  "header_quality": "low",
  "row_count": 34,
  "col_count": 5,
  "top_left_preview": [
    ["Consolidated Statements of Operations", "", "", "", ""],
    ["(in millions)", "2022", "2021", "2020", ""],
    ["Revenue", "158,057", "136,341", "95,023", ""]
  ],
  "keywords": [
    "revenue",
    "net income",
    "profit",
    "operating income",
    "income statement",
    "consolidated"
  ],
  "notes": "Headers appear on row 2; first row is a title line; verify scaling '(in millions)'."
}
```
### Indexing with FAISS (Text and Tables)
Two separate FAISS indexes were created:
- One for **text data** extracted from PDFs
- One for **table data**, with references to the underlying CSV files  

This separation helps improve retrieval accuracy for both descriptive and numeric queries.

### RAG Agent Using LangChain
A simple RAG agent was built using **LangChain**.  
User queries are analyzed to identify companies, years, and whether numeric data is required.  
Relevant text sections and table data are retrieved from FAISS and provided to the language model as evidence.  
The prompt enforces that answers must rely only on the retrieved evidence and clearly state when information is missing.  
Basic conversational memory is used to support follow-up questions while avoiding rigid assumptions.

---

## Issues Encountered

- **Tabular data extraction**: Inconsistent table formats across PDFs made reliable extraction difficult.
- **Agent prompting**: Balancing strict evidence usage with natural responses required multiple prompt iterations.
- **Query processing and retrieval quality**: Generic queries involving multiple companies and years were challenging to interpret accurately.
- **Context management**: Maintaining conversation context without incorrectly assuming company or year information.
- **Multi-company comparisons**: Handling uneven data availability across different companies.

---
## Files

- text_pdfplumber.py : extracts and cleans narrative text from company annual report PDFs and stores them as structured text chunks with metadata. This generates
-     ./data/processed/text_chunks.jsonl
- tables_tabula.py : Extracts  tables from annual report PDFs using Tabula and saves them as CSV files with table cards. This generates
-     ./data/processed/table_index.jsonl
- fiass_index_text.py: Creates a FAISS vector index over extracted PDF text chunks to enable semantic retrieval of narrative content. This generates:
-      ./data/processed/index/faiss_text/index.faiss
-      ./data/processed/index/faiss_text/index.pkl
-  fiass_index_tables.py: Creates a FAISS vector index over table reference metadata to enable semantic discovery of relevant financial tables. This generates:
-      .data/processed/index/faiss_tables/index.faiss
-      .data/processed/index/faiss_tables/index.pkl
-  rag_agent.py: Impelemnts RAG agent which accepts user queries and respond with evideance
-  agentrunoutputs.ipynb : Outputs of the questions executed as part of the task. (Recommended to see this and use this to run agents)
-  requirements.txt: Lists the required lib required to run this all above steps

