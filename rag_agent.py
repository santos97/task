





from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


FAISS_TEXT_DIR = Path("/kaggle/input/consigli/data/processed/index/faiss_text")
FAISS_TABLES_DIR = Path("/kaggle/input/consigli/data/processed/index/faiss_tables")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"

MAX_CHAT_HISTORY_TURNS = 8  

NUMERIC_KEYWORDS = {
    "revenue",
    "profit",
    "profits",
    "net income",
    "income",
    "ebit",
    "ebitda",
    "margin",
    "sales",
    "growth",
}

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_numeric_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in NUMERIC_KEYWORDS)


def extract_target_years(query: str) -> List[int]:
    """Years mentioned in the question (may be older than available report PDFs)."""
    return [int(y) for y in re.findall(r"\b(20\d{2})\b", query)]


def extract_companies(query: str) -> List[str]:
    """Extract company mentions from a query."""
    q = query.lower()
    companies = []
    if "bmw" in q:
        companies.append("BMW")
    if "ford" in q:
        companies.append("Ford")
    if "tesla" in q:
        companies.append("Tesla")
    return companies


def parse_context_only_company(user_message: str) -> Optional[str]:
    """
    Detect messages that are essentially just setting context
    Returns the company if detected, else None.
    """
    msg = normalize_text(user_message).lower()

    patterns = [
        r"^(it'?s\s+for\s+)(tesla|bmw|ford)$",
        r"^(for\s+)(tesla|bmw|ford)$",
        r"^(tesla|bmw|ford)$",
        r"^(tesla|bmw|ford)\s+(please|pls)$",
    ]

    for pat in patterns:
        m = re.match(pat, msg)
        if m:
            company_token = m.group(len(m.groups()))  # last group is company
            return company_token.capitalize() if company_token != "bmw" else "BMW"

    if "tesla" in msg and len(msg.split()) <= 6 and ("for" in msg or "its" in msg or "it's" in msg):
        return "Tesla"
    if "bmw" in msg and len(msg.split()) <= 6 and ("for" in msg or "its" in msg or "it's" in msg):
        return "BMW"
    if "ford" in msg and len(msg.split()) <= 6 and ("for" in msg or "its" in msg or "it's" in msg):
        return "Ford"

    return None


def rewrite_query_with_active_context(
    user_query: str,
    active_company: Optional[str],
    active_target_years: List[int],
) -> str:
    """
    If the user query doesn't name a company, inject the active company.
    If the user query doesn't include any years but we have active_target_years, optionally inject them.
    This improves retrieval and helps the LLM behave like it "remembers" the context.
    """
    rewritten = user_query.strip()

    if active_company and not extract_companies(rewritten):
        rewritten = f"{active_company}: {rewritten}"

    if active_target_years and not extract_target_years(rewritten):
        if is_numeric_query(rewritten) or "compare" in rewritten.lower() or "trend" in rewritten.lower():
            years_str = ", ".join(str(y) for y in active_target_years)
            rewritten = f"{rewritten} (years: {years_str})"

    return rewritten


def build_company_report_years_map(text_store: FAISS) -> Dict[str, List[int]]:
    years_map: Dict[str, set] = {}

    doc_dict = getattr(text_store.docstore, "_dict", {})
    for doc in doc_dict.values():
        meta = doc.metadata or {}
        company = meta.get("company")
        report_year = meta.get("report_year")
        if company and isinstance(report_year, int):
            years_map.setdefault(company, set()).add(report_year)

    return {c: sorted(list(yrs)) for c, yrs in years_map.items()}


def build_retrieval_plan(
    user_query: str,
    convo_state: Dict[str, Any],
    company_report_years: Dict[str, List[int]],
) -> Dict[str, Any]:
    """
    Distinguish:
    - target_years: years user asked about (can be 2017)
    - report_years_to_search: actual PDF report years we have (2021–2023)
    """
    companies = extract_companies(user_query) or ([convo_state["active_company"]] if convo_state.get("active_company") else [])
    target_years = extract_target_years(user_query) or convo_state.get("active_target_years", [])

    is_numeric = is_numeric_query(user_query)
    needs_comparison = len(companies) > 1 or len(target_years) > 1

    company_to_report_years: Dict[str, List[int]] = {}
    for company in companies:
        available_years = company_report_years.get(company, [])

        if not available_years:
            company_to_report_years[company] = [None]
            continue

        requested_report_years = [y for y in target_years if y in available_years]
        if requested_report_years:
            company_to_report_years[company] = sorted(requested_report_years)
        else:
            company_to_report_years[company] = available_years

    return {
        "is_numeric": is_numeric,
        "companies": companies,
        "target_years": target_years,
        "company_to_report_years": company_to_report_years,
        "needs_comparison": needs_comparison,
    }


def retrieve_text_context(
    store: FAISS,
    query: str,
    company: str,
    report_year: Optional[int],
    k: int = 7,
) -> List[Document]:
    docs = store.similarity_search(query, k=k)
    filtered: List[Document] = []
    for d in docs:
        meta = d.metadata or {}
        if company and meta.get("company") != company:
            continue
        if report_year is not None and meta.get("report_year") != report_year:
            continue
        filtered.append(d)
    return filtered


def retrieve_tables(
    store: FAISS,
    query: str,
    company: str,
    report_year: Optional[int],
    k: int = 12,
) -> List[Document]:
    docs = store.similarity_search(query, k=k)
    filtered: List[Document] = []
    for d in docs:
        meta = d.metadata or {}
        if company and meta.get("company") != company:
            continue
        if report_year is not None and meta.get("report_year") != report_year:
            continue
        filtered.append(d)
    return filtered


def score_table_with_text_and_target_year(
    table_doc: Document,
    text_docs: List[Document],
    target_years: List[int],
    prefer_report_year: Optional[int],
) -> int:
    score = 0
    tmeta = table_doc.metadata or {}
    ttext = (table_doc.page_content or "").lower()

    table_company = tmeta.get("company")
    table_report_year = tmeta.get("report_year")
    page_start = tmeta.get("page_start")
    page_end = tmeta.get("page_end")

    for y in target_years:
        if str(y) in ttext:
            score += 4

    if prefer_report_year is not None and table_report_year == prefer_report_year:
        score += 2

    for td in text_docs:
        meta = td.metadata or {}
        text_company = meta.get("company")
        text_report_year = meta.get("report_year")
        text_page = meta.get("page")

        if table_company and text_company and table_company == text_company:
            score += 2
        if prefer_report_year is not None and text_report_year == prefer_report_year:
            score += 1

        if isinstance(text_page, int) and isinstance(page_start, int) and isinstance(page_end, int):
            if page_start <= text_page <= page_end:
                score += 6

        text_content = (td.page_content or "").lower()
        for kw in NUMERIC_KEYWORDS:
            if kw in ttext and kw in text_content:
                score += 1

    return score


def select_best_tables(
    table_docs: List[Document],
    text_docs: List[Document],
    target_years: List[int],
    prefer_report_year: Optional[int],
    top_n: int = 2,
) -> List[Document]:
    if not table_docs:
        return []
    scored: List[Tuple[int, Document]] = []
    for t in table_docs:
        scored.append((score_table_with_text_and_target_year(t, text_docs, target_years, prefer_report_year), t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for s, t in scored[:top_n] if s > 0]

def build_csv_snippet(csv_path: Path, user_query: str, target_years: List[int]) -> str:
    if not csv_path.exists():
        return f"[Missing CSV: {csv_path.as_posix()}]"

    try:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
    except Exception as e:
        return f"[Failed to read CSV {csv_path.name}: {e}]"

    df = df.applymap(normalize_text)

    q = user_query.lower()
    metric_terms = [k for k in NUMERIC_KEYWORDS if k in q]
    target_year_strs = [str(y) for y in target_years]

    lines: List[str] = []
    lines.append(f"CSV_FILE: {csv_path.as_posix()}")
    lines.append("CSV_TOP_PREVIEW (first 12 rows, first 8 cols):")
    lines.append(df.iloc[:12, :8].to_string(index=False))

    if metric_terms:
        lines.append("\nCSV_METRIC_ROWS (heuristic match, first 8 cols):")
        hits = []
        for i in range(len(df)):
            row_text = " | ".join(df.iloc[i, :].tolist()).lower()
            if any(term in row_text for term in metric_terms):
                hits.append(i)
        for i in hits[:8]:
            lines.append(df.iloc[i:i+1, :8].to_string(index=False))

    if target_year_strs:
        lines.append("\nCSV_TARGET_YEAR_ROWS (rows containing target year, first 8 cols):")
        hits = []
        for i in range(len(df)):
            row_text = " | ".join(df.iloc[i, :].tolist())
            if any(y in row_text for y in target_year_strs):
                hits.append(i)
        for i in hits[:6]:
            lines.append(df.iloc[i:i+1, :8].to_string(index=False))

    return "\n".join(lines)

def build_prompt(
    user_query: str,
    chat_history: List[Dict[str, str]],
    text_docs: List[Document],
    table_docs: List[Document],
    csv_snippets: List[str],
    active_company: Optional[str],
) -> List[Dict[str, str]]:
    """
    Important: we explicitly tell the model the active company context,
    so it stops asking "which company?" unless truly ambiguous.
    """
    text_blocks = [
        f"[PDF:{d.metadata.get('source_file')} p.{d.metadata.get('page')}] {normalize_text(d.page_content)[:700]}"
        for d in text_docs[:10]
    ]

    table_blocks = [
        f"[TABLE:{t.metadata.get('table_id')}] "
        f"company={t.metadata.get('company')} report_year={t.metadata.get('report_year')} "
        f"pages={t.metadata.get('page_start')}-{t.metadata.get('page_end')} "
        f"csv={t.metadata.get('csv_path')}"
        for t in table_docs[:6]
    ]

    system = (
        "You are a supportive, professional financial analyst assistant.\n"
        "Your job is to help the user get to the right answer with a friendly tone.\n"
        "Rules:\n"
        "1) Use ONLY the evidence provided.\n"
        "2) Cite narrative claims using [PDF:FILE p.PAGE].\n"
        "3) For numeric answers, cite the CSV file path and specify units/currency if visible.\n"
        "4) If asked to compare multiple years/companies, list the values used before concluding.\n"
        "5) If evidence is missing, say so and state what is missing instead of guessing. Dont give technical details as which CSV is missing or what is to be done next. \n"
        "6) If the user does not specify a company, assume the active company context provided.\n"
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(chat_history[-MAX_CHAT_HISTORY_TURNS:])

    context_line = f"ACTIVE_COMPANY_CONTEXT: {active_company}" if active_company else "ACTIVE_COMPANY_CONTEXT: (none)"

    user_msg = f"""
{context_line}

User question:
{user_query}

PDF EVIDENCE:
{chr(10).join(text_blocks) if text_blocks else "(none)"}

TABLE INDEX EVIDENCE:
{chr(10).join(table_blocks) if table_blocks else "(none)"}

CSV EVIDENCE:
{chr(10).join(csv_snippets) if csv_snippets else "(none)"}
"""
    messages.append({"role": "user", "content": user_msg.strip()})
    return messages



def main() -> None:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    text_store = FAISS.load_local(
        str(FAISS_TEXT_DIR), embeddings, allow_dangerous_deserialization=True
    )
    table_store = FAISS.load_local(
        str(FAISS_TABLES_DIR), embeddings, allow_dangerous_deserialization=True
    )

    company_report_years = build_company_report_years_map(text_store)

    api_key =  secret_value_0
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set. Please export GROQ_API_KEY first.")

    llm = ChatOpenAI(
        model=GROQ_MODEL_NAME,
        api_key=api_key,
        base_url=GROQ_BASE_URL,
        temperature=0.2,
    )

    chat_history: List[Dict[str, str]] = []
    convo_state: Dict[str, Any] = {
        "active_company": None, # current conversation context
        "active_target_years": [], # last referenced years (for follow-ups)
        "pending_question": None,  # last question awaiting company clarification
    }

    print("\nChatbot ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You----> ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        context_company = parse_context_only_company(user_input)
        if context_company:
            convo_state["active_company"] = context_company

            if convo_state.get("pending_question"):
                user_query = convo_state["pending_question"]
                convo_state["pending_question"] = None
            else:
                print(f"\nAssistant>\nUnderstood. I’ll assume {context_company} going forward.\n")
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": f"Understood. Active company is now {context_company}."})
                continue
        else:
            user_query = user_input


        rewritten_query = rewrite_query_with_active_context(
            user_query=user_query,
            active_company=convo_state.get("active_company"),
            active_target_years=convo_state.get("active_target_years", []),
        )

        if not extract_companies(rewritten_query) and not convo_state.get("active_company"):
            convo_state["pending_question"] = user_query
            assistant_text = "Which company is this for (Tesla, BMW, or Ford)?"
            print(f"\nAssistant>\n{assistant_text}\n")
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": assistant_text})
            continue

        explicit_years = extract_target_years(user_query)
        if explicit_years:
            convo_state["active_target_years"] = explicit_years


        plan = build_retrieval_plan(
            user_query=rewritten_query,
            convo_state=convo_state,
            company_report_years=company_report_years,
        )


        merged_text_docs: List[Document] = []
        merged_table_docs: List[Document] = []
        merged_csv_snippets: List[str] = []

        for company in plan["companies"]:
            report_years_to_search = plan["company_to_report_years"].get(company, [None])

            for report_year in report_years_to_search:
                text_docs = retrieve_text_context(text_store, rewritten_query, company, report_year, k=7)

                best_tables: List[Document] = []
                if plan["is_numeric"]:
                    table_candidates = retrieve_tables(table_store, rewritten_query, company, report_year, k=12)
                    best_tables = select_best_tables(
                        table_docs=table_candidates,
                        text_docs=text_docs,
                        target_years=plan["target_years"],
                        prefer_report_year=report_year,
                        top_n=2,
                    )

                    for t in best_tables:
                        csv_path = Path(t.metadata["csv_path"])
                        merged_csv_snippets.append(build_csv_snippet(csv_path, rewritten_query, plan["target_years"]))

                merged_text_docs.extend(text_docs)
                merged_table_docs.extend(best_tables)

        messages = build_prompt(
            user_query=rewritten_query,
            chat_history=chat_history,
            text_docs=merged_text_docs,
            table_docs=merged_table_docs,
            csv_snippets=merged_csv_snippets,
            active_company=convo_state.get("active_company"),
        )

        try:
            response_text = llm.invoke(messages).content.strip()
        except Exception as e:
            response_text = f"[ERROR] LLM call failed: {e}"

        print(f"\nAssistant----------->{response_text}")
        print("--"*25)

        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()

