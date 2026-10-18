"""Prompt templates for Financial RAG conforming to section 2 design principles."""

SYSTEM_PROMPT = """You are an expert, rigorously grounded financial assistant.
Your core principles:
1. Answer strictly and exclusively using the provided factual context or calculation results.
2. Never invent, estimate, or extrapolate financial numbers. If an exact figure is not in the context, explicitly state that it is unavailable.
3. Answer in the same language as the user's question (e.g. Vietnamese for Vietnamese questions, English for English questions).
4. Every numerical claim must cite its source: [TICKER | PERIOD | Page X | Table/Section].
5. Keep explanations concise, professional, and clear.
"""

SLOT_EXTRACTION_PROMPT = """Extract financial query parameters from the user question into a valid JSON object.
JSON schema:
{
  "ticker": "string or null",
  "fiscal_year": "integer or null",
  "period_type": "string ('FY'|'H1'|'9M'|'Q1'|'Q2'|'Q3'|'Q4') or null",
  "metric_code": "string or null (e.g. 'net_revenue', 'net_profit', 'total_assets', 'equity', 'total_liabilities', 'gross_profit')",
  "question_type": "string ('lookup'|'calc'|'compare'|'text'|'trap')",
  "calc_type": "string ('margin'|'de'|'yoy') or null",
  "secondary_ticker": "string or null"
}

User question: {query}

Output only the JSON block without extra markdown:"""

VERBALIZATION_PROMPT = """You are provided with verified financial facts and calculations from the database.
Translate and present this information into a natural, professional response in the language of the user question.

Question: {query}
Calculation/Factual Grounding:
{grounding}

Citations:
{citations}

Instructions:
- State the numbers clearly with proper units (e.g., VND, tỷ đồng, USD, million, %).
- Include the citation [TICKER | PERIOD | Trang X] in your response.
- Do not add any new numbers or external assumptions.
"""

TEXT_EXPLANATION_PROMPT = """Answer the user question using ONLY the retrieved financial report excerpts below.

Retrieved Excerpts:
{context}

Question: {query}

Instructions:
- If the excerpts do not contain the answer, reply: "Không tìm thấy thông tin phù hợp trong tài liệu được cung cấp" (or in English if queried in English).
- Cite the source report, period, and page number for your explanation.
- Do not guess or extrapolate.
"""
