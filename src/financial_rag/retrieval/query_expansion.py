"""Query expansion module using bilingual Vietnamese-English financial glossary."""

import re
from typing import List, Set
from financial_rag.db.repository import FinancialRepository


class QueryExpander:
    """Expands query terms across Vietnamese and English using the metric glossary."""

    def __init__(self, repo: FinancialRepository):
        self.repo = repo
        self.metrics = self.repo.get_all_metrics()

    def expand(self, query: str) -> str:
        """
        Expand terms found in query with corresponding aliases.
        E.g. 'Doanh thu thuần' -> 'Doanh thu thuần OR "Net revenue" OR "Revenue"'
        """
        q_lower = query.lower()
        matched_aliases: Set[str] = set()

        for m in self.metrics:
            names = [m["name_vi"].lower(), m["name_en"].lower()]
            aliases = [a.strip().lower() for a in (m["aliases"] or "").split("|") if a.strip()]
            all_terms = names + aliases

            # Check if any term is in the query
            for term in all_terms:
                if len(term) >= 3 and term in q_lower:
                    # Add top English and Vietnamese counterparts
                    matched_aliases.add(m["name_vi"])
                    matched_aliases.add(m["name_en"])
                    # Add first 2 aliases
                    for a in aliases[:2]:
                        matched_aliases.add(a)
                    break

        if not matched_aliases:
            return query

        expanded_terms = [f'"{term}"' for term in matched_aliases]
        return f"{query} {' '.join(expanded_terms)}"
