"""Extracts structured financial facts from parsed tables and reports."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from financial_rag.db.repository import FinancialRepository
from financial_rag.parser.number_normalizer import NumberNormalizer
from financial_rag.parser.pdf_parser import DocumentParseResult, ParsedTable


logger = logging.getLogger(__name__)


class FactExtractor:
    """Extracts standardized and raw financial facts from document tables."""

    def __init__(self, repo: FinancialRepository):
        self.repo = repo
        self.metrics = self.repo.get_all_metrics()

    def determine_column_roles(self, headers: List[str], fiscal_year: int) -> Tuple[int, List[Optional[str]]]:
        """
        Identify (item_col_idx, roles) where roles[col_idx] is 'current', 'prior_year_same_period', 'year_start', or None.
        """
        prior_year = fiscal_year - 1
        item_col_idx = 0

        # Check if first column is code/STT
        if headers:
            h0 = str(headers[0]).lower().strip()
            if any(kw in h0 for kw in ["mã số", "stt", "code", "no."]) and len(headers) > 1:
                item_col_idx = 1

        roles: List[Optional[str]] = [None] * len(headers)
        data_col_count = 0

        for idx, h in enumerate(headers):
            if idx == item_col_idx:
                roles[idx] = None
                continue

            h_clean = str(h).lower().strip()
            if any(kw in h_clean for kw in ["mã số", "stt", "thuyết minh", "note", "notes", "tm"]):
                roles[idx] = None
                continue

            # Prior year check
            if str(prior_year) in h_clean or any(kw in h_clean for kw in ["năm trước", "kỳ trước", "prior", "cùng kỳ", "previous"]):
                roles[idx] = "prior_year_same_period"
            # Year start check
            elif any(kw in h_clean for kw in ["đầu năm", "beginning", "số đầu kỳ", "year start"]):
                roles[idx] = "year_start"
            # Current year check
            elif str(fiscal_year) in h_clean or any(kw in h_clean for kw in ["năm nay", "kỳ này", "current", "cuối kỳ", "số cuối năm"]):
                roles[idx] = "current"
            else:
                # Default by appearance after item column
                if idx > item_col_idx:
                    data_col_count += 1
                    if data_col_count == 1:
                        roles[idx] = "current"
                    elif data_col_count == 2:
                        roles[idx] = "prior_year_same_period"

        return item_col_idx, roles

    def extract_from_parsed_document(
        self,
        parse_result: DocumentParseResult,
        report_id: int,
        fiscal_year: int,
        unit_scale: int = 1,
    ) -> List[Dict[str, Any]]:
        extracted_facts: List[Dict[str, Any]] = []

        for page in parse_result.pages:
            for table in page.tables:
                table_facts = self.extract_from_table(
                    table=table,
                    report_id=report_id,
                    fiscal_year=fiscal_year,
                    unit_scale=unit_scale,
                )
                extracted_facts.extend(table_facts)

            # Also scan narrative lines with strong pattern: "Doanh thu thuần: 1.234 tỷ"
            text_facts = self.extract_from_narrative_text(
                text=page.text,
                page_num=page.page_number,
                report_id=report_id,
                fiscal_year=fiscal_year,
                unit_scale=unit_scale,
            )
            extracted_facts.extend(text_facts)

        return extracted_facts

    def extract_from_table(
        self,
        table: ParsedTable,
        report_id: int,
        fiscal_year: int,
        unit_scale: int = 1,
    ) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        if not table.headers or not table.rows:
            return facts

        item_col_idx, roles = self.determine_column_roles(table.headers, fiscal_year)

        for row in table.rows:
            if not row:
                continue

            # Case A: All text and numbers merged into row[0] (borderless PDF tables)
            if len(row) == 1 or all(not str(c).strip() for c in row[1:]):
                raw_cell = str(row[0]).strip()
                num_matches = list(re.finditer(r"(?:\(?\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?\s*\)?|\b\d+\b)", raw_cell))
                if num_matches:
                    first_num_pos = num_matches[0].start()
                    clean_item = raw_cell[:first_num_pos].rstrip("$€£: \t-—")
                    if len(clean_item) >= 3:
                        matched_metric = self.repo.match_metric_alias(clean_item)
                        metric_id = matched_metric["metric_id"] if matched_metric else None
                        nums = [m.group(0).strip() for m in num_matches]
                        if len(nums) >= 1:
                            v1 = NumberNormalizer.parse_number(nums[0])
                            if v1 is not None and v1 != 0:
                                facts.append({
                                    "report_id": report_id,
                                    "metric_id": metric_id,
                                    "line_item": clean_item,
                                    "column_role": "current",
                                    "value": v1 * unit_scale,
                                    "value_raw": nums[0],
                                    "page": table.page,
                                    "table_ref": table.table_id,
                                })
                        if len(nums) >= 2:
                            v2 = NumberNormalizer.parse_number(nums[1])
                            if v2 is not None and v2 != 0:
                                facts.append({
                                    "report_id": report_id,
                                    "metric_id": metric_id,
                                    "line_item": clean_item,
                                    "column_role": "prior_year_same_period",
                                    "value": v2 * unit_scale,
                                    "value_raw": nums[1],
                                    "page": table.page,
                                    "table_ref": table.table_id,
                                })
                continue

            # Case B: Multi-column table
            if len(row) <= item_col_idx:
                continue

            line_item = str(row[item_col_idx]).strip()
            # Clean leading roman numerals / bullets (e.g., "1. Doanh thu" -> "Doanh thu")
            clean_item = re.sub(r"^(?:[A-ZIVX\d]+\.|\-|\*)\s*", "", line_item).strip()

            # Skip empty or header-like rows
            if not clean_item or clean_item.startswith("---") or clean_item.lower() in ("chỉ tiêu", "item", "stt", "tài sản", "nguồn vốn"):
                continue

            matched_metric = self.repo.match_metric_alias(clean_item) or self.repo.match_metric_alias(line_item)
            metric_id = matched_metric["metric_id"] if matched_metric else None

            # Extract numbers for assigned roles
            for col_idx, role in enumerate(roles):
                if role is None or col_idx >= len(row):
                    continue

                raw_val = str(row[col_idx]).strip()
                val = NumberNormalizer.parse_number(raw_val)
                if val is not None:
                    scaled_val = val * unit_scale
                    facts.append({
                        "report_id": report_id,
                        "metric_id": metric_id,
                        "line_item": clean_item,
                        "column_role": role,
                        "value": scaled_val,
                        "value_raw": raw_val,
                        "page": table.page,
                        "table_ref": table.table_id,
                    })

        return facts

    def extract_from_narrative_text(
        self,
        text: str,
        page_num: int,
        report_id: int,
        fiscal_year: int,
        unit_scale: int = 1,
    ) -> List[Dict[str, Any]]:
        facts: List[Dict[str, Any]] = []
        lines = text.splitlines()

        for line in lines:
            line_str = line.strip()
            if not line_str or len(line_str) > 200:
                continue

            # Look for colon separator: "Doanh thu thuần: 4.225.864.218.423 đồng"
            if ":" in line_str:
                parts = line_str.split(":", 1)
                item_name = parts[0].strip()
                val_part = parts[1].strip()

                matched_metric = self.repo.match_metric_alias(item_name)
                if matched_metric:
                    val = NumberNormalizer.parse_number(val_part)
                    if val is not None:
                        facts.append({
                            "report_id": report_id,
                            "metric_id": matched_metric["metric_id"],
                            "line_item": item_name,
                            "column_role": "current",
                            "value": val * unit_scale,
                            "value_raw": val_part,
                            "page": page_num,
                            "table_ref": f"narrative_p{page_num}",
                        })

        return facts
