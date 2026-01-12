"""Validation functions for extraction integrity."""
import json
import os
from datetime import datetime
from typing import List, Tuple, Optional
from pydantic import ValidationError

from .schemas import ExtractionSchema, TableDataLLM
from .config import DEBUG_DIR


class ValidationError(Exception):
    """Raised when extraction output fails validation."""
    pass


def validate_table_invariants(table: TableDataLLM) -> List[str]:
    """
    Validate table structure invariants.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    # 1. provenance_map length must match rows
    if len(table.provenance_map) != len(table.rows):
        errors.append(
            f"Table '{table.title}': provenance_map length ({len(table.provenance_map)}) "
            f"!= rows length ({len(table.rows)})"
        )
    
    # 2. Each provenance row must match corresponding data row
    for i, (prov_row, data_row) in enumerate(zip(table.provenance_map, table.rows)):
        if len(prov_row) != len(data_row):
            errors.append(
                f"Table '{table.title}' row {i}: provenance width ({len(prov_row)}) "
                f"!= data width ({len(data_row)})"
            )
    
    # 3. Headers must match row widths
    for i, row in enumerate(table.rows):
        if len(table.headers) != len(row):
            errors.append(
                f"Table '{table.title}' row {i}: headers ({len(table.headers)}) "
                f"!= row width ({len(row)})"
            )
            break  # Only report once per table
    
    # 4. No empty headers
    empty_headers = [i for i, h in enumerate(table.headers) if not h or not str(h).strip()]
    if empty_headers:
        errors.append(
            f"Table '{table.title}': empty headers at positions {empty_headers}"
        )
    
    # 5. No duplicate headers (after normalization)
    normalized = [str(h).strip().lower() for h in table.headers]
    seen = {}
    duplicates = []
    for i, h in enumerate(normalized):
        if h in seen:
            duplicates.append((h, seen[h], i))
        else:
            seen[h] = i
    if duplicates:
        errors.append(
            f"Table '{table.title}': duplicate headers {duplicates}"
        )
    
    return errors


def validate_extraction(result: ExtractionSchema) -> Tuple[bool, List[str]]:
    """
    Validate full extraction result.
    
    Returns (is_valid, list_of_errors).
    """
    all_errors = []
    
    # Validate each table
    for table in result.tables:
        table_errors = validate_table_invariants(table)
        all_errors.extend(table_errors)
    
    return len(all_errors) == 0, all_errors


def write_validation_debug(
    errors: List[str],
    extraction_result: dict,
    url: str
) -> str:
    """
    Write debug dump when validation fails.
    
    Returns path to debug file.
    """
    os.makedirs(DEBUG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_path = os.path.join(DEBUG_DIR, f"validation_fail_{timestamp}.json")
    
    debug_data = {
        "url": url,
        "timestamp": timestamp,
        "errors": errors,
        "extraction_result": extraction_result,
    }
    
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, indent=2, ensure_ascii=False)
    
    return debug_path


def check_tables_sanity(tables: list) -> bool:
    """
    Check if Crawl4AI result.tables are sane enough to use directly.
    
    Returns True if tables look usable.
    """
    if not tables:
        return False
    
    for table in tables:
        # Must have headers
        if not hasattr(table, 'headers') or not table.headers:
            return False
        
        # Must have rows
        if not hasattr(table, 'rows') or not table.rows:
            return False
        
        # Check row width consistency
        header_width = len(table.headers)
        for row in table.rows:
            if len(row) != header_width:
                return False
    
    return True
