"""Deterministic network JSON to table extraction (no LLM for numbers)."""
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field


# Common keys that often contain tabular data
TABULAR_KEYS = {"data", "results", "items", "records", "rows", "list", "values", 
                "entries", "ministries", "sectors", "categories", "datasets"}


@dataclass
class TableCandidate:
    """A candidate table extracted from network JSON."""
    title_guess: str
    headers: List[str]
    rows: List[List[Any]]
    provenance: str = "network"
    source_network_file: str = ""
    source_network_url: str = ""
    key_path: str = ""
    row_count: int = 0
    confidence: float = 0.0
    
    def __post_init__(self):
        self.row_count = len(self.rows)


def infer_type(value: Any) -> Any:
    """
    Infer and convert type for cleaner output.
    
    - digit-only strings → int
    - float-like strings → float
    - else keep as string
    """
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        s = value.strip()
        # Integer check
        if s.isdigit() or (s.startswith('-') and s[1:].isdigit()):
            return int(s)
        # Float check
        try:
            if re.match(r'^-?\d+\.?\d*$', s) or re.match(r'^-?\d*\.\d+$', s):
                return float(s)
        except:
            pass
    return value


def normalize_row(row: List[Any]) -> List[Any]:
    """Apply type inference to all values in a row."""
    return [infer_type(v) for v in row]


def is_consistent_list_of_dicts(data: list, min_items: int = 2) -> bool:
    """Check if list of dicts has consistent keys."""
    if not data or not isinstance(data, list):
        return False
    if len(data) < min_items:
        return len(data) >= 1 and isinstance(data[0], dict) and len(data[0]) > 0
    
    sample = data[:10]
    if not all(isinstance(item, dict) for item in sample):
        return False
    
    first_keys = set(sample[0].keys())
    if len(first_keys) == 0:
        return False
        
    for item in sample[1:]:
        overlap = len(first_keys & set(item.keys())) / max(len(first_keys), 1)
        if overlap < 0.7:
            return False
    return True


def extract_table_from_list_of_dicts(
    data: list, 
    key_path: str = "",
    source_file: str = "",
    source_url: str = ""
) -> Optional[TableCandidate]:
    """Convert a list of dicts to a TableCandidate with type normalization."""
    if not is_consistent_list_of_dicts(data, min_items=1):
        return None
    
    # Collect all unique headers from all items
    all_headers = set()
    for item in data[:50]:  # Check first 50 items
        all_headers.update(item.keys())
    headers = sorted(list(all_headers))
    
    if not headers:
        return None
    
    # Extract rows with type normalization
    rows = []
    for item in data:
        row = [item.get(h, None) for h in headers]
        rows.append(normalize_row(row))
    
    # Generate title from key path
    title = key_path.split(".")[-1] if key_path else "extracted_data"
    title = title.replace("_", " ").replace("-", " ").title()
    
    # Calculate confidence
    confidence = 0.5
    if len(rows) >= 5:
        confidence += 0.2
    if len(headers) >= 3:
        confidence += 0.1
    if any(isinstance(v, (int, float)) for v in rows[0] if v is not None):
        confidence += 0.2
    
    return TableCandidate(
        title_guess=title,
        headers=headers,
        rows=rows,
        key_path=key_path,
        source_network_file=source_file,
        source_network_url=source_url,
        confidence=min(confidence, 1.0),
    )


def extract_all_tables_from_json(
    obj: Any, 
    source_file: str = "",
    source_url: str = "",
    key_path: str = "",
    depth: int = 0,
    max_depth: int = 4
) -> List[TableCandidate]:
    """
    Extract ALL tabular datasets from JSON, one per key.
    
    For {"ministries": [...], "sectors": [...]}
    Returns two separate TableCandidates, not merged.
    """
    candidates = []
    
    if depth > max_depth:
        return candidates
    
    if isinstance(obj, list) and is_consistent_list_of_dicts(obj, min_items=2):
        candidate = extract_table_from_list_of_dicts(obj, key_path, source_file, source_url)
        if candidate and candidate.row_count >= 2:
            candidates.append(candidate)
    
    elif isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{key_path}.{key}" if key_path else key
            
            # Check if value is directly a tabular list
            if isinstance(value, list) and is_consistent_list_of_dicts(value, min_items=1):
                candidate = extract_table_from_list_of_dicts(value, new_path, source_file, source_url)
                if candidate and candidate.row_count >= 1:
                    # Boost confidence for known keys
                    if key.lower() in TABULAR_KEYS:
                        candidate.confidence += 0.15
                    candidates.append(candidate)
            else:
                # Recurse
                sub = extract_all_tables_from_json(value, source_file, source_url, new_path, depth + 1, max_depth)
                candidates.extend(sub)
    
    return candidates


def parse_network_files(raw_network_dir: Path) -> List[TableCandidate]:
    """Parse all network files and extract ALL table candidates."""
    all_candidates = []
    
    if not raw_network_dir.exists():
        return all_candidates
    
    for json_file in sorted(raw_network_dir.glob("*.json")):
        if json_file.name.endswith(".meta.json"):
            continue
        
        try:
            # Load metadata
            meta_file = json_file.with_suffix(".meta.json")
            source_url = ""
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                    source_url = meta.get("response_url", "")
            
            # Parse JSON
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            candidates = extract_all_tables_from_json(
                data, 
                source_file=json_file.name,
                source_url=source_url
            )
            all_candidates.extend(candidates)
            
        except Exception:
            continue
    
    # Sort by confidence, then row count
    all_candidates.sort(key=lambda c: (c.confidence, c.row_count), reverse=True)
    
    return all_candidates


def candidate_to_table_dict(candidate: TableCandidate) -> Dict[str, Any]:
    """Convert TableCandidate to full table dict with table-level metadata."""
    return {
        "title": candidate.title_guess,
        "headers": candidate.headers,
        "rows": candidate.rows,
        "row_count": candidate.row_count,
        # Table-level provenance (not just cell-level)
        "extraction_source": "network",
        "provenance_map": [["network"] * len(candidate.headers) for _ in candidate.rows],
        "source_network_files": [candidate.source_network_file] if candidate.source_network_file else [],
        "source_network_urls": [candidate.source_network_url] if candidate.source_network_url else [],
        "json_key_path": candidate.key_path,
        "confidence": round(candidate.confidence, 2),
    }
