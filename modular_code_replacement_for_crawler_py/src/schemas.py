from pydantic import BaseModel, Field
from typing import List, Any, Dict, Optional


# --- 1. LLM Extraction Schema (Strict: NO Defaults) ---
# Used inside client.models.generate_content(..., response_schema=ExtractionSchema)
# CRITICAL: No default values allowed - causes "Default value not supported" API error

class TableDataLLM(BaseModel):
    """Single table extracted from web content."""
    title: str
    headers: List[str]
    # Raw data matrix
    rows: List[List[Any]]
    # Provenance Matrix: "network", "dom", "vision"
    # Must match 'rows' dimensions (Row x Col)
    provenance_map: List[List[str]]


class ExtractionSchema(BaseModel):
    """Main LLM response schema for structured extraction."""
    summary: str
    tables: List[TableDataLLM]
    # LLM identifies the "Best" source file for the data
    used_network_files: List[str]


# --- 2. Internal Application Schema (Flexible: With Defaults) ---
# Used for final JSON output and processing

class TableDataInternal(TableDataLLM):
    """Extended table data with optional metadata."""
    confidence_score: float = 1.0
    metadata: Dict[str, Any] = {}
