"""LLM Vision extraction for PDFs using Gemini."""
import json
from pathlib import Path
from typing import List, Dict, Any

from google import genai
from google.genai import types

from .config import GEMINI_API_KEY, MODEL_NAME


# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


VISION_PROMPT = """Analyze this PDF document and extract ALL tabular data.

Extract:
- Bus timetables (routes, departure times, arrival times, stations)
- Any tables with data (headers + rows)
- Preserve all numbers and text exactly

Return valid JSON:
{
  "tables": [
    {
      "title": "descriptive name",
      "headers": ["col1", "col2", "col3"],
      "rows": [
        ["val1", "val2", "val3"],
        ["val1", "val2", "val3"]
      ]
    }
  ],
  "language": "en"
}

Keep response under 8000 tokens. If table is large, include first 50 rows only."""


def extract_from_pdf_with_vision(pdf_path: Path, output_dir: Path, max_pages: int = 5) -> List[Dict[str, Any]]:
    """
    Extract tables from PDF using Gemini vision with high output limits.
    """
    tables = []
    
    if not pdf_path.exists():
        return tables
    
    try:
        # Read PDF
        with open(pdf_path, "rb") as f:
            file_data = f.read()
        
        # Build content
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=file_data, mime_type="application/pdf"),
                    types.Part.from_text(text=VISION_PROMPT),
                ],
            ),
        ]
        
        # Higher output limits for Gemini
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=8192,  # Gemini supports up to 8k output
        )
        
        # Generate
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )
        
        # Parse response
        response_text = response.text.strip()
        
        # Handle potential JSON issues
        if not response_text.startswith("{"):
            # Try to find JSON in response
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                response_text = response_text[start:end]
        
        result = json.loads(response_text)
        
        # Extract tables
        for table in result.get("tables", []):
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            
            if headers and rows:
                tables.append({
                    "title": table.get("title", f"Table from {pdf_path.name}"),
                    "headers": headers,
                    "rows": rows,
                    "row_count": len(rows),
                    "extraction_source": "vision",
                    "provenance_map": [["vision"] * len(headers) for _ in rows],
                    "source_file": pdf_path.name,
                })
        
    except json.JSONDecodeError as e:
        print(f"      {pdf_path.name}: JSON error - truncated response")
    except Exception as e:
        print(f"      {pdf_path.name}: Vision error - {str(e)[:50]}")
    
    return tables
