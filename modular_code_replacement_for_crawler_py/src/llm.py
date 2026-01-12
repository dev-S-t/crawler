"""Google Gemini LLM client for structured data extraction."""
import os
import json
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Optional, Any
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from .schemas import ExtractionSchema, TableDataLLM
from .config import GEMINI_API_KEY, MODEL_NAME, THINKING_LEVEL, MAX_MARKDOWN_LENGTH, LLM_TIMEOUT_SECONDS


class LLMTimeoutError(Exception):
    """Raised when LLM call times out."""
    pass


def call_with_timeout(func, timeout_seconds=LLM_TIMEOUT_SECONDS):
    """Execute a function with timeout. Returns None on timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            print(f"   ⏰ LLM timeout after {timeout_seconds}s")
            return None


# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


def tables_to_schema(
    crawl_tables: list,
    summary: str = "Tables extracted directly from DOM."
) -> ExtractionSchema:
    """
    Convert Crawl4AI result.tables to ExtractionSchema.
    
    This bypasses LLM for table DATA - uses canonical numbers from DOM.
    Gemini is NOT used for number extraction here.
    """
    converted_tables = []
    
    for table in crawl_tables:
        headers = list(table.headers) if hasattr(table, 'headers') else []
        rows = [list(row) for row in table.rows] if hasattr(table, 'rows') else []
        
        # Build provenance_map - all "dom" since from crawl4ai
        provenance_map = [["dom"] * len(headers) for _ in rows]
        
        # Generate title from caption or position
        title = getattr(table, 'caption', None) or f"Table {len(converted_tables) + 1}"
        
        converted_tables.append(TableDataLLM(
            title=title,
            headers=headers,
            rows=rows,
            provenance_map=provenance_map,
        ))
    
    return ExtractionSchema(
        summary=summary,
        tables=converted_tables,
        used_network_files=[],
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
def normalize_tables_with_llm(
    tables: List[TableDataLLM],
    markdown_context: str,
    network_summaries: list = None
) -> ExtractionSchema:
    """
    Use Gemini to normalize/label pre-extracted tables.
    
    LLM is a "schema mapper" - it only adds titles, summaries, metadata.
    Numbers come from the pre-extracted tables (canonical).
    """
    if network_summaries is None:
        network_summaries = []
    
    # Convert tables to JSON for prompt
    tables_json = [t.model_dump() for t in tables]
    
    prompt = f"""
    You are given pre-extracted tables from a web page. Your job is to:
    1. Write a concise summary of what data these tables contain
    2. Improve table titles if they are generic (e.g., "Table 1")
    3. Do NOT modify the numbers/values in rows - they are canonical
    4. Keep headers and provenance_map exactly as provided
    
    Context (Markdown snippet for understanding):
    {markdown_context[:10000]}
    
    Pre-extracted Tables (canonical data):
    {json.dumps(tables_json, indent=2)}
    
    Available Network Data (for reference):
    {json.dumps(network_summaries, indent=2) if network_summaries else "None"}
    
    Return the tables with improved titles and a summary.
    """
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        response_mime_type="application/json",
        response_schema=ExtractionSchema,
    )
    
    def do_call():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generate_content_config,
        )
    
    response = call_with_timeout(do_call)
    if response is None:
        # Return empty schema on timeout
        return ExtractionSchema(summary="LLM timeout", tables=[], used_network_files=[])
    
    return response.parsed


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_from_markdown(markdown_text: str, network_summaries: list = None) -> ExtractionSchema:
    """
    Fallback: Extract tables from markdown when result.tables is unavailable/broken.
    
    This is the original approach - LLM does full extraction.
    Higher hallucination risk, but handles messy pages.
    """
    if network_summaries is None:
        network_summaries = []
    
    truncated_markdown = markdown_text[:MAX_MARKDOWN_LENGTH]
    
    prompt = f"""
    Analyze this web content and extract all statistical data tables.
    
    Context (Markdown):
    {truncated_markdown}
    
    Available Network Data (JSON Files):
    {json.dumps(network_summaries, indent=2) if network_summaries else "None available"}
    
    Goal: Extract statistical tables with their data.
    
    Instructions:
    1. Extract all tables visible in the content
    2. For each table, provide title, headers, and data rows
    3. Set provenance_map to "dom" for data extracted from markdown
    4. If network data matches table content, prefer network values and set provenance to "network"
    5. List any network files used in used_network_files
    6. Write a brief summary of what data was found
    """
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=prompt)],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        response_mime_type="application/json",
        response_schema=ExtractionSchema,
    )
    
    def do_call():
        return client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=generate_content_config,
        )
    
    response = call_with_timeout(do_call)
    if response is None:
        return ExtractionSchema(summary="LLM timeout", tables=[], used_network_files=[])
    
    return response.parsed


# Keep original function for backward compatibility
def extract_structured_data(markdown_text: str, network_summaries: list = None) -> ExtractionSchema:
    """Original extraction function - kept for backward compatibility."""
    return extract_from_markdown(markdown_text, network_summaries)
