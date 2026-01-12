#!/usr/bin/env python3
"""
███████████████████████████████████████████████████████████████████████████████
█                                                                             █
█   WEB DATA EXTRACTOR - SINGLE FILE CRAWLER                                  █
█   ==========================================                                █
█                                                                             █
█   A production-ready Python script for extracting structured data from     █
█   complex web sources. Handles JavaScript-rendered content, charts,        █
█   tables, PDFs, and provides clean JSON output.                            █
█                                                                             █
█   INTERNSHIP CHALLENGE: WEB DATA EXTRACTION                                █
█   CloudSufi / Gemini Assessment                                            █
█                                                                             █
███████████████████████████████████████████████████████████████████████████████

Usage:
    python crawler.py <URL>                          # Basic crawl
    python crawler.py <URL> --mode smart             # Smart extraction (default)
    python crawler.py <URL> --mode fast              # Fast mode (static)
    python crawler.py <URL> --mode dynamic           # Full JS rendering
    python crawler.py <URL> --depth 1                # Crawl 1 level of subpages
    python crawler.py <URL> --depth 2                # Crawl 2 levels deep (max 10 pages)
    python crawler.py <URL> --skip-llm               # Disable LLM processing
    python crawler.py <URL> --network off            # Disable network capture

Examples:
    python crawler.py "https://www.worldometers.info/world-population/"
    python crawler.py "https://www.statista.com/chart/18794/estimated-global-plastics-production/"

Output:
    Creates output/<domain>_<timestamp>/ containing:
    - data.json       : Extracted tables, images, and structured data
    - page.md         : Full page content as markdown
    - page.json       : Links, metadata, and block structure  
    - page.jpg        : Full page screenshot
    - crawl_meta.json : Crawl configuration and timing
    - raw_network/    : Captured XHR/fetch responses (JSON, CSV)
    - downloads/      : Downloaded PDFs and CSVs
    - subpages/       : Crawled subpages (if --depth used)

Requirements:
    - Python 3.10+  
    - GEMINI_API_KEY environment variable
    - pip install crawl4ai google-genai pdfplumber tenacity python-dotenv requests

Author: CloudSufi Internship Candidate
"""

# =============================================================================
# IMPORTS
# =============================================================================
import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
import logging

import requests
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

# Suppress noisy crawl4ai network capture logs
logging.getLogger("crawl4ai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
except ImportError:
    print("ERROR: crawl4ai not installed. Run: pip install crawl4ai")
    sys.exit(1)

try:
    import pdfplumber
except ImportError:
    pdfplumber = None  # Optional


# =============================================================================
# CONFIGURATION
# =============================================================================
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LLM_AVAILABLE = bool(GEMINI_API_KEY)
if not LLM_AVAILABLE:
    print("⚠️  WARNING: GEMINI_API_KEY not set. LLM/Vision features disabled.")
    print("   Set it with: echo 'GEMINI_API_KEY=your_key' > .env")
    print("   Continuing with DOM/Network extraction only...\n")

MODEL_NAME = "gemini-2.0-flash"
THINKING_LEVEL = "MEDIUM"

OUTPUT_DIR = "output"
MAX_MARKDOWN_LENGTH = 50000
LLM_TIMEOUT_SECONDS = 30
CRAWL_TIMEOUT = 90
BROWSER_LIMIT = 3
MAX_EXTRACTIONS = 10

BROWSER_CONFIG = BrowserConfig(
    viewport_width=1920,
    viewport_height=1080,
    headless=True,
)

CRAWLER_CONFIG_FAST = CrawlerRunConfig(
    magic=True,
    screenshot=True,
    screenshot_wait_for=1.0,
    table_score_threshold=5,  # Filter low-quality tables (layout vs data)
    excluded_tags=["nav", "aside", "footer", "header"],  # Focus on main content
)

CRAWLER_CONFIG_DYNAMIC = CrawlerRunConfig(
    magic=True,
    wait_until="networkidle",
    page_timeout=60000,
    scan_full_page=True,
    scroll_delay=0.5,
    delay_before_return_html=2.0,
    screenshot=True,
    screenshot_wait_for=3.0,
    table_score_threshold=5,  # Filter low-quality tables
    excluded_tags=["nav", "aside", "footer", "header"],  # Focus on main content
)

PLACEHOLDER_PATTERNS = ["retrieving data", "loading...", "please wait", "fetching"]

# Initialize Gemini client (if API key available)
client = genai.Client(api_key=GEMINI_API_KEY) if LLM_AVAILABLE else None
browser_semaphore = asyncio.Semaphore(BROWSER_LIMIT)


# =============================================================================
# SCHEMAS (Pydantic models for LLM structured output)
# =============================================================================
class TableDataLLM(BaseModel):
    title: str = ""
    headers: List[str] = []
    rows: List[List[Any]] = []
    provenance_map: List[List[str]] = []
    row_count: int = 0
    extraction_source: str = "llm"


class ExtractionSchema(BaseModel):
    summary: str = ""
    tables: List[TableDataLLM] = []
    used_network_files: List[str] = []


# =============================================================================
# NETWORK CAPTURE (hooks.py)
# =============================================================================
MAX_CAPTURES = 50
MAX_BODY_SIZE = 5 * 1024 * 1024

ALLOWED_CONTENT_TYPES = {
    "application/json", "application/ld+json", "application/hal+json",
    "application/vnd.api+json", "application/graphql-response+json",
}


@dataclass
class CapturedResponse:
    response_url: str
    method: str
    status: int
    content_type: str
    body: bytes
    sha256: str
    captured_at: str
    resource_type: str = ""
    request_url: str = ""


@dataclass
class NetworkCaptureState:
    page_url: str
    output_dir: Path
    captures: List[CapturedResponse] = field(default_factory=list)
    pending_tasks: List[asyncio.Task] = field(default_factory=list)
    total_bytes: int = 0
    seen_hashes: Set[str] = field(default_factory=set)

    def is_at_limit(self):
        return len(self.captures) >= MAX_CAPTURES


def is_json_content_type(content_type: str) -> bool:
    if not content_type:
        return False
    ct = content_type.lower().split(";")[0].strip()
    if ct in ALLOWED_CONTENT_TYPES:
        return True
    if "json" in ct:
        return True
    return False


def is_json_like_body(body: bytes) -> bool:
    if not body or len(body) < 2:
        return False
    text = body[:100].strip()
    return text.startswith(b"{") or text.startswith(b"[")


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


async def capture_response(response, state: NetworkCaptureState):
    if state.is_at_limit():
        return
    try:
        resource_type = getattr(response.request, 'resource_type', '')
        if resource_type not in ("xhr", "fetch"):
            return
        content_type = response.headers.get("content-type", "")
        if not is_json_content_type(content_type):
            return
        body = await response.body()
        if len(body) < 10 or len(body) > MAX_BODY_SIZE:
            return
        if not is_json_like_body(body):
            return
        sha = compute_sha256(body)
        if sha in state.seen_hashes:
            return
        state.seen_hashes.add(sha)
        capture = CapturedResponse(
            response_url=response.url,
            method=response.request.method,
            status=response.status,
            content_type=content_type,
            body=body,
            sha256=sha,
            captured_at=datetime.now().isoformat(),
            resource_type=resource_type,
            request_url=response.request.url,
        )
        state.captures.append(capture)
        state.total_bytes += len(body)
    except Exception:
        pass


def save_captured_response(capture: CapturedResponse, state: NetworkCaptureState) -> Dict:
    state.output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{capture.sha256}.json"
    file_path = state.output_dir / file_name
    try:
        data = json.loads(capture.body)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except:
        with open(file_path, "wb") as f:
            f.write(capture.body)
    meta_path = state.output_dir / f"{capture.sha256}.meta.json"
    meta = {
        "url": capture.response_url,
        "method": capture.method,
        "status": capture.status,
        "content_type": capture.content_type,
        "size": len(capture.body),
        "sha256": capture.sha256,
        "captured_at": capture.captured_at,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return {"file": str(file_path), "meta": str(meta_path), "size": len(capture.body)}


class NetworkHooks:
    def __init__(self, page_url: str, output_dir: Path):
        self.state = NetworkCaptureState(page_url=page_url, output_dir=output_dir)

    async def on_page_context_created(self, page, context, **kwargs):
        async def on_response(response):
            task = asyncio.create_task(capture_response(response, self.state))
            self.state.pending_tasks.append(task)
        page.on("response", on_response)

    async def before_return_html(self, page, html, **kwargs):
        if self.state.pending_tasks:
            await asyncio.gather(*self.state.pending_tasks, return_exceptions=True)
            self.state.pending_tasks.clear()
        for capture in self.state.captures:
            save_captured_response(capture, self.state)
        return html

    def get_stats(self):
        return {
            "network_files_captured": len(self.state.captures),
            "network_total_bytes": self.state.total_bytes,
        }


# =============================================================================
# NETWORK PARSER
# =============================================================================
def parse_network_files(network_dir: Path) -> List[Dict]:
    candidates = []
    if not network_dir.exists():
        return candidates
    for json_file in network_dir.glob("*.json"):
        if json_file.name.endswith(".meta.json"):
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta_file = json_file.with_suffix("").with_suffix(".meta.json")
            meta = {}
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    meta = json.load(f)
            row_count = 0
            if isinstance(data, list):
                row_count = len(data)
            elif isinstance(data, dict):
                for key in ["data", "rows", "items", "results", "records"]:
                    if key in data and isinstance(data[key], list):
                        row_count = len(data[key])
                        break
            if row_count > 0:
                candidates.append({
                    "file": str(json_file),
                    "row_count": row_count,
                    "size": json_file.stat().st_size,
                    "url": meta.get("url", ""),
                    "data": data,
                })
        except:
            pass
    return sorted(candidates, key=lambda x: x["row_count"], reverse=True)


def candidate_to_table_dict(candidate: Dict) -> Dict:
    data = candidate.get("data", {})
    rows_data = []
    if isinstance(data, list):
        rows_data = data
    elif isinstance(data, dict):
        for key in ["data", "rows", "items", "results", "records"]:
            if key in data and isinstance(data[key], list):
                rows_data = data[key]
                break
    if not rows_data:
        return None
    if isinstance(rows_data[0], dict):
        headers = list(rows_data[0].keys())
        rows = [[row.get(h, "") for h in headers] for row in rows_data]
    elif isinstance(rows_data[0], list):
        headers = [f"col_{i}" for i in range(len(rows_data[0]))]
        rows = rows_data
    else:
        return None
    return {
        "title": Path(candidate["file"]).stem,
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
        "extraction_source": "network",
        "provenance_map": [["network"] * len(headers) for _ in rows],
    }


# =============================================================================
# LLM FUNCTIONS
# =============================================================================
def call_with_timeout(func, timeout_seconds=90):
    """Execute function with 90s timeout. Returns None on timeout."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_seconds)
        except FuturesTimeoutError:
            print(f"   ⏰ LLM timeout after {timeout_seconds}s")
            return None


def tables_to_schema(crawl_tables: list, summary: str = "Tables from DOM.") -> ExtractionSchema:
    converted = []
    for table in crawl_tables:
        headers = list(table.headers) if hasattr(table, 'headers') else []
        rows = [list(row) for row in table.rows] if hasattr(table, 'rows') else []
        provenance_map = [["dom"] * len(headers) for _ in rows]
        title = getattr(table, 'caption', None) or f"Table {len(converted) + 1}"
        converted.append(TableDataLLM(
            title=title, headers=headers, rows=rows, provenance_map=provenance_map
        ))
    return ExtractionSchema(summary=summary, tables=converted, used_network_files=[])


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=4, max=10))
def extract_from_markdown(markdown_text: str, network_summaries: list = None) -> ExtractionSchema:
    if network_summaries is None:
        network_summaries = []
    truncated = markdown_text[:MAX_MARKDOWN_LENGTH]
    prompt = f"""
    Analyze this web content and extract all statistical data tables.
    Context: {truncated}
    Network Data: {json.dumps(network_summaries, indent=2) if network_summaries else "None"}
    Extract tables with title, headers, rows. Set provenance_map to "dom".
    """
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
        response_mime_type="application/json",
        response_schema=ExtractionSchema,
    )
    def do_call():
        return client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
    response = call_with_timeout(do_call)
    if response is None:
        return ExtractionSchema(summary="LLM timeout", tables=[], used_network_files=[])
    return response.parsed


# =============================================================================
# VALIDATORS
# =============================================================================
def check_tables_sanity(tables) -> bool:
    if not tables:
        return False
    for table in tables:
        if hasattr(table, 'rows') and len(table.rows) > 0:
            return True
    return False


def validate_extraction(schema: ExtractionSchema) -> bool:
    return len(schema.tables) > 0


# =============================================================================
# PARSER
# =============================================================================
def parse_markdown_to_blocks(markdown: str) -> List[Dict]:
    blocks = []
    lines = markdown.split('\n')
    current_block = {"type": "text", "content": []}
    for line in lines:
        if line.startswith('#'):
            if current_block["content"]:
                blocks.append(current_block)
            level = len(line) - len(line.lstrip('#'))
            blocks.append({"type": "heading", "level": level, "content": line.lstrip('# ').strip()})
            current_block = {"type": "text", "content": []}
        elif line.startswith('|'):
            if current_block["type"] != "table":
                if current_block["content"]:
                    blocks.append(current_block)
                current_block = {"type": "table", "content": []}
            current_block["content"].append(line)
        else:
            if current_block["type"] == "table":
                blocks.append(current_block)
                current_block = {"type": "text", "content": []}
            if line.strip():
                current_block["content"].append(line)
    if current_block["content"]:
        blocks.append(current_block)
    return blocks


def build_page_envelope(result, blocks, links, media, net_manifest) -> Dict:
    return {
        "url": result.url if hasattr(result, 'url') else "",
        "title": result.title if hasattr(result, 'title') else "",
        "blocks": blocks,
        "links": links,
        "media": media,
        "network": net_manifest,
    }


# =============================================================================
# IMAGE/CHART EXTRACTION
# =============================================================================
SKIP_PATTERNS = [
    "flag", "logo", "icon", "avatar", "profile", "badge", "medal",
    "button", "arrow", "chevron", "facebook", "twitter", "linkedin",
    "ad", "banner", "pixel", "bg", "background", "pattern",
]

IMAGE_ANALYSIS_PROMPT = """Analyze this image and extract data.
If DATA VISUALIZATION: extract numbers, labels, trends.
If DECORATIVE: return {"is_data": false, "description": "brief"}
Return JSON: {"is_data": true/false, "image_type": "chart|graph|...", "title": "", "description": "", "data_points": [], "insights": ""}"""

PAGE_ANALYSIS_PROMPT = """Analyze this webpage screenshot for charts/graphs.
Find: bar charts, line graphs, pie charts, tables, infographics.
Return JSON: {"charts_found": N, "visualizations": [{"type": "", "title": "", "data_points": [], "insights": ""}]}"""


def filter_meaningful_images(media: List[Dict], max_images: int = 5) -> List[Dict]:
    meaningful = []
    for item in media:
        url = item.get("src", item.get("url", ""))
        alt = item.get("alt", "")
        if not url or not url.startswith("http"):
            continue
        combined = f"{url.lower()} {alt.lower()}"
        skip = any(p in combined for p in SKIP_PATTERNS)
        if skip:
            continue
        if url.lower().endswith((".ico", ".svg", ".gif")):
            continue
        try:
            w = int(item.get("width", 0) or 0)
            h = int(item.get("height", 0) or 0)
            if w > 0 and h > 0 and (w < 100 or h < 100):
                continue
        except:
            pass
        meaningful.append({"url": url, "alt": alt})
        if len(meaningful) >= max_images:
            break
    return meaningful


def extract_image_info(image_url: str) -> Optional[Dict]:
    try:
        print(f"      Downloading: {image_url[:50]}...")
        resp = requests.get(image_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        mime = "image/jpeg"
        if image_url.lower().endswith(".png"):
            mime = "image/png"
        elif image_url.lower().endswith(".webp"):
            mime = "image/webp"
        contents = [types.Content(role="user", parts=[
            types.Part.from_bytes(data=resp.content, mime_type=mime),
            types.Part.from_text(text=IMAGE_ANALYSIS_PROMPT),
        ])]
        config = types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.1, max_output_tokens=2048
        )
        result = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
        parsed = json.loads(result.text)
        parsed["source_url"] = image_url
        return parsed
    except Exception as e:
        print(f"      Image error: {e}")
        return None


def extract_images_from_page(media: List[Dict], max_images: int = 3) -> List[Dict]:
    extracted = []
    meaningful = filter_meaningful_images(media, max_images)
    if meaningful:
        print(f"   → Found {len(meaningful)} images, analyzing...")
    for img in meaningful:
        result = extract_image_info(img["url"])
        if result and result.get("is_data"):
            extracted.append({
                "source_url": img["url"],
                "image_type": result.get("image_type", "chart"),
                "title": result.get("title", ""),
                "description": result.get("description", ""),
                "data_points": result.get("data_points", []),
                "insights": result.get("insights", ""),
                "extraction_source": "vision",
            })
    return extracted


def extract_from_screenshot(screenshot_path: Path) -> List[Dict]:
    if not screenshot_path.exists():
        return []
    try:
        print(f"   → Analyzing screenshot for charts...")
        with open(screenshot_path, "rb") as f:
            image_data = f.read()
        mime = "image/jpeg" if str(screenshot_path).endswith(".jpg") else "image/png"
        contents = [types.Content(role="user", parts=[
            types.Part.from_bytes(data=image_data, mime_type=mime),
            types.Part.from_text(text=PAGE_ANALYSIS_PROMPT),
        ])]
        config = types.GenerateContentConfig(
            response_mime_type="application/json", temperature=0.1, max_output_tokens=8192
        )
        result = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
        try:
            parsed = json.loads(result.text.strip())
        except:
            parsed = {"charts_found": 0, "visualizations": []}
        
        # Handle case where parsed is a list instead of dict
        if isinstance(parsed, list):
            visualizations = parsed
        else:
            visualizations = parsed.get("visualizations", [])
        
        extracted = []
        for viz in visualizations:
            if isinstance(viz, dict):
                extracted.append({
                    "source_url": str(screenshot_path),
                    "image_type": viz.get("type", "chart"),
                    "title": viz.get("title", ""),
                    "data_points": viz.get("data_points", []),
                    "insights": viz.get("insights", ""),
                    "extraction_source": "screenshot",
                })
        if extracted:
            print(f"      Found {len(extracted)} charts")
        return extracted
    except Exception as e:
        print(f"      Screenshot error: {e}")
        return []


# =============================================================================
# PDF EXTRACTION
# =============================================================================
def detect_downloadable_links(links: List[Dict], base_url: str) -> List[Dict]:
    downloadable = []
    extensions = (".pdf", ".csv", ".xlsx", ".xls")
    for link in links:
        href = link.get("href", "")
        if any(href.lower().endswith(ext) for ext in extensions):
            full_url = urljoin(base_url, href)
            downloadable.append({"url": full_url, "text": link.get("text", ""), "type": href.split(".")[-1]})
    return downloadable


async def download_file(url: str, output_dir: Path) -> Optional[Dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            filename = Path(urlparse(url).path).name or "download"
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return {"url": url, "path": str(filepath), "size": len(resp.content)}
    except:
        pass
    return None


def extract_tables_from_pdf(pdf_path: Path) -> tuple:
    """Extract tables from PDF. Returns (tables, has_text) tuple."""
    if not pdfplumber:
        return [], False
    tables = []
    full_text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:10]):
                # Try table extraction
                for table in page.extract_tables():
                    if table and len(table) > 1:
                        headers = [str(h or "") for h in table[0]]
                        rows = [[str(c or "") for c in row] for row in table[1:]]
                        tables.append({
                            "title": f"{pdf_path.name} - Page {i+1}",
                            "headers": headers, "rows": rows,
                            "row_count": len(rows),
                            "extraction_source": "pdf",
                            "provenance_map": [["pdf"] * len(headers) for _ in rows],
                        })
                # Also collect text
                page_text = page.extract_text() or ""
                full_text += page_text + "\n"
        
        # If no tables but has text, create a text-based "table"
        if not tables and len(full_text.strip()) > 100:
            # Create simple key-value extraction from text
            lines = [l.strip() for l in full_text.split("\n") if l.strip()]
            if lines:
                tables.append({
                    "title": f"{pdf_path.name} - Text Content",
                    "headers": ["Content"],
                    "rows": [[line] for line in lines[:50]],  # First 50 lines
                    "row_count": min(len(lines), 50),
                    "extraction_source": "pdf_text",
                    "provenance_map": [["pdf_text"] for _ in lines[:50]],
                })
                return tables, True
    except:
        pass
    return tables, bool(tables)


def extract_csv_data(csv_path: Path) -> List[Dict]:
    import csv
    tables = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if rows:
                headers = rows[0]
                data_rows = rows[1:]
                tables.append({
                    "title": csv_path.name,
                    "headers": headers, "rows": data_rows,
                    "row_count": len(data_rows),
                    "extraction_source": "csv",
                    "provenance_map": [["csv"] * len(headers) for _ in data_rows],
                })
    except:
        pass
    return tables


def extract_pdf_with_vision(pdf_path: Path) -> List[Dict]:
    """Extract tables from image-based PDFs using Gemini vision."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []
    
    tables = []
    try:
        doc = fitz.open(pdf_path)
        # Process first 2 pages max
        for page_num in range(min(2, len(doc))):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for clarity
            img_data = pix.tobytes("png")
            
            # Use Gemini vision to extract tables
            prompt = """Extract all tables from this PDF page as structured data.
Return JSON with format: {"tables": [{"title": "...", "headers": [...], "rows": [[...]]}]}
Focus on any tabular data, schedules, or structured information."""
            
            contents = [types.Content(role="user", parts=[
                types.Part.from_bytes(data=img_data, mime_type="image/png"),
                types.Part.from_text(text=prompt),
            ])]
            config = types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.1, max_output_tokens=4096
            )
            
            result = call_with_timeout(lambda: client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            ), timeout_seconds=60)
            
            if result and result.text:
                try:
                    parsed = json.loads(result.text.strip())
                    for t in parsed.get("tables", []):
                        if isinstance(t, dict) and t.get("headers"):
                            tables.append({
                                "title": f"{pdf_path.name} - Page {page_num + 1}: {t.get('title', '')}",
                                "headers": t.get("headers", []),
                                "rows": t.get("rows", []),
                                "row_count": len(t.get("rows", [])),
                                "extraction_source": "pdf_vision",
                                "provenance_map": [["vision"] * len(t.get("headers", [])) for _ in t.get("rows", [])],
                            })
                except:
                    pass
        doc.close()
    except Exception as e:
        print(f"      Vision PDF error: {e}")
    return tables


# =============================================================================
# UTILITIES
# =============================================================================
def generate_run_id(url: str) -> str:
    domain = urlparse(url).netloc.replace(".", "_").replace("www_", "")
    return f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def compress_screenshot(base64_data: str) -> str:
    try:
        from PIL import Image
        data = base64.b64decode(base64_data)
        img = Image.open(BytesIO(data))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        buffer = BytesIO()
        img.save(buffer, format='JPEG', quality=85, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode()
    except:
        return base64_data


def should_retry_with_dynamic(markdown: str) -> bool:
    text = markdown.lower()
    return any(p in text for p in PLACEHOLDER_PATTERNS)


def get_markdown_content(result) -> str:
    for attr in ['markdown_v2', 'markdown']:
        val = getattr(result, attr, None)
        if val:
            if hasattr(val, 'raw_markdown'):
                return val.raw_markdown or ""
            return str(val)
    return ""


def extract_media_from_result(result) -> List[Dict]:
    media = getattr(result, 'media', None)
    if media and isinstance(media, dict):
        items = []
        for mtype, mlist in media.items():
            if isinstance(mlist, list):
                for m in mlist:
                    if isinstance(m, dict):
                        items.append({**m, "type": mtype})
        return items
    return []


def extract_links_from_result(result) -> List[Dict]:
    links = getattr(result, 'links', None)
    if links and isinstance(links, dict):
        items = []
        for ltype, llist in links.items():
            if isinstance(llist, list):
                for l in llist:
                    if isinstance(l, dict):
                        items.append({**l, "link_type": ltype})
        return items
    return []


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().replace("_", " ").replace("-", " ")


def map_tables_to_context(tables: List[Dict], blocks: List[Dict]) -> List[Dict]:
    """
    Add context mapping to extracted tables using fuzzy matching.
    Matches table titles to page blocks to find related sections.
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        return tables  # Fallback if rapidfuzz not installed
    
    mapped_tables = []
    
    # Collect searchable content from blocks
    block_texts = []
    for block in blocks:
        content = block.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        block_type = block.get("type", "")
        
        if content and isinstance(content, str):
            block_texts.append({
                "text": normalize_text(content),
                "content": content[:100],
                "type": block_type,
            })
    
    for table in tables:
        title = table.get("title", "")
        if not title:
            mapped_tables.append(table)
            continue
        
        title_normalized = normalize_text(title)
        
        # Find matching blocks using fuzzy matching
        related_blocks = []
        best_section = None
        best_score = 0
        
        for bt in block_texts:
            score = max(
                fuzz.partial_ratio(title_normalized, bt["text"]),
                fuzz.token_set_ratio(title_normalized, bt["text"]),
            )
            
            if score >= 60:  # Threshold
                related_blocks.append({
                    "score": score,
                    "matched_content": bt["content"],
                })
                
                if score > best_score and bt["type"] == "heading":
                    best_score = score
                    best_section = bt["content"]
        
        # Add context to table
        table_with_context = table.copy()
        table_with_context["section_context"] = best_section
        table_with_context["context_score"] = best_score if best_score > 0 else None
        
        mapped_tables.append(table_with_context)
    
    return mapped_tables


def url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:8]


# =============================================================================
# MAIN PROCESS URL
# =============================================================================
async def process_url(
    url: str,
    mode: str = "smart",
    skip_llm: bool = False,
    network_capture: bool = True,
) -> Dict[str, Any]:
    """Process a single URL and extract data."""
    
    # Auto-skip LLM if API key not available
    if not LLM_AVAILABLE:
        skip_llm = True
    
    run_id = generate_run_id(url)
    run_dir = Path(OUTPUT_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"URL: {url}")
    print(f"Run: {run_id} | Mode: {mode} | Network: {'on' if network_capture else 'off'}")
    print(f"{'='*60}")
    
    # Setup network capture using crawl4ai's built-in feature
    network_dir = run_dir / "raw_network"
    
    # Select config - use capture_network_requests flag instead of hooks
    if mode == "dynamic":
        config = CrawlerRunConfig(
            magic=True,
            wait_until="networkidle",
            page_timeout=60000,
            scan_full_page=True,
            scroll_delay=0.5,
            delay_before_return_html=2.0,
            screenshot=True,
            screenshot_wait_for=3.0,
            capture_network_requests=network_capture,
            verbose=False,
        )
        profile = "dynamic"
    else:
        config = CrawlerRunConfig(
            magic=True,
            screenshot=True,
            screenshot_wait_for=1.0,
            capture_network_requests=network_capture,
            verbose=False,
        )
        profile = "fast"
    
    # Crawl
    start_time = datetime.now()
    print(f"[INIT].... → Crawl4AI")
    
    async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
        try:
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=config),
                timeout=CRAWL_TIMEOUT
            )
        except asyncio.TimeoutError:
            print(f"   ⏰ Crawl timeout after {CRAWL_TIMEOUT}s")
            return {"url": url, "run_id": run_id, "success": False, "error": "timeout"}
    
    crawl_time = (datetime.now() - start_time).total_seconds()
    print(f"   Crawl: {crawl_time:.1f}s ({profile})")
    
    # Get network requests from result (if captured)
    network_requests = getattr(result, 'network_requests', []) or []
    net_stats = {"network_files_captured": len(network_requests), "network_total_bytes": 0}
    
    # Save captured network requests to files for parsing later
    if network_requests and network_capture:
        network_dir.mkdir(parents=True, exist_ok=True)
        for i, req in enumerate(network_requests):
            if isinstance(req, dict) and req.get('response_body'):
                body = req.get('response_body', b'')
                if isinstance(body, str):
                    body = body.encode('utf-8')
                sha = hashlib.sha256(body).hexdigest()[:16]
                file_path = network_dir / f"{sha}.json"
                try:
                    data = json.loads(body)
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    net_stats["network_total_bytes"] += len(body)
                except:
                    pass
    
    print(f"   Network: {net_stats['network_files_captured']} requests")
    
    # Smart mode: retry with dynamic if placeholders detected
    markdown_content = get_markdown_content(result)
    if mode == "smart" and should_retry_with_dynamic(markdown_content) and profile == "fast":
        print(f"   ⟳ Retry: placeholder detected")
        dynamic_config = CrawlerRunConfig(
            magic=True,
            wait_until="networkidle",
            page_timeout=60000,
            scan_full_page=True,
            scroll_delay=0.5,
            delay_before_return_html=2.0,
            screenshot=True,
            screenshot_wait_for=3.0,
            capture_network_requests=network_capture,
            verbose=False,
        )
        async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
            try:
                result = await asyncio.wait_for(
                    crawler.arun(url=url, config=dynamic_config),
                    timeout=CRAWL_TIMEOUT
                )
                profile = "dynamic"
                markdown_content = get_markdown_content(result)
            except:
                pass
    
    # Save screenshot
    if hasattr(result, 'screenshot') and result.screenshot:
        try:
            compressed = compress_screenshot(result.screenshot)
            with open(run_dir / "page.jpg", "wb") as f:
                f.write(base64.b64decode(compressed))
        except:
            pass
    
    # Parse page content
    blocks = parse_markdown_to_blocks(markdown_content)
    media = extract_media_from_result(result)
    links = extract_links_from_result(result)
    
    # Save markdown
    with open(run_dir / "page.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    # Save page.json
    net_manifest = []  # Simplified
    page_envelope = build_page_envelope(result, blocks, links, media, net_manifest)
    with open(run_dir / "page.json", "w", encoding="utf-8") as f:
        json.dump(page_envelope, f, indent=2, ensure_ascii=False, default=str)
    
    # Save crawl metadata
    crawl_meta = {
        "url": url, "run_id": run_id, "mode": mode, "profile": profile,
        "crawl_time": crawl_time, "network": net_stats,
        "timestamp": datetime.now().isoformat(),
    }
    with open(run_dir / "crawl_meta.json", "w") as f:
        json.dump(crawl_meta, f, indent=2)
    
    # EXTRACTION PIPELINE
    print("   Extracting...")
    extracted_tables = []
    extraction_source = "none"
    
    # 1. Try network tables first
    if network_capture:
        candidates = parse_network_files(network_dir)
        if candidates:
            print(f"   → Network: {len(candidates)} candidates")
            for c in candidates[:5]:
                table = candidate_to_table_dict(c)
                if table:
                    extracted_tables.append(table)
            if extracted_tables:
                extraction_source = "network"
    
    # 2. Try DOM tables
    crawl_tables = getattr(result, 'tables', [])
    if not extracted_tables and check_tables_sanity(crawl_tables):
        print(f"   → DOM tables")
        ext = tables_to_schema(crawl_tables)
        extracted_tables = [t.model_dump() for t in ext.tables]
        extraction_source = "dom"
    
    # 3. LLM fallback
    elif not extracted_tables and not skip_llm:
        print(f"   → LLM extraction")
        try:
            ext = extract_from_markdown(markdown_content, [])
            extracted_tables = [t.model_dump() for t in ext.tables]
            extraction_source = "llm"
        except Exception as e:
            print(f"   LLM error: {e}")
    
    # 4. PDF downloads
    downloaded_files = []
    downloadable = detect_downloadable_links(links, url)
    if downloadable:
        print(f"   → Found {len(downloadable)} downloadable files (processing first 10)")
        downloads_dir = run_dir / "downloads"
        vision_fallback_count = 0
        max_vision_fallback = 2  # Use vision on first 2 PDFs that fail text extraction
        
        for dl in downloadable[:10]:  # Download first 10
            result_dl = await download_file(dl["url"], downloads_dir)
            if result_dl:
                downloaded_files.append(result_dl)
                fpath = Path(result_dl["path"])
                if fpath.suffix.lower() == ".pdf":
                    # Try pdfplumber text/table extraction first
                    pdf_tables, has_content = extract_tables_from_pdf(fpath)
                    if pdf_tables:
                        extracted_tables.extend(pdf_tables)
                        src = "text" if any(t.get("extraction_source") == "pdf_text" for t in pdf_tables) else "table"
                        print(f"      PDF: {fpath.name} → {len(pdf_tables)} items ({src})")
                    elif vision_fallback_count < max_vision_fallback and not skip_llm:
                        # Vision fallback for image-based PDFs (no text at all)
                        print(f"      PDF: {fpath.name} → trying vision extraction...")
                        vision_tables = extract_pdf_with_vision(fpath)
                        if vision_tables:
                            extracted_tables.extend(vision_tables)
                            print(f"      PDF: {fpath.name} → {len(vision_tables)} tables (vision)")
                        vision_fallback_count += 1
                elif fpath.suffix.lower() == ".csv":
                    csv_tables = extract_csv_data(fpath)
                    extracted_tables.extend(csv_tables)
    
    # 5. Image extraction
    page_images = []
    for m in media:
        mtype = str(m.get("type", "")).lower()
        src = str(m.get("src", ""))
        if "image" in mtype or "img" in mtype or any(src.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            page_images.append(m)
    
    if page_images:
        print(f"   → Found {len(page_images)} images")
    
    image_data = []
    if page_images and not skip_llm:
        image_data = extract_images_from_page(page_images, max_images=3)
        if image_data:
            extraction_source = "vision" if extraction_source == "none" else extraction_source
    
    # Screenshot fallback for JS charts
    if not image_data and not skip_llm:
        screenshot_path = run_dir / "page.jpg"
        if screenshot_path.exists():
            image_data = extract_from_screenshot(screenshot_path)
            if image_data:
                extraction_source = "screenshot" if extraction_source == "none" else extraction_source
    
    # 6. Map tables to page context using fuzzy matching
    if extracted_tables and blocks:
        extracted_tables = map_tables_to_context(extracted_tables, blocks)
    
    # Save data.json
    data_output = {
        "summary": f"Extracted from {url}",
        "tables": extracted_tables,
        "extraction_source": extraction_source,
        "downloaded_files": downloaded_files,
        "images": image_data,
    }
    with open(run_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(data_output, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"✅ {run_id} | src={extraction_source} tables={len(extracted_tables)} images={len(image_data)} net={net_stats['network_files_captured']}")
    
    print(f"\n{'='*60}")
    print(f"📁 OUTPUT DIRECTORY: {run_dir}")
    print(f"{'='*60}")
    print(f"\n📊 EXTRACTION STATISTICS:")
    print(f"   • Extraction source: {extraction_source}")
    print(f"   • Tables extracted: {len(extracted_tables)}")
    print(f"   • Images/charts analyzed: {len(image_data)}")
    print(f"   • Network files captured: {net_stats['network_files_captured']}")
    print(f"   • Downloaded files: {len(downloaded_files)}")
    
    data_size = (run_dir / "data.json").stat().st_size // 1024 if (run_dir / "data.json").exists() else 0
    md_size = (run_dir / "page.md").stat().st_size // 1024 if (run_dir / "page.md").exists() else 0
    jpg_size = (run_dir / "page.jpg").stat().st_size // 1024 if (run_dir / "page.jpg").exists() else 0
    
    print(f"\n📄 OUTPUT FILES:")
    print(f"   • data.json      - Extracted tables & images ({data_size}KB)")
    print(f"   • page.md        - Full page markdown ({md_size}KB)")
    print(f"   • page.json      - Links, metadata, blocks")
    print(f"   • page.jpg       - Full page screenshot ({jpg_size}KB)")
    print(f"   • crawl_meta.json - Crawl configuration & timing")
    if (run_dir / "raw_network").exists():
        print(f"   • raw_network/   - {len(list((run_dir / 'raw_network').glob('*')))} captured files")
    if (run_dir / "downloads").exists():
        print(f"   • downloads/     - {len(list((run_dir / 'downloads').glob('*')))} downloaded files")
    print(f"{'='*60}\n")
    
    return {
        "url": url, "run_id": run_id, "success": True,
        "run_dir": str(run_dir),
        "extraction_source": extraction_source,
        "tables": len(extracted_tables),
        "images": len(image_data),
    }


# =============================================================================
# DEPTH CRAWLING
# =============================================================================
def filter_crawlable_links(links: List[Dict], base_url: str) -> List[Tuple[str, str]]:
    """Filter links to same-domain pages."""
    base_domain = urlparse(base_url).netloc
    crawlable = []
    seen = set()
    skip_ext = ('.pdf', '.jpg', '.png', '.gif', '.css', '.js', '.zip')
    
    for link in links:
        href = link.get("href", "")
        if not href or href.startswith("#") or href.startswith("javascript:"):
            continue
        full_url = urljoin(base_url, href)
        parsed = urlparse(full_url)
        if parsed.netloc != base_domain:
            continue
        if any(full_url.lower().endswith(ext) for ext in skip_ext):
            continue
        if full_url in seen:
            continue
        seen.add(full_url)
        text = link.get("text", "")[:50]
        crawlable.append((full_url, text))
    
    return crawlable[:20]


async def crawl_with_depth(
    url: str,
    mode: str = "smart",
    depth_mode: str = "0",
    skip_llm: bool = False,
    network: bool = True,
    extraction_count: List[int] = None,
):
    """Crawl with optional depth (subpages)."""
    
    if extraction_count is None:
        extraction_count = [0]
    
    max_depth = 0
    if depth_mode == "1":
        max_depth = 1
    elif depth_mode == "2":
        max_depth = 2
    elif depth_mode == "smart":
        max_depth = 1
    
    # Check extraction limit
    if extraction_count[0] >= MAX_EXTRACTIONS:
        print(f"   ⚠️ Extraction limit ({MAX_EXTRACTIONS}) reached")
        return {"url": url, "success": False, "error": "limit_reached"}
    
    # Crawl main page
    result = await process_url(url, mode, skip_llm, network)
    extraction_count[0] += 1
    
    if not result.get("success") or max_depth == 0:
        return result
    
    # Get links for subpages
    run_dir = Path(result["run_dir"])
    page_json = run_dir / "page.json"
    if not page_json.exists():
        return result
    
    with open(page_json, "r", encoding="utf-8") as f:
        page_data = json.load(f)
    
    links = page_data.get("links", [])
    crawlable = filter_crawlable_links(links, url)
    
    if not crawlable:
        return result
    
    # For smart mode, use LLM to select most relevant links
    if depth_mode == "smart" and len(crawlable) > 3 and not skip_llm:
        try:
            # Prepare link info for LLM
            link_info = "\n".join([f"- [{text}]({url_})" for url_, text in crawlable[:15]])
            prompt = f"""Select the 5 most important links from this list that likely contain data/statistics:
{link_info}

Return JSON: {{"selected_urls": ["url1", "url2", ...]}}
Only include URLs that will have useful data tables or charts."""
            
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            config = types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.1, max_output_tokens=1024
            )
            response = call_with_timeout(lambda: client.models.generate_content(
                model=MODEL_NAME, contents=contents, config=config
            ), timeout_seconds=15)
            
            if response:
                parsed = json.loads(response.text)
                selected = set(parsed.get("selected_urls", []))
                # Filter to selected URLs
                crawlable = [(u, t) for u, t in crawlable if u in selected]
                print(f"   → Smart: LLM selected {len(crawlable)} data-rich links")
        except Exception as e:
            print(f"   → Smart selection failed, using top links: {e}")
    
    # Limit links per depth
    max_subpages = 6 if max_depth == 1 else 3
    crawlable = crawlable[:max_subpages]
    
    print(f"\n   → Crawling {len(crawlable)} subpages in parallel (depth={max_depth})...")
    
    subpages_dir = run_dir / "subpages"
    subpages_dir.mkdir(exist_ok=True)
    subpage_results = []
    
    async def crawl_subpage(link_url, link_text):
        if extraction_count[0] >= MAX_EXTRACTIONS:
            return None
        
        async with browser_semaphore:
            try:
                sub_result = await asyncio.wait_for(
                    process_url(link_url, "fast", skip_llm=True, network_capture=network),
                    timeout=30
                )
                extraction_count[0] += 1
                return {"url": link_url, "text": link_text, **sub_result}
            except Exception as e:
                return {"url": link_url, "text": link_text, "success": False, "error": str(e)}
    
    tasks = [crawl_subpage(u, t) for u, t in crawlable]
    results = await asyncio.gather(*tasks)
    subpage_results = [r for r in results if r]
    
    # Save depth summary
    depth_summary = {
        "parent_url": url,
        "subpages": subpage_results,
        "total_extractions": extraction_count[0],
    }
    with open(subpages_dir / "_depth_summary.json", "w") as f:
        json.dump(depth_summary, f, indent=2)
    
    result["subpages"] = len(subpage_results)
    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Web Data Extractor - Extract structured data from web pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python crawler.py "https://www.worldometers.info/world-population/"
    python crawler.py "https://www.statista.com/chart/18794/" --mode smart
    python crawler.py "https://example.com" --depth 1
        """
    )
    parser.add_argument("url", nargs="?", help="URL to crawl")
    parser.add_argument("--mode", choices=["fast", "dynamic", "smart"], default="smart",
                        help="Extraction mode (default: smart)")
    parser.add_argument("--depth", choices=["0", "1", "2", "smart"], default="0",
                        help="Depth of subpage crawling")
    parser.add_argument("--skip-llm", action="store_true", help="Disable LLM processing")
    parser.add_argument("--network", choices=["on", "off", "capture"], default="on",
                        help="Network capture mode")
    
    args = parser.parse_args()
    
    if not args.url:
        parser.print_help()
        sys.exit(1)
    
    network_capture = args.network != "off"
    
    if args.depth != "0":
        result = asyncio.run(crawl_with_depth(
            args.url, args.mode, args.depth, args.skip_llm, network_capture
        ))
    else:
        result = asyncio.run(process_url(
            args.url, args.mode, args.skip_llm, network_capture
        ))
    
    if result.get("success"):
        print(f"\n✅ Extraction complete!")
    else:
        print(f"\n❌ Extraction failed: {result.get('error', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
