"""PDF/document download and extraction pipeline.

Simple strategy:
1. Download 10 PDFs max
2. Try pdfplumber (text extraction) on all
3. If all are scanned (no text extracted) → send 2 to Gemini LLM
4. If text works → extract from all 10
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib.parse import urljoin, urlparse
import httpx

# PDF text extraction
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False


# Settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
DOWNLOAD_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".zip"}


def detect_downloadable_links(
    links: List[Dict[str, Any]],
    base_url: str
) -> List[Dict[str, str]]:
    """Detect all downloadable file links (no cap on detection)."""
    base_domain = urlparse(base_url).netloc
    downloadable = []
    
    for link in links:
        href = link.get("href", "")
        if not href:
            continue
        
        if href.startswith("/"):
            href = urljoin(base_url, href)
        elif not href.startswith("http"):
            href = urljoin(base_url, href)
        
        link_domain = urlparse(href).netloc
        if link_domain != base_domain:
            continue
        
        path = urlparse(href).path.lower()
        ext = Path(path).suffix
        if ext not in DOWNLOAD_EXTENSIONS:
            continue
        
        downloadable.append({
            "url": href,
            "extension": ext,
            "text": link.get("text", ""),
        })
    
    return downloadable


async def download_file(
    url: str,
    output_dir: Path,
    timeout: float = 60.0
) -> Optional[Dict[str, Any]]:
    """Download a file."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
            response = await client.get(url)
            
            if response.status_code != 200:
                return None
            
            content = response.content
            if len(content) > MAX_FILE_SIZE:
                return None
            
            sha = hashlib.sha256(content).hexdigest()[:12]
            ext = Path(urlparse(url).path).suffix or ".bin"
            filename = f"{sha}{ext}"
            
            output_dir.mkdir(parents=True, exist_ok=True)
            filepath = output_dir / filename
            with open(filepath, "wb") as f:
                f.write(content)
            
            return {
                "url": url,
                "filename": filename,
                "path": str(filepath),
                "size_bytes": len(content),
            }
            
    except Exception:
        return None


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    if not HAS_PDFPLUMBER:
        return ""
    
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:5]:
                page_text = page.extract_text() or ""
                text += page_text
        return text.strip()
    except Exception:
        return ""


def extract_tables_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract tables from text-based PDF using pdfplumber."""
    tables = []
    
    if not HAS_PDFPLUMBER:
        return tables
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages[:5]):
                page_tables = page.extract_tables()
                
                for table_data in page_tables:
                    if not table_data or len(table_data) < 2:
                        continue
                    
                    headers = [str(h or "").strip() for h in table_data[0]]
                    rows = [[str(c or "").strip() for c in row] for row in table_data[1:]]
                    
                    if headers and rows:
                        tables.append({
                            "title": f"Table from {pdf_path.name} (page {page_num + 1})",
                            "headers": headers,
                            "rows": rows,
                            "row_count": len(rows),
                            "extraction_source": "pdf",
                            "provenance_map": [["pdf"] * len(headers) for _ in rows],
                            "source_file": pdf_path.name,
                        })
    except Exception:
        pass
    
    return tables


def extract_csv_data(csv_path: Path) -> List[Dict[str, Any]]:
    """Extract table from CSV file."""
    tables = []
    
    try:
        import csv
        
        with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            rows_list = list(reader)
            
            if len(rows_list) >= 2:
                headers = rows_list[0]
                rows = rows_list[1:]
                
                tables.append({
                    "title": f"Data from {csv_path.name}",
                    "headers": headers,
                    "rows": rows[:1000],
                    "row_count": min(len(rows), 1000),
                    "extraction_source": "file",
                    "provenance_map": [["file"] * len(headers) for _ in rows[:1000]],
                    "source_file": csv_path.name,
                })
    except Exception:
        pass
    
    return tables
