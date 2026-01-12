"""Link selection and filtering for depth crawling."""
import json
import hashlib
from urllib.parse import urlparse, urljoin
from typing import List, Dict, Any

from google import genai
from google.genai import types

from .config import GEMINI_API_KEY, MODEL_NAME


# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)


# Extensions to skip
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".pdf", ".zip", ".doc", ".docx", ".xls", ".xlsx",
    ".mp3", ".mp4", ".avi", ".mov", ".wmv",
}

# Path patterns to skip
SKIP_PATTERNS = {
    "login", "logout", "signin", "signout", "signup", "register",
    "auth", "oauth", "sso", "password", "forgot",
    "about", "contact", "privacy", "terms", "legal", "cookie",
    "careers", "jobs", "sitemap", "feed", "rss",
    "cart", "checkout", "account", "profile", "settings",
}


def normalize_url(url: str) -> str:
    """Normalize URL for deduplication."""
    parsed = urlparse(url)
    # Remove trailing slash, fragments, common query params
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def url_hash(url: str) -> str:
    """Create short hash of URL for folder naming."""
    return hashlib.md5(normalize_url(url).encode()).hexdigest()[:10]


def filter_crawlable_links(
    links: List[Dict[str, Any]], 
    base_url: str,
    max_links: int = 20
) -> List[Dict[str, str]]:
    """
    Filter links to crawlable same-domain pages.
    
    Returns list of {url, text} dicts.
    """
    base_domain = urlparse(base_url).netloc
    seen = set()
    filtered = []
    
    for link in links:
        href = link.get("href", "")
        text = link.get("text", "").strip()[:100]
        
        if not href:
            continue
        
        # Skip fragments and javascript
        if href.startswith("#") or href.startswith("javascript:"):
            continue
        
        # Resolve relative URLs
        if href.startswith("/"):
            href = urljoin(base_url, href)
        elif not href.startswith("http"):
            href = urljoin(base_url, href)
        
        # Check domain
        link_domain = urlparse(href).netloc
        if link_domain != base_domain:
            continue
        
        # Check extension
        path = urlparse(href).path.lower()
        ext = path.split(".")[-1] if "." in path.split("/")[-1] else ""
        if f".{ext}" in SKIP_EXTENSIONS:
            continue
        
        # Check skip patterns
        path_lower = path.lower()
        if any(pattern in path_lower for pattern in SKIP_PATTERNS):
            continue
        
        # Deduplicate
        normalized = normalize_url(href)
        if normalized in seen or normalized == normalize_url(base_url):
            continue
        seen.add(normalized)
        
        filtered.append({
            "url": href,
            "text": text or path.split("/")[-1],
        })
        
        if len(filtered) >= max_links:
            break
    
    return filtered


def select_links_with_llm(
    links: List[Dict[str, str]], 
    page_title: str,
    max_links: int = 10
) -> List[Dict[str, str]]:
    """
    Use LLM to select most relevant links for data extraction.
    """
    if not links:
        return []
    
    # Build link list for prompt
    link_list = "\n".join([
        f"{i+1}. {l['text']} - {l['url']}" 
        for i, l in enumerate(links[:30])  # Cap input
    ])
    
    prompt = f"""You are selecting links to crawl for data extraction.

Page context: "{page_title}"

Available links:
{link_list}

Select links most likely to contain:
- Data tables
- Product catalogs
- Lists of items/records
- Statistics/numbers

Return JSON:
{{"selected": [1, 5, 7, ...]}}

Select up to {max_links} links by their number. Prioritize data-rich pages."""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=500,
            ),
        )
        
        result = json.loads(response.text)
        selected_indices = result.get("selected", [])
        
        # Map indices back to links
        selected = []
        for idx in selected_indices:
            if 1 <= idx <= len(links):
                selected.append(links[idx - 1])
        
        return selected[:max_links]
        
    except Exception as e:
        print(f"   LLM link selection error: {e}")
        # Fallback to first N links
        return links[:max_links]
