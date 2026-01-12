"""Network hooks for capturing API responses during crawl."""
import asyncio
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field


# Capture limits
MAX_NETWORK_FILES = 50
MAX_NETWORK_TOTAL_BYTES = 30 * 1024 * 1024  # 30MB
MIN_BODY_SIZE = 500
MAX_BODY_SIZE = 5 * 1024 * 1024  # 5MB

# Content-type allowlist
ALLOWED_CONTENT_TYPES = {
    "application/json",
    "application/ld+json",
    "application/hal+json",
    "application/vnd.api+json",
    "application/graphql-response+json",
}


@dataclass
class CapturedResponse:
    """A captured network response."""
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
    """State for network capture during a crawl."""
    page_url: str
    output_dir: Path
    captures: List[CapturedResponse] = field(default_factory=list)
    pending_tasks: List[asyncio.Task] = field(default_factory=list)
    total_bytes: int = 0
    seen_hashes: Set[str] = field(default_factory=set)
    
    def is_at_limit(self) -> bool:
        return (len(self.captures) >= MAX_NETWORK_FILES or 
                self.total_bytes >= MAX_NETWORK_TOTAL_BYTES)


def is_json_content_type(content_type: str) -> bool:
    """Check if content-type indicates JSON."""
    if not content_type:
        return False
    ct_lower = content_type.lower().split(";")[0].strip()
    
    # Exact match
    if ct_lower in ALLOWED_CONTENT_TYPES:
        return True
    
    # Pattern match for *+json
    if ct_lower.endswith("+json"):
        return True
    
    # text/plain might be JSON
    if ct_lower == "text/plain":
        return True  # Will verify body starts with { or [
    
    return False


def is_json_like_body(body: bytes) -> bool:
    """Check if body looks like JSON."""
    if not body:
        return False
    try:
        text = body.decode('utf-8', errors='ignore').strip()
        return text.startswith('{') or text.startswith('[')
    except:
        return False


def compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of data."""
    return hashlib.sha256(data).hexdigest()[:16]


async def capture_response(response, state: NetworkCaptureState) -> Optional[CapturedResponse]:
    """
    Capture a network response if it meets criteria.
    
    Criteria:
    - resource_type is xhr or fetch
    - content-type is JSON-like
    - body size in valid range
    - body looks like JSON
    """
    try:
        # Get resource type
        request = response.request
        resource_type = request.resource_type if hasattr(request, 'resource_type') else ""
        
        # Filter: only xhr/fetch
        if resource_type not in {"xhr", "fetch", "other"}:
            return None
        
        # Get content-type
        headers = response.headers if hasattr(response, 'headers') else {}
        content_type = headers.get("content-type", "")
        
        # Filter: JSON content-type
        if not is_json_content_type(content_type):
            return None
        
        # Get body
        try:
            body = await response.body()
        except Exception:
            return None
        
        # Filter: size limits
        body_len = len(body)
        if body_len < MIN_BODY_SIZE or body_len > MAX_BODY_SIZE:
            return None
        
        # Filter: looks like JSON
        if not is_json_like_body(body):
            return None
        
        # Compute hash for dedup
        sha = compute_sha256(body)
        if sha in state.seen_hashes:
            return None
        state.seen_hashes.add(sha)
        
        # Check limits
        if state.is_at_limit():
            return None
        
        captured = CapturedResponse(
            response_url=response.url,
            method=request.method if hasattr(request, 'method') else "GET",
            status=response.status,
            content_type=content_type,
            body=body,
            sha256=sha,
            captured_at=datetime.now().isoformat(),
            resource_type=resource_type,
            request_url=request.url if hasattr(request, 'url') else "",
        )
        
        state.captures.append(captured)
        state.total_bytes += body_len
        
        return captured
        
    except Exception as e:
        return None


def save_captured_response(capture: CapturedResponse, state: NetworkCaptureState) -> Dict[str, str]:
    """
    Save captured response to disk.
    
    Returns dict with file paths.
    """
    raw_network_dir = state.output_dir / "raw_network"
    raw_network_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%H%M%S%f")[:10]
    base_name = f"{timestamp}_{capture.sha256}"
    
    # Determine extension
    try:
        json.loads(capture.body.decode('utf-8'))
        ext = ".json"
    except:
        ext = ".txt"
    
    body_file = raw_network_dir / f"{base_name}{ext}"
    meta_file = raw_network_dir / f"{base_name}.meta.json"
    
    # Save body
    with open(body_file, "wb") as f:
        f.write(capture.body)
    
    # Save metadata
    meta = {
        "page_url": state.page_url,
        "response_url": capture.response_url,
        "request_url": capture.request_url,
        "method": capture.method,
        "status": capture.status,
        "content_type": capture.content_type,
        "bytes": len(capture.body),
        "sha256": capture.sha256,
        "captured_at": capture.captured_at,
        "resource_type": capture.resource_type,
    }
    
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    return {
        "body_file": str(body_file.relative_to(state.output_dir)),
        "meta_file": str(meta_file.relative_to(state.output_dir)),
    }


def get_top_keys(body: bytes, max_keys: int = 5) -> List[str]:
    """Extract top-level keys from JSON for manifest."""
    try:
        data = json.loads(body.decode('utf-8'))
        if isinstance(data, dict):
            return list(data.keys())[:max_keys]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            return list(data[0].keys())[:max_keys]
        return []
    except:
        return []


def build_network_manifest(state: NetworkCaptureState) -> List[Dict[str, Any]]:
    """Build network manifest for page.json."""
    manifest = []
    
    for capture in state.captures:
        files = save_captured_response(capture, state)
        
        manifest.append({
            "file": files["body_file"],
            "response_url": capture.response_url,
            "content_type": capture.content_type,
            "bytes": len(capture.body),
            "sha256": capture.sha256,
            "top_keys": get_top_keys(capture.body),
            "status": capture.status,
        })
    
    return manifest


class NetworkHooks:
    """
    Crawl4AI hooks for network capture.
    
    Usage with Crawl4AI:
        hooks = NetworkHooks(page_url, output_dir)
        config = CrawlerRunConfig(
            hooks={
                "on_page_context_created": hooks.on_page_context_created,
                "before_return_html": hooks.before_return_html,
            }
        )
    """
    
    def __init__(self, page_url: str, output_dir: Path):
        self.state = NetworkCaptureState(page_url=page_url, output_dir=output_dir)
    
    async def on_page_context_created(self, page, context, **kwargs):
        """Called when page context is created - attach response listener."""
        async def on_response(response):
            if not self.state.is_at_limit():
                task = asyncio.create_task(capture_response(response, self.state))
                self.state.pending_tasks.append(task)
        
        page.on("response", on_response)
    
    async def before_return_html(self, page, html, **kwargs):
        """Called before returning HTML - flush pending captures."""
        # Wait for all pending capture tasks
        if self.state.pending_tasks:
            await asyncio.gather(*self.state.pending_tasks, return_exceptions=True)
            self.state.pending_tasks.clear()
        
        return html
    
    def get_manifest(self) -> List[Dict[str, Any]]:
        """Get network manifest after crawl completes."""
        return build_network_manifest(self.state)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics."""
        return {
            "network_files_captured": len(self.state.captures),
            "network_bytes_captured": self.state.total_bytes,
            "at_limit": self.state.is_at_limit(),
        }
