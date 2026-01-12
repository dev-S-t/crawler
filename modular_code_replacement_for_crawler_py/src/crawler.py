"""Crawler wrapper with generalized dynamic stabilization."""
import io
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from PIL import Image
from crawl4ai import AsyncWebCrawler, CrawlResult

from .config import (
    BROWSER_CONFIG, 
    CRAWLER_CONFIG_FAST, 
    CRAWLER_CONFIG_DYNAMIC,
    PLACEHOLDER_PATTERNS,
)


@dataclass
class CrawlMetrics:
    """Metrics extracted from a crawl for comparison."""
    markdown_length: int = 0
    block_count: int = 0
    table_count: int = 0
    link_count: int = 0
    media_count: int = 0


@dataclass 
class RetrySignal:
    """Reason for triggering dynamic retry."""
    triggered: bool
    reason: str  # placeholder_text | thin_content | busy_dom | none
    detail: str  # specific pattern/threshold


@dataclass
class CrawlResultWithMeta:
    """Extended crawl result with metadata about strategy."""
    result: CrawlResult
    used_dynamic: bool
    retry_signal: RetrySignal
    fast_metrics: Optional[CrawlMetrics] = None
    dynamic_metrics: Optional[CrawlMetrics] = None


# --- Thresholds for "thin content" heuristic ---
MIN_MARKDOWN_LENGTH = 8000  # Less than this = likely incomplete
MIN_BLOCK_COUNT = 15        # Less than this = likely incomplete  
MIN_TABLE_COUNT = 0         # 0 tables when expecting some = retry


def extract_metrics(result: CrawlResult) -> CrawlMetrics:
    """Extract metrics from crawl result for comparison."""
    markdown = ""
    if hasattr(result, 'markdown'):
        if hasattr(result.markdown, 'fit_markdown'):
            markdown = result.markdown.fit_markdown or result.markdown.raw_markdown or ""
        elif isinstance(result.markdown, str):
            markdown = result.markdown
    
    # Count blocks (rough: count markdown lines that look structural)
    lines = markdown.split('\n')
    block_count = sum(1 for l in lines if l.strip() and (
        l.strip().startswith('#') or 
        l.strip().startswith('|') or
        l.strip().startswith('-') or
        l.strip().startswith('*')
    ))
    
    tables = getattr(result, 'tables', None) or []
    links = getattr(result, 'links', None)
    media = getattr(result, 'media', None)
    
    link_count = 0
    if isinstance(links, dict):
        link_count = sum(len(v) for v in links.values() if isinstance(v, list))
    elif isinstance(links, list):
        link_count = len(links)
    
    media_count = 0
    if isinstance(media, dict):
        media_count = sum(len(v) for v in media.values() if isinstance(v, list))
    elif isinstance(media, list):
        media_count = len(media)
    
    return CrawlMetrics(
        markdown_length=len(markdown),
        block_count=block_count,
        table_count=len(tables),
        link_count=link_count,
        media_count=media_count,
    )


def check_placeholder_text(result: CrawlResult) -> Optional[str]:
    """Signal A: Check for known placeholder text patterns."""
    markdown = ""
    if hasattr(result, 'markdown'):
        if hasattr(result.markdown, 'fit_markdown'):
            markdown = result.markdown.fit_markdown or result.markdown.raw_markdown or ""
        elif isinstance(result.markdown, str):
            markdown = result.markdown
    
    markdown_lower = markdown.lower()
    html_lower = (getattr(result, 'html', "") or "").lower()
    
    for pattern in PLACEHOLDER_PATTERNS:
        if pattern in markdown_lower or pattern in html_lower:
            return pattern
    return None


def check_thin_content(metrics: CrawlMetrics) -> Optional[str]:
    """Signal B: Check if content looks too thin (likely incomplete)."""
    if metrics.markdown_length < MIN_MARKDOWN_LENGTH:
        return f"markdown_length={metrics.markdown_length}<{MIN_MARKDOWN_LENGTH}"
    if metrics.block_count < MIN_BLOCK_COUNT:
        return f"block_count={metrics.block_count}<{MIN_BLOCK_COUNT}"
    return None


def check_busy_dom(result: CrawlResult) -> Optional[str]:
    """Signal C: Check for busy/loading DOM indicators."""
    html = getattr(result, 'html', "") or ""
    html_lower = html.lower()
    
    busy_indicators = [
        ('aria-busy="true"', 'aria_busy'),
        ('role="progressbar"', 'progressbar'),
        ('class="skeleton"', 'skeleton_class'),
        ('class="shimmer"', 'shimmer_class'),
        ('class="loading"', 'loading_class'),
        ('class="placeholder"', 'placeholder_class'),
    ]
    
    for pattern, indicator_name in busy_indicators:
        if pattern in html_lower:
            return indicator_name
    
    # Check for skeleton/loading class patterns with regex
    skeleton_patterns = [
        r'class="[^"]*skeleton[^"]*"',
        r'class="[^"]*shimmer[^"]*"',
        r'class="[^"]*loading[^"]*"',
        r'class="[^"]*spinner[^"]*"',
    ]
    for pattern in skeleton_patterns:
        if re.search(pattern, html_lower):
            return f"class_pattern:{pattern[:20]}"
    
    return None


def should_retry_with_dynamic(result: CrawlResult, metrics: CrawlMetrics) -> RetrySignal:
    """
    Determine if we should retry with dynamic profile.
    
    Checks 3 signals:
    A) Placeholder text detected
    B) Content too thin (short markdown, few blocks)
    C) Busy DOM indicators (aria-busy, skeleton classes)
    """
    # Signal A: Placeholder text
    placeholder = check_placeholder_text(result)
    if placeholder:
        return RetrySignal(
            triggered=True,
            reason="placeholder_text",
            detail=placeholder
        )
    
    # Signal B: Thin content
    thin = check_thin_content(metrics)
    if thin:
        return RetrySignal(
            triggered=True,
            reason="thin_content", 
            detail=thin
        )
    
    # Signal C: Busy DOM
    busy = check_busy_dom(result)
    if busy:
        return RetrySignal(
            triggered=True,
            reason="busy_dom",
            detail=busy
        )
    
    return RetrySignal(triggered=False, reason="none", detail="")


def compress_screenshot(screenshot_data: bytes, quality: int = 85) -> bytes:
    """Compress screenshot to JPEG format."""
    try:
        img = Image.open(io.BytesIO(screenshot_data))
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        return output.getvalue()
    except Exception:
        return screenshot_data


async def crawl_url(
    url: str, 
    mode: str = "smart"
) -> CrawlResultWithMeta:
    """
    Crawl a URL with configurable strategy.
    
    Modes:
    - "fast": Use fast profile only
    - "dynamic": Use dynamic profile only
    - "smart": Fast first, retry with dynamic if needed (default)
    
    Smart mode checks 3 signals before retry:
    A) Placeholder text (loading, retrieving, etc.)
    B) Thin content (short markdown, few blocks)
    C) Busy DOM (aria-busy, skeleton classes)
    """
    async with AsyncWebCrawler(config=BROWSER_CONFIG) as crawler:
        
        # Mode: dynamic only
        if mode == "dynamic":
            result = await crawler.arun(url=url, config=CRAWLER_CONFIG_DYNAMIC)
            metrics = extract_metrics(result)
            return CrawlResultWithMeta(
                result=result,
                used_dynamic=True,
                retry_signal=RetrySignal(False, "forced_dynamic", ""),
                dynamic_metrics=metrics,
            )
        
        # Mode: fast only
        if mode == "fast":
            result = await crawler.arun(url=url, config=CRAWLER_CONFIG_FAST)
            metrics = extract_metrics(result)
            return CrawlResultWithMeta(
                result=result,
                used_dynamic=False,
                retry_signal=RetrySignal(False, "forced_fast", ""),
                fast_metrics=metrics,
            )
        
        # Mode: smart (default)
        # Step 1: Try fast
        result = await crawler.arun(url=url, config=CRAWLER_CONFIG_FAST)
        fast_metrics = extract_metrics(result)
        
        if not result.success:
            return CrawlResultWithMeta(
                result=result,
                used_dynamic=False,
                retry_signal=RetrySignal(False, "crawl_failed", ""),
                fast_metrics=fast_metrics,
            )
        
        # Step 2: Check if retry needed
        signal = should_retry_with_dynamic(result, fast_metrics)
        
        if signal.triggered:
            print(f"   ⟳ Retry triggered ({signal.reason}: {signal.detail})")
            result = await crawler.arun(url=url, config=CRAWLER_CONFIG_DYNAMIC)
            dynamic_metrics = extract_metrics(result)
            return CrawlResultWithMeta(
                result=result,
                used_dynamic=True,
                retry_signal=signal,
                fast_metrics=fast_metrics,
                dynamic_metrics=dynamic_metrics,
            )
        
        # Fast succeeded without retry
        return CrawlResultWithMeta(
            result=result,
            used_dynamic=False,
            retry_signal=signal,
            fast_metrics=fast_metrics,
        )
