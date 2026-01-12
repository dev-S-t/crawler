"""
Web Data Extractor v3 - Main Orchestrator

Day 2: Network hooks + deterministic extraction

Usage:
    uv run python main.py <URL>
    uv run python main.py <URL> --mode fast|dynamic|smart
    uv run python main.py --suite urls.json
    uv run python main.py <URL> --network off|capture
"""
import argparse
import asyncio
import sys
import json
import base64
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

from src.crawler import compress_screenshot, extract_metrics, should_retry_with_dynamic, RetrySignal
from src.llm import tables_to_schema, extract_from_markdown
from src.validators import validate_extraction, check_tables_sanity
from src.parser import parse_markdown_to_blocks, build_page_envelope
from src.hooks import NetworkHooks
from src.network_parser import parse_network_files, candidate_to_table_dict
from src.context_mapper import map_tables_to_context
from src.pdf_pipeline import (
    detect_downloadable_links, download_file, extract_tables_from_pdf, 
    extract_csv_data, extract_text_from_pdf
)
from src.vision_extractor import extract_from_pdf_with_vision
from src.graph_extractor import extract_images_from_page, extract_from_screenshot
from src.link_selector import filter_crawlable_links, select_links_with_llm, url_hash
from src.config import OUTPUT_DIR, BROWSER_CONFIG, CRAWLER_CONFIG_FAST, CRAWLER_CONFIG_DYNAMIC


# Parallel crawling: max 3 concurrent browsers
BROWSER_LIMIT = 3
browser_semaphore = asyncio.Semaphore(BROWSER_LIMIT)


def generate_run_id(url: str) -> str:
    domain = urlparse(url).netloc.replace(".", "_").replace("www_", "")
    return f"{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def get_markdown_content(result) -> str:
    if hasattr(result, 'markdown'):
        if hasattr(result.markdown, 'fit_markdown'):
            return result.markdown.fit_markdown or result.markdown.raw_markdown or ""
        elif isinstance(result.markdown, str):
            return result.markdown
    return str(result.markdown) if result.markdown else ""


def save_screenshots(result, run_dir: Path) -> dict:
    screenshot_data = getattr(result, 'screenshot', None)
    if not screenshot_data:
        return {"saved": False}
    try:
        if isinstance(screenshot_data, str):
            image_data = base64.b64decode(screenshot_data)
        else:
            image_data = screenshot_data
        compressed = compress_screenshot(image_data, quality=85)
        with open(run_dir / "page.jpg", "wb") as f:
            f.write(compressed)
        return {"saved": True, "jpg_bytes": len(compressed)}
    except:
        return {"saved": False}


def extract_media_from_result(result) -> list:
    """Extract all media (images, videos, etc) from crawl result."""
    media = []
    
    # Method 1: From result.media dict
    raw_media = getattr(result, 'media', None)
    if raw_media and isinstance(raw_media, dict):
        for mtype, items in raw_media.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        src = item.get("src", item.get("url", ""))
                        if src:
                            media.append({
                                "type": mtype, 
                                "src": src, 
                                "alt": item.get("alt", ""),
                                "width": item.get("width", 0),
                                "height": item.get("height", 0),
                            })
    
    # Method 2: From result.links (some images are in links)
    links = getattr(result, 'links', None) or {}
    images_in_links = links.get('images', [])
    for item in images_in_links:
        if isinstance(item, dict):
            src = item.get("src", item.get("href", ""))
            if src and src not in [m.get("src") for m in media]:
                media.append({
                    "type": "images",
                    "src": src,
                    "alt": item.get("alt", item.get("text", "")),
                })
    
    return media


def extract_links_from_result(result) -> list:
    links = []
    raw_links = getattr(result, 'links', None)
    if raw_links and isinstance(raw_links, dict):
        for ltype, items in raw_links.items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        links.append({"type": ltype, "href": item.get("href", ""), "text": item.get("text", "")})
    return links


def convert_tables_to_dict(tables) -> list:
    if not tables:
        return []
    return [{
        "table_id": f"table_{i+1}",
        "caption": getattr(t, 'caption', None) or f"Table {i+1}",
        "headers": list(t.headers) if hasattr(t, 'headers') else [],
        "rows": [list(r) for r in t.rows] if hasattr(t, 'rows') else [],
        "source": "dom",
    } for i, t in enumerate(tables)]


async def process_url(
    url: str, 
    mode: str = "smart", 
    skip_llm: bool = False,
    network_capture: bool = True
) -> Dict[str, Any]:
    """Process a single URL with network capture via hooks."""
    run_id = generate_run_id(url)
    run_dir = Path(OUTPUT_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"URL: {url}")
    print(f"Run: {run_id} | Mode: {mode} | Network: {'on' if network_capture else 'off'}")
    print(f"{'='*60}")
    
    # Setup hooks
    hooks_instance = NetworkHooks(url, run_dir) if network_capture else None
    
    start_time = datetime.now()
    used_dynamic = False
    retry_signal = RetrySignal(False, "none", "")
    fast_metrics = None
    dynamic_metrics = None
    
    # Timeout for entire crawl operation (prevents hanging)
    CRAWL_TIMEOUT = 90  # seconds
    
    # Create crawler and set hooks
    crawler = AsyncWebCrawler(config=BROWSER_CONFIG)
    
    if hooks_instance:
        crawler.crawler_strategy.set_hook("on_page_context_created", hooks_instance.on_page_context_created)
        crawler.crawler_strategy.set_hook("before_return_html", hooks_instance.before_return_html)
    
    await crawler.start()
    
    try:
        if mode == "dynamic":
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=CRAWLER_CONFIG_DYNAMIC),
                timeout=CRAWL_TIMEOUT
            )
            used_dynamic = True
            dynamic_metrics = extract_metrics(result)
        elif mode == "fast":
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=CRAWLER_CONFIG_FAST),
                timeout=CRAWL_TIMEOUT
            )
            fast_metrics = extract_metrics(result)
        else:  # smart
            result = await asyncio.wait_for(
                crawler.arun(url=url, config=CRAWLER_CONFIG_FAST),
                timeout=CRAWL_TIMEOUT
            )
            fast_metrics = extract_metrics(result)
            
            if result.success:
                retry_signal = should_retry_with_dynamic(result, fast_metrics)
                if retry_signal.triggered:
                    print(f"   ⟳ Retry: {retry_signal.reason}")
                    # Reset hooks for retry
                    if network_capture:
                        hooks_instance = NetworkHooks(url, run_dir)
                        crawler.crawler_strategy.set_hook("on_page_context_created", hooks_instance.on_page_context_created)
                        crawler.crawler_strategy.set_hook("before_return_html", hooks_instance.before_return_html)
                    result = await asyncio.wait_for(
                        crawler.arun(url=url, config=CRAWLER_CONFIG_DYNAMIC),
                        timeout=CRAWL_TIMEOUT
                    )
                    used_dynamic = True
                    dynamic_metrics = extract_metrics(result)
    except asyncio.TimeoutError:
        await crawler.close()
        print(f"   ⏰ Crawl timeout after {CRAWL_TIMEOUT}s")
        return {"url": url, "run_id": run_id, "success": False, "error": f"Timeout after {CRAWL_TIMEOUT}s"}
    finally:
        await crawler.close()
    
    crawl_time = (datetime.now() - start_time).total_seconds()
    
    if not result.success:
        return {"url": url, "run_id": run_id, "success": False, "error": result.error_message}
    
    profile = "dynamic" if used_dynamic else "fast"
    print(f"   Crawl: {crawl_time:.1f}s ({profile})")
    
    # Save markdown
    markdown_content = get_markdown_content(result)
    with open(run_dir / "page.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    save_screenshots(result, run_dir)
    
    # Parse
    blocks = parse_markdown_to_blocks(markdown_content)
    media = extract_media_from_result(result)
    links = extract_links_from_result(result)
    crawl_tables = getattr(result, 'tables', None) or []
    tables_dict = convert_tables_to_dict(crawl_tables)
    
    # Network stats
    network_manifest = hooks_instance.get_manifest() if hooks_instance else []
    net_stats = hooks_instance.get_stats() if hooks_instance else {"network_files_captured": 0, "network_bytes_captured": 0}
    print(f"   Network: {net_stats['network_files_captured']} files, {net_stats['network_bytes_captured']//1024}KB")
    
    # Build metadata
    crawl_meta = {
        "url": url, "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "crawl_time_seconds": crawl_time,
        "success": True, "profile_used": profile,
        "retry_signal": {"triggered": retry_signal.triggered, "reason": retry_signal.reason, "detail": retry_signal.detail},
        "fast_metrics": fast_metrics.__dict__ if fast_metrics else None,
        "dynamic_metrics": dynamic_metrics.__dict__ if dynamic_metrics else None,
        **net_stats,
    }
    
    # Build page envelope
    page_envelope = build_page_envelope(url, markdown_content, blocks, tables_dict, media, links, crawl_meta)
    page_envelope["network_manifest"] = network_manifest
    
    with open(run_dir / "page.json", "w", encoding="utf-8") as f:
        json.dump(page_envelope, f, indent=2, ensure_ascii=False)
    with open(run_dir / "crawl_meta.json", "w", encoding="utf-8") as f:
        json.dump(crawl_meta, f, indent=2)
    
    # Extraction: network > DOM > LLM
    extracted_tables = []
    extraction_source = "none"
    
    print(f"   Extracting...")
    
    # 1. Network tables (always try, deterministic)
    network_candidates = parse_network_files(run_dir / "raw_network")
    if network_candidates:
        print(f"   → Network: {len(network_candidates)} candidates")
        for c in network_candidates:
            extracted_tables.append(candidate_to_table_dict(c))
        extraction_source = "network"
    # 2. DOM tables (deterministic)
    elif check_tables_sanity(crawl_tables):
        print(f"   → DOM tables")
        ext = tables_to_schema(crawl_tables)
        extracted_tables = [t.model_dump() for t in ext.tables]
        extraction_source = "dom"
    # 3. LLM (only if not skipped)
    elif not skip_llm:
        print(f"   → LLM")
        try:
            ext = extract_from_markdown(markdown_content, [])
            extracted_tables = [t.model_dump() for t in ext.tables]
            extraction_source = "llm"
        except Exception as e:
            print(f"   LLM error: {e}")
    else:
        print(f"   → Skipped (no network/DOM data, LLM disabled)")
    
    # 4. PDF/download detection with smart LLM/OCR choice
    downloadable = detect_downloadable_links(links, url)
    downloaded_files = []
    pdf_files = []
    
    if downloadable:
        print(f"   → Found {len(downloadable)} downloadable files")
        downloads_dir = run_dir / "downloads"
        
        # Download up to 10 files
        for dl in downloadable[:10]:
            result = await download_file(dl["url"], downloads_dir)
            if result:
                downloaded_files.append(result)
                fpath = Path(result["path"])
                if fpath.suffix.lower() == ".pdf":
                    pdf_files.append(fpath)
                elif fpath.suffix.lower() == ".csv":
                    csv_tables = extract_csv_data(fpath)
                    extracted_tables.extend(csv_tables)
        
        if downloaded_files:
            print(f"   Downloaded: {len(downloaded_files)} files")
        
        # PDF Strategy: text first, LLM for scanned
        if pdf_files:
            text_extracted = 0
            scanned_pdfs = []
            
            # First: try text extraction on all PDFs
            for fpath in pdf_files:
                text = extract_text_from_pdf(fpath)
                if len(text) > 100:  # Has readable text
                    pdf_tables = extract_tables_from_pdf(fpath)
                    if pdf_tables:
                        extracted_tables.extend(pdf_tables)
                        text_extracted += 1
                        extraction_source = "pdf" if extraction_source == "none" else extraction_source
                        print(f"      {fpath.name}: {len(pdf_tables)} tables (text)")
                    else:
                        print(f"      {fpath.name}: text found, no tables")
                else:
                    scanned_pdfs.append(fpath)
            
            # If all are scanned, send 2 to LLM
            if scanned_pdfs and text_extracted == 0 and not skip_llm:
                print(f"   → All {len(scanned_pdfs)} PDFs are scanned, using LLM on 2")
                for fpath in scanned_pdfs[:2]:
                    vision_tables = extract_from_pdf_with_vision(fpath, downloads_dir)
                    if vision_tables:
                        extracted_tables.extend(vision_tables)
                        extraction_source = "vision" if extraction_source == "none" else extraction_source
                        print(f"      {fpath.name}: {len(vision_tables)} tables (LLM)")
                    else:
                        print(f"      {fpath.name}: No tables (LLM)")
            elif scanned_pdfs:
                print(f"   → {len(scanned_pdfs)} scanned PDFs skipped")
    
    # 5. Image/graph extraction (generic - find meaningful images on page)
    # media contains {type, src, alt} - type can be "images", "img", etc
    page_images = []
    for m in media:
        mtype = str(m.get("type", "")).lower()
        src = str(m.get("src", ""))
        # Include if type contains 'image' or src looks like an image URL
        if "image" in mtype or "img" in mtype or any(src.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp", ".gif"]):
            page_images.append(m)
    
    if page_images:
        print(f"   → Found {len(page_images)} images, checking for data visualizations...")
    
    if page_images and not skip_llm:
        image_data = extract_images_from_page(page_images, max_images=3)
        if image_data:
            extraction_source = "vision" if extraction_source == "none" else extraction_source
    else:
        image_data = []
    
    # Screenshot fallback: When no chart images found, analyze page.jpg for JS-rendered charts
    if not image_data and not skip_llm:
        screenshot_path = run_dir / "page.jpg"
        if screenshot_path.exists():
            image_data = extract_from_screenshot(screenshot_path)
            if image_data:
                extraction_source = "screenshot" if extraction_source == "none" else extraction_source
    
    # 6. Map tables to page context (Day 2.2)
    if extracted_tables and blocks:
        extracted_tables = map_tables_to_context(extracted_tables, blocks)
    
    data_output = {
        "summary": f"Extracted from {url}",
        "tables": extracted_tables,
        "extraction_source": extraction_source,
        "downloaded_files": downloaded_files,
        "images": image_data,
    }
    with open(run_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(data_output, f, indent=2, ensure_ascii=False)
    
    print(f"✅ {run_id} | src={extraction_source} tables={len(extracted_tables)} images={len(image_data)} net={net_stats['network_files_captured']}")
    
    # Print detailed run summary
    print(f"\n{'='*60}")
    print(f"📁 OUTPUT DIRECTORY: {run_dir}")
    print(f"{'='*60}")
    print(f"\n📊 EXTRACTION STATISTICS:")
    print(f"   • Extraction source: {extraction_source}")
    print(f"   • Tables extracted: {len(extracted_tables)}")
    print(f"   • Images/charts analyzed: {len(image_data)}")
    print(f"   • Network files captured: {net_stats['network_files_captured']}")
    print(f"   • Downloaded files: {len(downloaded_files)}")
    
    # Calculate sizes
    data_json_size = (run_dir / "data.json").stat().st_size if (run_dir / "data.json").exists() else 0
    page_md_size = (run_dir / "page.md").stat().st_size if (run_dir / "page.md").exists() else 0
    page_jpg_size = (run_dir / "page.jpg").stat().st_size if (run_dir / "page.jpg").exists() else 0
    
    print(f"\n📄 OUTPUT FILES:")
    print(f"   • data.json      - Extracted tables & images ({data_json_size//1024}KB)")
    print(f"   • page.md        - Full page markdown ({page_md_size//1024}KB)")
    print(f"   • page.json      - Links, metadata, blocks")
    print(f"   • page.jpg       - Full page screenshot ({page_jpg_size//1024}KB)")
    print(f"   • crawl_meta.json - Crawl configuration & timing")
    if (run_dir / "raw_network").exists():
        net_files = list((run_dir / "raw_network").glob("*"))
        print(f"   • raw_network/   - {len(net_files)} captured network files")
    if (run_dir / "downloads").exists():
        dl_files = list((run_dir / "downloads").glob("*"))
        print(f"   • downloads/     - {len(dl_files)} downloaded files (PDFs, CSVs)")
    if (run_dir / "subpages").exists():
        sub_dirs = [d for d in (run_dir / "subpages").iterdir() if d.is_dir()]
        print(f"   • subpages/      - {len(sub_dirs)} crawled subpages")
    
    print(f"{'='*60}\n")
    
    return {
        "url": url, "run_id": run_id, "success": True,
        "run_dir": str(run_dir),
        "profile_used": profile, "extraction_source": extraction_source,
        "tables": len(extracted_tables), "network_files": net_stats["network_files_captured"],
    }


async def run_suite(urls_file: str, mode: str = "smart", skip_llm: bool = False, network: bool = True):
    with open(urls_file, "r") as f:
        urls_config = json.load(f)
    urls = urls_config if isinstance(urls_config, list) else urls_config.get("urls", [])
    
    results = []
    for entry in urls:
        url = entry if isinstance(entry, str) else entry.get("url")
        try:
            results.append(await process_url(url, mode, skip_llm, network))
        except Exception as e:
            results.append({"url": url, "success": False, "error": str(e)})
    
    report = Path(OUTPUT_DIR) / f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report, "w") as f:
        json.dump({"results": results}, f, indent=2)
    
    print(f"\n{'='*60}\nSuite Summary")
    for r in results:
        s = "✅" if r.get("success") else "❌"
        print(f"{s} {r['url'][:50]:50} src={r.get('extraction_source', 'N/A'):8} net={r.get('network_files', 0)}")


async def crawl_with_depth(
    url: str,
    mode: str = "smart",
    skip_llm: bool = False,
    network: bool = True,
    depth_mode: str = "0",
    current_depth: int = 0,
    max_depth: int = 2,
    parent_dir: Path = None,
    extraction_count: list = None  # Mutable counter [count] shared across recursive calls
) -> Dict[str, Any]:
    """
    Crawl URL with optional depth (follow links).
    
    depth_mode:
        "0" = single page only
        "1" = crawl links on main page
        "2" = crawl 2 levels deep (true recursion)
        "smart" = LLM selects which links to crawl (2 levels)
    
    MAX 10 extractions total across all levels.
    """
    # Initialize extraction counter on first call
    if extraction_count is None:
        extraction_count = [0]
    
    # Check if already at max extractions
    MAX_EXTRACTIONS = 10
    if extraction_count[0] >= MAX_EXTRACTIONS:
        return {"url": url, "success": False, "error": "Max extractions reached"}
    
    # Determine max depth
    if depth_mode == "0":
        max_depth = 0
    elif depth_mode == "1":
        max_depth = 1
    elif depth_mode == "2":
        max_depth = 2
    elif depth_mode == "smart":
        max_depth = 2
    
    # Crawl main page
    result = await process_url(url, mode, skip_llm, network)
    
    # Increment extraction counter (main page counts as 1)
    extraction_count[0] += 1
    
    if not result.get("success"):
        return result
    
    # Check if we should go deeper
    if current_depth >= max_depth or depth_mode == "0":
        return result
    
    # Get links from result
    run_dir = Path(result.get("run_dir", ""))
    page_json = run_dir / "page.json"
    
    if not page_json.exists():
        return result
    
    with open(page_json, "r", encoding="utf-8") as f:
        page_data = json.load(f)
    
    links = page_data.get("links_flat", [])
    
    # Depth-based link limits: 6 at depth 0, 3 at depth 1
    depth_limits = {0: 6, 1: 3}
    max_links_for_depth = depth_limits.get(current_depth, 3)
    
    # Filter to crawlable links
    crawlable = filter_crawlable_links(links, url, max_links=20)
    
    if not crawlable:
        print(f"   → No crawlable links found")
        return result
    
    # Smart mode: LLM picks links with strict limit
    if depth_mode == "smart" and not skip_llm:
        page_title = page_data.get("title", url)
        crawlable = select_links_with_llm(crawlable, page_title, max_links=max_links_for_depth)
        print(f"   → Smart: LLM selected {len(crawlable)} links (max {max_links_for_depth})")
    else:
        # Non-smart: just take first N
        crawlable = crawlable[:max_links_for_depth]
        print(f"   → Depth {current_depth + 1}: {len(crawlable)} links (max {max_links_for_depth})")
    
    # Create subpages directory
    subpages_dir = run_dir / "subpages"
    subpages_dir.mkdir(exist_ok=True)
    
    # Cap at max 6 subpages for depth 0, 3 for depth 1
    max_subpages = 6 if current_depth == 0 else 3
    links_to_crawl = crawlable[:max_subpages]
    
    print(f"   → Crawling {len(links_to_crawl)} links in parallel (max {BROWSER_LIMIT} browsers)")
    
    # Async helper to crawl one subpage with semaphore
    async def crawl_subpage(link: Dict[str, str], idx: int) -> Dict[str, Any]:
        # Check if we hit max extractions
        if extraction_count[0] >= MAX_EXTRACTIONS:
            return {"url": link["url"], "success": False, "error": "Max extractions reached"}
        
        link_url = link["url"]
        link_text = link.get("text", "")[:40]
        link_hash = url_hash(link_url)
        subpage_dir = subpages_dir / link_hash
        subpage_dir.mkdir(exist_ok=True)
        
        async with browser_semaphore:  # Max 3 concurrent
            # Double check counter inside semaphore
            if extraction_count[0] >= MAX_EXTRACTIONS:
                return {"url": link_url, "success": False, "error": "Max extractions reached"}
            
            print(f"   [{idx+1}/{len(links_to_crawl)}] {link_text}... (total: {extraction_count[0]+1})")
            
            try:
                crawler = AsyncWebCrawler(config=BROWSER_CONFIG)
                await crawler.start()
                
                try:
                    crawl_result = await asyncio.wait_for(
                        crawler.arun(url=link_url, config=CRAWLER_CONFIG_FAST),
                        timeout=30.0
                    )
                    
                    # Increment counter
                    extraction_count[0] += 1
                    
                    # Extract and save data
                    tables = getattr(crawl_result, 'tables', None) or []
                    table_count = len(tables)
                    
                    # Save markdown
                    md_content = ""
                    if hasattr(crawl_result, 'markdown'):
                        if hasattr(crawl_result.markdown, 'raw_markdown'):
                            md_content = crawl_result.markdown.raw_markdown
                        elif isinstance(crawl_result.markdown, str):
                            md_content = crawl_result.markdown
                    
                    with open(subpage_dir / "page.md", "w", encoding="utf-8") as f:
                        f.write(md_content)
                    
                    # Save tables
                    tables_data = convert_tables_to_dict(tables)
                    with open(subpage_dir / "data.json", "w", encoding="utf-8") as f:
                        json.dump({"tables": tables_data, "url": link_url}, f, indent=2)
                    
                    print(f"      ✓ {table_count} tables saved")
                    
                    result = {"url": link_url, "text": link_text, "success": True, "tables": table_count, "dir": link_hash}
                    
                    # Recursive: if we should go deeper AND have room left
                    if current_depth + 1 < max_depth and extraction_count[0] < MAX_EXTRACTIONS:
                        # Extract links from this page and crawl level 2
                        links_in_page = getattr(crawl_result, 'links', {})
                        internal_links = links_in_page.get('internal', [])
                        if internal_links:
                            level2_links = filter_crawlable_links(internal_links, link_url, max_links=5)[:3]
                            if level2_links:
                                print(f"      → Level 2: {len(level2_links)} links")
                                # Create level2 subdir
                                level2_dir = subpage_dir / "subpages"
                                level2_dir.mkdir(exist_ok=True)
                                
                                level2_results = []
                                for l2_link in level2_links:
                                    if extraction_count[0] >= MAX_EXTRACTIONS:
                                        break
                                    l2_url = l2_link["url"]
                                    l2_hash = url_hash(l2_url)
                                    l2_subdir = level2_dir / l2_hash
                                    l2_subdir.mkdir(exist_ok=True)
                                    
                                    # Simple crawl for level 2
                                    try:
                                        l2_crawler = AsyncWebCrawler(config=BROWSER_CONFIG)
                                        await l2_crawler.start()
                                        l2_result = await asyncio.wait_for(
                                            l2_crawler.arun(url=l2_url, config=CRAWLER_CONFIG_FAST),
                                            timeout=30.0
                                        )
                                        await l2_crawler.close()
                                        
                                        extraction_count[0] += 1
                                        l2_tables = getattr(l2_result, 'tables', None) or []
                                        print(f"         ✓ L2: {len(l2_tables)} tables")
                                        level2_results.append({
                                            "url": l2_url, "success": True, 
                                            "tables": len(l2_tables), "dir": l2_hash
                                        })
                                    except Exception as l2_e:
                                        print(f"         ❌ L2 error: {l2_e}")
                                        level2_results.append({"url": l2_url, "success": False, "error": str(l2_e)})
                                
                                result["level2"] = level2_results
                    
                    return result
                    
                except asyncio.TimeoutError:
                    print(f"      ⏰ Timeout")
                    return {"url": link_url, "text": link_text, "success": False, "error": "timeout"}
                finally:
                    await crawler.close()
                    
            except Exception as e:
                print(f"      ❌ Error: {e}")
                return {"url": link_url, "text": link_text, "success": False, "error": str(e)}
    
    # Run all subpage crawls in parallel
    tasks = [crawl_subpage(link, i) for i, link in enumerate(links_to_crawl)]
    subpage_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to error dicts
    subpage_results = [
        r if isinstance(r, dict) else {"success": False, "error": str(r)}
        for r in subpage_results
    ]
    
    # Save depth summary
    summary_path = subpages_dir / "_depth_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "parent_url": url,
            "depth_mode": depth_mode,
            "current_depth": current_depth,
            "subpages": subpage_results,
            "total_tables": sum(r.get("tables", 0) for r in subpage_results),
        }, f, indent=2)
    
    # Ensure result is a dict (process_url returns dict)
    if not isinstance(result, dict):
        result = {"success": True, "url": url, "run_dir": str(run_dir)}
    
    result["subpages"] = subpage_results
    result["total_subpage_tables"] = sum(r.get("tables", 0) for r in subpage_results)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Web Data Extractor v3")
    parser.add_argument("url", nargs="?")
    parser.add_argument("--mode", choices=["fast", "dynamic", "smart"], default="smart")
    parser.add_argument("--suite", metavar="FILE")
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--network", choices=["off", "capture"], default="capture")
    parser.add_argument("--depth", choices=["0", "1", "2", "smart"], default="0",
        help="Crawl depth: 0=single, 1=links, 2=links+sublinks, smart=LLM picks")
    args = parser.parse_args()
    
    if args.suite:
        asyncio.run(run_suite(args.suite, args.mode, args.skip_llm, args.network == "capture"))
    elif args.url:
        if args.depth == "0":
            asyncio.run(process_url(args.url, args.mode, args.skip_llm, args.network == "capture"))
        else:
            asyncio.run(crawl_with_depth(
                args.url, 
                args.mode, 
                args.skip_llm, 
                args.network == "capture",
                args.depth
            ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

