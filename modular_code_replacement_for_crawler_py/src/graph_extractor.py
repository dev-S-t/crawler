"""Generic image extraction - identify important images and extract info via LLM."""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types

from .config import GEMINI_API_KEY, MODEL_NAME


client = genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# FILTERING: Skip decorative images, keep meaningful ones
# ============================================================

SKIP_PATTERNS = [
    # Flags, icons, logos
    "flag", "logo", "icon", "avatar", "profile", "badge", "medal", 
    "trophy", "seal", "emblem", "favicon",
    # UI elements
    "button", "arrow", "chevron", "caret", "close", "menu", 
    "hamburger", "search", "loading", "spinner",
    # Social media icons
    "facebook", "twitter", "linkedin", "instagram", "youtube", 
    "whatsapp", "telegram", "pinterest", "reddit",
    # Ads, tracking, spacers
    "ad", "banner", "pixel", "tracker", "1x1", "2x2", "spacer", 
    "blank", "transparent",
    # Common decorative
    "bg", "background", "pattern", "texture", "divider", "separator",
]

# Large images likely to contain data
MIN_MEANINGFUL_SIZE = 10000  # 10KB
LIKELY_DATA_SIZE = 50000     # 50KB


def is_meaningful_image(
    url: str,
    alt_text: str = "",
    width: int = 0,
    height: int = 0
) -> bool:
    """
    Determine if an image is likely meaningful (data viz, infographic, etc).
    
    Filters out: flags, icons, logos, UI elements, tiny images.
    """
    url_lower = url.lower()
    alt_lower = alt_text.lower()
    combined = f"{url_lower} {alt_lower}"
    
    # Skip if matches decorative patterns
    for pattern in SKIP_PATTERNS:
        if pattern in combined:
            return False
    
    # Skip common icon extensions
    if url_lower.endswith((".ico", ".svg", ".gif")):
        return False
    
    # Skip tiny images (likely icons)
    if width > 0 and height > 0:
        if width < 100 or height < 100:
            return False
        if width * height < 10000:  # Less than 100x100
            return False
    
    # Keep images with data-related alt text
    data_keywords = ["chart", "graph", "plot", "diagram", "statistics", 
                     "trend", "comparison", "infographic", "data", "rate",
                     "growth", "decline", "percentage", "population"]
    for keyword in data_keywords:
        if keyword in alt_lower:
            return True
    
    # Keep larger images (more likely to be meaningful)
    return True


def filter_meaningful_images(
    media: List[Dict[str, Any]],
    max_images: int = 5
) -> List[Dict[str, Any]]:
    """
    Filter page media to only meaningful images.
    
    Returns at most max_images that are likely data visualizations or important content.
    """
    meaningful = []
    
    for item in media:
        url = item.get("src", item.get("url", ""))
        alt = item.get("alt", "")
        # Safe type conversion for width/height (can be strings or None)
        try:
            width = int(item.get("width", 0) or 0)
        except (ValueError, TypeError):
            width = 0
        try:
            height = int(item.get("height", 0) or 0)
        except (ValueError, TypeError):
            height = 0
        
        if not url or not url.startswith("http"):
            continue
        
        if is_meaningful_image(url, alt, width, height):
            meaningful.append({
                "url": url,
                "alt": alt,
                "width": width,
                "height": height,
            })
        
        if len(meaningful) >= max_images:
            break
    
    return meaningful


# ============================================================
# LLM EXTRACTION: Send images to Gemini for analysis
# ============================================================

IMAGE_ANALYSIS_PROMPT = """Analyze this image and extract any useful information.

If this is a DATA VISUALIZATION (chart, graph, plot, map):
- Describe what data it shows
- Extract any visible numbers, labels, or values
- Identify trends or key insights

If this is an INFOGRAPHIC or DIAGRAM:
- Describe the main message or information
- Extract any text or data points

If this is just a DECORATIVE IMAGE (photo, illustration without data):
- Return {"is_data": false, "description": "brief description"}

Return JSON:
{
  "is_data": true/false,
  "image_type": "chart|graph|map|infographic|photo|other",
  "title": "title if visible",
  "description": "what the image shows",
  "data_points": [{"label": "name", "value": "123"}],
  "insights": "key insights or trends"
}"""


def extract_image_info(image_url: str) -> Optional[Dict[str, Any]]:
    """
    Extract information from an image using Gemini vision.
    
    Downloads the image and sends to Gemini for analysis.
    Returns extracted data or None if decorative/error.
    """
    import requests
    
    try:
        # Download image first
        print(f"      Downloading: {image_url[:50]}...")
        response = requests.get(image_url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        if response.status_code != 200:
            print(f"      Download failed: {response.status_code}")
            return None
        
        image_data = response.content
        
        # Detect mime type
        mime_type = "image/jpeg"
        if image_url.lower().endswith(".png"):
            mime_type = "image/png"
        elif image_url.lower().endswith(".webp"):
            mime_type = "image/webp"
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    types.Part.from_text(text=IMAGE_ANALYSIS_PROMPT),
                ],
            ),
        ]
        
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=2048,
        )
        
        result = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )
        
        parsed = json.loads(result.text)
        parsed["source_url"] = image_url
        return parsed
        
    except Exception as e:
        print(f"      Image analysis error: {e}")
        return None


def extract_images_from_page(
    media: List[Dict[str, Any]],
    max_images: int = 3
) -> List[Dict[str, Any]]:
    """
    Extract information from meaningful images on a page.
    
    1. Filters out decorative images (flags, icons, etc)
    2. Sends meaningful images to LLM for analysis
    3. Returns extracted data/descriptions
    """
    extracted = []
    
    meaningful = filter_meaningful_images(media, max_images)
    
    if meaningful:
        print(f"   → Found {len(meaningful)} meaningful images, analyzing...")
    
    for img in meaningful:
        result = extract_image_info(img["url"])
        if result and result.get("is_data"):
            extracted.append({
                "source_url": img["url"],
                "alt_text": img.get("alt", ""),
                "image_type": result.get("image_type", "unknown"),
                "title": result.get("title", ""),
                "description": result.get("description", ""),
                "data_points": result.get("data_points", []),
                "insights": result.get("insights", ""),
                "extraction_source": "vision",
            })
            print(f"      Extracted: {result.get('image_type', 'image')} - {result.get('title', 'untitled')[:50]}")
    
    return extracted


# ============================================================
# SCREENSHOT FALLBACK: Analyze page.jpg when no images found
# ============================================================

PAGE_ANALYSIS_PROMPT = """Analyze this webpage screenshot and extract data from any charts, graphs, or infographics visible.

Look for:
1. Bar charts, line graphs, pie charts, area charts
2. Data tables rendered visually
3. Infographics with statistics
4. Maps with data overlays

For EACH visualization found, extract:
- Title or caption
- Data values (numbers, percentages, labels)
- Axes labels and scales
- Key insights or trends

Return JSON:
{
  "charts_found": 2,
  "visualizations": [
    {
      "type": "line_chart|bar_chart|pie_chart|table|map|infographic",
      "title": "visible title",
      "context": "text around the chart describing it",
      "data_points": [{"label": "2020", "value": "100M"}, ...],
      "axes": {"x": "Year", "y": "Population"},
      "insights": "key trend or finding"
    }
  ]
}

If no data visualizations found, return {"charts_found": 0, "visualizations": []}"""


def extract_from_screenshot(screenshot_path: Path) -> List[Dict[str, Any]]:
    """
    Analyze a full page screenshot to extract chart/graph data.
    
    Used as fallback when:
    1. No meaningful images found via media URLs
    2. Charts are JS-rendered (canvas/SVG) not <img> tags
    
    Returns list of extracted visualizations.
    """
    if not screenshot_path.exists():
        return []
    
    try:
        print(f"   → Analyzing page screenshot for charts...")
        
        with open(screenshot_path, "rb") as f:
            image_data = f.read()
        
        # Detect mime type
        mime_type = "image/jpeg"
        if str(screenshot_path).lower().endswith(".png"):
            mime_type = "image/png"
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=image_data, mime_type=mime_type),
                    types.Part.from_text(text=PAGE_ANALYSIS_PROMPT),
                ],
            ),
        ]
        
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
            max_output_tokens=8192,  # Increased for complex pages
        )
        
        result = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        )
        
        response_text = result.text.strip()
        
        # Try to repair truncated JSON
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to find valid JSON structure
            if '{"charts_found":' in response_text:
                # Find visualizations array and try to parse what we can
                try:
                    # Look for partial valid structure
                    if '"visualizations": [' in response_text:
                        # Extract until we find closing structure
                        viz_start = response_text.find('"visualizations": [')
                        # Try to find a complete visualization object
                        parsed = {"charts_found": 1, "visualizations": []}
                        print("      Partial JSON - attempting recovery...")
                except:
                    pass
            parsed = {"charts_found": 0, "visualizations": []}
        
        charts_found = parsed.get("charts_found", 0)
        if charts_found > 0:
            print(f"      Found {charts_found} chart(s) in screenshot")
        
        # Convert to standard format
        extracted = []
        for viz in parsed.get("visualizations", []):
            extracted.append({
                "source_url": str(screenshot_path),
                "alt_text": viz.get("context", ""),
                "image_type": viz.get("type", "chart"),
                "title": viz.get("title", ""),
                "description": viz.get("context", ""),
                "data_points": viz.get("data_points", []),
                "axes": viz.get("axes", {}),
                "insights": viz.get("insights", ""),
                "extraction_source": "screenshot",
            })
        
        return extracted
        
    except Exception as e:
        print(f"      Screenshot analysis error: {e}")
        return []
