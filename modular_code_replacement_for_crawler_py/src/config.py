"""Configuration settings for the web data extractor."""
import os
from dotenv import load_dotenv
from crawl4ai import BrowserConfig, CrawlerRunConfig

# Load environment variables
load_dotenv()

# --- API Keys ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# --- Model Configuration ---
MODEL_NAME = "gemini-3-flash-preview"
THINKING_LEVEL = "MEDIUM"

# --- Browser Configuration ---
BROWSER_CONFIG = BrowserConfig(
    viewport_width=1920,
    viewport_height=1080,
    headless=True,
)

# --- Crawler Run Configurations ---

# FAST profile: Default for static/simple pages
CRAWLER_CONFIG_FAST = CrawlerRunConfig(
    magic=True,
    screenshot=True,
    screenshot_wait_for=1.0,
)

# DYNAMIC profile: For JS-heavy pages with lazy loading
# Used when fast profile detects placeholders
CRAWLER_CONFIG_DYNAMIC = CrawlerRunConfig(
    magic=True,
    # Wait for network to settle (JS finished)
    wait_until="networkidle",
    # Timeout for wait conditions
    page_timeout=60000,
    # Scroll through entire page to trigger lazy loading
    scan_full_page=True,
    scroll_delay=0.5,
    # Extra pause before capturing HTML
    delay_before_return_html=2.0,
    # Screenshot settings
    screenshot=True,
    screenshot_wait_for=3.0,  # Extra wait for charts/graphs
)

# Legacy alias for backward compatibility
CRAWLER_RUN_CONFIG = CRAWLER_CONFIG_FAST

# --- Placeholder Detection ---
# Patterns that indicate content is still loading
PLACEHOLDER_PATTERNS = [
    "retrieving data",
    "loading...",
    "please wait",
    "fetching",
]

# --- Output Directories ---
OUTPUT_DIR = "output"
RAW_NETWORK_DIR = os.path.join(OUTPUT_DIR, "raw_network")
SCREENSHOTS_DIR = os.path.join(OUTPUT_DIR, "screenshots")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "debug")

# --- Extraction Settings ---
MAX_MARKDOWN_LENGTH = 50000
LLM_TIMEOUT_SECONDS = 30  # Timeout for LLM calls


def create_config_with_hooks(base_config: CrawlerRunConfig, hooks: dict) -> CrawlerRunConfig:
    """Create a new config with hooks injected."""
    # CrawlerRunConfig accepts hooks parameter
    return CrawlerRunConfig(
        magic=base_config.magic,
        screenshot=base_config.screenshot,
        screenshot_wait_for=getattr(base_config, 'screenshot_wait_for', 1.0),
        wait_until=getattr(base_config, 'wait_until', None),
        page_timeout=getattr(base_config, 'page_timeout', 30000),
        scan_full_page=getattr(base_config, 'scan_full_page', False),
        scroll_delay=getattr(base_config, 'scroll_delay', 0.2),
        delay_before_return_html=getattr(base_config, 'delay_before_return_html', 0.0),
        hooks=hooks,
    )

