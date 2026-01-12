"""Utility functions for the web data extractor."""
import os
import json
import glob
from typing import List, Dict, Any

from .config import RAW_NETWORK_DIR


def generate_network_manifest() -> List[Dict[str, Any]]:
    """
    Scan raw_network directory and generate a manifest of captured JSON files.
    
    For each JSON file, extracts the top-level keys to help LLM understand
    what data is available.
    
    Returns:
        List of dicts: [{filename: "...", keys: ["GDP", "Year"], url: "..."}]
    """
    manifest = []
    
    # Get all JSON files (excluding .meta.json files)
    pattern = os.path.join(RAW_NETWORK_DIR, "*.json")
    json_files = [f for f in glob.glob(pattern) if not f.endswith(".meta.json")]
    
    for filepath in json_files:
        filename = os.path.basename(filepath)
        entry = {
            "filename": filename,
            "keys": [],
            "url": None,
        }
        
        try:
            # Extract keys from JSON
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            if isinstance(data, dict):
                entry["keys"] = list(data.keys())[:20]  # Limit keys shown
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                entry["keys"] = list(data[0].keys())[:20]
            
            # Try to get URL from sidecar metadata
            meta_path = filepath.replace(".json", ".meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    entry["url"] = meta.get("url")
                    
        except Exception:
            pass  # Silent fail for corrupted files
        
        manifest.append(entry)
    
    return manifest


def ensure_output_directories() -> None:
    """Create output directories if they don't exist."""
    from .config import OUTPUT_DIR, RAW_NETWORK_DIR, SCREENSHOTS_DIR, DEBUG_DIR
    
    for directory in [OUTPUT_DIR, RAW_NETWORK_DIR, SCREENSHOTS_DIR, DEBUG_DIR]:
        os.makedirs(directory, exist_ok=True)
