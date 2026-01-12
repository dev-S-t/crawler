"""Context mapping: match network tables to page blocks using fuzzy matching."""
from typing import List, Dict, Any, Optional
from rapidfuzz import fuzz, process


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return text.lower().strip().replace("_", " ").replace("-", " ")


def find_related_blocks(
    table_title: str,
    blocks: List[Any],
    threshold: int = 60
) -> Dict[str, Any]:
    """
    Find blocks that are related to a table title using fuzzy matching.
    
    Returns:
        Dict with related_block_ids, section_path_guess, and match scores
    """
    title_normalized = normalize_text(table_title)
    
    # Collect searchable content from blocks
    block_texts = []
    for block in blocks:
        # Handle both dict and dataclass Block objects
        if hasattr(block, 'content'):
            content = block.content
            block_id = block.block_id
            block_type = block.block_type  # Block dataclass uses block_type
            section_path = block.section_path
        else:
            content = block.get("content", "")
            block_id = block.get("block_id", "")
            block_type = block.get("type", "")
            section_path = block.get("section_path", [])
        
        search_text = normalize_text(content)
        
        block_texts.append({
            "block_id": block_id,
            "text": search_text,
            "content": content,
            "type": block_type,
            "section_path": section_path,
        })
    
    # Find matches using fuzzy matching
    related_blocks = []
    best_section_path = None
    best_score = 0
    
    for bt in block_texts:
        # Try different matching strategies
        score = max(
            fuzz.partial_ratio(title_normalized, bt["text"]),
            fuzz.token_set_ratio(title_normalized, bt["text"]),
        )
        
        if score >= threshold:
            related_blocks.append({
                "block_id": bt["block_id"],
                "score": score,
                "matched_content": bt["content"][:100],
            })
            
            # Track best section path from headings
            if score > best_score and bt["type"] == "heading":
                best_score = score
                best_section_path = bt["section_path"]
    
    # Sort by score
    related_blocks.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "related_block_ids": [b["block_id"] for b in related_blocks[:5]],
        "section_path_guess": best_section_path,
        "match_details": related_blocks[:3] if related_blocks else None,
    }


def map_tables_to_context(
    tables: List[Dict[str, Any]],
    blocks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Add context mapping to extracted tables.
    
    For each table, adds:
    - related_block_ids: blocks that match the table title
    - section_path_guess: likely section in page structure
    """
    mapped_tables = []
    
    for table in tables:
        title = table.get("title", table.get("title_guess", ""))
        
        # Find related blocks
        context = find_related_blocks(title, blocks)
        
        # Add context to table
        table_with_context = table.copy()
        table_with_context["related_block_ids"] = context["related_block_ids"]
        table_with_context["section_path_guess"] = context["section_path_guess"]
        
        mapped_tables.append(table_with_context)
    
    return mapped_tables
