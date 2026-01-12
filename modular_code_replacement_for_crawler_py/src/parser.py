"""Deterministic Markdown-to-blocks parser for context preservation."""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class Link:
    """Link extracted from markdown."""
    anchor_text: str
    url: str
    line_number: int


@dataclass
class Block:
    """A structural block from markdown content."""
    block_id: str
    block_type: str  # heading | paragraph | list_item | table_block | code | quote
    content: str
    section_path: List[str]  # ["H1 text", "H2 text", "H3 text"]
    line_start: int
    line_end: int
    links: List[Link] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "block_id": self.block_id,
            "type": self.block_type,
            "content": self.content,
            "section_path": self.section_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "links": [{"anchor_text": l.anchor_text, "url": l.url, "line": l.line_number} for l in self.links]
        }


def extract_links_from_text(text: str, base_line: int) -> List[Link]:
    """Extract markdown links from text."""
    links = []
    # Match [anchor](url) pattern
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    for match in re.finditer(pattern, text):
        links.append(Link(
            anchor_text=match.group(1),
            url=match.group(2),
            line_number=base_line
        ))
    return links


def parse_markdown_to_blocks(markdown: str) -> List[Block]:
    """
    Parse markdown into structured blocks with section context.
    
    Each block contains:
    - block_id: unique identifier
    - type: heading, paragraph, list_item, table_block, code, quote
    - section_path: hierarchy of headings above this block
    - line_start, line_end: source line numbers
    - links: extracted links with context
    """
    blocks = []
    lines = markdown.split('\n')
    
    # Track current section hierarchy
    section_stack = []  # [(level, heading_text), ...]
    
    block_counter = 0
    i = 0
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        if not stripped:
            i += 1
            continue
        
        block_counter += 1
        block_id = f"block_{block_counter:04d}"
        line_start = i + 1  # 1-indexed
        
        # Current section path
        current_section = [s[1] for s in section_stack]
        
        # === HEADING ===
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            
            # Update section stack
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, heading_text))
            
            links = extract_links_from_text(heading_text, line_start)
            blocks.append(Block(
                block_id=block_id,
                block_type="heading",
                content=heading_text,
                section_path=current_section.copy(),
                line_start=line_start,
                line_end=line_start,
                links=links
            ))
            i += 1
            continue
        
        # === CODE BLOCK ===
        if stripped.startswith('```'):
            code_lines = [line]
            code_start = i
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                code_lines.append(lines[i])
                i += 1
            
            blocks.append(Block(
                block_id=block_id,
                block_type="code",
                content='\n'.join(code_lines),
                section_path=current_section.copy(),
                line_start=code_start + 1,
                line_end=i,
                links=[]
            ))
            continue
        
        # === QUOTE ===
        if stripped.startswith('>'):
            quote_lines = []
            quote_start = i
            while i < len(lines) and lines[i].strip().startswith('>'):
                quote_lines.append(lines[i].strip()[1:].strip())
                i += 1
            
            content = '\n'.join(quote_lines)
            links = extract_links_from_text(content, quote_start + 1)
            blocks.append(Block(
                block_id=block_id,
                block_type="quote",
                content=content,
                section_path=current_section.copy(),
                line_start=quote_start + 1,
                line_end=i,
                links=links
            ))
            continue
        
        # === TABLE ===
        if '|' in stripped and stripped.startswith('|'):
            table_lines = []
            table_start = i
            while i < len(lines) and '|' in lines[i]:
                table_lines.append(lines[i])
                i += 1
            
            content = '\n'.join(table_lines)
            links = extract_links_from_text(content, table_start + 1)
            blocks.append(Block(
                block_id=block_id,
                block_type="table_block",
                content=content,
                section_path=current_section.copy(),
                line_start=table_start + 1,
                line_end=i,
                links=links
            ))
            continue
        
        # === LIST ITEM ===
        list_match = re.match(r'^(\s*)[-*+]|\d+\.', stripped)
        if list_match:
            list_lines = []
            list_start = i
            while i < len(lines):
                l = lines[i].strip()
                if not l:
                    break
                if re.match(r'^[-*+]|\d+\.', l) or lines[i].startswith('  '):
                    list_lines.append(lines[i])
                    i += 1
                else:
                    break
            
            content = '\n'.join(list_lines)
            links = extract_links_from_text(content, list_start + 1)
            blocks.append(Block(
                block_id=block_id,
                block_type="list_item",
                content=content,
                section_path=current_section.copy(),
                line_start=list_start + 1,
                line_end=i,
                links=links
            ))
            continue
        
        # === PARAGRAPH (default) ===
        para_lines = []
        para_start = i
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('#'):
            para_lines.append(lines[i])
            i += 1
        
        content = '\n'.join(para_lines)
        links = extract_links_from_text(content, para_start + 1)
        blocks.append(Block(
            block_id=block_id,
            block_type="paragraph",
            content=content,
            section_path=current_section.copy(),
            line_start=para_start + 1,
            line_end=i,
            links=links
        ))
    
    return blocks


def build_page_envelope(
    url: str,
    markdown: str,
    blocks: List[Block],
    tables: List[Dict],
    media: List[Dict],
    links: List[Dict],
    crawl_meta: Dict
) -> Dict[str, Any]:
    """
    Build the complete page.json envelope.
    """
    return {
        "url": url,
        "timestamp": crawl_meta.get("timestamp"),
        "crawl_time_seconds": crawl_meta.get("crawl_time_seconds"),
        "blocks": [b.to_dict() for b in blocks],
        "tables": tables,
        "media": media,
        "links_flat": links,
        "stats": {
            "block_count": len(blocks),
            "table_count": len(tables),
            "media_count": len(media),
            "link_count": len(links),
            "markdown_length": len(markdown),
        }
    }
