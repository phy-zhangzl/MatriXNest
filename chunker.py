"""Table-aware chunking for budget documents with cross-page table handling."""

import re
from typing import Optional


class TableAwareChunker:
    """Chunks markdown text while preserving table context and structure."""
    
    def __init__(self, max_chunk_size: int = 1500, overlap: int = 200):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def _is_table_row(self, line: str) -> bool:
        """Check if a line is a table row (contains | characters)."""
        return "|" in line and line.strip().startswith("|")
    
    def _is_table_separator(self, line: str) -> bool:
        """Check if a line is a table separator (|---|---|)."""
        return bool(re.match(r"^\|[\s\-:]+\|", line.strip()))
    
    def _is_table_header(self, line: str, next_line: Optional[str] = None) -> bool:
        """Check if a line is a table header (followed by separator)."""
        if not self._is_table_row(line):
            return False
        if next_line and self._is_table_separator(next_line):
            return True
        return False
    
    def _is_table_continuation(self, text: str) -> bool:
        """Check if text starts with table rows but no header."""
        lines = text.strip().split("\n")
        if not lines:
            return False
        
        # Look at first few non-empty lines
        table_rows = []
        for line in lines[:5]:
            line = line.strip()
            if not line:
                continue
            if self._is_table_row(line):
                table_rows.append(line)
            elif line.startswith("#"):
                return False  # Starts with heading, not continuation
            else:
                break
        
        # If we found table rows but no separator after first row, it's a continuation
        if len(table_rows) >= 2:
            if not self._is_table_separator(table_rows[1]):
                return True
        
        return False
    
    def _table_is_complete(self, text: str) -> bool:
        """Check if text ends with a complete table (or no table)."""
        lines = text.strip().split("\n")
        if not lines:
            return True
        
        # Find last table row
        last_table_idx = -1
        for i in range(len(lines) - 1, -1, -1):
            if self._is_table_row(lines[i]):
                last_table_idx = i
                break
        
        if last_table_idx == -1:
            return True  # No table found
        
        # Check if there's content after the last table row
        for i in range(last_table_idx + 1, len(lines)):
            line = lines[i].strip()
            if line and not self._is_table_row(line):
                return True  # Non-table content after table
        
        # Table is at the end - check if it looks complete
        # (This is heuristic - a table ending mid-page might still be complete)
        return True
    
    def _extract_table_header(self, text: str) -> Optional[str]:
        """Extract the table header (first row + separator) from text."""
        lines = text.split("\n")
        header_lines = []
        
        for i, line in enumerate(lines):
            if self._is_table_row(line):
                if i + 1 < len(lines) and self._is_table_separator(lines[i + 1]):
                    return line + "\n" + lines[i + 1]
                elif header_lines:
                    # We're past the header
                    break
        
        return None
    
    def merge_cross_page_tables(self, pages: list[dict]) -> list[dict]:
        """Detect and merge tables that span multiple pages."""
        if not pages:
            return []
        
        merged = []
        pending = None
        
        for page in pages:
            text = page["text"]
            page_num = page["page"]
            
            # Check if this page starts with a table continuation
            starts_with_continuation = self._is_table_continuation(text)
            
            if starts_with_continuation and pending:
                # Merge with previous page
                pending["text"] += "\n" + text
                pending["end_page"] = page_num
            else:
                # Save pending if exists
                if pending:
                    merged.append(pending)
                
                # Start new pending
                pending = {
                    "text": text,
                    "start_page": page_num,
                    "end_page": page_num
                }
        
        # Don't forget the last pending
        if pending:
            merged.append(pending)
        
        return merged
    
    def chunk_with_table_context(self, text: str, metadata: dict) -> list[dict]:
        """Chunk text while preserving table headers in each chunk."""
        chunks = []
        current_section = metadata.get("section", "")
        current_table_header = None
        
        lines = text.split("\n")
        current_chunk_lines = []
        current_size = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Track section headers
            if line.strip().startswith("#"):
                current_section = line.strip().lstrip("#").strip()
            
            # Detect table header
            next_line = lines[i + 1] if i + 1 < len(lines) else None
            if self._is_table_header(line, next_line):
                current_table_header = line + "\n" + next_line
            
            # Add line to current chunk
            current_chunk_lines.append(line)
            current_size += len(line) + 1  # +1 for newline
            
            # Check if chunk is full
            if current_size >= self.max_chunk_size:
                chunk_text = "\n".join(current_chunk_lines)
                
                chunks.append({
                    "text": chunk_text,
                    "section": current_section,
                    "table_header": current_table_header,
                    **metadata
                })
                
                # Start new chunk with overlap and table context
                overlap_start = max(0, len(current_chunk_lines) - 5)  # Keep last 5 lines for overlap
                current_chunk_lines = current_chunk_lines[overlap_start:]
                
                # Prepend table header if we're in a table
                if current_table_header and self._is_table_row(line):
                    header_lines = current_table_header.split("\n")
                    current_chunk_lines = header_lines + current_chunk_lines
                
                current_size = sum(len(l) + 1 for l in current_chunk_lines)
            
            i += 1
        
        # Don't forget the last chunk
        if current_chunk_lines:
            chunk_text = "\n".join(current_chunk_lines)
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "section": current_section,
                    "table_header": current_table_header,
                    **metadata
                })
        
        return chunks
    
    def process_pages(self, pages: list[dict]) -> list[dict]:
        """Full processing pipeline: merge tables, then chunk."""
        # Step 1: Merge cross-page tables
        merged_pages = self.merge_cross_page_tables(pages)
        
        # Step 2: Chunk each merged section
        all_chunks = []
        for page_data in merged_pages:
            metadata = {
                "start_page": page_data["start_page"],
                "end_page": page_data["end_page"]
            }
            chunks = self.chunk_with_table_context(page_data["text"], metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
