"""Document ingestion pipeline: OCR -> Chunking -> Embedding -> Vector Store."""

import base64
import json
import io
import time
from pathlib import Path
from mistralai import Mistral
import chromadb
import fitz  # PyMuPDF

from config import (
    MISTRAL_API_KEY,
    OCR_MODEL,
    EMBEDDING_MODEL,
    PDF_PATH,
    VECTORSTORE_DIR,
    MAX_CHUNK_SIZE,
    CHUNK_OVERLAP
)
from chunker import TableAwareChunker


class MistralEmbeddingFunction:
    """Custom embedding function for ChromaDB using Mistral."""
    
    def __init__(self, api_key: str):
        self.client = Mistral(api_key=api_key)
        self.model = EMBEDDING_MODEL
    
    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        # Handle empty input
        if not input:
            return []
        
        # Mistral has a limit on batch size, process in batches
        batch_size = 10
        all_embeddings = []
        
        for i in range(0, len(input), batch_size):
            batch = input[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                inputs=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings


def extract_text_with_mistral_ocr(pdf_path: Path, api_key: str, progress_file: Path = None) -> list[dict]:
    """Extract text from PDF using Mistral OCR API, processing page by page for large PDFs.
    
    Supports resuming from a progress file if the process was interrupted.
    """
    print(f"Starting OCR processing for: {pdf_path}")
    
    client = Mistral(api_key=api_key)
    
    # Open PDF with PyMuPDF
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    print(f"PDF has {total_pages} pages")
    
    # Check for existing progress
    pages_text = []
    start_page = 0
    
    if progress_file and progress_file.exists():
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                pages_text = json.load(f)
            start_page = len(pages_text)
            if start_page > 0:
                print(f"Resuming from page {start_page + 1} (found {start_page} previously processed pages)")
        except Exception as e:
            print(f"Could not load progress file: {e}")
            pages_text = []
            start_page = 0
    
    if start_page >= total_pages:
        print("All pages already processed!")
        doc.close()
        return pages_text
    
    print(f"Processing pages {start_page + 1} to {total_pages} individually...")
    
    failed_pages = []
    
    # Process each page individually
    for page_num in range(start_page, total_pages):
        try:
            # Extract single page as PDF
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            
            # Convert to bytes
            pdf_bytes = single_page_doc.tobytes()
            pdf_base64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
            
            single_page_doc.close()
            
            # Call Mistral OCR API for this page
            response = client.ocr.process(
                model=OCR_MODEL,
                document={
                    "type": "document_url",
                    "document_url": f"data:application/pdf;base64,{pdf_base64}"
                }
            )
            
            # Extract text from response
            if response.pages:
                page_text = response.pages[0].markdown
            else:
                page_text = ""
            
            pages_text.append({
                "page": page_num + 1,
                "text": page_text
            })
            
            print(f"  ✓ Page {page_num + 1}/{total_pages} - {len(page_text)} chars")
            
            # Save progress every 10 pages
            if progress_file and (page_num + 1) % 10 == 0:
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(pages_text, f, ensure_ascii=False)
                print(f"    [Progress saved at page {page_num + 1}]")
            
            # Small delay to avoid rate limiting
            if (page_num + 1) % 10 == 0:
                time.sleep(1)
                
        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Page {page_num + 1}/{total_pages} - Error: {error_msg[:80]}")
            failed_pages.append(page_num + 1)
            pages_text.append({
                "page": page_num + 1,
                "text": f"[OCR failed for page {page_num + 1}]"
            })
            
            # Save progress on error
            if progress_file:
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(pages_text, f, ensure_ascii=False)
            
            # Wait longer after errors (rate limiting)
            if "429" in error_msg or "rate" in error_msg.lower():
                print("    Rate limited, waiting 30 seconds...")
                time.sleep(30)
            else:
                time.sleep(2)
    
    doc.close()
    
    # Save final progress
    if progress_file:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(pages_text, f, ensure_ascii=False)
    
    print(f"\nOCR complete. Extracted {len(pages_text)} pages.")
    if failed_pages:
        print(f"Warning: {len(failed_pages)} pages failed: {failed_pages[:10]}{'...' if len(failed_pages) > 10 else ''}")
    
    return pages_text


def create_vector_store(chunks: list[dict], api_key: str, persist_dir: Path):
    """Create and populate ChromaDB with Mistral embeddings."""
    print(f"\nCreating vector store with {len(chunks)} chunks...")
    
    # Initialize ChromaDB with persistence
    client = chromadb.PersistentClient(path=str(persist_dir))
    
    # Delete existing collection if exists
    try:
        client.delete_collection(name="tunnel_budget")
        print("Deleted existing collection.")
    except Exception:
        pass
    
    # Create collection with Mistral embeddings
    embedding_fn = MistralEmbeddingFunction(api_key)
    collection = client.create_collection(
        name="tunnel_budget",
        embedding_function=embedding_fn,
        metadata={"description": "Tunnel infrastructure budget document"}
    )
    
    # Prepare data for insertion
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [{
        "start_page": c.get("start_page", 0),
        "end_page": c.get("end_page", 0),
        "section": c.get("section", "") or "",
        "table_header": c.get("table_header", "") or ""
    } for c in chunks]
    
    # Add chunks in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        collection.add(
            ids=ids[i:end_idx],
            documents=documents[i:end_idx],
            metadatas=metadatas[i:end_idx]
        )
        print(f"  Added chunks {i+1} to {end_idx}")
    
    print(f"Vector store created with {collection.count()} chunks.")
    return collection


def save_extracted_text(pages: list[dict], output_path: Path):
    """Save extracted text to JSON for inspection."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"Saved extracted text to: {output_path}")


def main():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("RAG Document Ingestion Pipeline")
    print("=" * 60)
    
    # Validate API key
    if not MISTRAL_API_KEY:
        print("\nERROR: MISTRAL_API_KEY not found!")
        print("Please create a .env file with your Mistral API key:")
        print("  MISTRAL_API_KEY=your_key_here")
        return
    
    # Check if PDF exists
    if not PDF_PATH.exists():
        print(f"\nERROR: PDF not found at {PDF_PATH}")
        print("Please copy 'Tunnel budget.pdf' to the data/ folder.")
        return
    
    # Step 1: Extract text with OCR
    print("\n[Step 1/3] Extracting text with Mistral OCR...")
    progress_file = VECTORSTORE_DIR / "ocr_progress.json"
    pages = extract_text_with_mistral_ocr(PDF_PATH, MISTRAL_API_KEY, progress_file)
    
    # Save extracted text for inspection (final version)
    save_extracted_text(pages, VECTORSTORE_DIR / "extracted_pages.json")
    
    # Step 2: Chunk with table awareness
    print("\n[Step 2/3] Chunking with table context preservation...")
    chunker = TableAwareChunker(
        max_chunk_size=MAX_CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    chunks = chunker.process_pages(pages)
    print(f"Created {len(chunks)} chunks from {len(pages)} pages.")
    
    # Save chunks for inspection
    chunks_preview = [{
        "id": i,
        "text_preview": c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
        "start_page": c.get("start_page"),
        "end_page": c.get("end_page"),
        "section": c.get("section"),
        "has_table_header": bool(c.get("table_header"))
    } for i, c in enumerate(chunks)]
    
    with open(VECTORSTORE_DIR / "chunks_preview.json", "w", encoding="utf-8") as f:
        json.dump(chunks_preview, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks preview to: {VECTORSTORE_DIR / 'chunks_preview.json'}")
    
    # Step 3: Create vector store
    print("\n[Step 3/3] Creating vector store with embeddings...")
    collection = create_vector_store(chunks, MISTRAL_API_KEY, VECTORSTORE_DIR)
    
    print("\n" + "=" * 60)
    print("Ingestion complete!")
    print(f"  - Pages processed: {len(pages)}")
    print(f"  - Chunks created: {len(chunks)}")
    print(f"  - Vector store location: {VECTORSTORE_DIR}")
    print("\nYou can now run the app with: streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
