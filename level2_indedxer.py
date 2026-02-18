import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from pypdf import PdfReader
from pinecone import Pinecone
import time
import json
from typing import List, Dict

# --- CONFIGURATION BLOCK ---
load_dotenv() 

# Get environment variables
VERTEX_AI_SERVICE_ACCOUNT_PATH = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Settings
PINECONE_INDEX_NAME = "document-summaries"
EMBEDDING_MODEL = "text-embedding-004"
LOCATION = "us-central1"
DATA_DIR = "./data"

# Chunking settings
CHUNK_SIZE = 400  # words per chunk
CHUNK_OVERLAP = 50  # words of overlap between chunks

# Cache file for detail chunks
DETAIL_CACHE_FILE = "document_details_cache.json"

# TOC detection settings
TOC_SEARCH_PAGES = 25  # TOC is in first 25 pages
MIN_CHAPTER_LENGTH = 1000


def initialize_genai_client():
    """Initialize Google GenAI client with Vertex AI credentials."""
    try:
        from google.oauth2 import service_account
        
        credentials = service_account.Credentials.from_service_account_file(
            VERTEX_AI_SERVICE_ACCOUNT_PATH,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        client = genai.Client(
            vertexai=True,
            project=GOOGLE_CLOUD_PROJECT,
            location=LOCATION,
            credentials=credentials
        )
        print(f"✓ Google GenAI client initialized")
        return client
    except Exception as e:
        print(f"Error initializing GenAI client: {e}")
        return None


def extract_chapter_titles_from_toc(pdf_path: str) -> List[Dict]:
    """Extract ONLY chapter titles and numbers from TOC (without page numbers)."""
    print("\n" + "="*60)
    print("STEP 1: EXTRACTING CHAPTER TITLES FROM TOC")
    print("="*60)
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"✓ PDF loaded: {total_pages} pages")
        
        print(f"\nSearching first {TOC_SEARCH_PAGES} pages for Table of Contents...")
        toc_pages = []
        
        for page_num in range(min(TOC_SEARCH_PAGES, total_pages)):
            text = reader.pages[page_num].extract_text()
            
            if re.search(r'\b(Contents|TABLE OF CONTENTS|Table of Contents)\b', text, re.IGNORECASE):
                toc_pages.append(text)
                if len(toc_pages) == 1:
                    print(f"  ✓ Found TOC starting at page {page_num + 1}")
                
                if len(toc_pages) >= 10:
                    break
        
        if not toc_pages:
            print("  ✗ Could not find Table of Contents")
            return []
        
        print(f"  ✓ Reading TOC from {len(toc_pages)} pages")
        
        full_toc_text = "\n".join(toc_pages)
        
        # Extract chapter titles
        chapter_pattern = re.compile(
            r'^\s*Chapter\s*(\d+)\s+([A-Za-z][^\d\n]+?)(?:\s+\d+)?\s*$',
            re.MULTILINE
        )
        
        chapters = []
        seen_chapters = set()
        
        for match in chapter_pattern.finditer(full_toc_text):
            chapter_num = match.group(1)
            title_raw = match.group(2)
            
            title = re.sub(r'\.{2,}', '', title_raw).strip()
            title = re.sub(r'\s+', ' ', title)
            
            if chapter_num in seen_chapters:
                continue
            
            seen_chapters.add(chapter_num)
            
            chapters.append({
                'chapter_num': chapter_num,
                'title': title,
            })
            
            print(f"  ✓ Found Chapter {chapter_num}: {title}")
        
        if not chapters:
            print("  ✗ No chapters found in TOC")
            return []
        
        chapters.sort(key=lambda x: int(x['chapter_num']))
        
        print(f"\n✓ Extracted {len(chapters)} chapter titles from TOC")
        return chapters
        
    except Exception as e:
        print(f"✗ Error extracting from TOC: {e}")
        import traceback
        print(traceback.format_exc())
        return []


def find_chapter_pages_from_headers(pdf_path: str, chapters: List[Dict]) -> List[Dict]:
    """
    Scan through PDF to find where each chapter starts by looking at page headers.
    Chapters start 1 page BEFORE where the header appears.
    """
    print("\n" + "="*60)
    print("STEP 2: FINDING CHAPTER START PAGES FROM HEADERS")
    print("="*60)
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        print(f"Scanning pages 26-{total_pages} for chapter headers (alternating pages)...")
        print("(This may take a minute...)\n")
        
        for chapter in chapters:
            chapter_num = chapter['chapter_num']
            title = chapter['title']
            
            pattern = re.compile(
                rf'Chapter\s*{chapter_num}\s+{re.escape(title[:20])}',
                re.IGNORECASE
            )
            
            found = False
            
            # Scan alternating pages starting from page 26
            for page_num in range(25, total_pages, 2):
                text = reader.pages[page_num].extract_text()
                
                header_section = text[:int(len(text) * 0.2)]
                
                if pattern.search(header_section):
                    # Chapter starts 1 page before the header
                    chapter['start_page'] = page_num
                    print(f"  ✓ Chapter {chapter_num} ({title}) starts at page {page_num}")
                    found = True
                    break
            
            if not found:
                print(f"  ⚠ Chapter {chapter_num} not found in headers - will estimate")
                chapter['start_page'] = None
        
        # Estimate missing chapters
        for i, chapter in enumerate(chapters):
            if chapter['start_page'] is None:
                prev_page = 1
                next_page = total_pages
                
                for j in range(i - 1, -1, -1):
                    if chapters[j]['start_page'] is not None:
                        prev_page = chapters[j]['start_page']
                        break
                
                for j in range(i + 1, len(chapters)):
                    if chapters[j]['start_page'] is not None:
                        next_page = chapters[j]['start_page']
                        break
                
                estimated_page = (prev_page + next_page) // 2
                chapter['start_page'] = estimated_page
                print(f"  ⚠ Estimated Chapter {chapter['chapter_num']} at page {estimated_page}")
        
        # Calculate end pages
        for i in range(len(chapters)):
            if i < len(chapters) - 1:
                chapters[i]['end_page'] = chapters[i+1]['start_page'] - 1
            else:
                chapters[i]['end_page'] = total_pages
        
        print(f"\n✓ Successfully determined page ranges for all {len(chapters)} chapters")
        return chapters
        
    except Exception as e:
        print(f"✗ Error finding chapter pages: {e}")
        import traceback
        print(traceback.format_exc())
        return []


def extract_chapter_text(pdf_path: str, chapters: List[Dict]) -> List[Dict]:
    """Extract the actual text for each chapter based on page ranges."""
    print("\n" + "="*60)
    print("STEP 3: EXTRACTING TEXT FOR EACH CHAPTER")
    print("="*60)
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        for chapter in chapters:
            print(f"\n  Extracting Chapter {chapter['chapter_num']}: {chapter['title']}")
            print(f"    Pages {chapter['start_page']}-{chapter['end_page']}")
            
            chapter_text = ""
            
            for page_idx in range(chapter['start_page'] - 1, min(chapter['end_page'], total_pages)):
                if page_idx < total_pages:
                    chapter_text += reader.pages[page_idx].extract_text() + "\n"
            
            chapter['text'] = chapter_text
            chapter['word_count'] = len(chapter_text.split())
            
            print(f"    ✓ Extracted {chapter['word_count']} words")
        
        valid_chapters = [ch for ch in chapters if ch['word_count'] >= MIN_CHAPTER_LENGTH]
        
        if len(valid_chapters) < len(chapters):
            removed = len(chapters) - len(valid_chapters)
            print(f"\n  ⚠ Filtered out {removed} chapters with < {MIN_CHAPTER_LENGTH} words")
        
        print(f"\n✓ Successfully extracted text for {len(valid_chapters)} chapters")
        return valid_chapters
        
    except Exception as e:
        print(f"✗ Error extracting text: {e}")
        import traceback
        print(traceback.format_exc())
        return []


def extract_chapters_from_toc(pdf_path: str) -> List[Dict]:
    """
    Main function: Extract chapters using 3-step process:
    1. Get chapter titles from TOC
    2. Find actual page numbers by scanning headers
    3. Extract text for each chapter
    """
    chapters = extract_chapter_titles_from_toc(pdf_path)
    
    if not chapters:
        return []
    
    chapters = find_chapter_pages_from_headers(pdf_path, chapters)
    
    if not chapters:
        return []
    
    chapters = extract_chapter_text(pdf_path, chapters)
    
    return chapters


def detect_chapters_by_pattern(pdf_path: str) -> List[Dict]:
    """Fallback: Pattern-based chapter detection."""
    print("\n⚠ Using fallback: Pattern-based chapter detection")
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        page_texts = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            page_texts.append({'page_num': page_num + 1, 'text': text})
        
        chapter_pattern = re.compile(
            r'^\s*(?:CHAPTER|Chapter)\s*(\d+)',
            re.MULTILINE | re.IGNORECASE
        )
        
        chapters = []
        
        for page_data in page_texts:
            page_num = page_data['page_num']
            text = page_data['text']
            
            text_sample = text[:int(len(text) * 0.3)]
            match = chapter_pattern.search(text_sample)
            
            if match:
                chapter_num = match.group(1)
                
                lines = text.split('\n')
                title = f"Chapter {chapter_num}"
                for i, line in enumerate(lines):
                    if re.search(rf'(?:CHAPTER|Chapter)\s*{chapter_num}', line, re.IGNORECASE):
                        if i + 1 < len(lines):
                            potential_title = lines[i + 1].strip()
                            if len(potential_title) > 5 and len(potential_title) < 100:
                                title = potential_title
                        break
                
                chapters.append({
                    'chapter_num': chapter_num,
                    'start_page': page_num,
                    'title': title
                })
        
        seen = {}
        for ch in chapters:
            if ch['chapter_num'] not in seen:
                seen[ch['chapter_num']] = ch
        
        chapters = list(seen.values())
        chapters.sort(key=lambda x: int(x['chapter_num']))
        
        for i in range(len(chapters)):
            if i < len(chapters) - 1:
                chapters[i]['end_page'] = chapters[i+1]['start_page'] - 1
            else:
                chapters[i]['end_page'] = total_pages
        
        for chapter in chapters:
            chapter_text = ""
            for page_num in range(chapter['start_page'], chapter['end_page'] + 1):
                if page_num <= total_pages:
                    chapter_text += page_texts[page_num - 1]['text'] + "\n"
            
            chapter['text'] = chapter_text
            chapter['word_count'] = len(chapter_text.split())
        
        chapters = [ch for ch in chapters if ch['word_count'] >= MIN_CHAPTER_LENGTH]
        
        return chapters
        
    except Exception as e:
        print(f"✗ Error in pattern detection: {e}")
        return []


def fallback_split_by_pages(pdf_path: str, pages_per_section: int = 50) -> List[Dict]:
    """Last resort: Page-based split."""
    print("\n⚠ Using last resort: Page-based split")
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        chapters = []
        section_num = 1
        
        for start_page in range(1, total_pages + 1, pages_per_section):
            end_page = min(start_page + pages_per_section - 1, total_pages)
            
            section_text = ""
            for page_num in range(start_page - 1, end_page):
                section_text += reader.pages[page_num].extract_text() + "\n"
            
            chapters.append({
                'chapter_num': str(section_num),
                'start_page': start_page,
                'end_page': end_page,
                'title': f"Section {section_num} (Pages {start_page}-{end_page})",
                'text': section_text,
                'word_count': len(section_text.split())
            })
            
            section_num += 1
        
        return chapters
        
    except Exception as e:
        print(f"✗ Error in fallback split: {e}")
        return []


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks based on word count."""
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        start += (chunk_size - overlap)
    
    return chunks


def load_detail_cache() -> List[Dict]:
    """Load cached detail chunks if they exist."""
    if os.path.exists(DETAIL_CACHE_FILE):
        try:
            with open(DETAIL_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load detail cache: {e}")
    return []


def save_detail_cache(detail_data: List[Dict]):
    """Save detail chunks to cache file."""
    try:
        cache_data = [
            {k: v for k, v in item.items() if k != 'chunk_text'}
            for item in detail_data
        ]
        with open(DETAIL_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Detail cache saved to {DETAIL_CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def get_embeddings(client, texts: List[str]) -> List[List[float]]:
    """Generate embeddings for text chunks with rate limiting and retry logic."""
    try:
        embeddings = []
        batch_size = 5
        delay_between_batches = 5
        max_retries = 3
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            current_batch = i//batch_size + 1
            total_batches = (len(texts)-1)//batch_size + 1
            
            print(f"  Processing embedding batch {current_batch}/{total_batches}...")
            
            retry_delay = delay_between_batches
            for attempt in range(max_retries):
                try:
                    result = client.models.embed_content(
                        model=EMBEDDING_MODEL,
                        contents=batch
                    )
                    
                    for embedding in result.embeddings:
                        embeddings.append(embedding.values)
                    
                    break
                    
                except ClientError as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt < max_retries - 1:
                            retry_delay = min(retry_delay * 2, 60)
                            print(f"  ⚠ Rate limit hit! Retrying in {retry_delay}s...")
                            time.sleep(retry_delay)
                        else:
                            print(f"  ✗ Max retries reached.")
                            raise
                    else:
                        raise
            
            if i + batch_size < len(texts):
                print(f"  ⏳ Waiting {delay_between_batches} seconds...")
                time.sleep(delay_between_batches)
        
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []


def create_detail_chunks_from_single_pdf(data_dir: str = DATA_DIR) -> List[Dict]:
    """Create Level 2 (detail) chunks from single PDF with header-based detection."""
    print("\n" + "="*60)
    print("READING PDF FROM DATA FOLDER")
    print("="*60)
    
    if not os.path.exists(data_dir):
        print(f"✗ Error: Data directory not found: {data_dir}")
        return []
    
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"✗ Error: No PDF files found in {data_dir}")
        return []
    
    if len(pdf_files) > 1:
        print(f"\n⚠ Warning: Found {len(pdf_files)} PDFs. Using first one: {pdf_files[0]}")
    
    pdf_path = os.path.join(data_dir, pdf_files[0])
    print(f"✓ Processing: {pdf_files[0]}")
    
    # Check cache
    cached_details = load_detail_cache()
    if cached_details:
        print(f"\n✓ Found {len(cached_details)} cached detail chunks!")
        use_cache = input("Do you want to use cached detail chunks? (y/n): ").lower().strip()
        if use_cache == 'y':
            print("✓ Using cached detail chunks.")
            return cached_details
    
    # Try Method 1: TOC + Header scanning (BEST)
    print("\n" + "="*60)
    print("METHOD 1: TOC Extraction + Header Scanning")
    print("="*60)
    chapters = extract_chapters_from_toc(pdf_path)
    
    # Try Method 2: Pattern matching
    if not chapters:
        print("\n" + "="*60)
        print("METHOD 2: Pattern-Based Detection")
        print("="*60)
        chapters = detect_chapters_by_pattern(pdf_path)
    
    # Try Method 3: Page split
    if not chapters:
        print("\n" + "="*60)
        print("METHOD 3: Page-Based Splitting")
        print("="*60)
        use_fallback = input("No chapters detected. Split by pages? (y/n): ").lower().strip()
        if use_fallback == 'y':
            chapters = fallback_split_by_pages(pdf_path)
        else:
            print("✗ Cannot proceed without chapters.")
            return []
    
    if not chapters:
        print("✗ Failed to detect chapters.")
        return []
    
    # Show detected chapters
    print(f"\n{'='*60}")
    print(f"DETECTED {len(chapters)} CHAPTERS:")
    print(f"{'='*60}")
    for ch in chapters:
        pages = ch['end_page'] - ch['start_page'] + 1
        print(f"  Chapter {ch['chapter_num']}: {ch['title']}")
        print(f"    Pages {ch['start_page']}-{ch['end_page']} ({pages} pages)")
    
    print(f"\n{'='*60}")
    print("CHUNKING CHAPTERS INTO DETAILS")
    print(f"{'='*60}")
    print(f"Chunk size: {CHUNK_SIZE} words")
    print(f"Chunk overlap: {CHUNK_OVERLAP} words\n")
    
    all_detail_chunks = []
    
    # Process each chapter
    for doc_idx, chapter in enumerate(chapters):
        document_id = f"D{doc_idx:02d}"
        title = f"Chapter {chapter['chapter_num']}: {chapter['title']}"
        
        print(f"\n--- Processing {title} ---")
        print(f"  Text length: {len(chapter['text'])} characters ({chapter['word_count']} words)")
        
        # Chunk the chapter text
        chunks = chunk_text(chapter['text'], CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"  Created {len(chunks)} chunks")
        
        # Create metadata for each chunk
        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{document_id}-CHUNK-{chunk_idx:03d}"
            
            detail_item = {
                "id": chunk_id,
                "document_id": document_id,
                "file_name": pdf_files[0],
                "title": title,
                "chapter_num": chapter['chapter_num'],
                "chunk_index": chunk_idx,
                "chunk_text": chunk,
                "type": "detail"
            }
            
            all_detail_chunks.append(detail_item)
        
        print(f"  ✓ Generated {len(chunks)} detail chunks for {document_id}")
    
    print(f"\n{'='*60}")
    print(f"✓ Total detail chunks created: {len(all_detail_chunks)}")
    print(f"{'='*60}\n")
    
    save_detail_cache(all_detail_chunks)
    
    return all_detail_chunks


def store_details_in_pinecone(detail_data: List[Dict], genai_client) -> object:
    """Store detail chunks in Pinecone."""
    if not detail_data:
        print("No detail chunks to store.")
        return None
    
    print("\n" + "="*60)
    print("STORING DETAILS IN PINECONE")
    print("="*60)
    
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✓ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return None
    
    print(f"\nGenerating embeddings for {len(detail_data)} detail chunks...")
    print("⚠ This may take a while...")
    
    chunk_texts = [item['chunk_text'] for item in detail_data]
    embeddings = get_embeddings(genai_client, chunk_texts)
    
    if not embeddings or len(embeddings) != len(chunk_texts):
        print("✗ Failed to generate embeddings")
        return None
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    print("\nPreparing vectors for upload...")
    vectors_to_upsert = []
    
    for i, item in enumerate(detail_data):
        vector_data = {
            "id": item['id'],
            "values": embeddings[i],
            "metadata": {
                "document_id": item['document_id'],
                "file_name": item['file_name'],
                "title": item['title'],
                "chapter_num": item.get('chapter_num', ''),
                "chunk_index": item['chunk_index'],
                "type": "detail",
                "text_preview": item['chunk_text'][:500]
            }
        }
        vectors_to_upsert.append(vector_data)
    
    print(f"\nUploading {len(vectors_to_upsert)} vectors to Pinecone...")
    batch_size = 100
    
    try:
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            index.upsert(vectors=batch)
            print(f"  Upserted batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
            time.sleep(1)
        
        print("\nWaiting for index to update...")
        time.sleep(5)
        
        stats = index.describe_index_stats()
        print(f"\n✓ Upload complete!")
        print(f"  Total vectors in index now: {stats['total_vector_count']}")
        print(f"  (This includes both summaries and details)")
        
        return index
    except Exception as e:
        print(f"Error uploading to Pinecone: {e}")
        return None


def main():
    """Main execution flow."""
    print("="*60)
    print("RAG SYSTEM: LEVEL 2 (DETAIL) INDEXING - HEADER-BASED")
    print("="*60)
    
    if not all([VERTEX_AI_SERVICE_ACCOUNT_PATH, GOOGLE_CLOUD_PROJECT, PINECONE_API_KEY]):
        print("\n✗ Error: Missing required environment variables!")
        return
    
    print(f"\n✓ Configuration:")
    print(f"  - Project: {GOOGLE_CLOUD_PROJECT}")
    print(f"  - Pinecone Index: {PINECONE_INDEX_NAME}")
    print(f"  - Chunk Size: {CHUNK_SIZE} words")
    print(f"  - Chunk Overlap: {CHUNK_OVERLAP} words")
    
    print("\n" + "="*60)
    print("STEP 1: Initialize Google GenAI Client")
    print("="*60)
    client = initialize_genai_client()
    if not client:
        print("✗ Failed to initialize client")
        return
    
    print("\n" + "="*60)
    print("STEP 2: Create Detail Chunks from PDF")
    print("="*60)
    detail_chunks = create_detail_chunks_from_single_pdf(DATA_DIR)
    
    if not detail_chunks:
        print("✗ No detail chunks created")
        return
    
    print("\n" + "="*60)
    print("STEP 3: Store Details in Pinecone")
    print("="*60)
    result = store_details_in_pinecone(detail_chunks, client)
    
    if result:
        print("\n" + "="*60)
        print("✓ LEVEL 2 INDEXING COMPLETE!")
        print("="*60)
        print("\nYour Pinecone index now contains:")
        print("  - Level 1: Chapter summaries (type='summary')")
        print("  - Level 2: Detailed chunks (type='detail')")
        print("\nNext step: Build the query router and retrieval logic!")
    else:
        print("\n✗ Failed to complete Level 2 indexing")


if __name__ == "__main__":
    main()