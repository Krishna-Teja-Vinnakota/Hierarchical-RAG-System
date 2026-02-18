import os
import re
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
import time
import json
from typing import List, Dict, Tuple

# --- CONFIGURATION BLOCK ---
load_dotenv() 

# Get environment variables
VERTEX_AI_SERVICE_ACCOUNT_PATH = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Settings
DATA_DIR = "./data"
PINECONE_INDEX_NAME = "document-summaries"
EMBEDDING_DIMENSION = 768
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
EMBEDDING_MODEL = "text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash-exp"
LOCATION = "us-central1"
DELAY_BETWEEN_REQUESTS = 15

# Cache file
CACHE_FILE = "document_summaries_cache.json"

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
        print(f"✓ Google GenAI client initialized for project: {GOOGLE_CLOUD_PROJECT}")
        return client
    except Exception as e:
        print(f"Error initializing GenAI client: {e}")
        return None


def extract_chapter_titles_from_toc(pdf_path: str) -> List[Dict]:
    """
    Extract ONLY chapter titles and numbers from TOC (without page numbers).
    We'll find the actual page numbers by scanning headers later.
    """
    print("\n" + "="*60)
    print("STEP 1: EXTRACTING CHAPTER TITLES FROM TOC")
    print("="*60)
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"✓ PDF loaded: {total_pages} pages")
        
        # Find TOC pages
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
        
        # Extract chapter titles (without worrying about page numbers)
        # Pattern: "Chapter1 Introduction" or "Chapter 1 Introduction"
        chapter_pattern = re.compile(
            r'^\s*Chapter\s*(\d+)\s+([A-Za-z][^\d\n]+?)(?:\s+\d+)?\s*$',
            re.MULTILINE
        )
        
        chapters = []
        seen_chapters = set()
        
        for match in chapter_pattern.finditer(full_toc_text):
            chapter_num = match.group(1)
            title_raw = match.group(2)
            
            # Clean up title
            title = re.sub(r'\.{2,}', '', title_raw).strip()
            title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
            
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
        
        # Sort by chapter number
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
    Headers appear on alternating pages after page 25.
    """
    print("\n" + "="*60)
    print("STEP 2: FINDING CHAPTER START PAGES FROM HEADERS")
    print("="*60)
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        
        print(f"Scanning pages 26-{total_pages} for chapter headers (alternating pages)...")
        print("(This may take a minute...)\n")
        
        # Create search patterns for each chapter
        for chapter in chapters:
            chapter_num = chapter['chapter_num']
            title = chapter['title']
            
            # Create flexible pattern to match header
            # Example: "Chapter 1 Introduction" or "Chapter1 Introduction"
            # We'll look in the first 20% of each page (where headers usually are)
            
            pattern = re.compile(
                rf'Chapter\s*{chapter_num}\s+{re.escape(title[:20])}',  # Match first 20 chars of title
                re.IGNORECASE
            )
            
            found = False
            
            # Scan alternating pages starting from page 26 (skip TOC pages 1-25)
            # Check every 2nd page since headers appear on alternating pages
            for page_num in range(25, total_pages, 2):  # Start at page 26 (index 25), step by 2
                text = reader.pages[page_num].extract_text()
                
                # Check first 20% of page (where headers are)
                header_section = text[:int(len(text) * 0.2)]
                
                if pattern.search(header_section):
                    # Chapter actually starts 1 page BEFORE where we find the header
                    # Header is on page_num+1, so chapter starts on page_num
                    chapter['start_page'] = page_num  # This is the page before the header
                    print(f"  ✓ Chapter {chapter_num} ({title}) starts at page {page_num}")
                    found = True
                    break
            
            if not found:
                print(f"  ⚠ Chapter {chapter_num} not found in headers - will estimate")
                chapter['start_page'] = None
        
        # Handle chapters where we couldn't find start page
        # Estimate based on position between found chapters
        for i, chapter in enumerate(chapters):
            if chapter['start_page'] is None:
                # Find nearest chapters with known pages
                prev_page = 1
                next_page = total_pages
                
                # Look backwards
                for j in range(i - 1, -1, -1):
                    if chapters[j]['start_page'] is not None:
                        prev_page = chapters[j]['start_page']
                        break
                
                # Look forwards
                for j in range(i + 1, len(chapters)):
                    if chapters[j]['start_page'] is not None:
                        next_page = chapters[j]['start_page']
                        break
                
                # Estimate as midpoint
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
    """
    Extract the actual text for each chapter based on page ranges.
    """
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
        
        # Filter out very short chapters
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
    # Step 1: Get chapter titles from TOC
    chapters = extract_chapter_titles_from_toc(pdf_path)
    
    if not chapters:
        return []
    
    # Step 2: Find actual page numbers from headers
    chapters = find_chapter_pages_from_headers(pdf_path, chapters)
    
    if not chapters:
        return []
    
    # Step 3: Extract text
    chapters = extract_chapter_text(pdf_path, chapters)
    
    return chapters


def detect_chapters_by_pattern(pdf_path: str) -> List[Dict]:
    """
    Fallback: Detect chapters by searching for chapter markers in content.
    """
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
    """Last resort: Split PDF into equal sections."""
    print("\n⚠ Using last resort: Page-based split")
    print(f"Creating sections of {pages_per_section} pages each")
    
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


def load_cache() -> List[Dict]:
    """Load cached summaries if they exist."""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file: {e}")
    return []


def save_cache(summary_data: List[Dict]):
    """Save summaries to cache file."""
    try:
        cache_data = [
            {k: v for k, v in item.items() if k not in ['raw_text_chunk']}
            for item in summary_data
        ]
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"✓ Progress saved to {CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Could not save cache: {e}")


def get_embeddings(client, texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Google GenAI."""
    try:
        embeddings = []
        batch_size = 5
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"  Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}...")
            
            result = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch
            )
            
            for embedding in result.embeddings:
                embeddings.append(embedding.values)
            
            if i + batch_size < len(texts):
                time.sleep(2)
        
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []


def process_single_pdf(data_dir=DATA_DIR) -> Tuple[List[Dict], genai.Client]:
    """Process a single large PDF with header-based chapter detection."""
    client = initialize_genai_client()
    if not client:
        return [], None

    if not os.path.exists(data_dir):
        print(f"✗ Error: Data directory not found at {data_dir}")
        return [], client

    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"✗ Error: No PDF files found in '{data_dir}'")
        return [], client
    
    if len(pdf_files) > 1:
        print(f"\n⚠ Warning: Found {len(pdf_files)} PDFs. Using first one: {pdf_files[0]}")
    
    pdf_path = os.path.join(data_dir, pdf_files[0])
    print(f"\n✓ Processing: {pdf_files[0]}")
    
    # Check for cached progress
    cached_data = load_cache()
    if cached_data:
        print(f"\n✓ Found {len(cached_data)} cached chapter summaries!")
        use_cache = input("Do you want to use cached summaries? (y/n): ").lower().strip()
        if use_cache == 'y':
            print("✓ Using cached summaries.")
            return cached_data, client
    
    # Try Method 1: TOC + Header scanning (BEST)
    print("\n" + "="*60)
    print("METHOD 1: TOC Extraction + Header Scanning")
    print("="*60)
    chapters = extract_chapters_from_toc(pdf_path)
    
    # Try Method 2: Pattern matching (FALLBACK)
    if not chapters:
        print("\n" + "="*60)
        print("METHOD 2: Pattern-Based Chapter Detection")
        print("="*60)
        chapters = detect_chapters_by_pattern(pdf_path)
    
    # Try Method 3: Page split (LAST RESORT)
    if not chapters:
        print("\n" + "="*60)
        print("METHOD 3: Page-Based Splitting")
        print("="*60)
        use_fallback = input("No chapters detected. Split by pages? (y/n): ").lower().strip()
        if use_fallback == 'y':
            chapters = fallback_split_by_pages(pdf_path, pages_per_section=50)
        else:
            print("✗ Cannot proceed without chapters. Exiting.")
            return [], client
    
    if not chapters:
        print("✗ Failed to detect any chapters.")
        return [], client
    
    # Show detected chapters
    print(f"\n{'='*60}")
    print(f"DETECTED {len(chapters)} CHAPTERS:")
    print(f"{'='*60}")
    for ch in chapters:
        pages = ch['end_page'] - ch['start_page'] + 1
        print(f"  Chapter {ch['chapter_num']}: {ch['title']}")
        print(f"    Pages {ch['start_page']}-{ch['end_page']} ({pages} pages, {ch['word_count']} words)")
    
    print(f"\n{'='*60}")
    proceed = input("Does this look correct? Proceed with summarization? (y/n): ").lower().strip()
    if proceed != 'y':
        print("✗ Aborted by user.")
        return [], client
    
    # Generate summaries
    print(f"\n{'='*60}")
    print("GENERATING SUMMARIES")
    print(f"{'='*60}")
    
    summary_data = []
    
    for idx, chapter in enumerate(chapters):
        doc_id = f"D{idx:02d}"
        chapter_title = f"Chapter {chapter['chapter_num']}: {chapter['title']}"
        
        print(f"\n--- Processing {chapter_title} ---")
        
        text_for_llm = chapter['text'][:15000]
        
        prompt = f"""You are a system designed to create study guides. 
Analyze the following text from {chapter_title}. Your task is to generate a comprehensive, 
dense summary (around 800-1000 characters) that captures all key concepts, definitions, and themes. 
Do NOT use Markdown formatting (like bullet points).

---
{text_for_llm}
---
"""
        
        max_retries = 3
        retry_delay = DELAY_BETWEEN_REQUESTS
        
        for attempt in range(max_retries):
            try:
                if idx > 0 or attempt > 0:
                    print(f"  ⏳ Waiting {retry_delay} seconds...")
                    time.sleep(retry_delay)
                
                response = client.models.generate_content(
                    model=GENERATION_MODEL,
                    contents=prompt
                )
                
                if not response or not response.text:
                    print(f"  ✗ Empty response. Skipping.")
                    break
                
                summary_data.append({
                    "id": f"{doc_id}-C00",
                    "document_id": doc_id,
                    "file_name": pdf_files[0],
                    "title": chapter_title,
                    "chapter_num": chapter['chapter_num'],
                    "start_page": chapter['start_page'],
                    "end_page": chapter['end_page'],
                    "summary_text": response.text.strip(),
                    "raw_text_chunk": chapter['text']
                })
                print(f"  ✓ Success! ID: {doc_id}-C00")
                
                save_cache(summary_data)
                break
                
            except ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    if attempt < max_retries - 1:
                        retry_delay = min(retry_delay * 2, 120)
                        print(f"  ⚠ Rate limit. Retrying in {retry_delay}s...")
                    else:
                        print(f"  ✗ Max retries reached.")
                else:
                    print(f"  ✗ API Error: {str(e)[:150]}")
                    break
            except Exception as e:
                print(f"  ✗ Error: {str(e)[:150]}")
                break
    
    print(f"\n{'='*60}")
    print(f"✓ Summary generation complete!")
    print(f"  Total summaries generated: {len(summary_data)}")
    print(f"{'='*60}\n")
    
    return summary_data, client


def store_in_pinecone(summary_data: List[Dict], genai_client):
    """Store summaries in Pinecone with embeddings."""
    if not summary_data:
        print("No data to store.")
        return None

    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        print(f"✓ Pinecone client initialized")
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        return None
    
    try:
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            print(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
            )
            print("Waiting for index to be ready...")
            time.sleep(10)
        else:
            print(f"✓ Using existing index: {PINECONE_INDEX_NAME}")
        
        index = pc.Index(PINECONE_INDEX_NAME)
        print(f"✓ Connected to index")
        
    except Exception as e:
        print(f"Error with Pinecone index: {e}")
        return None
    
    print(f"\nGenerating embeddings for {len(summary_data)} chapters...")
    documents = [item['summary_text'] for item in summary_data]
    
    embeddings = get_embeddings(genai_client, documents)
    if not embeddings or len(embeddings) != len(documents):
        print("Failed to generate embeddings.")
        return None
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    vectors_to_upsert = []
    for i, item in enumerate(summary_data):
        vector_data = {
            "id": item['id'], 
            "values": embeddings[i],
            "metadata": {
                "document_id": item['document_id'],
                "file_name": item['file_name'],
                "title": item['title'],
                "chapter_num": item.get('chapter_num', ''),
                "start_page": item.get('start_page', 0),
                "end_page": item.get('end_page', 0),
                "type": "summary",
                "text_preview": item['summary_text'][:500]
            }
        }
        vectors_to_upsert.append(vector_data)
    
    try:
        print(f"Upserting {len(vectors_to_upsert)} vectors to Pinecone...")
        for i in range(0, len(vectors_to_upsert), 100):
            batch = vectors_to_upsert[i:i + 100]
            index.upsert(vectors=batch)
            print(f"  Upserted batch {i//100 + 1}/{(len(vectors_to_upsert)-1)//100 + 1}")
        
        time.sleep(5)
        stats = index.describe_index_stats()
        print(f"✓ Indexing Complete! Total vectors: {stats['total_vector_count']}")
        
        return index
    except Exception as e:
        print(f"Error upserting to Pinecone: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("RAG SYSTEM: SINGLE PDF with HEADER-BASED DETECTION")
    print("=" * 60)
    
    if not all([VERTEX_AI_SERVICE_ACCOUNT_PATH, GOOGLE_CLOUD_PROJECT, PINECONE_API_KEY]):
        print("\n✗ Error: Missing required environment variables!")
        exit(1)
    
    print(f"\n✓ Environment configured:")
    print(f"  - Project: {GOOGLE_CLOUD_PROJECT}")
    print(f"  - Data Folder: {DATA_DIR}")
    print(f"  - Pinecone Index: {PINECONE_INDEX_NAME}")
    
    print("\n" + "=" * 60)
    print("STEP 1: Detect Chapters & Generate Summaries")
    print("=" * 60)
    result = process_single_pdf(DATA_DIR)
    
    if result:
        final_data, client = result
        
        if final_data:
            print("\n" + "=" * 60)
            print("STEP 2: Storing Summaries in Pinecone")
            print("=" * 60)
            store_in_pinecone(final_data, client)
            print("\n✓ RAG system Level 1 complete!")
        else:
            print("\n✗ No summaries generated.")
    else:
        print("\n✗ Failed to process PDF.")