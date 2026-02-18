"""
Entity Extraction Module
Extracts OS concepts from textbook chapters and populates Neo4j graph
Two-pass approach:
  Pass 1: Extract concepts from each chapter
  Pass 2: Extract relationships between concepts
"""

import os
import json
import re
import time
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from pypdf import PdfReader
from neo4j_handler import Neo4jHandler

# --- CONFIGURATION ---
load_dotenv()

VERTEX_AI_SERVICE_ACCOUNT_PATH = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")

GENERATION_MODEL = "gemini-2.0-flash-exp"
LOCATION = "us-central1"
DATA_DIR = "./data"

# Cache files
CONCEPTS_CACHE_FILE = "extracted_concepts_cache.json"
RELATIONSHIPS_CACHE_FILE = "extracted_relationships_cache.json"

# Settings
DELAY_BETWEEN_GEMINI_CALLS = 5  # Seconds between Gemini API calls
MAX_RETRIES = 3
RETRY_DELAY = 15


def initialize_genai_client():
    """Initialize Google GenAI client with Vertex AI credentials."""
    print("  Initializing GenAI client...")
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
        
        print("  ✓ GenAI client initialized")
        return client
    except Exception as e:
        print(f"  ✗ Error initializing GenAI client: {e}")
        return None


def load_concepts_cache() -> List[Dict]:
    """Load cached concepts if they exist."""
    if os.path.exists(CONCEPTS_CACHE_FILE):
        try:
            with open(CONCEPTS_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load concepts cache: {e}")
    return []


def save_concepts_cache(concepts: List[Dict]):
    """Save extracted concepts to cache file."""
    try:
        with open(CONCEPTS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(concepts, f, indent=2, ensure_ascii=False)
        print(f"✓ Concepts cache saved to {CONCEPTS_CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Could not save concepts cache: {e}")


def load_relationships_cache() -> List[Dict]:
    """Load cached relationships if they exist."""
    if os.path.exists(RELATIONSHIPS_CACHE_FILE):
        try:
            with open(RELATIONSHIPS_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load relationships cache: {e}")
    return []


def save_relationships_cache(relationships: List[Dict]):
    """Save extracted relationships to cache file."""
    try:
        with open(RELATIONSHIPS_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(relationships, f, indent=2, ensure_ascii=False)
        print(f"✓ Relationships cache saved to {RELATIONSHIPS_CACHE_FILE}")
    except Exception as e:
        print(f"Warning: Could not save relationships cache: {e}")


def extract_chapters_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract chapters from PDF.
    Uses the same logic as level_1_summaries.py to be consistent.
    """
    print("\n" + "="*60)
    print("STEP 1: EXTRACTING CHAPTERS FROM PDF")
    print("="*60)
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        print(f"✓ PDF loaded: {total_pages} pages")
        
        # Find TOC
        print(f"\nSearching for Table of Contents...")
        toc_pages = []
        
        for page_num in range(min(25, total_pages)):
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
            chapters.append({'chapter_num': chapter_num, 'title': title})
            print(f"  ✓ Found Chapter {chapter_num}: {title}")
        
        if not chapters:
            print("  ✗ No chapters found")
            return []
        
        chapters.sort(key=lambda x: int(x['chapter_num']))
        
        # Find chapter pages from headers
        print(f"\nScanning for chapter headers...")
        for chapter in chapters:
            chapter_num = chapter['chapter_num']
            title = chapter['title']
            
            pattern = re.compile(
                rf'Chapter\s*{chapter_num}\s+{re.escape(title[:20])}',
                re.IGNORECASE
            )
            
            found = False
            for page_num in range(25, total_pages, 2):
                text = reader.pages[page_num].extract_text()
                header_section = text[:int(len(text) * 0.2)]
                
                if pattern.search(header_section):
                    chapter['start_page'] = page_num
                    found = True
                    break
            
            if not found:
                chapter['start_page'] = None
        
        # Estimate missing pages
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
                
                chapter['start_page'] = (prev_page + next_page) // 2
        
        # Calculate end pages
        for i in range(len(chapters)):
            if i < len(chapters) - 1:
                chapters[i]['end_page'] = chapters[i+1]['start_page'] - 1
            else:
                chapters[i]['end_page'] = total_pages
        
        # Extract chapter text
        print(f"\nExtracting chapter text...")
        for chapter in chapters:
            chapter_text = ""
            for page_idx in range(chapter['start_page'] - 1, min(chapter['end_page'], total_pages)):
                if page_idx < total_pages:
                    chapter_text += reader.pages[page_idx].extract_text() + "\n"
            
            chapter['text'] = chapter_text
            chapter['word_count'] = len(chapter_text.split())
            print(f"  ✓ Chapter {chapter['chapter_num']}: {chapter['word_count']} words")
        
        # Filter short chapters
        chapters = [ch for ch in chapters if ch['word_count'] >= 1000]
        
        print(f"\n✓ Extracted {len(chapters)} chapters")
        return chapters
        
    except Exception as e:
        print(f"✗ Error extracting chapters: {e}")
        import traceback
        traceback.print_exc()
        return []


def extract_concepts_from_chapter(chapter: Dict, genai_client) -> List[Dict]:
    """
    Extract concepts from a single chapter using Gemini.
    
    Args:
        chapter: Dictionary with chapter text and metadata
        genai_client: Initialized GenAI client
    
    Returns:
        List of concepts with name, definition, and importance
    """
    chapter_num = chapter['chapter_num']
    title = chapter['title']
    chapter_text = chapter['text']
    
    # Limit text to first 30KB to avoid token limits
    text_for_llm = chapter_text[:30000]
    
    prompt = f"""You are an expert in Operating Systems textbooks.
Analyze the following chapter and extract ALL key OS concepts.

For each concept, provide:
1. Concept name (concise, 1-3 words)
2. Definition (1-2 sentences, clear and precise)
3. Importance level (high/medium/low)

Chapter: {title}

Chapter text:
{text_for_llm}

Return the results in this EXACT JSON format (and ONLY JSON, no other text):
{{
  "concepts": [
    {{
      "name": "Process",
      "definition": "A program in execution with its own memory space and resources.",
      "importance": "high"
    }},
    {{
      "name": "Thread",
      "definition": "A lightweight unit of execution within a process.",
      "importance": "high"
    }}
  ]
}}

Extract concepts now:"""

    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                retry_delay = min(RETRY_DELAY * (2 ** (attempt - 1)), 60)
                print(f"    ⏳ Retry attempt {attempt + 1}/{MAX_RETRIES}. Waiting {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"  ⏳ Extracting concepts from Chapter {chapter_num}...")
                time.sleep(DELAY_BETWEEN_GEMINI_CALLS)
            
            response = genai_client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )
            
            if not response or not response.text:
                print(f"    ✗ Empty response")
                continue
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Try to extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                print(f"    ✗ No JSON found in response")
                continue
            
            json_text = json_match.group(0)
            data = json.loads(json_text)
            
            if 'concepts' not in data:
                print(f"    ✗ No concepts key in JSON")
                continue
            
            concepts = data['concepts']
            
            # Add chapter info to each concept
            for concept in concepts:
                concept['first_mentioned_chapter'] = chapter_num
            
            print(f"    ✓ Extracted {len(concepts)} concepts")
            return concepts
            
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < MAX_RETRIES - 1:
                    continue
                else:
                    print(f"    ✗ Rate limit exceeded after retries")
                    return []
            else:
                print(f"    ✗ API Error: {str(e)[:150]}")
                return []
        except json.JSONDecodeError as e:
            print(f"    ✗ JSON parse error: {e}")
            continue
        except Exception as e:
            print(f"    ✗ Error: {e}")
            return []
    
    return []


def extract_all_concepts(chapters: List[Dict], genai_client) -> List[Dict]:
    """
    Extract concepts from all chapters.
    
    Args:
        chapters: List of chapter dictionaries
        genai_client: Initialized GenAI client
    
    Returns:
        List of all unique concepts
    """
    print("\n" + "="*60)
    print("PASS 1: EXTRACTING CONCEPTS FROM CHAPTERS")
    print("="*60)
    
    # Check cache
    cached_concepts = load_concepts_cache()
    if cached_concepts:
        print(f"\n✓ Found {len(cached_concepts)} cached concepts!")
        use_cache = input("Use cached concepts? (y/n): ").strip().lower()
        if use_cache == 'y':
            print("✓ Using cached concepts")
            return cached_concepts
    
    all_concepts = []
    concept_names = set()
    
    for idx, chapter in enumerate(chapters):
        chapter_num = chapter['chapter_num']
        
        print(f"\nChapter {chapter_num}/{len(chapters)}")
        
        concepts = extract_concepts_from_chapter(chapter, genai_client)
        
        for concept in concepts:
            concept_name = concept['name']
            
            # Avoid duplicates
            if concept_name.lower() not in concept_names:
                all_concepts.append(concept)
                concept_names.add(concept_name.lower())
        
        # Add delay between chapters
        if idx < len(chapters) - 1:
            print(f"  ⏳ Waiting {DELAY_BETWEEN_GEMINI_CALLS}s before next chapter...")
            time.sleep(DELAY_BETWEEN_GEMINI_CALLS)
    
    print(f"\n{'='*60}")
    print(f"✓ Total unique concepts extracted: {len(all_concepts)}")
    print(f"{'='*60}")
    
    # Show concepts
    print("\nExtracted concepts:")
    for i, concept in enumerate(all_concepts, 1):
        print(f"  {i}. {concept['name']} ({concept['importance']}) - {concept['definition'][:50]}...")
    
    save_concepts_cache(all_concepts)
    return all_concepts


def extract_relationships(all_concepts: List[Dict], genai_client) -> List[Dict]:
    """
    Extract relationships between concepts using Gemini.
    
    Args:
        all_concepts: List of all extracted concepts
        genai_client: Initialized GenAI client
    
    Returns:
        List of relationships (concept1, concept2, relationship_type)
    """
    print("\n" + "="*60)
    print("PASS 2: EXTRACTING RELATIONSHIPS BETWEEN CONCEPTS")
    print("="*60)
    
    # Check cache
    cached_relationships = load_relationships_cache()
    if cached_relationships:
        print(f"\n✓ Found {len(cached_relationships)} cached relationships!")
        use_cache = input("Use cached relationships? (y/n): ").strip().lower()
        if use_cache == 'y':
            print("✓ Using cached relationships")
            return cached_relationships
    
    concept_names = [c['name'] for c in all_concepts]
    concepts_str = "\n".join([f"- {c['name']}: {c['definition']}" for c in all_concepts])
    
    prompt = f"""You are an expert in Operating Systems.
Given these concepts from an OS textbook, identify relationships between them.

Concepts:
{concepts_str}

For EACH relationship that exists, provide:
1. Concept1 name
2. Concept2 name
3. Relationship type (choose ONE):
   - PREREQUISITE_FOR (must learn Concept1 before Concept2)
   - RELATES_TO (concepts are connected/related)
   - SOLVES_PROBLEM_OF (Concept1 solves the problem of Concept2)

Return ONLY JSON format (no other text):
{{
  "relationships": [
    {{
      "concept1": "Process",
      "concept2": "Thread",
      "type": "PREREQUISITE_FOR"
    }},
    {{
      "concept1": "Mutex",
      "concept2": "Race Condition",
      "type": "SOLVES_PROBLEM_OF"
    }}
  ]
}}

Identify relationships now:"""

    print("\n⏳ Sending all concepts to Gemini for relationship extraction...")
    time.sleep(DELAY_BETWEEN_GEMINI_CALLS)
    
    for attempt in range(MAX_RETRIES):
        try:
            if attempt > 0:
                retry_delay = min(RETRY_DELAY * (2 ** (attempt - 1)), 60)
                print(f"  ⏳ Retry attempt {attempt + 1}/{MAX_RETRIES}. Waiting {retry_delay}s...")
                time.sleep(retry_delay)
            
            response = genai_client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )
            
            if not response or not response.text:
                print(f"  ✗ Empty response")
                continue
            
            # Parse JSON response
            response_text = response.text.strip()
            
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if not json_match:
                print(f"  ✗ No JSON found in response")
                continue
            
            json_text = json_match.group(0)
            data = json.loads(json_text)
            
            if 'relationships' not in data:
                print(f"  ✗ No relationships key in JSON")
                continue
            
            relationships = data['relationships']
            
            # Validate relationships (both concepts must exist)
            valid_relationships = []
            for rel in relationships:
                if rel['concept1'] in concept_names and rel['concept2'] in concept_names:
                    valid_relationships.append(rel)
            
            print(f"  ✓ Extracted {len(valid_relationships)} relationships")
            return valid_relationships
            
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < MAX_RETRIES - 1:
                    continue
                else:
                    print(f"  ✗ Rate limit exceeded after retries")
                    return []
            else:
                print(f"  ✗ API Error: {str(e)[:150]}")
                return []
        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parse error: {e}")
            continue
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return []
    
    return []


def populate_neo4j_graph(chapters: List[Dict], concepts: List[Dict], 
                        relationships: List[Dict], neo4j_handler: Neo4jHandler):
    """
    Populate Neo4j graph with chapters, concepts, and relationships.
    
    Args:
        chapters: List of chapters
        concepts: List of concepts
        relationships: List of relationships
        neo4j_handler: Neo4jHandler instance
    """
    print("\n" + "="*60)
    print("POPULATING NEO4J GRAPH")
    print("="*60)
    
    # Create chapter nodes
    print("\n1. Creating Chapter nodes...")
    for chapter in chapters:
        neo4j_handler.create_chapter_node(
            chapter['chapter_num'],
            chapter['title'],
            f"Chapter {chapter['chapter_num']}: {chapter['title']}"
        )
    print(f"   ✓ Created {len(chapters)} chapter nodes")
    
    # Create concept nodes
    print("\n2. Creating Concept nodes...")
    for concept in concepts:
        neo4j_handler.create_concept_node(
            concept['name'],
            concept['definition'],
            concept['importance'],
            concept['first_mentioned_chapter']
        )
    print(f"   ✓ Created {len(concepts)} concept nodes")
    
    # Create CONTAINS relationships
    print("\n3. Creating CONTAINS relationships...")
    contains_count = 0
    for concept in concepts:
        chapter_num = concept['first_mentioned_chapter']
        if neo4j_handler.chapter_exists(chapter_num):
            neo4j_handler.create_contains_relationship(chapter_num, concept['name'])
            contains_count += 1
    print(f"   ✓ Created {contains_count} CONTAINS relationships")
    
    # Create other relationships
    print("\n4. Creating concept relationships...")
    for rel in relationships:
        rel_type = rel['type']
        concept1 = rel['concept1']
        concept2 = rel['concept2']
        
        if rel_type == "PREREQUISITE_FOR":
            neo4j_handler.create_prerequisite_relationship(concept1, concept2)
        elif rel_type == "RELATES_TO":
            neo4j_handler.create_relates_to_relationship(concept1, concept2)
        elif rel_type == "SOLVES_PROBLEM_OF":
            neo4j_handler.create_solves_problem_relationship(concept1, concept2)
    
    print(f"   ✓ Created {len(relationships)} concept relationships")
    
    # Get statistics
    print("\n5. Graph Statistics:")
    stats = neo4j_handler.get_graph_stats()
    print(f"   - Total Chapters: {stats['total_chapters']}")
    print(f"   - Total Concepts: {stats['total_concepts']}")
    print(f"   - Total Relationships: {stats['total_relationships']}")


def main():
    """Main execution flow."""
    print("="*60)
    print("ENTITY EXTRACTION FOR NEO4J GRAPH")
    print("="*60)
    
    # Initialize clients
    print("\nInitializing clients...")
    print("-" * 60)
    
    genai_client = initialize_genai_client()
    if not genai_client:
        print("✗ Failed to initialize GenAI client")
        return
    
    neo4j_handler = Neo4jHandler()
    
    try:
        # Step 1: Extract chapters from PDF
        if not os.path.exists(DATA_DIR):
            print(f"✗ Data directory not found: {DATA_DIR}")
            return
        
        pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"✗ No PDF files found in {DATA_DIR}")
            return
        
        pdf_path = os.path.join(DATA_DIR, pdf_files[0])
        print(f"\n✓ Processing: {pdf_files[0]}")
        
        chapters = extract_chapters_from_pdf(pdf_path)
        if not chapters:
            print("✗ Failed to extract chapters")
            return
        
        # Step 2: Extract concepts from chapters
        concepts = extract_all_concepts(chapters, genai_client)
        if not concepts:
            print("✗ Failed to extract concepts")
            return
        
        # Step 3: Extract relationships
        relationships = extract_relationships(concepts, genai_client)
        if not relationships:
            print("⚠ Warning: No relationships extracted (this is okay)")
            relationships = []
        
        # Step 4: Populate Neo4j
        proceed = input("\nProceed to populate Neo4j? (y/n): ").strip().lower()
        if proceed == 'y':
            populate_neo4j_graph(chapters, concepts, relationships, neo4j_handler)
            
            print("\n" + "="*60)
            print("✓ ENTITY EXTRACTION COMPLETE!")
            print("="*60)
            print("\nYour Neo4j graph is now populated with:")
            print(f"  - {len(chapters)} chapters")
            print(f"  - {len(concepts)} concepts")
            print(f"  - {len(relationships)} relationships")
            print("\nYou can now use this graph in Phase 4 for intelligent retrieval!")
        else:
            print("✗ Aborted by user")
    
    finally:
        neo4j_handler.close()


if __name__ == "__main__":
    main()