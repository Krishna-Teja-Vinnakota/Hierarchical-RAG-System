
import os
import re
import sys
from pypdf import PdfReader

# Use "Data" matching the filesystem listing, though Windows is permissive
DATA_DIR = "./Data"

def log(msg):
    print(msg)
    with open("extraction_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log file
with open("extraction_log.txt", "w", encoding="utf-8") as f:
    f.write("Starting extraction debug...\n")

def extract_chapters_from_pdf(pdf_path: str):
    log("\n" + "="*60)
    log("DEBUG: EXTRACTING CHAPTERS FROM PDF")
    log("="*60)
    
    try:
        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        log(f"✓ PDF loaded: {total_pages} pages")
        
        # Find TOC
        log(f"\nSearching for Table of Contents...")
        toc_pages = []
        
        for page_num in range(min(25, total_pages)):
            text = reader.pages[page_num].extract_text()
            # Loose check for TOC
            if re.search(r'\b(Contents|TABLE OF CONTENTS|Table of Contents)\b', text, re.IGNORECASE):
                toc_pages.append(text)
                if len(toc_pages) == 1:
                    log(f"  ✓ Found TOC starting at page {page_num + 1}")
                if len(toc_pages) >= 10:
                    break
        
        if not toc_pages:
            log("  ✗ Could not find Table of Contents")
            return []
        
        full_toc_text = "\n".join(toc_pages)
        log(f"DEBUG: Size of TOC text: {len(full_toc_text)} chars")
        
        # Extract chapter titles
        # Original Regex
        chapter_pattern = re.compile(
            r'^\s*Chapter\s*(\d+)\s+([A-Za-z][^\d\n]+?)(?:\s+\d+)?\s*$',
            re.MULTILINE
        )
        
        chapters = []
        seen_chapters = set()
        
        log("\nDEBUG: Scanning for chapters pattern matches in TOC...")
        matches_found = 0
        for match in chapter_pattern.finditer(full_toc_text):
            matches_found += 1
            chapter_num = match.group(1)
            title_raw = match.group(2)
            
            title = re.sub(r'\.{2,}', '', title_raw).strip()
            title = re.sub(r'\s+', ' ', title)
            
            if chapter_num in seen_chapters:
                continue
            
            seen_chapters.add(chapter_num)
            chapters.append({'chapter_num': chapter_num, 'title': title})
            log(f"  ✓ Found Chapter {chapter_num}: {title}")
        
        log(f"DEBUG: Global found in TOC: {matches_found} matches, {len(chapters)} unique chapters.")

        if not chapters:
            log("  ✗ No chapters found")
            return []
        
        chapters.sort(key=lambda x: int(x['chapter_num']))
        
        # Find chapter pages from headers
        log(f"\nScanning for chapter headers...")
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
                log(f"  ⚠ Warning: Could not find start page for Chapter {chapter_num}")
            else:
                 log(f"  ✓ Found start page {chapter['start_page']} for Chapter {chapter_num}")
        
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
                log(f"  ℹ Estimated start page {chapter['start_page']} for Chapter {chapter['chapter_num']}")
        
        # Calculate end pages
        for i in range(len(chapters)):
            if i < len(chapters) - 1:
                chapters[i]['end_page'] = chapters[i+1]['start_page'] - 1
            else:
                chapters[i]['end_page'] = total_pages
        
        # Extract chapter text
        log(f"\nExtracting chapter text...")
        for chapter in chapters:
            chapter_text = ""
            for page_idx in range(chapter['start_page'] - 1, min(chapter['end_page'], total_pages)):
                if page_idx < total_pages:
                    chapter_text += reader.pages[page_idx].extract_text() + "\n"
            
            chapter['text'] = chapter_text
            chapter['word_count'] = len(chapter_text.split())
            log(f"  ✓ Chapter {chapter['chapter_num']}: {chapter['word_count']} words")
        
        # Debug Filter
        log("\nDEBUG: Applying Filter (detected words >= 1000)...")
        passed_chapters = [ch for ch in chapters if ch['word_count'] >= 1000]
        
        for ch in chapters:
            status = "KEPT" if ch in passed_chapters else "DROPPED (Word count < 1000)"
            log(f"  Chapter {ch['chapter_num']}: {ch['word_count']} words -> {status}")

        log(f"\n✓ Extracted {len(passed_chapters)} chapters after filtering")
        return passed_chapters
        
    except Exception as e:
        log(f"✗ Error extracting chapters: {e}")
        import traceback
        traceback.print_exc()
        return []

def main():
    if not os.path.exists(DATA_DIR):
        log(f"✗ Data directory not found: {DATA_DIR}")
        return
    
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        log(f"✗ No PDF files found in {DATA_DIR}")
        return
    
    pdf_path = os.path.join(DATA_DIR, pdf_files[0])
    log(f"\n✓ Processing: {pdf_files[0]}")
    
    extract_chapters_from_pdf(pdf_path)

if __name__ == "__main__":
    main()
