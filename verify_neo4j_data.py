
from neo4j_handler import Neo4jHandler
import sys

def log(msg):
    print(msg)
    with open("verify_log_utf8.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log
with open("verify_log_utf8.txt", "w", encoding="utf-8") as f:
    f.write("")

def verify_data():
    log("="*60)
    log("VERIFYING NEO4J DATA")
    log("="*60)
    
    try:
        handler = Neo4jHandler()
        
        # Check Chapters
        log("\n1. Checking Chapters in DB...")
        chapters = handler.get_all_chapters()
        log(f"Total Chapters found: {len(chapters)}")
        
        chapter_nums = []
        for ch in chapters:
            # Count concepts for this chapter
            concepts = handler.get_concepts_in_chapter(ch['num'])
            log(f"  Chapter {ch['num']}: {ch['title']} -> {len(concepts)} concepts")
            chapter_nums.append(int(ch['num']))
        
        chapter_nums.sort()
        log(f"\nChapters present (sorted): {chapter_nums}")
        
        missing = [i for i in range(1, 21) if i not in chapter_nums]
        if missing:
            log(f"⚠ Missing Chapters: {missing}")
        else:
            log("✓ All chapters 1-20 are present.")

        handler.close()
        
    except Exception as e:
        log(f"Error during verification: {e}")

if __name__ == "__main__":
    verify_data()
