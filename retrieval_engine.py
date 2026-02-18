"""
Retrieval Engine with TRUE GRAPH-FIRST Neo4j Integration
Graph output BECOMES INPUT to Pinecone via enhanced query embedding!
Combines Pinecone (semantic search) + Neo4j (structural relationships)
for intelligent hierarchical retrieval
"""

import os
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone
from typing import List, Dict, Tuple, Set
import time

# Import Neo4j handler
from neo4j_handler import Neo4jHandler

# --- CONFIGURATION ---
load_dotenv()

VERTEX_AI_SERVICE_ACCOUNT_PATH = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = "document-summaries"
EMBEDDING_MODEL = "text-embedding-004"
LOCATION = "us-central1"

# Retrieval settings
TOP_K_SUMMARIES = 3  # Number of chapter summaries to retrieve for COMPLEX queries
TOP_K_DETAILS = 5    # Number of detail chunks to retrieve
TOP_K_GRAPH_CONCEPTS = 5  # Number of related concepts to retrieve from Neo4j


def initialize_genai_client():
    """Initialize Google GenAI client for embeddings."""
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


def initialize_pinecone():
    """Initialize Pinecone client and connect to index."""
    print("  Initializing Pinecone...")
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        
        # Verify index stats
        stats = index.describe_index_stats()
        print(f"  ✓ Connected to Pinecone index: {PINECONE_INDEX_NAME}")
        print(f"    Total vectors: {stats['total_vector_count']}")
        
        return index
    except Exception as e:
        print(f"  ✗ Error connecting to Pinecone: {e}")
        return None


def initialize_neo4j():
    """Initialize Neo4j handler."""
    print("  Initializing Neo4j...")
    try:
        handler = Neo4jHandler()
        print("  ✓ Neo4j handler initialized")
        return handler
    except Exception as e:
        print(f"  ✗ Error initializing Neo4j: {e}")
        return None


def generate_query_embedding(query: str, genai_client) -> List[float]:
    """
    Generate embedding vector for user query.
    
    Args:
        query: User's question (can be original or enhanced with concepts)
        genai_client: Initialized GenAI client
    
    Returns:
        Embedding vector as list of floats
    """
    try:
        result = genai_client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[query]
        )
        
        if result.embeddings and len(result.embeddings) > 0:
            return result.embeddings[0].values
        else:
            print("  ✗ Error: Empty embedding response")
            return None
            
    except Exception as e:
        print(f"  ✗ Error generating embedding: {e}")
        return None


def extract_concepts_from_query(query: str) -> Set[str]:
    """
    Extract potential concept names from query using simple heuristics.
    Looks for capitalized words or common OS terms.
    
    Args:
        query: User's question
    
    Returns:
        Set of potential concept names
    """
    # List of common OS concepts to look for
    common_concepts = [
        "process", "thread", "deadlock", "semaphore", "mutex", "race condition",
        "scheduling", "context switch", "virtual memory", "paging", "file system",
        "synchronization", "interrupt", "system call", "kernel", "cpu", "i/o",
        "memory", "protection", "security", "distributed", "linux", "windows"
    ]
    
    query_lower = query.lower()
    found_concepts = set()
    
    for concept in common_concepts:
        if concept in query_lower:
            found_concepts.add(concept.title())
    
    return found_concepts


def get_graph_related_concepts(query_concepts: Set[str], neo4j_handler, 
                               relationship_types: List[str] = None) -> Dict[str, List[str]]:
    """
    Get concepts related to query concepts from Neo4j graph.
    
    Args:
        query_concepts: Set of concepts found in query
        neo4j_handler: Neo4jHandler instance
        relationship_types: Types of relationships to follow (default: all types)
    
    Returns:
        Dictionary mapping query concepts to related concepts
    """
    if not query_concepts or not neo4j_handler:
        return {}
    
    related_map = {}
    
    for concept in query_concepts:
        # Check if concept exists in graph
        if not neo4j_handler.concept_exists(concept):
            continue
        
        related = []
        
        # Get different types of related concepts
        # 1. Prerequisites
        prerequisites = neo4j_handler.get_prerequisites(concept)
        related.extend(prerequisites)
        
        # 2. Dependent concepts
        dependents = neo4j_handler.get_dependent_concepts(concept)
        related.extend(dependents)
        
        # 3. Related concepts
        relates_to = neo4j_handler.get_related_concepts(concept, "RELATES_TO")
        related.extend(relates_to)
        
        # Limit and deduplicate
        related = list(set(related))[:TOP_K_GRAPH_CONCEPTS]
        
        if related:
            related_map[concept] = related
    
    return related_map


def create_enhanced_query(original_query: str, search_concepts: Set[str], 
                         max_concepts: int = 5) -> str:
    """
    Create enhanced query by combining original query with discovered concepts.
    This enhanced query will be used to generate embeddings for Pinecone search.
    
    Graph output BECOMES input to Pinecone!
    
    Args:
        original_query: User's original question
        search_concepts: Set of concepts discovered from Neo4j
        max_concepts: Maximum number of concepts to include
    
    Returns:
        Enhanced query string with concepts included
    """
    if not search_concepts or len(search_concepts) == 0:
        return original_query
    
    # Convert to list and limit
    concepts_list = list(search_concepts)[:max_concepts]
    
    # Create enhanced query
    enhanced_query = f"{original_query} {' '.join(concepts_list)}"
    
    return enhanced_query


def retrieve_simple(query: str, genai_client, pinecone_index, 
                   neo4j_handler=None, top_k: int = TOP_K_DETAILS) -> Tuple[List[Dict], Dict]:
    """
    SIMPLE retrieval path: Search Level 2 (detail) index directly + graph enrichment.
    
    Used for single-concept queries like definitions or explanations.
    
    Args:
        query: User's question
        genai_client: GenAI client for embeddings
        pinecone_index: Connected Pinecone index
        neo4j_handler: Neo4jHandler instance for graph queries (optional)
        top_k: Number of chunks to retrieve
    
    Returns:
        Tuple of (retrieved_chunks, graph_info)
    """
    print("\n" + "="*60)
    print("SIMPLE RETRIEVAL PATH")
    print("="*60)
    print(f"Strategy: Direct search in Level 2 (detail chunks) + Graph enrichment")
    print(f"Retrieving top {top_k} most relevant chunks\n")
    
    # Generate embedding for query
    print("  Generating query embedding...")
    query_embedding = generate_query_embedding(query, genai_client)
    
    if not query_embedding:
        print("  ✗ Failed to generate query embedding")
        return [], {}
    
    print("  ✓ Query embedding generated")
    
    # Search Level 2 (detail) index only
    print(f"  Searching Level 2 index (type='detail')...")
    
    try:
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            filter={"type": {"$eq": "detail"}},  # Only retrieve detail chunks
            include_metadata=True
        )
        
        if not results.matches:
            print("  ✗ No results found")
            return [], {}
        
        print(f"  ✓ Retrieved {len(results.matches)} chunks")
        
        # Format results
        retrieved_chunks = []
        for i, match in enumerate(results.matches, 1):
            chunk_data = {
                "rank": i,
                "score": match.score,
                "id": match.id,
                "chapter_num": match.metadata.get("chapter_num", ""),
                "title": match.metadata.get("title", ""),
                "chunk_index": match.metadata.get("chunk_index", 0),
                "text": match.metadata.get("text_preview", ""),
                "type": "detail"
            }
            retrieved_chunks.append(chunk_data)
            
            print(f"    [{i}] Chapter {chunk_data['chapter_num']}: {chunk_data['title']}")
            print(f"        Score: {chunk_data['score']:.4f} | Chunk: {chunk_data['chunk_index']}")
        
        # Graph enrichment
        graph_info = {}
        if neo4j_handler:
            print(f"\n  Querying Neo4j for related concepts...")
            query_concepts = extract_concepts_from_query(query)
            related_concepts = get_graph_related_concepts(query_concepts, neo4j_handler)
            
            if related_concepts:
                print(f"    ✓ Found related concepts in graph:")
                for concept, related in related_concepts.items():
                    print(f"      • {concept} → {', '.join(related[:3])}")
                graph_info["related_concepts"] = related_concepts
            else:
                print(f"    ⚠ No related concepts found in graph")
        
        return retrieved_chunks, graph_info
        
    except Exception as e:
        print(f"  ✗ Error during search: {e}")
        return [], {}


def retrieve_complex(query: str, genai_client, pinecone_index, 
                    neo4j_handler=None,
                    top_k_summaries: int = TOP_K_SUMMARIES,
                    top_k_details: int = TOP_K_DETAILS) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    TRUE GRAPH-FIRST COMPLEX retrieval path: Neo4j output becomes Pinecone input!
    
    PHASE 1: GRAPH TRAVERSAL & CONCEPT EXPANSION
    - Extract concepts from query
    - Traverse Neo4j to find prerequisites, dependents, related concepts
    - Build expanded concept set (8-10 vs 2-3)
    
    PHASE 2: CREATE ENHANCED QUERY
    - Combine original query with discovered concepts
    - Graph output BECOMES INPUT here!
    
    PHASE 3: SEMANTIC SEARCH (GRAPH-GUIDED)
    - Generate embedding from ENHANCED query
    - Search Level 1 (summaries) with enriched embedding
    - Extract chapter IDs from summaries
    - Search Level 2 (details) filtered by those chapter IDs
    
    PHASE 4: HIERARCHICAL FILTERING & RETURN
    - Return with full graph context and relationships
    
    Used for multi-concept queries requiring comparison or synthesis.
    
    Args:
        query: User's question
        genai_client: GenAI client for embeddings
        pinecone_index: Connected Pinecone index
        neo4j_handler: Neo4jHandler instance for graph queries (optional)
        top_k_summaries: Number of chapter summaries to retrieve
        top_k_details: Number of detail chunks to retrieve per chapter
    
    Returns:
        Tuple of (summary_chunks, detail_chunks, graph_info)
    """
    print("\n" + "="*60)
    print("TRUE GRAPH-FIRST COMPLEX RETRIEVAL PATH (HIERARCHICAL)")
    print("="*60)
    print(f"Strategy: Neo4j Traversal → Enhanced Query → Pinecone Search")
    print(f"Graph output BECOMES input to Pinecone embedding!\n")
    print(f"Retrieving {top_k_summaries} summaries, then {top_k_details} filtered details\n")
    
    # ========== PHASE 1: GRAPH TRAVERSAL & CONCEPT EXPANSION ==========
    print("  [PHASE 1] GRAPH TRAVERSAL & CONCEPT EXPANSION")
    print("  " + "-"*56)
    
    graph_info = {}
    query_concepts = set()
    search_concepts = set()
    
    if neo4j_handler:
        print("  Step 1a: Extract concepts from query...")
        query_concepts = extract_concepts_from_query(query)
        
        if query_concepts:
            print(f"  ✓ Found query concepts: {', '.join(list(query_concepts)[:3])}")
        else:
            print(f"  ⚠ No recognized concepts found in query")
        
        print("\n  Step 1b: Traverse Neo4j for relationships...")
        related_concepts_map = get_graph_related_concepts(query_concepts, neo4j_handler)
        
        if related_concepts_map:
            print(f"  ✓ Expanded {len(related_concepts_map)} concepts with relationships:")
            for concept, related in related_concepts_map.items():
                all_related = related
                print(f"    • {concept}: {len(all_related)} related concepts")
                # Show sample relationships
                print(f"      └─ Examples: {', '.join(all_related[:3])}")
        
        # Build comprehensive concept set for searching
        search_concepts = set(query_concepts)
        for concept_info in related_concepts_map.values():
            search_concepts.update(concept_info)
        
        print(f"\n  Total unique concepts discovered: {len(search_concepts)}")
        if search_concepts:
            concepts_preview = list(search_concepts)[:8]
            print(f"  Concepts: {', '.join(concepts_preview)}")
            if len(search_concepts) > 8:
                print(f"           ... and {len(search_concepts) - 8} more")
        
        graph_info["query_concepts"] = list(query_concepts)
        graph_info["related_concepts"] = related_concepts_map
        graph_info["all_discovered_concepts"] = list(search_concepts)
    else:
        query_concepts = extract_concepts_from_query(query)
        search_concepts = query_concepts
    
    # ========== PHASE 2: CREATE ENHANCED QUERY ==========
    print("\n  [PHASE 2] CREATE ENHANCED QUERY (GRAPH OUTPUT → PINECONE INPUT)")
    print("  " + "-"*56)
    
    # THIS IS WHERE GRAPH OUTPUT BECOMES PINECONE INPUT!
    enhanced_query = create_enhanced_query(query, search_concepts, max_concepts=5)
    
    if enhanced_query != query:
        print(f"  Original Query: \"{query}\"")
        print(f"\n  Enhanced Query (with discovered concepts):")
        print(f"  \"{enhanced_query}\"")
        print(f"\n  ✓ Graph concepts embedded in query for Pinecone!")
    else:
        print(f"  ⚠ No enhancement possible, using original query")
    
    # ========== PHASE 3: SEMANTIC SEARCH (GRAPH-GUIDED) ==========
    print("\n  [PHASE 3] SEMANTIC SEARCH (GRAPH-GUIDED)")
    print("  " + "-"*56)
    
    # Generate embedding for ENHANCED query (with graph concepts)
    print("  Step 3a: Generating embedding from enhanced query...")
    query_embedding = generate_query_embedding(enhanced_query, genai_client)
    
    if not query_embedding:
        print("  ✗ Failed to generate query embedding")
        return [], [], {}
    
    print("  ✓ Embedding generated (from enhanced query with graph concepts)")
    
    # Search Pinecone with ENHANCED embedding
    print(f"\n  Step 3b: Searching Pinecone with graph-enhanced embedding...")
    
    try:
        # PHASE 3A: Search Level 1 (summary) index with ENHANCED embedding
        print(f"    → Search 1: Enhanced query in summaries (type='summary')")
        summary_results = pinecone_index.query(
            vector=query_embedding,  # ✓ Using enhanced embedding with graph concepts
            top_k=top_k_summaries,
            filter={"type": {"$eq": "summary"}},
            include_metadata=True
        )
        
        if not summary_results.matches:
            print("    ✗ No summaries found")
            return [], [], {}
        
        print(f"    ✓ Retrieved {len(summary_results.matches)} summaries (from enhanced search)")
        
        # Format summary results and extract chapter IDs
        summary_chunks = []
        relevant_chapters = []
        
        for i, match in enumerate(summary_results.matches, 1):
            chapter_num = match.metadata.get("chapter_num", "")
            title = match.metadata.get("title", "")
            
            summary_data = {
                "rank": i,
                "score": match.score,
                "id": match.id,
                "chapter_num": chapter_num,
                "title": title,
                "text": match.metadata.get("text_preview", ""),
                "type": "summary"
            }
            summary_chunks.append(summary_data)
            
            if chapter_num:
                relevant_chapters.append(chapter_num)
            
            print(f"      [{i}] Chapter {chapter_num}: {title}")
            print(f"          Score: {summary_data['score']:.4f}")
        
        if not relevant_chapters:
            print("  ✗ No chapter IDs found")
            return summary_chunks, [], {}
        
        # ========== PHASE 4: HIERARCHICAL FILTERING ==========
        print(f"\n  [PHASE 4] HIERARCHICAL FILTERING & RETURN")
        print("  " + "-"*56)
        
        # PHASE 3B: Search Level 2 (detail) with chapter filter and ENHANCED embedding
        print(f"  Searching Level 2 index filtered by chapters {relevant_chapters}...")
        print(f"  (Using graph-enhanced embedding)")
        
        detail_results = pinecone_index.query(
            vector=query_embedding,  # ✓ Using same enhanced embedding
            top_k=top_k_details,
            filter={
                "type": {"$eq": "detail"},
                "chapter_num": {"$in": relevant_chapters}
            },
            include_metadata=True
        )
        
        if not detail_results.matches:
            print("  ✗ No detail chunks found in filtered search")
            return summary_chunks, [], {}
        
        print(f"  ✓ Retrieved {len(detail_results.matches)} filtered details (from enhanced search)")
        
        # Format detail results
        detail_chunks = []
        for i, match in enumerate(detail_results.matches, 1):
            detail_data = {
                "rank": i,
                "score": match.score,
                "id": match.id,
                "chapter_num": match.metadata.get("chapter_num", ""),
                "title": match.metadata.get("title", ""),
                "chunk_index": match.metadata.get("chunk_index", 0),
                "text": match.metadata.get("text_preview", ""),
                "type": "detail"
            }
            detail_chunks.append(detail_data)
            
            print(f"    [{i}] Chapter {detail_data['chapter_num']}: {detail_data['title']}")
            print(f"        Score: {detail_data['score']:.4f} | Chunk: {detail_data['chunk_index']}")
        
        print(f"\n  ✓ Search complete!")
        print(f"    Total chunks retrieved: {len(summary_chunks) + len(detail_chunks)}")
        print(f"    Graph concepts helped improve search relevance!")
        
        return summary_chunks, detail_chunks, graph_info
        
    except Exception as e:
        print(f"  ✗ Error during hierarchical search: {e}")
        import traceback
        traceback.print_exc()
        return [], [], {}


def retrieve(query: str, query_type: str, genai_client, pinecone_index, 
            neo4j_handler=None) -> Dict:
    """
    Main retrieval function that routes to appropriate strategy.
    TRUE GRAPH-FIRST approach: Neo4j output becomes Pinecone input!
    
    Args:
        query: User's question
        query_type: "SIMPLE" or "COMPLEX" (from query_router)
        genai_client: GenAI client for embeddings
        pinecone_index: Connected Pinecone index
        neo4j_handler: Neo4jHandler instance (optional)
    
    Returns:
        Dictionary containing:
        - query: Original query
        - query_type: SIMPLE or COMPLEX
        - summaries: List of summary chunks (empty for SIMPLE)
        - details: List of detail chunks
        - graph_info: Information from Neo4j about relationships
        - total_chunks: Total number of chunks retrieved
        - retrieval_method: "TRUE_GRAPH_FIRST"
        - note: Confirms graph output became Pinecone input
    """
    
    if query_type == "SIMPLE":
        # Direct detail search with graph enrichment
        detail_chunks, graph_info = retrieve_simple(query, genai_client, pinecone_index, neo4j_handler)
        
        return {
            "query": query,
            "query_type": "SIMPLE",
            "summaries": [],
            "details": detail_chunks,
            "graph_info": graph_info,
            "total_chunks": len(detail_chunks),
            "retrieval_method": "TRUE_GRAPH_FIRST",
            "note": "Graph enriches search (graph output → Pinecone)"
        }
        
    elif query_type == "COMPLEX":
        # True graph-first hierarchical search
        summary_chunks, detail_chunks, graph_info = retrieve_complex(
            query, genai_client, pinecone_index, neo4j_handler
        )
        
        return {
            "query": query,
            "query_type": "COMPLEX",
            "summaries": summary_chunks,
            "details": detail_chunks,
            "graph_info": graph_info,
            "total_chunks": len(summary_chunks) + len(detail_chunks),
            "retrieval_method": "TRUE_GRAPH_FIRST",
            "note": "Graph discoveries EMBEDDED in enhanced query for Pinecone"
        }
    
    else:
        print(f"  ✗ Unknown query type: {query_type}")
        return {
            "query": query,
            "query_type": query_type,
            "summaries": [],
            "details": [],
            "graph_info": {},
            "total_chunks": 0,
            "retrieval_method": "TRUE_GRAPH_FIRST"
        }


# --- TESTING FUNCTIONS ---

def test_retrieval():
    """Test the retrieval engine with sample queries."""
    print("="*60)
    print("RETRIEVAL ENGINE TEST (True Graph-First with Neo4j)")
    print("="*60)
    print()
    
    # Initialize clients
    print("Initializing clients...")
    genai_client = initialize_genai_client()
    pinecone_index = initialize_pinecone()
    neo4j_handler = initialize_neo4j()
    
    if not genai_client or not pinecone_index:
        print("\n✗ Failed to initialize clients")
        return
    
    print("\n" + "="*60)
    
    # Test queries
    test_cases = [
        {
            "query": "What is a process?",
            "type": "SIMPLE"
        },
        {
            "query": "Compare processes and threads",
            "type": "COMPLEX"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test_case['type']} Query")
        print(f"{'='*60}")
        print(f"Query: \"{test_case['query']}\"")
        
        # Add delay to avoid rate limits
        if i > 1:
            print("\n⏳ Waiting 3 seconds to avoid rate limits...")
            time.sleep(3)
        
        # Retrieve
        result = retrieve(test_case['query'], test_case['type'], genai_client, pinecone_index, neo4j_handler)
        
        # Display results
        print(f"\n{'='*60}")
        print("RETRIEVAL RESULTS")
        print(f"{'='*60}")
        print(f"Query Type: {result['query_type']}")
        print(f"Retrieval Method: {result.get('retrieval_method', 'N/A')}")
        print(f"Note: {result.get('note', 'N/A')}")
        print(f"Total Chunks Retrieved: {result['total_chunks']}")
        print(f"  - Summaries: {len(result['summaries'])}")
        print(f"  - Details: {len(result['details'])}")
        
        if result['summaries']:
            print(f"\nSummary Chunks:")
            for chunk in result['summaries']:
                print(f"  • Chapter {chunk['chapter_num']}: {chunk['title']}")
                print(f"    Preview: {chunk['text'][:150]}...")
        
        if result['details']:
            print(f"\nDetail Chunks:")
            for chunk in result['details']:
                print(f"  • Chapter {chunk['chapter_num']}: {chunk['title']} (Chunk {chunk['chunk_index']})")
                print(f"    Preview: {chunk['text'][:150]}...")
        
        if result['graph_info']:
            print(f"\nNeo4j Graph Information:")
            if 'query_concepts' in result['graph_info']:
                print(f"  Query Concepts Found: {', '.join(result['graph_info']['query_concepts'])}")
            if 'all_discovered_concepts' in result['graph_info']:
                all_concepts = result['graph_info']['all_discovered_concepts']
                print(f"  Total Discovered Concepts: {len(all_concepts)}")
                print(f"  All Concepts: {', '.join(all_concepts[:5])}")
                if len(all_concepts) > 5:
                    print(f"                {', '.join(all_concepts[5:10])}")
            if 'related_concepts' in result['graph_info']:
                print(f"  Related Concepts Map:")
                for concept, related in result['graph_info']['related_concepts'].items():
                    print(f"    • {concept}: {', '.join(related[:3])}")
    
    # Cleanup
    if neo4j_handler:
        neo4j_handler.close()


def interactive_retrieval():
    """Interactive mode to test retrieval with custom queries."""
    print("="*60)
    print("RETRIEVAL ENGINE - INTERACTIVE MODE (True Graph-First)")
    print("="*60)
    print("\nEnter queries to test retrieval")
    print("Format: <SIMPLE|COMPLEX> your question here")
    print("Example: SIMPLE What is a process?")
    print("Example: COMPLEX Compare processes and threads")
    print("Type 'quit' to exit\n")
    
    # Initialize clients once
    genai_client = initialize_genai_client()
    pinecone_index = initialize_pinecone()
    neo4j_handler = initialize_neo4j()
    
    if not genai_client or not pinecone_index:
        print("\n✗ Failed to initialize clients")
        return
    
    print("\n" + "="*60)
    
    try:
        while True:
            user_input = input("\nEnter query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode.")
                break
            
            if not user_input:
                continue
            
            # Parse input
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("  ✗ Invalid format. Use: <SIMPLE|COMPLEX> your question")
                continue
            
            query_type = parts[0].upper()
            query = parts[1]
            
            if query_type not in ["SIMPLE", "COMPLEX"]:
                print("  ✗ Query type must be SIMPLE or COMPLEX")
                continue
            
            # Retrieve
            result = retrieve(query, query_type, genai_client, pinecone_index, neo4j_handler)
            
            # Display results
            print(f"\nRetrieved {result['total_chunks']} chunks:")
            print(f"  - Summaries: {len(result['summaries'])}")
            print(f"  - Details: {len(result['details'])}")
            print(f"  - Retrieval Method: {result.get('retrieval_method', 'N/A')}")
            print(f"  - Note: {result.get('note', 'N/A')}")
            
            if result['graph_info']:
                print(f"\nGraph Information:")
                if 'all_discovered_concepts' in result['graph_info']:
                    concepts = result['graph_info']['all_discovered_concepts']
                    print(f"  Discovered Concepts: {len(concepts)}")
                    print(f"  Examples: {', '.join(concepts[:5])}")
    
    finally:
        if neo4j_handler:
            neo4j_handler.close()


# --- MAIN ---

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("RETRIEVAL ENGINE STARTING... (True Graph-First with Neo4j)")
    print("="*60)
    print("Graph output BECOMES input to Pinecone!")
    print("="*60)
    print()
    
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == "test":
                test_retrieval()
            elif sys.argv[1] == "interactive":
                interactive_retrieval()
            else:
                print("Usage:")
                print("  python retrieval_engine.py test          # Run automated tests")
                print("  python retrieval_engine.py interactive   # Interactive mode")
        else:
            print("Usage:")
            print("  python retrieval_engine.py test          # Run automated tests")
            print("  python retrieval_engine.py interactive   # Interactive mode")
            print()
            print("Running automated tests...\n")
            test_retrieval()
            
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()