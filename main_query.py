"""
Main Query Pipeline - Complete Hierarchical RAG System with Neo4j Integration
Combines Query Routing + Retrieval (Pinecone + Neo4j) + Answer Generation

🚀 NOW USING GRAPH-FIRST RETRIEVAL (Graph directs semantic search!)
"""

import os
from dotenv import load_dotenv
from google import genai
from pinecone import Pinecone

# Import our custom modules
from query_router import classify_query
from retrieval_engine import retrieve, initialize_pinecone, initialize_neo4j  # ✅ CHANGED TO GRAPH-FIRST!
from answer_generator import generate_answer, display_answer

# --- CONFIGURATION ---
load_dotenv()

VERTEX_AI_SERVICE_ACCOUNT_PATH = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-central1"


def initialize_genai_client():
    """Initialize Google GenAI client."""
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
        
        return client
    except Exception as e:
        print(f"✗ Error initializing GenAI client: {e}")
        return None


def process_query(query: str, genai_client, pinecone_index, neo4j_handler=None):
    """
    Process a single query through the complete RAG pipeline.
    
    Args:
        query: User's question
        genai_client: Initialized GenAI client
        pinecone_index: Connected Pinecone index
        neo4j_handler: Neo4j handler for graph queries (optional)
    
    Returns:
        Dictionary with answer and metadata
    """
    print("\n" + "="*60)
    print("PROCESSING QUERY")
    print("="*60)
    print(f"Question: {query}")
    
    # STEP 1: Classify query
    print("\n" + "="*60)
    print("STEP 1: CLASSIFYING QUERY")
    print("="*60)
    
    query_type = classify_query(query, genai_client)
    print(f"  ✓ Classification: {query_type}")
    
    # STEP 2: Retrieve relevant chunks (with Graph-First approach)
    print("\n" + "="*60)
    print("STEP 2: RETRIEVING RELEVANT INFORMATION")
    print("="*60)
    
    retrieval_result = retrieve(query, query_type, genai_client, pinecone_index, neo4j_handler)
    
    if retrieval_result['total_chunks'] == 0:
        print("  ✗ No relevant information found")
        return {
            "query": query,
            "query_type": query_type,
            "answer": "I couldn't find relevant information in the textbook to answer your question.",
            "sources": [],
            "num_chunks_used": 0
        }
    
    print(f"  ✓ Retrieved {retrieval_result['total_chunks']} chunks")
    print(f"    - Summaries: {len(retrieval_result['summaries'])}")
    print(f"    - Details: {len(retrieval_result['details'])}")
    
    # STEP 3: Generate answer (using graph-enriched context)
    print("\n" + "="*60)
    print("STEP 3: GENERATING ANSWER")
    print("="*60)
    
    answer_result = generate_answer(query, retrieval_result, genai_client)
    
    return answer_result


def interactive_mode():
    """
    Interactive mode - user can ask multiple questions.
    """
    print("="*60)
    print("ADAPTIVE STUDY GUIDE - HIERARCHICAL RAG SYSTEM")
    print("="*60)
    print("\nWelcome! Ask questions about Operating Systems.")
    print("The system will automatically classify your query and retrieve relevant information.")
    print("\n🚀 Using Graph-First Retrieval (Graph directs semantic search!)")
    print("\nType 'quit', 'exit', or 'q' to stop\n")
    
    # Initialize clients once
    print("Initializing system...")
    print("-" * 60)
    
    genai_client = initialize_genai_client()
    if not genai_client:
        print("✗ Failed to initialize GenAI client")
        return
    print("✓ GenAI client initialized")
    
    pinecone_index = initialize_pinecone()
    if not pinecone_index:
        print("✗ Failed to connect to Pinecone")
        return
    
    # Initialize Neo4j for graph enrichment
    neo4j_handler = initialize_neo4j()
    if not neo4j_handler:
        print("⚠ Warning: Neo4j not available, continuing without graph enrichment")
    
    print("\n" + "="*60)
    print("SYSTEM READY!")
    print("="*60)
    
    # Main query loop
    try:
        while True:
            print("\n" + "="*60)
            user_query = input("\n📚 Enter your question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q', '']:
                print("\n" + "="*60)
                print("Thank you for using the Adaptive Study Guide!")
                print("="*60)
                break
            
            # Process the query
            try:
                result = process_query(user_query, genai_client, pinecone_index, neo4j_handler)
                display_answer(result)
                
            except Exception as e:
                print(f"\n✗ Error processing query: {e}")
                import traceback
                traceback.print_exc()
                print("\nPlease try again with a different question.")
    
    finally:
        # Close Neo4j connection
        if neo4j_handler:
            neo4j_handler.close()


def single_query_mode(query: str):
    """
    Single query mode - process one question and exit.
    
    Args:
        query: User's question
    """
    print("="*60)
    print("ADAPTIVE STUDY GUIDE - SINGLE QUERY MODE")
    print("="*60)
    print("🚀 Using Graph-First Retrieval (Graph directs semantic search!)")
    
    # Initialize clients
    print("\nInitializing system...")
    
    genai_client = initialize_genai_client()
    if not genai_client:
        print("✗ Failed to initialize GenAI client")
        return
    print("✓ GenAI client initialized")
    
    pinecone_index = initialize_pinecone()
    if not pinecone_index:
        print("✗ Failed to connect to Pinecone")
        return
    
    # Initialize Neo4j for graph enrichment
    neo4j_handler = initialize_neo4j()
    if not neo4j_handler:
        print("⚠ Warning: Neo4j not available, continuing without graph enrichment")
    
    # Process the query
    try:
        result = process_query(query, genai_client, pinecone_index, neo4j_handler)
        display_answer(result)
        
    except Exception as e:
        print(f"\n✗ Error processing query: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close Neo4j connection
        if neo4j_handler:
            neo4j_handler.close()


def demo_mode():
    """
    Demo mode - run example queries to showcase the system.
    """
    print("="*60)
    print("ADAPTIVE STUDY GUIDE - DEMO MODE")
    print("="*60)
    print("🚀 Using Graph-First Retrieval (Graph directs semantic search!)")
    print("\nRunning example queries to demonstrate the system...\n")
    
    # Initialize clients
    print("Initializing system...")
    print("-" * 60)
    
    genai_client = initialize_genai_client()
    if not genai_client:
        print("✗ Failed to initialize GenAI client")
        return
    print("✓ GenAI client initialized")
    
    pinecone_index = initialize_pinecone()
    if not pinecone_index:
        print("✗ Failed to connect to Pinecone")
        return
    
    # Initialize Neo4j for graph enrichment
    neo4j_handler = initialize_neo4j()
    if not neo4j_handler:
        print("⚠ Warning: Neo4j not available, continuing without graph enrichment")
    
    # Demo queries
    demo_queries = [
        "What is a process?",  # SIMPLE
        "Compare processes and threads"  # COMPLEX
    ]
    
    try:
        for i, query in enumerate(demo_queries, 1):
            print("\n" + "="*60)
            print(f"DEMO QUERY {i}/{len(demo_queries)}")
            print("="*60)
            
            try:
                result = process_query(query, genai_client, pinecone_index, neo4j_handler)
                display_answer(result)
                
                # Pause between queries
                if i < len(demo_queries):
                    input("\n[Press Enter to continue to next demo query...]")
                    
            except Exception as e:
                print(f"\n✗ Error processing query: {e}")
                import traceback
                traceback.print_exc()
    
    finally:
        # Close Neo4j connection
        if neo4j_handler:
            neo4j_handler.close()


# --- MAIN ---

if __name__ == "__main__":
    import sys
    
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == "demo":
                # Run demo mode
                demo_mode()
            elif sys.argv[1] == "interactive" or sys.argv[1] == "i":
                # Run interactive mode
                interactive_mode()
            else:
                # Single query mode - treat args as query
                query = " ".join(sys.argv[1:])
                single_query_mode(query)
        else:
            # Default: Interactive mode
            interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("System interrupted by user")
        print("="*60)
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()