import os
from dotenv import load_dotenv
from google import genai
from typing import Literal
import time

# --- CONFIGURATION ---
load_dotenv()

VERTEX_AI_SERVICE_ACCOUNT_PATH = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION = "us-central1"
GENERATION_MODEL = "gemini-2.0-flash-exp"


def initialize_genai_client():
    """Initialize Google GenAI client with Vertex AI credentials."""
    print("  Initializing GenAI client...")
    try:
        from google.oauth2 import service_account
        
        if not VERTEX_AI_SERVICE_ACCOUNT_PATH:
            print(f"  ✗ Error: VERTEX_AI_SERVICE_ACCOUNT_PATH not set in .env")
            return None
            
        if not GOOGLE_CLOUD_PROJECT:
            print(f"  ✗ Error: GOOGLE_CLOUD_PROJECT not set in .env")
            return None
        
        print(f"  Using project: {GOOGLE_CLOUD_PROJECT}")
        print(f"  Using service account: {VERTEX_AI_SERVICE_ACCOUNT_PATH}")
        
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
        
        print("  ✓ Client initialized successfully")
        return client
    except Exception as e:
        print(f"  ✗ Error initializing GenAI client: {e}")
        import traceback
        traceback.print_exc()
        return None


def classify_query(query: str, client=None) -> Literal["SIMPLE", "COMPLEX"]:
    """
    Classify user query as SIMPLE or COMPLEX using Gemini.
    
    SIMPLE queries:
    - Ask for definitions, facts, or explanations of a single concept
    - Focus on one specific topic or term
    - Examples: "What is a process?", "Define semaphore", "Explain paging"
    
    COMPLEX queries:
    - Require comparison across multiple topics or chapters
    - Ask for synthesis, historical context, or relationships
    - Need high-level understanding before details
    - Examples: "Compare processes vs threads", "How did OS memory management evolve?"
    
    Args:
        query: User's question
        client: Initialized GenAI client (optional, will create if not provided)
    
    Returns:
        "SIMPLE" or "COMPLEX"
    """
    
    # Initialize client if not provided
    if client is None:
        client = initialize_genai_client()
        if not client:
            print("⚠ Warning: Could not initialize client. Defaulting to SIMPLE.")
            return "SIMPLE"
    
    # Classification prompt
    classification_prompt = f"""You are a query classifier for a Retrieval-Augmented Generation (RAG) system that answers questions about Operating Systems from a textbook.

Your task is to classify the user's query into ONE of these TWO categories:

**SIMPLE**: 
- Queries asking for definitions, facts, or explanations of a single concept
- Questions about one specific topic or term
- Direct "what is" or "explain" questions focused on a single subject
- Examples:
  * "What is a process?"
  * "Define semaphore"
  * "Explain the page replacement algorithm"
  * "What is deadlock?"
  * "How does virtual memory work?"

**COMPLEX**:
- Queries requiring comparison between multiple topics or chapters
- Questions asking for synthesis, historical evolution, or relationships between concepts
- Comparative questions with words like "compare", "difference", "vs", "versus"
- Questions requiring context from multiple chapters
- Examples:
  * "Compare processes and threads"
  * "What are the differences between paging and segmentation?"
  * "How did memory management evolve in operating systems?"
  * "Compare Linux and Windows scheduling approaches"
  * "Explain the relationship between deadlocks and synchronization"

**CRITICAL RULES**:
1. Your response must ONLY be the word "SIMPLE" or "COMPLEX" - nothing else
2. Do NOT explain your reasoning
3. Do NOT add punctuation or extra words
4. If unsure, default to "SIMPLE"

User Query: "{query}"

Classification:"""

    try:
        # Add delay to avoid rate limits
        print("  ⏳ Adding delay to avoid rate limits (3s)...")
        time.sleep(3)
        
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=classification_prompt
        )
        
        if not response or not response.text:
            print("⚠ Warning: Empty response from LLM. Defaulting to SIMPLE.")
            return "SIMPLE"
        
        # Extract and clean the response
        classification = response.text.strip().upper()
        
        # Validate response
        if "SIMPLE" in classification:
            return "SIMPLE"
        elif "COMPLEX" in classification:
            return "COMPLEX"
        else:
            # If response is unclear, default to SIMPLE
            print(f"⚠ Warning: Unexpected classification '{classification}'. Defaulting to SIMPLE.")
            return "SIMPLE"
            
    except Exception as e:
        print(f"⚠ Error during classification: {e}")
        print("Defaulting to SIMPLE.")
        return "SIMPLE"


def classify_query_with_explanation(query: str, client=None) -> dict:
    """
    Enhanced version that returns both classification and explanation.
    Useful for debugging and understanding the router's decisions.
    
    Args:
        query: User's question
        client: Initialized GenAI client (optional)
    
    Returns:
        Dictionary with 'classification' and 'explanation' keys
    """
    
    if client is None:
        client = initialize_genai_client()
        if not client:
            return {
                "classification": "SIMPLE",
                "explanation": "Could not initialize client. Defaulting to SIMPLE."
            }
    
    explanation_prompt = f"""You are a query classifier for a RAG system about Operating Systems.

Classify this query as SIMPLE or COMPLEX, then explain your reasoning in ONE sentence.

**SIMPLE**: Single-concept queries (definitions, facts, single-topic explanations)
**COMPLEX**: Multi-concept queries (comparisons, synthesis, cross-chapter relationships)

User Query: "{query}"

Respond in this EXACT format:
Classification: [SIMPLE or COMPLEX]
Reason: [One sentence explanation]"""

    try:
        # Add delay
        time.sleep(3)
        
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=explanation_prompt
        )
        
        if not response or not response.text:
            return {
                "classification": "SIMPLE",
                "explanation": "Empty response from LLM."
            }
        
        text = response.text.strip()
        
        # Parse response
        classification = "SIMPLE"
        explanation = "Could not parse response."
        
        lines = text.split('\n')
        for line in lines:
            if 'Classification:' in line:
                if 'COMPLEX' in line.upper():
                    classification = "COMPLEX"
                else:
                    classification = "SIMPLE"
            elif 'Reason:' in line:
                explanation = line.replace('Reason:', '').strip()
        
        return {
            "classification": classification,
            "explanation": explanation
        }
        
    except Exception as e:
        return {
            "classification": "SIMPLE",
            "explanation": f"Error during classification: {str(e)}"
        }


# --- TESTING FUNCTIONS ---

def test_router():
    """Test the query router with example queries."""
    print("="*60)
    print("QUERY ROUTER TEST")
    print("="*60)
    
    # Initialize client once for all tests
    client = initialize_genai_client()
    if not client:
        print("✗ Failed to initialize client")
        return
    
    print("✓ Client initialized\n")
    
    # Test queries
    test_queries = [
        # SIMPLE queries
        "What is a process?",
        "Define semaphore",
        "Explain virtual memory",
        "What is deadlock?",
        "How does the CPU scheduler work?",
        
        # COMPLEX queries
        "Compare processes and threads",
        "What are the differences between paging and segmentation?",
        "How did memory management evolve in operating systems?",
        "Compare Linux and Windows file systems",
        "Explain the relationship between deadlocks and synchronization"
    ]
    
    print("Testing classification (with explanations):\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        
        result = classify_query_with_explanation(query, client)
        
        print(f"  → Classification: {result['classification']}")
        print(f"  → Reason: {result['explanation']}")
        print()


def interactive_mode():
    """Interactive mode to test custom queries."""
    print("="*60)
    print("QUERY ROUTER - INTERACTIVE MODE")
    print("="*60)
    print("\nType your questions to classify them as SIMPLE or COMPLEX")
    print("Type 'quit' or 'exit' to stop\n")
    
    client = initialize_genai_client()
    if not client:
        print("✗ Failed to initialize client")
        return
    
    while True:
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("\nExiting interactive mode.")
            break
        
        if not query:
            continue
        
        print()
        result = classify_query_with_explanation(query, client)
        
        print(f"  Classification: {result['classification']}")
        print(f"  Reason: {result['explanation']}")
        print()


# --- MAIN ---

if __name__ == "__main__":
    import sys
    
    print("="*60)
    print("QUERY ROUTER STARTING...")
    print("="*60)
    print()
    
    try:
        if len(sys.argv) > 1:
            if sys.argv[1] == "test":
                # Run automated tests
                print("Mode: AUTOMATED TESTS\n")
                test_router()
            elif sys.argv[1] == "interactive":
                # Run interactive mode
                print("Mode: INTERACTIVE\n")
                interactive_mode()
            else:
                # Classify a single query passed as argument
                query = " ".join(sys.argv[1:])
                print(f"Mode: SINGLE QUERY")
                print(f"Query: {query}\n")
                
                client = initialize_genai_client()
                if client:
                    result = classify_query_with_explanation(query, client)
                    
                    print(f"Classification: {result['classification']}")
                    print(f"Reason: {result['explanation']}")
                else:
                    print("✗ Failed to initialize client")
        else:
            # Default: show usage
            print("Usage:")
            print("  python query_router.py test           # Run automated tests")
            print("  python query_router.py interactive    # Interactive mode")
            print('  python query_router.py "your query"   # Classify single query')
            print()
            print("Running automated tests...\n")
            test_router()
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback    
        traceback.print_exc()