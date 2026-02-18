"""
Answer Generator - Enhanced with Neo4j Graph Integration
Generates answers using both semantic (Pinecone) and structural (Neo4j) context
"""

import os
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from pinecone import Pinecone
from typing import Dict, List
import time

# --- CONFIGURATION ---
load_dotenv()

VERTEX_AI_SERVICE_ACCOUNT_PATH = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = "document-summaries"
EMBEDDING_MODEL = "text-embedding-004"
GENERATION_MODEL = "gemini-2.0-flash-exp"
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
        print(f"Error initializing GenAI client: {e}")
        return None


def get_embedding(client, text: str) -> List[float]:
    """Generate embedding for the given text."""
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=[text]
        )
        
        if result.embeddings and len(result.embeddings) > 0:
            return result.embeddings[0].values
        else:
            return None
            
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def build_context_with_graph(retrieval_result: Dict) -> str:
    """
    Build comprehensive context from retrieval result combining Pinecone + Neo4j.
    
    Args:
        retrieval_result: Output from retrieve() with graph_info
    
    Returns:
        Formatted context string for Gemini
    """
    context_parts = []
    
    # SECTION 1: Graph Relationships (NEW!)
    if retrieval_result.get('graph_info'):
        graph_info = retrieval_result['graph_info']
        
        if 'query_concepts' in graph_info and 'related_concepts' in graph_info:
            context_parts.append("="*60)
            context_parts.append("CONCEPT RELATIONSHIPS (from Knowledge Graph)")
            context_parts.append("="*60)
            
            for concept in graph_info['query_concepts']:
                related = graph_info['related_concepts'].get(concept, [])
                if related:
                    context_parts.append(f"\n{concept}:")
                    context_parts.append(f"  Related to: {', '.join(related)}")
                    context_parts.append(f"  (These concepts are connected in the knowledge graph)")
    
    # SECTION 2: Chapter Summaries (HIGH-LEVEL OVERVIEW)
    if retrieval_result.get('summaries'):
        context_parts.append("\n" + "="*60)
        context_parts.append("CHAPTER SUMMARIES (High-Level Overview)")
        context_parts.append("="*60)
        
        for chunk in retrieval_result['summaries']:
            context_parts.append(f"\n[Chapter {chunk['chapter_num']}: {chunk['title']}]")
            context_parts.append(f"Relevance Score: {chunk['score']:.4f}")
            context_parts.append(f"\n{chunk['text']}\n")
    
    # SECTION 3: Detailed Information
    if retrieval_result.get('details'):
        context_parts.append("\n" + "="*60)
        context_parts.append("DETAILED INFORMATION (From Textbook)")
        context_parts.append("="*60)
        
        for chunk in retrieval_result['details']:
            context_parts.append(f"\n[Chapter {chunk['chapter_num']}: {chunk['title']}, Chunk {chunk.get('chunk_index', 'N/A')}]")
            context_parts.append(f"Relevance Score: {chunk['score']:.4f}")
            context_parts.append(f"\n{chunk['text']}\n")
    
    return "\n".join(context_parts)


def generate_answer(query: str, retrieval_result: Dict, genai_client) -> Dict:
    """
    Generate answer using Gemini based on retrieval context (Pinecone + Neo4j).
    
    Args:
        query: User's question
        retrieval_result: Output from retrieve() with graph_info and chunks
        genai_client: Initialized GenAI client
        
    Returns:
        Dictionary with answer and metadata
    """
    
    # Build comprehensive context
    context = build_context_with_graph(retrieval_result)
    
    # Create prompt based on query type
    if retrieval_result['query_type'] == "SIMPLE":
        prompt = f"""You are an expert in Operating Systems, answering questions based on a comprehensive textbook.

The user is asking a single-concept question. Provide a clear, concise answer.

CONTEXT FROM TEXTBOOK:
{context}

QUESTION: {query}

Instructions:
1. Provide a direct, focused answer to the question
2. Use the detailed information from the textbook context
3. If related concepts are mentioned in the knowledge graph, briefly mention them for context
4. Be accurate and precise
5. Keep the answer organized and easy to understand

Answer:"""

    else:  # COMPLEX
        prompt = f"""You are an expert in Operating Systems, answering questions based on a comprehensive textbook.

The user is asking a complex question that requires comparing or synthesizing multiple concepts. 
Use the knowledge graph relationships to make connections between concepts.

CONTEXT FROM TEXTBOOK:
{context}

QUESTION: {query}

Instructions:
1. Provide a comprehensive answer that addresses all aspects of the question
2. Use the chapter summaries for high-level overview
3. Use detailed information for specific facts
4. IMPORTANT: Use the "Concept Relationships" section to make connections between concepts
5. Explain how concepts relate to each other using the knowledge graph
6. Compare and contrast concepts when relevant
7. Provide a well-structured answer with clear organization

Answer:"""

    # Generate answer with retry logic and delays
    max_retries = 3
    retry_delay = 15
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                print(f"  ⏳ Retry attempt {attempt + 1}/{max_retries}. Waiting {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                # Always add a small delay before first attempt to avoid rate limits
                print("  ⏳ Adding delay to avoid rate limits (5s)...")
                time.sleep(5)
            
            print("  Generating answer with Gemini...")
            response = genai_client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )
            
            if not response or not response.text:
                if attempt < max_retries - 1:
                    continue
                return {
                    "query": query,
                    "query_type": retrieval_result['query_type'],
                    "answer": "Sorry, I couldn't generate an answer at this time.",
                    "sources": [],
                    "graph_info": retrieval_result.get('graph_info', {}),
                    "num_chunks_used": retrieval_result.get('total_chunks', 0)
                }
            
            # Success! Compile sources and return
            sources = []
            
            # Add summary sources
            for chunk in retrieval_result.get('summaries', []):
                sources.append({
                    "type": "summary",
                    "chapter_num": chunk['chapter_num'],
                    "title": chunk['title'],
                    "score": chunk['score']
                })
            
            # Add detail sources
            for chunk in retrieval_result.get('details', []):
                sources.append({
                    "type": "detail",
                    "chapter_num": chunk['chapter_num'],
                    "title": chunk['title'],
                    "chunk_index": chunk.get('chunk_index', 'N/A'),
                    "score": chunk['score']
                })
            
            return {
                "query": query,
                "query_type": retrieval_result['query_type'],
                "answer": response.text.strip(),
                "sources": sources,
                "graph_info": retrieval_result.get('graph_info', {}),
                "num_chunks_used": retrieval_result.get('total_chunks', 0)
            }
            
        except ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    retry_delay = min(retry_delay * 2, 60)
                    continue
                else:
                    print(f"  ✗ Max retries reached. Rate limit exceeded.")
                    return {
                        "query": query,
                        "query_type": retrieval_result['query_type'],
                        "answer": "⚠ The system is experiencing high demand. Please wait 30 seconds and try again.",
                        "sources": [],
                        "graph_info": retrieval_result.get('graph_info', {}),
                        "num_chunks_used": retrieval_result.get('total_chunks', 0)
                    }
            else:
                print(f"  ✗ Error: {str(e)}")
                return {
                    "query": query,
                    "query_type": retrieval_result['query_type'],
                    "answer": f"Error generating answer: {str(e)[:200]}",
                    "sources": [],
                    "graph_info": retrieval_result.get('graph_info', {}),
                    "num_chunks_used": retrieval_result.get('total_chunks', 0)
                }
    
    # Should never reach here
    return {
        "query": query,
        "query_type": retrieval_result['query_type'],
        "answer": "Unable to generate answer after multiple retries.",
        "sources": [],
        "graph_info": retrieval_result.get('graph_info', {}),
        "num_chunks_used": retrieval_result.get('total_chunks', 0)
    }


def display_answer(result: Dict):
    """
    Display the answer with enhanced formatting including graph info.
    
    Args:
        result: Output from generate_answer() function
    """
    print("\n" + "="*60)
    print("ANSWER")
    print("="*60)
    print(f"\nQuery Type: {result['query_type']}")
    print(f"Chunks Used: {result['num_chunks_used']}")
    print("\n" + "-"*60)
    print(result['answer'])
    print("-"*60)
    
    # Display graph enrichment if available
    if result.get('graph_info'):
        graph_info = result['graph_info']
        
        if 'query_concepts' in graph_info or 'related_concepts' in graph_info:
            print("\n" + "="*60)
            print("KNOWLEDGE GRAPH ENRICHMENT")
            print("="*60)
            
            if 'query_concepts' in graph_info:
                print(f"\nConcepts Found in Your Query:")
                for concept in graph_info['query_concepts']:
                    print(f"  • {concept}")
            
            if 'related_concepts' in graph_info:
                print(f"\nRelated Concepts (from Knowledge Graph):")
                for concept, related in graph_info['related_concepts'].items():
                    print(f"  {concept}:")
                    for rel_concept in related[:5]:  # Show top 5 related
                        print(f"    → {rel_concept}")
    
    # Display sources
    if result.get('sources'):
        print("\n" + "="*60)
        print("SOURCES")
        print("="*60)
        
        # Group by type
        summaries = [s for s in result['sources'] if s['type'] == 'summary']
        details = [s for s in result['sources'] if s['type'] == 'detail']
        
        if summaries:
            print("\nChapter Summaries Used:")
            for i, source in enumerate(summaries, 1):
                print(f"  {i}. Chapter {source['chapter_num']}: {source['title']}")
                print(f"     Relevance Score: {source['score']:.4f}")
        
        if details:
            print("\nDetailed Information Used:")
            for i, source in enumerate(details, 1):
                print(f"  {i}. Chapter {source['chapter_num']}: {source['title']}")
                print(f"     Chunk: {source['chunk_index']} | Relevance Score: {source['score']:.4f}")
    
    print("\n" + "="*60)


# For backward compatibility with main_query.py
def answer_question(question: str, verbose: bool = True) -> dict:
    """
    Legacy function for compatibility - wraps the new pipeline.
    
    This is a simplified version that doesn't use the full routing logic.
    For full functionality, use the complete pipeline in main_query.py
    """
    print("\n⚠ Warning: Using simplified answer_question()")
    print("For full hierarchical RAG with graph integration, use main_query.py instead\n")
    
    # Initialize clients
    client = initialize_genai_client()
    if not client:
        return {
            "answer": "Failed to initialize GenAI client",
            "sources": [],
            "relevant_chapters": []
        }
    
    # Initialize Pinecone
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
    except Exception as e:
        return {
            "answer": f"Failed to connect to Pinecone: {e}",
            "sources": [],
            "relevant_chapters": []
        }
    
    # Generate query embedding
    query_embedding = get_embedding(client, question)
    if not query_embedding:
        return {
            "answer": "Failed to generate query embedding",
            "sources": [],
            "relevant_chapters": []
        }
    
    # Search summaries first
    if verbose:
        print("Searching summaries...")
    
    summary_results = index.query(
        vector=query_embedding,
        top_k=3,
        filter={"type": {"$eq": "summary"}},
        include_metadata=True
    )
    
    if not summary_results.matches:
        return {
            "answer": "No relevant chapters found",
            "sources": [],
            "relevant_chapters": []
        }
    
    relevant_chapters = [match.metadata.get("chapter_num", "") for match in summary_results.matches]
    
    if verbose:
        print(f"Found relevant chapters: {relevant_chapters}")
    
    # Search details with chapter filter
    if verbose:
        print("Searching detailed chunks...")
    
    detail_results = index.query(
        vector=query_embedding,
        top_k=5,
        filter={
            "type": {"$eq": "detail"},
            "chapter_num": {"$in": relevant_chapters}
        },
        include_metadata=True
    )
    
    # Build context
    context_parts = []
    for match in detail_results.matches:
        context_parts.append(
            f"[Chapter {match.metadata.get('chapter_num')}: {match.metadata.get('title')}]\n"
            f"{match.metadata.get('text_preview', '')}\n"
        )
    
    context = "\n".join(context_parts)
    
    # Generate answer with delay
    prompt = f"""Based on the following context from an Operating Systems textbook, answer the question.

Context:
{context}

Question: {question}

Provide a detailed answer based on the context above."""

    try:
        if verbose:
            print("Generating answer...")
            print("⏳ Adding delay to avoid rate limits (5s)...")
            time.sleep(5)
        
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )
        
        sources = []
        for match in detail_results.matches:
            sources.append({
                "chapter_number": match.metadata.get("chapter_num"),
                "chapter_title": match.metadata.get("title"),
                "page_range": f"{match.metadata.get('start_page', 'N/A')}-{match.metadata.get('end_page', 'N/A')}",
                "score": match.score
            })
        
        return {
            "answer": response.text.strip() if response and response.text else "No answer generated",
            "sources": sources,
            "relevant_chapters": relevant_chapters
        }
        
    except Exception as e:
        return {
            "answer": f"Error generating answer: {e}",
            "sources": [],
            "relevant_chapters": relevant_chapters
        }


if __name__ == "__main__":
    print("="*60)
    print("ANSWER GENERATOR - STANDALONE MODE")
    print("="*60)
    print("\n⚠ Note: For full hierarchical RAG with Neo4j graph integration,")
    print("please use main_query.py instead.\n")
    print("This standalone mode uses simplified retrieval.\n")
    
    # Interactive loop
    while True:
        question = input("\nYour question (or 'quit' to exit): ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        result = answer_question(question, verbose=True)
        
        print("\n" + "="*60)
        print("ANSWER:")
        print("="*60)
        print(result['answer'])
        
        if result['sources']:
            print("\n" + "="*60)
            print("SOURCES:")
            print("="*60)
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. Chapter {source['chapter_number']}: {source['chapter_title']}")
                print(f"   Pages {source['page_range']} (relevance: {source['score']:.4f})")