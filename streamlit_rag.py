"""
Streamlit UI for Graph-First Hierarchical RAG System
Integrates Query Router + Retrieval Engine + Answer Generator with Neo4j
Beautiful, interactive interface with full pipeline visualization
"""

import streamlit as st
import os
from dotenv import load_dotenv
import time
from typing import Dict, List
import json

# Import custom modules
from query_router import classify_query
from retrieval_engine import retrieve, initialize_pinecone, initialize_neo4j
from answer_generator import generate_answer, build_context_with_graph
from google import genai

# Load environment variables
load_dotenv()

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="📚 Operating Systems Study Guide",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Main styling */
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 30px;
    }
    
    .query-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    
    .classification-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    
    .simple-badge {
        background-color: #4CAF50;
        color: white;
    }
    
    .complex-badge {
        background-color: #2196F3;
        color: white;
    }
    
    .answer-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 15px 0;
    }
    
    .chunk-box {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #607D8B;
    }
    
    .graph-box {
        background-color: #F3E5F5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #9C27B0;
    }
    
    .source-box {
        background-color: #FFF3E0;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #FF9800;
    }
    
    .metric-container {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        margin: 10px;
    }
    
    .success-status {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .error-status {
        color: #F44336;
        font-weight: bold;
    }
    
    .warning-status {
        color: #FF9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE INITIALIZATION ====================
@st.cache_resource
def initialize_system():
    """Initialize all clients and return them"""
    try:
        # Initialize GenAI Client
        from google.oauth2 import service_account
        
        vertex_ai_path = os.getenv("VERTEX_AI_SERVICE_ACCOUNT_PATH")
        project = os.getenv("GOOGLE_CLOUD_PROJECT")
        
        credentials = service_account.Credentials.from_service_account_file(
            vertex_ai_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        genai_client = genai.Client(
            vertexai=True,
            project=project,
            location="us-central1",
            credentials=credentials
        )
        
        # Initialize Pinecone
        pinecone_index = initialize_pinecone()
        
        # Initialize Neo4j
        neo4j_handler = initialize_neo4j()
        
        return {
            "genai": genai_client,
            "pinecone": pinecone_index,
            "neo4j": neo4j_handler,
            "initialized": True
        }
    except Exception as e:
        st.error(f"❌ Initialization Error: {e}")
        return {"initialized": False}


# Initialize session state
if 'system' not in st.session_state:
    st.session_state.system = initialize_system()

if 'last_query' not in st.session_state:
    st.session_state.last_query = None

if 'last_result' not in st.session_state:
    st.session_state.last_result = None


# ==================== HEADER ====================
def display_header():
    """Display main header"""
    st.markdown(
        "<h1 class='main-header'>📚 Operating Systems Study Guide</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align: center; color: #666;'>"
        "Ask questions about Operating Systems. Powered by Graph-First Hierarchical RAG 🚀"
        "</p>",
        unsafe_allow_html=True
    )
    st.divider()


# ==================== SIDEBAR ====================
def display_sidebar():
    """Display sidebar with system status and instructions"""
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        
        if st.session_state.system.get("initialized"):
            st.markdown('<p class="success-status">✅ System Ready</p>', unsafe_allow_html=True)
            
            st.markdown("#### 📊 System Components")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("GenAI", "Active")
                st.metric("Pinecone", "Connected")
            with col2:
                st.metric("Neo4j", "Connected")
                st.metric("Retrieval", "Graph-First")
        else:
            st.markdown('<p class="error-status">❌ System Not Initialized</p>', unsafe_allow_html=True)
            st.warning("⚠️ System failed to initialize. Check environment variables and credentials.")
            return False
        
        st.divider()
        
        st.markdown("### 📖 How It Works")
        st.markdown("""
        **Pipeline:**
        1. **📝 Query Input** - You ask a question
        2. **🔍 Classification** - System identifies SIMPLE or COMPLEX
        3. **🧠 Neo4j Graph** - Graph finds related concepts
        4. **📊 Pinecone Search** - Graph-guided semantic search
        5. **✍️ Generation** - Gemini synthesizes answer
        6. **📚 Sources** - Shows which chapters were used
        
        **Features:**
        - ⚡ **Graph-First Retrieval**: Neo4j output becomes Pinecone input
        - 📊 **Hierarchical Search**: Summaries → Filtered details
        - 🔗 **Relationship Awareness**: Understands concept connections
        - 🎯 **Adaptive**: Routes SIMPLE queries differently than COMPLEX
        """)
        
        st.divider()
        
        st.markdown("### 🎯 Query Types")
        st.markdown("""
        **SIMPLE Queries:**
        - Single concept questions
        - Definitions, facts, explanations
        - Example: "What is a process?"
        
        **COMPLEX Queries:**
        - Comparisons, relationships
        - Multi-concept synthesis
        - Example: "Compare processes vs threads"
        """)
        
        return True


# ==================== MAIN CONTENT ====================
def process_query(query: str) -> Dict:
    """
    Process a query through the complete pipeline
    """
    system = st.session_state.system
    genai_client = system["genai"]
    pinecone_index = system["pinecone"]
    neo4j_handler = system["neo4j"]
    
    # Create columns for progress
    col1, col2, col3 = st.columns(3)
    
    # STEP 1: Classify Query
    with col1:
        with st.spinner("🔍 Classifying..."):
            time.sleep(1)  # Rate limiting
            query_type = classify_query(query, genai_client)
    
    col1.markdown(f"**Classification:** {query_type}")
    
    # STEP 2: Retrieve Information
    with col2:
        with st.spinner("📚 Retrieving..."):
            time.sleep(1)  # Rate limiting
            retrieval_result = retrieve(
                query,
                query_type,
                genai_client,
                pinecone_index,
                neo4j_handler
            )
    
    col2.markdown(f"**Chunks:** {retrieval_result['total_chunks']}")
    
    # STEP 3: Generate Answer
    with col3:
        with st.spinner("✍️ Generating..."):
            time.sleep(2)  # Rate limiting
            answer_result = generate_answer(query, retrieval_result, genai_client)
    
    col3.markdown(f"**Status:** ✅ Done")
    
    return {
        "query_type": query_type,
        "retrieval_result": retrieval_result,
        "answer_result": answer_result
    }


def display_classification(query_type: str):
    """Display query classification"""
    st.markdown("### 🔍 Query Classification")
    
    if query_type == "SIMPLE":
        st.markdown(
            '<span class="classification-badge simple-badge">SIMPLE QUERY</span>',
            unsafe_allow_html=True
        )
        st.info("Single-concept query: System will search Level 2 (details) with graph enrichment")
    else:
        st.markdown(
            '<span class="classification-badge complex-badge">COMPLEX QUERY</span>',
            unsafe_allow_html=True
        )
        st.info("Multi-concept query: System will use Graph-First hierarchical retrieval")


def display_retrieval_stats(retrieval_result: Dict):
    """Display retrieval statistics"""
    st.markdown("### 📊 Retrieval Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Summaries Retrieved",
            len(retrieval_result['summaries']),
            delta="chapters"
        )
    
    with col2:
        st.metric(
            "Details Retrieved",
            len(retrieval_result['details']),
            delta="chunks"
        )
    
    with col3:
        st.metric(
            "Total Chunks",
            retrieval_result['total_chunks'],
            delta="all"
        )
    
    with col4:
        st.metric(
            "Method",
            "Graph-First",
            delta="Enhanced"
        )
    
    # Display retrieval method note
    st.info(f"🚀 **Retrieval Method:** {retrieval_result.get('retrieval_method', 'N/A')}")
    st.caption(f"💡 {retrieval_result.get('note', 'Graph-First retrieval in action')}")


def display_retrieved_chunks(retrieval_result: Dict):
    """Display retrieved chunks in expandable sections"""
    st.markdown("### 📚 Retrieved Information")
    
    # Summaries
    if retrieval_result['summaries']:
        with st.expander("📄 Chapter Summaries (High-Level Overview)", expanded=False):
            for i, chunk in enumerate(retrieval_result['summaries'], 1):
                st.markdown(f"**[{i}] Chapter {chunk['chapter_num']}: {chunk['title']}**")
                st.caption(f"Relevance Score: {chunk['score']:.4f}")
                st.markdown("---")
    
    # Details
    if retrieval_result['details']:
        with st.expander(f"📝 Detail Chunks ({len(retrieval_result['details'])} chunks)", expanded=False):
            for i, chunk in enumerate(retrieval_result['details'], 1):
                st.markdown(f"**[{i}] Chapter {chunk['chapter_num']}: {chunk['title']}**")
                st.caption(
                    f"Chunk {chunk['chunk_index']} | Score: {chunk['score']:.4f}"
                )
                st.markdown("---")


def display_graph_info(retrieval_result: Dict):
    """Display Neo4j graph information"""
    graph_info = retrieval_result.get('graph_info', {})
    
    if not graph_info:
        return
    
    st.markdown("### 🔗 Knowledge Graph (Neo4j)")
    
    with st.expander("📊 Concept Discovery & Relationships", expanded=False):
        
        # Query Concepts
        if 'query_concepts' in graph_info:
            st.markdown("**Concepts Found in Your Query:**")
            for concept in graph_info['query_concepts']:
                st.markdown(f"• {concept}")
        
        # All Discovered Concepts
        if 'all_discovered_concepts' in graph_info:
            all_concepts = graph_info['all_discovered_concepts']
            st.markdown(f"**Total Discovered Concepts: {len(all_concepts)}**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Concept List:**")
                concepts_display = ", ".join(all_concepts[:10])
                st.caption(concepts_display)
                if len(all_concepts) > 10:
                    st.caption(f"... and {len(all_concepts) - 10} more")
        
        # Related Concepts Map
        if 'related_concepts' in graph_info:
            st.markdown("**Related Concepts (Graph Relationships):**")
            related_map = graph_info['related_concepts']
            
            for concept, related_list in related_map.items():
                with st.container():
                    st.markdown(f"**{concept}:**")
                    related_str = ", ".join(related_list[:5])
                    st.markdown(f"→ {related_str}")
                    if len(related_list) > 5:
                        st.caption(f"... and {len(related_list) - 5} more related concepts")


def display_answer(answer_result: Dict):
    """Display generated answer"""
    st.markdown("### ✅ Answer")
    
    st.markdown(
        f"""
        <div class='answer-box'>
        {answer_result['answer']}
        </div>
        """,
        unsafe_allow_html=True
    )


def display_sources(answer_result: Dict):
    """Display sources used in answer"""
    st.markdown("### 📚 Sources & Attribution")
    
    sources = answer_result.get('sources', [])
    
    if not sources:
        st.info("No sources found")
        return
    
    # Separate by type
    summaries = [s for s in sources if s['type'] == 'summary']
    details = [s for s in sources if s['type'] == 'detail']
    
    col1, col2 = st.columns(2)
    
    with col1:
        if summaries:
            st.markdown("**Chapter Summaries Used:**")
            for i, source in enumerate(summaries, 1):
                st.markdown(f"{i}. **Chapter {source['chapter_num']}:** {source['title']}")
                st.caption(f"   Relevance: {source['score']:.4f}")
    
    with col2:
        if details:
            st.markdown("**Detailed Sections Used:**")
            for i, source in enumerate(details, 1):
                st.markdown(f"{i}. **Chapter {source['chapter_num']}:** {source['title']}")
                st.caption(f"   Chunk {source['chunk_index']} | Score: {source['score']:.4f}")


def display_export_options(answer_result: Dict, query: str):
    """Display export/download options"""
    st.markdown("### 💾 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    # Export as text
    with col1:
        export_text = f"""QUESTION:
{query}

ANSWER:
{answer_result['answer']}

SOURCES:
"""
        for source in answer_result.get('sources', []):
            export_text += f"\n- Chapter {source['chapter_num']}: {source['title']}"
        
        st.download_button(
            label="📥 Download as TXT",
            data=export_text,
            file_name="answer.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    # Export as JSON
    with col2:
        export_json = {
            "question": query,
            "answer": answer_result['answer'],
            "query_type": answer_result.get('query_type', ''),
            "sources": answer_result.get('sources', []),
            "chunks_used": answer_result.get('num_chunks_used', 0)
        }
        
        st.download_button(
            label="📋 Download as JSON",
            data=json.dumps(export_json, indent=2),
            file_name="answer.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Copy to clipboard instruction
    with col3:
        st.info("💡 Select and copy answer text above using Ctrl+C / Cmd+C")


# ==================== MAIN FUNCTION ====================
def main():
    """Main Streamlit application"""
    
    # Display header
    display_header()
    
    # Display sidebar
    if not display_sidebar():
        st.error("❌ System initialization failed. Please check your environment variables.")
        return
    
    # Main content area
    st.markdown("### 🤔 Ask Your Question")
    
    # Query input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_query = st.text_input(
            "Enter your question about Operating Systems:",
            placeholder="E.g., What is a process? | Compare processes and threads...",
            label_visibility="collapsed",
            key="query_input"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    # Process query
    if search_button and user_query:
        st.session_state.last_query = user_query
        
        st.markdown("### ⏳ Processing Your Query...")
        
        try:
            # Process through pipeline
            result = process_query(user_query)
            st.session_state.last_result = result
            
            st.divider()
            
            # Display classification
            display_classification(result['query_type'])
            
            st.divider()
            
            # Display retrieval stats
            display_retrieval_stats(result['retrieval_result'])
            
            st.divider()
            
            # Display retrieved chunks
            display_retrieved_chunks(result['retrieval_result'])
            
            st.divider()
            
            # Display graph information
            display_graph_info(result['retrieval_result'])
            
            st.divider()
            
            # Display answer
            display_answer(result['answer_result'])
            
            st.divider()
            
            # Display sources
            display_sources(result['answer_result'])
            
            st.divider()
            
            # Display export options
            display_export_options(result['answer_result'], user_query)
            
            # Success message
            st.success("✅ Query processed successfully! You can now download or copy the answer above.")
            
        except Exception as e:
            st.error(f"❌ Error processing query: {e}")
            import traceback
            st.error(traceback.format_exc())
    
    # Show recent query info if available
    if st.session_state.last_query and not search_button:
        st.divider()
        st.markdown("### 📋 Last Query")
        st.caption(f"'{st.session_state.last_query}'")
        
        if st.button("🔄 Run Same Query Again"):
            st.rerun()


# ==================== RUN APP ====================
if __name__ == "__main__":
    main()