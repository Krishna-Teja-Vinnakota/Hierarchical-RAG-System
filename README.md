# Adaptive Study Guide - Hierarchical RAG System

An AI-powered question-answering system for Operating Systems textbook using hierarchical retrieval-augmented generation (H-RAG).

## 🎯 Features

- ✅ **Hierarchical Indexing**: Two-level granularity (chapter summaries + detailed chunks)
- ✅ **Smart Query Routing**: Automatically classifies queries as SIMPLE or COMPLEX
- ✅ **Metadata-Based Filtering**: Efficient chapter-level filtering for complex queries
- ✅ **Rate Limit Handling**: Automatic retry logic with exponential backoff
- ✅ **Cached Progress**: Resume indexing from where you left off

## 🏗️ Architecture
```
User Query
    ↓
Query Router (SIMPLE/COMPLEX Classification)
    ↓
Retrieval Engine
    ├── SIMPLE: Level 2 Search + Graph Enrichment
    └── COMPLEX: Neo4j Graph Traversal → Enhanced Query → Filtered Search
    ↓
Answer Generator (Gemini)
    ↓
Formatted Answer + Sources
```

## 📊 Technology Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Gemini 2.0 Flash Experimental (Vertex AI) |
| **Embeddings** | text-embedding-004 (Vertex AI) |
| **Vector Database** | Pinecone (Serverless) |
| **Graph Database** | Neo4j (AuraDB or Local) |
| **Frontend** | Streamlit |
| **PDF Processing** | PyPDF |
| **Language** | Python 3.11+ |

## 🚀 Setup Instructions

### Prerequisites

- Python 3.11 or higher
- Google Cloud Platform account with Vertex AI enabled
- Pinecone account
- Neo4j Database (AuraDB Free Tier recommended)
- Service account JSON key with Vertex AI permissions

### Installation

1. **Clone the repository**
```bash
   cd Rag_project
```

2. **Create virtual environment**
```bash
   python -m venv venv_rag
   
   # Windows
   venv_rag\Scripts\activate
   
   # Mac/Linux
   source venv_rag/bin/activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Configure environment variables**
   
   Create `.env` file in root directory:
```env
   VERTEX_AI_SERVICE_ACCOUNT_PATH=gen-lang-client.json
   GOOGLE_CLOUD_PROJECT=your-project-id
   PINECONE_API_KEY=your-pinecone-api-key
   GOOGLE_CLOUD_REGION=us-central1
   
   # Neo4j Configuration
   NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your-password
```

5. **Add your PDF**
   
   Place your PDF in the `data/` folder

## 📖 Usage

### Phase 1: Indexing (Run Once)

**Step 1: Create Level 1 Index (Chapter Summaries)**
```bash
python level1_indexer.py
```
- Extracts chapters from PDF
- Generates summaries using Gemini
- Stores in Pinecone with `type="summary"`

**Step 2: Create Level 2 Index (Detailed Chunks)**
```bash
python level2_indexer.py
```
- Chunks each chapter into smaller pieces
- Stores in Pinecone with `type="detail"`

### Phase 2: Querying (Run Multiple Times)

**Start Web Interface** (Recommended)
```bash
streamlit run streamlit_rag.py
```

**CLI Interactive Mode**
```bash
python main_query.py
```

**Single Query Mode**
```bash
python main_query.py "What is a process?"
```

**Demo Mode**
```bash
python main_query.py demo
```

## 🎓 Example Queries

### SIMPLE Queries (Direct Detail Search)
```
"What is a thread?"
"Define semaphore"
"Explain virtual memory"
"What is deadlock?"
```

### COMPLEX Queries (Hierarchical Search)
```
"Compare processes and threads"
"What are the differences between paging and segmentation?"
"How did memory management evolve in operating systems?"
"Compare Linux and Windows scheduling approaches"
```

## 🗂️ Project Structure
```
Rag_project/
├── .env                              # Environment configuration
├── gen-lang-client.json              # Vertex AI service account
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
│
├── data/                             # PDF storage
│   └── textbook.pdf
│
├── level1_indexer.py                 # Chapter summary indexing
├── level2_indexer.py                 # Detail chunk indexing
├── query_router.py                   # Query classification
├── retrieval_engine.py               # Hierarchical retrieval
├── answer_generator.py               # Answer generation
├── main_query.py                     # Main entry point
│
├── document_summaries_cache.json     # Level 1 cache
└── document_details_cache.json       # Level 2 cache
```

## 🔧 Configuration

### Indexing Settings

**Level 1 (Summaries)**
- Model: `gemini-2.0-flash-exp`
- Summary length: 800-1000 characters
- Delay between requests: 15 seconds

**Level 2 (Details)**
- Chunk size: 400 words
- Chunk overlap: 50 words
- Embedding model: `text-embedding-004`

### Query Settings

- Classification delay: 3 seconds
- Answer generation delay: 5 seconds
- Max retries: 3
- Retry backoff: 15s → 30s → 60s

## 📈 System Metrics

- **Total Vectors**: 1,078
  - Level 1 (Summaries): 20 vectors
  - Level 2 (Details): 1,058 vectors
- **Embedding Dimension**: 768
- **Index Type**: Pinecone Serverless (AWS us-east-1)

## 🐛 Troubleshooting

### Quota Exceeded Error
```
Error: 429 RESOURCE_EXHAUSTED
```
**Solution**: System automatically retries with delays. Wait for completion or increase delays in code.

### Model Not Found Error
```
Error: 404 NOT_FOUND - Model not found
```
**Solution**: Model name is correct for Vertex AI. Ensure:
- Vertex AI API is enabled in GCP
- Service account has proper permissions
- Project has access to Gemini models

### No Chapters Detected
**Solution**: System tries 3 methods:
1. TOC + Header scanning (best)
2. Pattern-based detection (fallback)
3. Page-based splitting (last resort)

## 📚 How It Works

### Query Classification
```python
"What is a thread?" → SIMPLE
"Compare processes and threads" → COMPLEX
```

### Retrieval Strategy

**SIMPLE Path:**
```
Query → Embedding → Search Level 2 → Top 5 chunks → Answer
```

**COMPLEX Path:**
```
Query → Embedding → Search Level 1 → Extract chapter IDs
      → Filter Level 2 by chapters → Top 5 chunks → Answer
```

### Metadata Filtering
```python
# Level 1 (Summaries)
{
  "type": "summary",
  "chapter_num": "4",
  "title": "Chapter 4: Threads"
}

# Level 2 (Details)
{
  "type": "detail",
  "chapter_num": "4",
  "chunk_index": 0
}
```

## 🎯 Future Enhancements

- [ ] Add caching for repeated queries
- [ ] Implement conversation history
- [ ] Support multiple PDFs
- [ ] Add web interface
- [ ] Export answers to PDF

## 👤 Author

Harshith Yaranagi Yadav
- Project: Hierarchical RAG for Educational Content
- Tech Stack: Vertex AI, Pinecone, Python

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Google Vertex AI for Gemini models
- Pinecone for vector database
- Operating Systems textbook content