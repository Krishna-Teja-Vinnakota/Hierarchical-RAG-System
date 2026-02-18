# Adaptive Study Guide

## Overview
The Adaptive Study Guide is an AI-powered question-answering system designed to help users navigate and understand complex textbook content, specifically Operating Systems. It employs a **Hierarchical Retrieval-Augmented Generation (H-RAG)** architecture that processes information at two levels: high-level chapter summaries and detailed text chunks. The system intelligently routes queries between these levels and utilizes a **Graph RAG** approach with Neo4j to understand relationships between concepts, ensuring accurate answers for both simple definitions and complex comparative questions.

This repository contains the full pipeline for indexing PDF content, managing vector and graph databases, and running the interactive Streamlit application.

## Technologies

*   **UI / App**: Streamlit (Python)
*   **AI / LLM**: Google Gemini 2.0 Flash (via `google-genai` / Vertex AI)
*   **Persistence**:
    *   **Vector Database**: Pinecone (Serverless)
    *   **Graph Database**: Neo4j (AuraDB or Local)
    *   **Local Cache**: JSON files (`document_summaries_cache.json`, `document_details_cache.json`)
*   **PDF Processing**: `pypdf`

## Prerequisites

*   **Python 3.11** or higher
*   **pip** package manager
*   **Git** (for cloning repository)
*   **Google Cloud Platform** account with Vertex AI enabled and a service account JSON key
*   **Pinecone** account and API Key
*   **Neo4j** Database instance (AuraDB Free Tier is sufficient)

## Usage

### Clone the repository
```bash
git clone <your-repository-url>
cd <your-repo-folder>
```

### Create and activate a virtual environment (recommended)
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### Install required packages
Install the dependencies using the requirements file:
```bash
pip install -r requirements.txt
```

### Configure credentials and environment
Create a `.env` file in the root directory and set the following required variables:

*   `VERTEX_AI_SERVICE_ACCOUNT_PATH` — Path to your Google service account JSON file
*   `GOOGLE_CLOUD_PROJECT` — Your GCP Project ID
*   `GOOGLE_CLOUD_REGION` — GCP Region (e.g., `us-central1`)
*   `PINECONE_API_KEY` — Your Pinecone API Key
*   `NEO4J_URI` — URI for your Neo4j instance (e.g., `neo4j+s://...`)
*   `NEO4J_USER` — Neo4j username (usually `neo4j`)
*   `NEO4J_PASSWORD` — Neo4j password

### Start the app
To run the main interactive interface:
```bash
streamlit run streamlit_rag.py
```

Open the local Streamlit URL (printed in the terminal, usually `http://localhost:8501`). You can then enter queries to interact with the indexed content.

## Output and persistence

*   **`document_summaries_cache.json`**: Local cache of generated chapter summaries (Level 1 index).
*   **`document_details_cache.json`**: Local cache of detailed text chunks (Level 2 index).
*   **Pinecone Index**: Stores vector embeddings for both summaries and detailed chunks for semantic retrieval.
*   **Neo4j Graph**: Stores extracted entities and their relationships to support complex query reasoning.

## Example quick usage

1.  Clone repo and create a virtual environment.
2.  Install dependencies with `pip install -r requirements.txt`.
3.  Place your textbook PDF in the `data/` folder.
4.  Set up your `.env` file with API keys and credentials.
5.  **First-time setup**: Run the indexers to process the PDF:
    ```bash
    python level1_indexer.py  # Generates summaries
    python level2_indexer.py  # Generates detailed chunks
    ```
6.  Run `streamlit run streamlit_rag.py` and ask a question like "What is the difference between a process and a thread?".