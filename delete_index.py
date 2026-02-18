import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "document-summaries"

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if index exists
    if INDEX_NAME in pc.list_indexes().names():
        print(f"Deleting Pinecone index: {INDEX_NAME}")
        pc.delete_index(INDEX_NAME)
        print("✓ Index deleted successfully!")
    else:
        print(f"Index '{INDEX_NAME}' does not exist. Nothing to delete.")
        
except Exception as e:
    print(f"Error: {e}")