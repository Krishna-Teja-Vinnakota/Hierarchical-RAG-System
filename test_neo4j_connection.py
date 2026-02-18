"""
Simple test script to verify Neo4j connection
Run this to make sure your credentials and URI are correct
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Neo4j credentials from .env
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

print("="*60)
print("NEO4J CONNECTION TEST")
print("="*60)

# Check if credentials are loaded
print("\n1. Checking if credentials are loaded from .env...")
if NEO4J_URI and NEO4J_USER and NEO4J_PASSWORD:
    print("   ✓ All credentials loaded:")
    print(f"   - URI: {NEO4J_URI}")
    print(f"   - USER: {NEO4J_USER}")
    print(f"   - PASSWORD: {'*' * len(NEO4J_PASSWORD)} (hidden)")
else:
    print("   ✗ Error: Credentials not loaded from .env")
    if not NEO4J_URI:
        print("   - Missing: NEO4J_URI")
    if not NEO4J_USER:
        print("   - Missing: NEO4J_USER")
    if not NEO4J_PASSWORD:
        print("   - Missing: NEO4J_PASSWORD")
    exit(1)

# Try to import neo4j driver
print("\n2. Checking if neo4j package is installed...")
try:
    from neo4j import GraphDatabase
    print("   ✓ neo4j package found")
except ImportError:
    print("   ✗ Error: neo4j package not installed")
    print("   Run: pip install neo4j")
    exit(1)

# Try to connect to Neo4j
print("\n3. Attempting to connect to Neo4j...")
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    print("   ✓ Connected to Neo4j!")
except Exception as e:
    print(f"   ✗ Error: Could not connect to Neo4j")
    print(f"   Error details: {e}")
    exit(1)

# Try to run a test query
print("\n4. Running test query...")
try:
    with driver.session() as session:
        result = session.run("RETURN 'Neo4j is working!' as message")
        for record in result:
            message = record["message"]
            print(f"   ✓ Query successful!")
            print(f"   ✓ Message from Neo4j: {message}")
except Exception as e:
    print(f"   ✗ Error: Could not execute test query")
    print(f"   Error details: {e}")
    driver.close()
    exit(1)

# Close driver
driver.close()

print("\n" + "="*60)
print("✓ ALL TESTS PASSED!")
print("="*60)
print("\nYour Neo4j connection is working correctly!")
print("You can proceed to Phase 2 of the setup.")
