"""
Neo4j Handler - Manages all graph database operations
Handles node/relationship creation, queries, and graph traversal
"""

import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import List, Dict, Tuple, Optional
import time

# Load environment variables
load_dotenv()

# Neo4j Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


class Neo4jHandler:
    """
    Handler class for all Neo4j operations
    Manages connection, node creation, relationships, and queries
    """
    
    def __init__(self):
        """Initialize Neo4j connection"""
        try:
            self.driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD)
            )
            # Verify connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print("✓ Neo4j connection established")
        except Exception as e:
            print(f"✗ Error connecting to Neo4j: {e}")
            raise
    
    def close(self):
        """Close Neo4j connection"""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j connection closed")
    
    # ==================== NODE CREATION FUNCTIONS ====================
    
    def create_chapter_node(self, chapter_num: str, title: str, summary: str = "") -> bool:
        """
        Create a Chapter node in the graph
        
        Args:
            chapter_num: Chapter number (e.g., "1", "2", "3")
            title: Chapter title (e.g., "Introduction to OS")
            summary: Optional chapter summary
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (c:Chapter {chapter_num: $chapter_num})
                    SET c.title = $title, c.summary = $summary
                    RETURN c
                    """,
                    chapter_num=chapter_num,
                    title=title,
                    summary=summary
                )
            return True
        except Exception as e:
            print(f"Error creating chapter node: {e}")
            return False
    
    def create_concept_node(self, name: str, definition: str = "", 
                           importance_level: str = "medium", 
                           first_mentioned_chapter: str = "") -> bool:
        """
        Create a Concept node in the graph
        
        Args:
            name: Concept name (e.g., "Process", "Thread", "Deadlock")
            definition: Concept definition/explanation
            importance_level: "high", "medium", "low"
            first_mentioned_chapter: Which chapter first introduces this concept
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MERGE (con:Concept {name: $name})
                    SET con.definition = $definition,
                        con.importance_level = $importance_level,
                        con.first_mentioned_chapter = $first_mentioned_chapter,
                        con.created_at = timestamp()
                    RETURN con
                    """,
                    name=name,
                    definition=definition,
                    importance_level=importance_level,
                    first_mentioned_chapter=first_mentioned_chapter
                )
            return True
        except Exception as e:
            print(f"Error creating concept node: {e}")
            return False
    
    # ==================== RELATIONSHIP CREATION FUNCTIONS ====================
    
    def create_contains_relationship(self, chapter_num: str, concept_name: str) -> bool:
        """
        Create a CONTAINS relationship: Chapter contains Concept
        
        Args:
            chapter_num: Chapter number
            concept_name: Concept name
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (c:Chapter {chapter_num: $chapter_num})
                    MATCH (con:Concept {name: $concept_name})
                    MERGE (c)-[:CONTAINS]->(con)
                    """,
                    chapter_num=chapter_num,
                    concept_name=concept_name
                )
            return True
        except Exception as e:
            print(f"Error creating CONTAINS relationship: {e}")
            return False
    
    def create_prerequisite_relationship(self, concept1_name: str, concept2_name: str) -> bool:
        """
        Create a PREREQUISITE_FOR relationship
        concept1 PREREQUISITE_FOR concept2 means:
        You must understand concept1 before concept2
        
        Args:
            concept1_name: Prerequisites concept
            concept2_name: Dependent concept
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (con1:Concept {name: $concept1_name})
                    MATCH (con2:Concept {name: $concept2_name})
                    MERGE (con1)-[:PREREQUISITE_FOR]->(con2)
                    """,
                    concept1_name=concept1_name,
                    concept2_name=concept2_name
                )
            return True
        except Exception as e:
            print(f"Error creating PREREQUISITE_FOR relationship: {e}")
            return False
    
    def create_relates_to_relationship(self, concept1_name: str, concept2_name: str) -> bool:
        """
        Create a RELATES_TO relationship (bidirectional)
        Used for concepts that are related but not in prerequisite order
        
        Args:
            concept1_name: First concept
            concept2_name: Second concept
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (con1:Concept {name: $concept1_name})
                    MATCH (con2:Concept {name: $concept2_name})
                    MERGE (con1)-[:RELATES_TO]->(con2)
                    MERGE (con2)-[:RELATES_TO]->(con1)
                    """,
                    concept1_name=concept1_name,
                    concept2_name=concept2_name
                )
            return True
        except Exception as e:
            print(f"Error creating RELATES_TO relationship: {e}")
            return False
    
    def create_solves_problem_relationship(self, concept1_name: str, concept2_name: str) -> bool:
        """
        Create a SOLVES_PROBLEM_OF relationship
        concept1 SOLVES_PROBLEM_OF concept2 means:
        Concept1 is a solution to the problem of concept2
        
        Args:
            concept1_name: Solution concept
            concept2_name: Problem concept
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                session.run(
                    """
                    MATCH (con1:Concept {name: $concept1_name})
                    MATCH (con2:Concept {name: $concept2_name})
                    MERGE (con1)-[:SOLVES_PROBLEM_OF]->(con2)
                    """,
                    concept1_name=concept1_name,
                    concept2_name=concept2_name
                )
            return True
        except Exception as e:
            print(f"Error creating SOLVES_PROBLEM_OF relationship: {e}")
            return False
    
    # ==================== QUERY FUNCTIONS ====================
    
    def get_concept_by_name(self, concept_name: str) -> Optional[Dict]:
        """
        Get a concept and its properties
        
        Args:
            concept_name: Name of the concept
        
        Returns:
            Dictionary with concept properties or None if not found
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (con:Concept {name: $concept_name})
                    RETURN con.name as name,
                           con.definition as definition,
                           con.importance_level as importance,
                           con.first_mentioned_chapter as chapter
                    """,
                    concept_name=concept_name
                )
                record = result.single()
                if record:
                    return {
                        "name": record["name"],
                        "definition": record["definition"],
                        "importance": record["importance"],
                        "chapter": record["chapter"]
                    }
            return None
        except Exception as e:
            print(f"Error getting concept: {e}")
            return None
    
    def get_concepts_in_chapter(self, chapter_num: str) -> List[Dict]:
        """
        Get all concepts in a specific chapter
        
        Args:
            chapter_num: Chapter number
        
        Returns:
            List of concepts with their properties
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Chapter {chapter_num: $chapter_num})-[:CONTAINS]->(con:Concept)
                    RETURN con.name as name,
                           con.definition as definition,
                           con.importance_level as importance
                    ORDER BY con.importance_level DESC
                    """,
                    chapter_num=chapter_num
                )
                concepts = []
                for record in result:
                    concepts.append({
                        "name": record["name"],
                        "definition": record["definition"],
                        "importance": record["importance"]
                    })
                return concepts
        except Exception as e:
            print(f"Error getting concepts in chapter: {e}")
            return []
    
    def get_related_concepts(self, concept_name: str, relationship_type: str = "RELATES_TO") -> List[str]:
        """
        Get concepts related to a given concept
        
        Args:
            concept_name: Name of the concept
            relationship_type: Type of relationship to follow
                               Options: "RELATES_TO", "PREREQUISITE_FOR", "SOLVES_PROBLEM_OF"
        
        Returns:
            List of related concept names
        """
        try:
            with self.driver.session() as session:
                query = f"""
                    MATCH (con:Concept {{name: $concept_name}})-[:{relationship_type}]->(related:Concept)
                    RETURN related.name as name
                    """
                result = session.run(query, concept_name=concept_name)
                related = [record["name"] for record in result]
                return related
        except Exception as e:
            print(f"Error getting related concepts: {e}")
            return []
    
    def get_prerequisites(self, concept_name: str) -> List[str]:
        """
        Get all concepts that must be understood before this concept
        
        Args:
            concept_name: Name of the concept
        
        Returns:
            List of prerequisite concept names
        """
        return self.get_related_concepts(concept_name, "PREREQUISITE_FOR")
    
    def get_dependent_concepts(self, concept_name: str) -> List[str]:
        """
        Get all concepts that depend on understanding this concept
        (reverse of PREREQUISITE_FOR)
        
        Args:
            concept_name: Name of the concept
        
        Returns:
            List of dependent concept names
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (con:Concept {name: $concept_name})<-[:PREREQUISITE_FOR]-(dependent:Concept)
                    RETURN dependent.name as name
                    """,
                    concept_name=concept_name
                )
                dependent = [record["name"] for record in result]
                return dependent
        except Exception as e:
            print(f"Error getting dependent concepts: {e}")
            return []
    
    def find_concept_path(self, concept1_name: str, concept2_name: str, max_depth: int = 5) -> Optional[List[str]]:
        """
        Find a path between two concepts through relationships
        Useful for understanding how concepts are connected
        
        Args:
            concept1_name: Starting concept
            concept2_name: Ending concept
            max_depth: Maximum relationship depth to search
        
        Returns:
            List of concept names forming a path, or None if no path found
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    f"""
                    MATCH path = shortestPath(
                        (con1:Concept {{name: $concept1_name}})-[*..{max_depth}]-(con2:Concept {{name: $concept2_name}})
                    )
                    RETURN [node in nodes(path) | node.name] as concept_path
                    """,
                    concept1_name=concept1_name,
                    concept2_name=concept2_name
                )
                record = result.single()
                if record:
                    return record["concept_path"]
            return None
        except Exception as e:
            print(f"Error finding concept path: {e}")
            return None
    
    def get_all_concepts(self) -> List[Dict]:
        """
        Get all concepts in the graph
        
        Returns:
            List of all concepts with their properties
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (con:Concept)
                    RETURN con.name as name,
                           con.definition as definition,
                           con.importance_level as importance,
                           con.first_mentioned_chapter as chapter
                    ORDER BY con.name
                    """
                )
                concepts = []
                for record in result:
                    concepts.append({
                        "name": record["name"],
                        "definition": record["definition"],
                        "importance": record["importance"],
                        "chapter": record["chapter"]
                    })
                return concepts
        except Exception as e:
            print(f"Error getting all concepts: {e}")
            return []
    
    def get_all_chapters(self) -> List[Dict]:
        """
        Get all chapters in the graph
        
        Returns:
            List of all chapters with their properties
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (c:Chapter)
                    RETURN c.chapter_num as num,
                           c.title as title,
                           c.summary as summary
                    ORDER BY c.chapter_num
                    """
                )
                chapters = []
                for record in result:
                    chapters.append({
                        "num": record["num"],
                        "title": record["title"],
                        "summary": record["summary"]
                    })
                return chapters
        except Exception as e:
            print(f"Error getting all chapters: {e}")
            return []
    
    # ==================== GRAPH STATISTICS ====================
    
    def get_graph_stats(self) -> Dict:
        """
        Get statistics about the graph
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (con:Concept)
                    WITH count(con) as concept_count
                    MATCH (c:Chapter)
                    WITH concept_count, count(c) as chapter_count
                    MATCH ()-[r]->()
                    RETURN concept_count, chapter_count, count(r) as relationship_count
                    """
                )
                record = result.single()
                if record:
                    return {
                        "total_concepts": record["concept_count"],
                        "total_chapters": record["chapter_count"],
                        "total_relationships": record["relationship_count"]
                    }
                return {"total_concepts": 0, "total_chapters": 0, "total_relationships": 0}
        except Exception as e:
            print(f"Error getting graph stats: {e}")
            return {}
    
    # ==================== UTILITY FUNCTIONS ====================
    
    def clear_graph(self) -> bool:
        """
        Delete all nodes and relationships from the graph
        WARNING: This deletes everything! Use carefully for testing/reset
        
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
            print("✓ Graph cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing graph: {e}")
            return False
    
    def concept_exists(self, concept_name: str) -> bool:
        """
        Check if a concept already exists in the graph
        
        Args:
            concept_name: Name of the concept
        
        Returns:
            True if exists, False otherwise
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (con:Concept {name: $concept_name}) RETURN count(con) > 0 as exists",
                    concept_name=concept_name
                )
                record = result.single()
                return record["exists"] if record else False
        except Exception as e:
            print(f"Error checking if concept exists: {e}")
            return False
    
    def chapter_exists(self, chapter_num: str) -> bool:
        """
        Check if a chapter already exists in the graph
        
        Args:
            chapter_num: Chapter number
        
        Returns:
            True if exists, False otherwise
        """
        try:
            with self.driver.session() as session:
                result = session.run(
                    "MATCH (c:Chapter {chapter_num: $chapter_num}) RETURN count(c) > 0 as exists",
                    chapter_num=chapter_num
                )
                record = result.single()
                return record["exists"] if record else False
        except Exception as e:
            print(f"Error checking if chapter exists: {e}")
            return False


# ==================== STANDALONE TESTING ====================

def test_neo4j_handler():
    """Test the Neo4j handler with sample data"""
    print("\n" + "="*60)
    print("NEO4J HANDLER TEST")
    print("="*60)
    
    handler = Neo4jHandler()
    
    try:
        # Clear any existing data
        print("\n1. Clearing graph...")
        handler.clear_graph()
        
        # Create chapters
        print("\n2. Creating chapter nodes...")
        handler.create_chapter_node("1", "Introduction to Operating Systems", "Basic OS concepts")
        handler.create_chapter_node("2", "Process Management", "Processes and their management")
        handler.create_chapter_node("3", "Concurrency and Synchronization", "Threading and synchronization")
        print("   ✓ Created 3 chapters")
        
        # Create concepts
        print("\n3. Creating concept nodes...")
        concepts_data = [
            ("Process", "A program in execution with its own memory space", "high", "1"),
            ("Thread", "A lightweight unit of execution within a process", "high", "2"),
            ("CPU Scheduler", "Determines which process gets CPU time", "high", "1"),
            ("Context Switch", "Switching execution from one process/thread to another", "medium", "1"),
            ("Mutex", "Mutual exclusion lock to prevent race conditions", "high", "3"),
            ("Semaphore", "Synchronization primitive with counter mechanism", "high", "3"),
            ("Race Condition", "Problem when multiple threads access shared data without synchronization", "high", "2"),
            ("Deadlock", "Circular wait where processes are stuck forever", "high", "3"),
        ]
        
        for name, definition, importance, chapter in concepts_data:
            handler.create_concept_node(name, definition, importance, chapter)
        print(f"   ✓ Created {len(concepts_data)} concepts")
        
        # Create CONTAINS relationships
        print("\n4. Creating CONTAINS relationships...")
        handler.create_contains_relationship("1", "Process")
        handler.create_contains_relationship("1", "CPU Scheduler")
        handler.create_contains_relationship("1", "Context Switch")
        handler.create_contains_relationship("2", "Thread")
        handler.create_contains_relationship("2", "Race Condition")
        handler.create_contains_relationship("3", "Mutex")
        handler.create_contains_relationship("3", "Semaphore")
        handler.create_contains_relationship("3", "Deadlock")
        print("   ✓ Created CONTAINS relationships")
        
        # Create PREREQUISITE_FOR relationships
        print("\n5. Creating PREREQUISITE_FOR relationships...")
        handler.create_prerequisite_relationship("Process", "Thread")
        handler.create_prerequisite_relationship("Context Switch", "Thread")
        handler.create_prerequisite_relationship("Race Condition", "Mutex")
        print("   ✓ Created PREREQUISITE_FOR relationships")
        
        # Create RELATES_TO relationships
        print("\n6. Creating RELATES_TO relationships...")
        handler.create_relates_to_relationship("Mutex", "Semaphore")
        handler.create_relates_to_relationship("CPU Scheduler", "Context Switch")
        print("   ✓ Created RELATES_TO relationships")
        
        # Create SOLVES_PROBLEM_OF relationships
        print("\n7. Creating SOLVES_PROBLEM_OF relationships...")
        handler.create_solves_problem_relationship("Mutex", "Race Condition")
        handler.create_solves_problem_relationship("Semaphore", "Race Condition")
        handler.create_solves_problem_relationship("Semaphore", "Deadlock")
        print("   ✓ Created SOLVES_PROBLEM_OF relationships")
        
        # Test queries
        print("\n8. Testing queries...")
        
        # Get all concepts
        all_concepts = handler.get_all_concepts()
        print(f"   ✓ Total concepts in graph: {len(all_concepts)}")
        
        # Get concepts in a chapter
        ch1_concepts = handler.get_concepts_in_chapter("1")
        print(f"   ✓ Concepts in Chapter 1: {[c['name'] for c in ch1_concepts]}")
        
        # Get related concepts
        mutex_related = handler.get_related_concepts("Mutex", "RELATES_TO")
        print(f"   ✓ Concepts related to Mutex: {mutex_related}")
        
        # Get prerequisites
        thread_prereqs = handler.get_prerequisites("Thread")
        print(f"   ✓ Prerequisites for Thread: {thread_prereqs}")
        
        # Get dependent concepts
        process_dependents = handler.get_dependent_concepts("Process")
        print(f"   ✓ Concepts that depend on Process: {process_dependents}")
        
        # Get graph stats
        stats = handler.get_graph_stats()
        print(f"\n9. Graph Statistics:")
        print(f"   - Total Chapters: {stats['total_chapters']}")
        print(f"   - Total Concepts: {stats['total_concepts']}")
        print(f"   - Total Relationships: {stats['total_relationships']}")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        
    finally:
        handler.close()


if __name__ == "__main__":
    test_neo4j_handler()