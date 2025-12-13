"""
Quick test script for MedGraphRAG Text-to-Cypher engine.
Run: python test_query.py
"""

import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

# Simple test without the full Text-to-Cypher engine
# This tests basic Neo4j connectivity and data

from neo4j import GraphDatabase


def test_neo4j_connection():
    """Test basic Neo4j connectivity and run sample queries."""
    
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    
    print(f"\n{'='*60}")
    print("MEDGRAPHRAG - CONNECTION TEST")
    print(f"{'='*60}")
    print(f"Connecting to: {uri}")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    try:
        with driver.session() as session:
            # Test 1: Count drugs
            result = session.run("MATCH (d:Drug) RETURN count(d) as count")
            drug_count = result.single()["count"]
            print(f"\n✅ Drugs in database: {drug_count:,}")
            
            # Test 2: Count interactions
            result = session.run("MATCH ()-[i:INTERACTS_WITH]->() RETURN count(i) as count")
            interaction_count = result.single()["count"]
            print(f"✅ Interactions in database: {interaction_count:,}")
            
            # Test 3: Sample interaction
            print(f"\n{'='*60}")
            print("SAMPLE INTERACTIONS")
            print(f"{'='*60}")
            
            result = session.run("""
                MATCH (d1:Drug)-[i:INTERACTS_WITH]->(d2:Drug)
                RETURN d1.name as drug1, d2.name as drug2, 
                       i.severity as severity, 
                       substring(i.description, 0, 100) as description
                LIMIT 5
            """)
            
            for record in result:
                print(f"\n• {record['drug1']} ↔ {record['drug2']}")
                print(f"  Severity: {record['severity']}")
                print(f"  Description: {record['description']}...")
            
            # Test 4: Search for specific drug
            print(f"\n{'='*60}")
            print("SEARCH TEST: Finding 'Aspirin' interactions")
            print(f"{'='*60}")
            
            result = session.run("""
                MATCH (d:Drug)-[i:INTERACTS_WITH]-(other:Drug)
                WHERE toLower(d.name) CONTAINS 'aspirin'
                RETURN d.name as drug, other.name as interacts_with, i.severity
                LIMIT 10
            """)
            
            records = list(result)
            if records:
                for record in records:
                    print(f"• Aspirin ↔ {record['interacts_with']} ({record['severity']})")
            else:
                # Try different search
                result = session.run("""
                    MATCH (d:Drug)
                    WHERE toLower(d.name) CONTAINS 'aspirin' 
                       OR toLower(d.name) CONTAINS 'ibuprofen'
                       OR toLower(d.name) CONTAINS 'warfarin'
                    RETURN d.name LIMIT 5
                """)
                found = [r['d.name'] for r in result]
                print(f"Aspirin not found. Sample drugs in DB: {found}")
            
            # Test 5: Show severity distribution
            print(f"\n{'='*60}")
            print("SEVERITY DISTRIBUTION")
            print(f"{'='*60}")
            
            result = session.run("""
                MATCH ()-[i:INTERACTS_WITH]->()
                RETURN i.severity as severity, count(*) as count
                ORDER BY count DESC
            """)
            
            for record in result:
                print(f"• {record['severity'] or 'unknown'}: {record['count']:,}")
            
            print(f"\n{'='*60}")
            print("✅ ALL TESTS PASSED!")
            print(f"{'='*60}")
            print("\nNext steps:")
            print("1. Run: streamlit run app/main.py")
            print("2. Open: http://localhost:8501")
            
    finally:
        driver.close()


def test_text_to_cypher():
    """Test the Text-to-Cypher engine (requires Ollama running)."""
    
    print(f"\n{'='*60}")
    print("TEXT-TO-CYPHER TEST")
    print(f"{'='*60}")
    
    try:
        from src.retrieval.text_to_cypher import TextToCypherEngine, QueryRouter
        
        engine = TextToCypherEngine(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"),
            llm_model="llama3.1:8b"
        )
        
        test_questions = [
            "What drugs interact with Aspirin?",
            "Show me major drug interactions",
            "Does Warfarin have any interactions?",
        ]
        
        for question in test_questions:
            print(f"\n📝 Question: {question}")
            print(f"   Complexity: {QueryRouter.classify(question).value}")
            
            result = engine.query(question)
            
            print(f"   Success: {result.success}")
            print(f"   Attempts: {result.attempts}")
            
            if result.success:
                print(f"   Results: {len(result.results)} found")
                if result.results:
                    print(f"   Sample: {result.results[0]}")
            else:
                print(f"   Error: {result.error}")
            
            print(f"\n   Generated Cypher:")
            print(f"   {result.query[:200]}...")
        
        engine.close()
        print(f"\n✅ Text-to-Cypher test complete!")
        
    except ImportError as e:
        print(f"⚠️  Could not import Text-to-Cypher engine: {e}")
        print("   Make sure src/retrieval/text_to_cypher.py exists")
    except Exception as e:
        print(f"⚠️  Text-to-Cypher test failed: {e}")
        print("   Make sure Ollama is running with llama3.2:3b model")


if __name__ == "__main__":
    # Test 1: Basic Neo4j connection
    test_neo4j_connection()
    
    # Test 2: Text-to-Cypher (optional - requires Ollama)
    print("\n" + "="*60)
    response = input("Test Text-to-Cypher engine? (requires Ollama) [y/N]: ")
    if response.lower() == 'y':
        test_text_to_cypher()