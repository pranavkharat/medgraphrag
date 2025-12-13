"""
Text-to-Cypher Engine with Self-Healing (v2)
=============================================
FIXED: Schema now matches actual database (Drug + INTERACTS_WITH only)
"""

from dataclasses import dataclass
from typing import Optional
import re
import logging
from enum import Enum

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class CypherResult:
    query: str
    results: list[dict]
    success: bool
    error: Optional[str] = None
    attempts: int = 1
    complexity: Optional[str] = None


# ==============================================================
# ACTUAL SCHEMA (Matches loaded Kaggle data)
# ==============================================================

GRAPH_SCHEMA = """
DATABASE SCHEMA:

Node Types:
- (:Drug) - Properties: name, drugbank_id, description, smiles, name_lower

Relationship Types:
- (:Drug)-[:INTERACTS_WITH]->(:Drug)
  Properties on INTERACTS_WITH:
  - severity: "major", "moderate", "minor", or "unknown"
  - description: Text explaining the interaction
  - action: Type of interaction effect
  - source: Data source (e.g., "DrugBank")

IMPORTANT QUERY PATTERNS:
1. Drug names are case-sensitive in the database. Always use toLower() for matching:
   WHERE toLower(d.name) CONTAINS toLower('aspirin')

2. INTERACTS_WITH is stored in one direction but represents bidirectional interaction.
   Use undirected pattern to find all interactions:
   MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)

3. For exact name matching:
   WHERE toLower(d.name) = toLower('Warfarin')

4. For partial/fuzzy matching:
   WHERE toLower(d.name) CONTAINS toLower('war')

DATABASE STATISTICS:
- Approximately 1,258 Drug nodes
- Approximately 161,771 INTERACTS_WITH relationships
"""


# ==============================================================
# FEW-SHOT EXAMPLES (Realistic for actual schema)
# ==============================================================

FEW_SHOT_EXAMPLES = """
Example 1: Check if two drugs interact
Question: Does Lisinopril interact with Metoprolol?
Cypher:
MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
WHERE toLower(d1.name) CONTAINS toLower('lisinopril')
  AND toLower(d2.name) CONTAINS toLower('metoprolol')
RETURN d1.name AS drug1, d2.name AS drug2, i.severity AS severity, i.description AS description
LIMIT 1

Example 2: Find all interactions for a single drug
Question: What drugs interact with Metformin?
Cypher:
MATCH (d:Drug)-[i:INTERACTS_WITH]-(other:Drug)
WHERE toLower(d.name) CONTAINS toLower('metformin')
RETURN d.name AS searched_drug, other.name AS interacts_with, i.severity AS severity, i.description AS description
ORDER BY CASE i.severity WHEN 'major' THEN 1 WHEN 'moderate' THEN 2 WHEN 'minor' THEN 3 ELSE 4 END
LIMIT 25

Example 3: Find only major interactions for a drug
Question: What are the major interactions with Lisinopril?
Cypher:
MATCH (d:Drug)-[i:INTERACTS_WITH]-(other:Drug)
WHERE toLower(d.name) CONTAINS toLower('lisinopril')
  AND i.severity = 'major'
RETURN d.name AS drug, other.name AS interacts_with, i.description AS description
LIMIT 20

Example 4: Count interactions for a drug
Question: How many drugs interact with Omeprazole?
Cypher:
MATCH (d:Drug)-[i:INTERACTS_WITH]-(other:Drug)
WHERE toLower(d.name) CONTAINS toLower('omeprazole')
RETURN d.name AS drug, count(DISTINCT other) AS interaction_count

Example 5: Check multiple drugs for interactions (polypharmacy)
Question: Do Lisinopril, Metformin, and Metoprolol interact with each other?
Cypher:
MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
WHERE toLower(d1.name) IN ['lisinopril', 'metformin', 'metoprolol']
  AND toLower(d2.name) IN ['lisinopril', 'metformin', 'metoprolol']
  AND d1.name < d2.name
RETURN d1.name AS drug1, d2.name AS drug2, i.severity AS severity, i.description AS description

Example 6: Find drugs with most interactions
Question: Which drugs have the most interactions?
Cypher:
MATCH (d:Drug)-[i:INTERACTS_WITH]-()
RETURN d.name AS drug, count(i) AS interaction_count
ORDER BY interaction_count DESC
LIMIT 10

Example 7: Search for a drug by partial name
Question: Find interactions for drugs containing "statin"
Cypher:
MATCH (d:Drug)-[i:INTERACTS_WITH]-(other:Drug)
WHERE toLower(d.name) CONTAINS 'statin'
RETURN DISTINCT d.name AS statin_drug, count(other) AS interaction_count
ORDER BY interaction_count DESC
LIMIT 10

Example 8: Get severity distribution for a drug
Question: What's the severity breakdown of Metoprolol interactions?
Cypher:
MATCH (d:Drug)-[i:INTERACTS_WITH]-()
WHERE toLower(d.name) CONTAINS toLower('metoprolol')
RETURN i.severity AS severity, count(*) AS count
ORDER BY count DESC
"""


# ==============================================================
# PROMPT TEMPLATES
# ==============================================================

CYPHER_GENERATION_PROMPT = PromptTemplate(
    input_variables=["schema", "examples", "question"],
    template="""You are a Neo4j Cypher expert for a drug interaction database.

TASK: Convert the natural language question into a valid Cypher query.

{schema}

CRITICAL RULES:
1. ONLY use the node type Drug and relationship type INTERACTS_WITH - nothing else exists
2. ALWAYS use toLower() for case-insensitive name matching
3. Use undirected relationship pattern: (d1)-[i:INTERACTS_WITH]-(d2)
4. Include LIMIT clause to prevent huge result sets
5. Return descriptive column aliases
6. If the question cannot be answered with this schema, return exactly: UNSUPPORTED_QUERY

EXAMPLES:
{examples}

USER QUESTION: {question}

Return ONLY the Cypher query, no explanations or markdown:"""
)


CYPHER_REPAIR_PROMPT = PromptTemplate(
    input_variables=["schema", "original_query", "error_message", "question"],
    template="""You are a Neo4j Cypher debugger. Fix this failed query.

{schema}

ORIGINAL QUESTION: {question}

FAILED QUERY:
{original_query}

ERROR:
{error_message}

COMMON FIXES:
- "Unknown function" → Only use built-in Cypher functions
- "Type mismatch" → Ensure toLower() is used on string properties
- "Variable not defined" → Check all variables are defined in MATCH
- Empty results → Try broader CONTAINS instead of exact match

Return ONLY the corrected Cypher query:"""
)


# ==============================================================
# TEXT-TO-CYPHER ENGINE
# ==============================================================

class TextToCypherEngine:
    """
    Converts natural language to Cypher with self-healing.
    
    Recruiter Talking Point:
    "I implemented a self-healing query engine that uses LLM-powered 
    error correction. When a generated Cypher query fails, the system 
    feeds the error message back to the LLM with schema context, 
    achieving automatic repair in up to 3 retry cycles."
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        llm_model: str = "llama3.2:3b",
        ollama_base_url: str = "http://localhost:11434",
        max_retries: int = 3,
        temperature: float = 0
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.llm = OllamaLLM(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=temperature
        )
        self.max_retries = max_retries
        self.generation_prompt = CYPHER_GENERATION_PROMPT
        self.repair_prompt = CYPHER_REPAIR_PROMPT
        
        # Test connection on init
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Neo4j connection works."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection verified")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
    
    def generate_cypher(self, question: str) -> str:
        """Generate Cypher from natural language."""
        prompt = self.generation_prompt.format(
            schema=GRAPH_SCHEMA,
            examples=FEW_SHOT_EXAMPLES,
            question=question
        )
        
        response = self.llm.invoke(prompt)
        cypher = self._clean_cypher(response)
        
        logger.info(f"Generated: {cypher[:100]}...")
        return cypher
    
    def _clean_cypher(self, response: str) -> str:
        """Extract clean Cypher from LLM response."""
        # Remove markdown code blocks
        response = re.sub(r'```(?:cypher)?\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Remove explanatory text
        lines = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('//') or line.startswith('#'):
                continue
            if any(line.lower().startswith(x) for x in ['note:', 'this query', 'explanation:']):
                break
            lines.append(line)
        
        return '\n'.join(lines).strip()
    
    def validate_cypher(self, cypher: str) -> tuple[bool, Optional[str]]:
        """Validate Cypher syntax using EXPLAIN."""
        if cypher == "UNSUPPORTED_QUERY":
            return False, "This question cannot be answered with the current database schema."
        
        if not cypher or len(cypher) < 10:
            return False, "Query too short or empty"
        
        # Check for dangerous operations
        dangerous = ['DELETE', 'DETACH', 'DROP', 'CREATE', 'SET', 'REMOVE', 'MERGE']
        for op in dangerous:
            if op in cypher.upper() and 'RETURN' not in cypher.upper():
                return False, f"Dangerous operation detected: {op}"
        
        try:
            with self.driver.session() as session:
                session.run(f"EXPLAIN {cypher}")
            return True, None
        except Exception as e:
            return False, str(e)
    
    def repair_cypher(self, original: str, error: str, question: str) -> str:
        """Attempt to repair failed Cypher."""
        prompt = self.repair_prompt.format(
            schema=GRAPH_SCHEMA,
            original_query=original,
            error_message=error,
            question=question
        )
        
        response = self.llm.invoke(prompt)
        repaired = self._clean_cypher(response)
        
        logger.info(f"Repaired: {repaired[:100]}...")
        return repaired
    
    def execute_cypher(self, cypher: str) -> tuple[list[dict], Optional[str]]:
        """Execute Cypher and return results."""
        try:
            with self.driver.session() as session:
                result = session.run(cypher)
                records = [dict(record) for record in result]
                return records, None
        except Exception as e:
            return [], str(e)
    
    def query(self, question: str) -> CypherResult:
        """
        Main entry: NL question → Cypher → Execute → Results
        With self-healing retry loop.
        """
        complexity = QueryRouter.classify(question)
        attempts = 0
        current_query = self.generate_cypher(question)
        last_error = None
        
        while attempts < self.max_retries:
            attempts += 1
            
            # Validate
            is_valid, val_error = self.validate_cypher(current_query)
            
            if not is_valid:
                logger.warning(f"Attempt {attempts} validation failed: {val_error}")
                last_error = val_error
                
                if attempts < self.max_retries:
                    current_query = self.repair_cypher(current_query, val_error, question)
                    continue
                else:
                    return CypherResult(
                        query=current_query,
                        results=[],
                        success=False,
                        error=val_error,
                        attempts=attempts,
                        complexity=complexity.value
                    )
            
            # Execute
            results, exec_error = self.execute_cypher(current_query)
            
            if exec_error:
                logger.warning(f"Attempt {attempts} execution failed: {exec_error}")
                last_error = exec_error
                
                if attempts < self.max_retries:
                    current_query = self.repair_cypher(current_query, exec_error, question)
                    continue
            else:
                return CypherResult(
                    query=current_query,
                    results=results,
                    success=True,
                    attempts=attempts,
                    complexity=complexity.value
                )
        
        return CypherResult(
            query=current_query,
            results=[],
            success=False,
            error=last_error,
            attempts=attempts,
            complexity=complexity.value
        )
    
    def close(self):
        self.driver.close()


# ==============================================================
# QUERY ROUTER
# ==============================================================

class QueryRouter:
    """Classifies query complexity for routing decisions."""
    
    COMPLEX_PATTERNS = [
        r'\b(and|with|plus|,)\s+\w+\s+(and|with|plus|,)',  # Multiple drugs
        r'all\s+interactions',
        r'most\s+(common|frequent|dangerous)',
        r'compare',
        r'between\s+\w+\s+and\s+\w+\s+and',  # 3+ drugs
    ]
    
    MODERATE_PATTERNS = [
        r'interact\w*\s+with',
        r'major|severe|dangerous',
        r'how\s+many',
        r'count',
        r'list\s+all',
    ]
    
    @classmethod
    def classify(cls, question: str) -> QueryComplexity:
        q = question.lower()
        
        for pattern in cls.COMPLEX_PATTERNS:
            if re.search(pattern, q):
                return QueryComplexity.COMPLEX
        
        for pattern in cls.MODERATE_PATTERNS:
            if re.search(pattern, q):
                return QueryComplexity.MODERATE
        
        return QueryComplexity.SIMPLE


# ==============================================================
# STANDALONE TEST
# ==============================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    engine = TextToCypherEngine(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password123")
    )
    
    questions = [
        "Does Lisinopril interact with Metoprolol?",
        "What are the major interactions with Metformin?",
        "How many drugs interact with Lisinopril?",
    ]
    
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = engine.query(q)
        print(f"Cypher: {result.query}")
        print(f"Success: {result.success} | Attempts: {result.attempts}")
        print(f"Results: {len(result.results)} records")
        if result.results:
            print(f"Sample: {result.results[0]}")
    
    engine.close()