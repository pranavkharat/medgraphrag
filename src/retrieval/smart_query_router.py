"""
Smart Query Router for MedGraphRAG
==================================
Routes natural language questions to the appropriate handler:
1. Graph queries (drug interactions) → Text-to-Cypher
2. Medical knowledge queries → Direct LLM
3. Patient-context queries → Patient data + LLM

This enables answering a wide range of medication questions,
not just what's stored in the knowledge graph.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from enum import Enum

from langchain_ollama import OllamaLLM

if TYPE_CHECKING:
    from src.models.patient import PatientProfile
    from src.retrieval.text_to_cypher import TextToCypherEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries the system can handle."""
    DRUG_INTERACTION = "drug_interaction"      # Graph query
    DRUG_INFO = "drug_info"                    # LLM general knowledge
    CONDITION_WARNING = "condition_warning"    # LLM medical knowledge
    TIMING_ADMIN = "timing_admin"              # LLM dosing info
    SIDE_EFFECTS = "side_effects"              # LLM knowledge
    ALTERNATIVES = "alternatives"              # LLM + maybe graph
    PATIENT_SPECIFIC = "patient_specific"      # Patient context + LLM
    GENERAL_MEDICAL = "general_medical"        # LLM fallback


@dataclass
class QueryResult:
    """Result from smart query routing."""
    query_type: QueryType
    answer: str
    source: str  # "graph", "llm", "hybrid"
    cypher_query: Optional[str] = None
    confidence: str = "medium"  # low, medium, high
    disclaimer: bool = True


# ==============================================================
# QUERY CLASSIFICATION PATTERNS
# ==============================================================

INTERACTION_PATTERNS = [
    r"(?:does|do|can|will|would)\s+\w+\s+interact",
    r"interaction(?:s)?\s+(?:between|with|of)",
    r"(?:take|mix|combine)\s+\w+\s+(?:with|and)",
    r"safe\s+to\s+(?:take|combine|mix)",
    r"together\s+with",
    r"drug[- ]drug",
]

DRUG_INFO_PATTERNS = [
    r"what\s+(?:is|does)\s+\w+(?:\s+used\s+for)?",
    r"what\s+(?:is|are)\s+\w+\s+for",
    r"tell\s+me\s+about\s+\w+",
    r"explain\s+\w+",
    r"how\s+does\s+\w+\s+work",
    r"mechanism\s+of",
    r"what\s+kind\s+of\s+(?:drug|medication|medicine)",
]

CONDITION_WARNING_PATTERNS = [
    r"(?:what|which)\s+(?:should|to)\s+(?:a\s+)?(?:person|patient|someone)\s+with\s+\w+\s+avoid",
    r"avoid\s+(?:if|when|with)\s+\w+",
    r"safe\s+(?:for|with)\s+(?:diabetes|kidney|liver|heart|pregnancy)",
    r"(?:diabetes|diabetic|kidney|renal|liver|hepatic|pregnant|elderly)",
    r"contraindicated\s+(?:for|in|with)",
    r"(?:should|can)\s+(?:a\s+)?diabetic",
    r"risk(?:s|y)?\s+for\s+(?:elderly|seniors|older)",
]

TIMING_PATTERNS = [
    r"when\s+(?:should|to|do)\s+(?:i|you)\s+take",
    r"(?:take|taken)\s+(?:with|without)\s+food",
    r"morning\s+or\s+(?:night|evening)",
    r"before\s+or\s+after\s+(?:meals?|eating|food)",
    r"how\s+(?:often|many\s+times)",
    r"best\s+time\s+to\s+take",
    r"empty\s+stomach",
]

SIDE_EFFECT_PATTERNS = [
    r"side\s+effects?\s+of",
    r"adverse\s+(?:effects?|reactions?)",
    r"(?:what|any)\s+side\s+effects",
    r"(?:cause|causes)\s+(?:drowsiness|nausea|dizziness)",
    r"make\s+(?:me|you)\s+(?:sleepy|tired|dizzy)",
]

ALTERNATIVE_PATTERNS = [
    r"(?:what|any)\s+alternatives?\s+to",
    r"instead\s+of\s+\w+",
    r"substitute\s+for",
    r"replace\s+\w+\s+with",
    r"switch\s+from\s+\w+",
    r"other\s+(?:options|medications|drugs)",
]

PATIENT_SPECIFIC_PATTERNS = [
    r"(?:my|for\s+me|based\s+on\s+my)",
    r"given\s+my\s+(?:condition|age|kidney|liver)",
    r"considering\s+(?:my|that\s+i)",
    r"with\s+my\s+(?:history|conditions|allergies)",
    r"in\s+my\s+(?:case|situation)",
]


class SmartQueryRouter:
    """
    Routes natural language queries to appropriate handlers.
    
    Recruiter Talking Point:
    "I built a hybrid query system that intelligently routes questions
    to either the knowledge graph or the LLM based on query classification.
    Drug interaction questions go to Neo4j for precise answers, while
    medical knowledge questions use the LLM's training data. This gives
    users a conversational interface without sacrificing accuracy."
    """
    
    def __init__(
        self,
        text_to_cypher_engine: "TextToCypherEngine",
        llm_model: str = "llama3.2:3b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.cypher_engine = text_to_cypher_engine
        self.llm = OllamaLLM(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0.3
        )
    
    def classify_query(self, question: str) -> QueryType:
        """Classify the type of query."""
        q_lower = question.lower()
        
        # Check patterns in priority order
        for pattern in PATIENT_SPECIFIC_PATTERNS:
            if re.search(pattern, q_lower):
                return QueryType.PATIENT_SPECIFIC
        
        for pattern in INTERACTION_PATTERNS:
            if re.search(pattern, q_lower):
                return QueryType.DRUG_INTERACTION
        
        for pattern in CONDITION_WARNING_PATTERNS:
            if re.search(pattern, q_lower):
                return QueryType.CONDITION_WARNING
        
        for pattern in SIDE_EFFECT_PATTERNS:
            if re.search(pattern, q_lower):
                return QueryType.SIDE_EFFECTS
        
        for pattern in TIMING_PATTERNS:
            if re.search(pattern, q_lower):
                return QueryType.TIMING_ADMIN
        
        for pattern in ALTERNATIVE_PATTERNS:
            if re.search(pattern, q_lower):
                return QueryType.ALTERNATIVES
        
        for pattern in DRUG_INFO_PATTERNS:
            if re.search(pattern, q_lower):
                return QueryType.DRUG_INFO
        
        return QueryType.GENERAL_MEDICAL
    
    def route_query(
        self, 
        question: str, 
        patient: Optional["PatientProfile"] = None
    ) -> QueryResult:
        """Route query to appropriate handler."""
        query_type = self.classify_query(question)
        
        logger.info(f"Query classified as: {query_type.value}")
        
        if query_type == QueryType.DRUG_INTERACTION:
            return self._handle_interaction_query(question)
        
        elif query_type == QueryType.DRUG_INFO:
            return self._handle_drug_info_query(question)
        
        elif query_type == QueryType.CONDITION_WARNING:
            return self._handle_condition_warning_query(question)
        
        elif query_type == QueryType.TIMING_ADMIN:
            return self._handle_timing_query(question)
        
        elif query_type == QueryType.SIDE_EFFECTS:
            return self._handle_side_effects_query(question)
        
        elif query_type == QueryType.ALTERNATIVES:
            return self._handle_alternatives_query(question)
        
        elif query_type == QueryType.PATIENT_SPECIFIC:
            return self._handle_patient_specific_query(question, patient)
        
        else:
            return self._handle_general_query(question)
    
    # ==============================================================
    # QUERY HANDLERS
    # ==============================================================
    
    def _handle_interaction_query(self, question: str) -> QueryResult:
        """Handle drug interaction queries using the graph."""
        result = self.cypher_engine.query(question)
        
        if result.success and result.results:
            # Format results nicely
            answer = self._format_interaction_results(result.results)
            return QueryResult(
                query_type=QueryType.DRUG_INTERACTION,
                answer=answer,
                source="graph",
                cypher_query=result.query,
                confidence="high"
            )
        elif result.success:
            return QueryResult(
                query_type=QueryType.DRUG_INTERACTION,
                answer="No interactions found between these medications in our database. However, always consult your pharmacist or doctor for complete information.",
                source="graph",
                cypher_query=result.query,
                confidence="medium"
            )
        else:
            # Fallback to LLM if graph query fails
            return self._handle_general_query(question)
    
    def _format_interaction_results(self, results: list) -> str:
        """Format graph query results into readable text."""
        if not results:
            return "No interactions found."
        
        lines = []
        for r in results[:5]:
            drug1 = r.get('drug1') or r.get('drug') or r.get('d1.name', 'Drug 1')
            drug2 = r.get('drug2') or r.get('interacts_with') or r.get('d2.name', 'Drug 2')
            severity = r.get('severity', 'unknown')
            desc = r.get('description', 'No description available')
            
            severity_emoji = {'major': '🔴', 'moderate': '🟡', 'minor': '🟢'}.get(severity, '⚪')
            lines.append(f"{severity_emoji} **{drug1} ↔ {drug2}** ({severity})\n   {desc[:200]}")
        
        return "\n\n".join(lines)
    
    def _handle_drug_info_query(self, question: str) -> QueryResult:
        """Handle drug information queries using LLM."""
        prompt = f"""You are a helpful pharmacist assistant. Answer this question about medications clearly and accurately.

Question: {question}

Provide a helpful answer that includes:
1. What the medication is/does
2. What conditions it treats
3. How it works (in simple terms)

Keep your answer concise (3-4 sentences). Use simple language.

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return QueryResult(
            query_type=QueryType.DRUG_INFO,
            answer=answer.strip(),
            source="llm",
            confidence="medium"
        )
    
    def _handle_condition_warning_query(self, question: str) -> QueryResult:
        """Handle condition-based warning queries."""
        prompt = f"""You are a pharmacist providing medication safety information.

Question: {question}

Provide practical advice about:
1. Which types/classes of medications may be problematic for this condition
2. Why these medications can be risky
3. What to discuss with a doctor

Be specific about drug classes (e.g., "NSAIDs like ibuprofen" not just "some painkillers").
Keep it concise but informative (4-5 sentences).

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return QueryResult(
            query_type=QueryType.CONDITION_WARNING,
            answer=answer.strip(),
            source="llm",
            confidence="medium"
        )
    
    def _handle_timing_query(self, question: str) -> QueryResult:
        """Handle medication timing/administration queries."""
        prompt = f"""You are a pharmacist advising on medication administration.

Question: {question}

Provide practical advice about:
1. Best time to take the medication
2. Whether to take with or without food
3. Any important timing considerations

Keep it concise and actionable (3-4 sentences).

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return QueryResult(
            query_type=QueryType.TIMING_ADMIN,
            answer=answer.strip(),
            source="llm",
            confidence="medium"
        )
    
    def _handle_side_effects_query(self, question: str) -> QueryResult:
        """Handle side effects queries."""
        prompt = f"""You are a pharmacist explaining medication side effects.

Question: {question}

Explain:
1. Common side effects (most people experience)
2. Serious side effects to watch for
3. When to contact a doctor

Be balanced - don't scare the patient but be honest. Keep it to 4-5 sentences.

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return QueryResult(
            query_type=QueryType.SIDE_EFFECTS,
            answer=answer.strip(),
            source="llm",
            confidence="medium"
        )
    
    def _handle_alternatives_query(self, question: str) -> QueryResult:
        """Handle medication alternative queries."""
        prompt = f"""You are a pharmacist discussing medication alternatives.

Question: {question}

Provide information about:
1. What drug class this medication belongs to
2. Other medications in the same class that might be alternatives
3. Note that switching should always be discussed with a doctor

Be helpful but emphasize the need for medical guidance. Keep it to 4-5 sentences.

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return QueryResult(
            query_type=QueryType.ALTERNATIVES,
            answer=answer.strip(),
            source="llm",
            confidence="low"  # Lower confidence for alternatives
        )
    
    def _handle_patient_specific_query(
        self, 
        question: str, 
        patient: Optional["PatientProfile"]
    ) -> QueryResult:
        """Handle patient-specific queries using patient context."""
        
        if patient is None:
            return QueryResult(
                query_type=QueryType.PATIENT_SPECIFIC,
                answer="To answer patient-specific questions, please select a patient first in the Patient Analysis tab.",
                source="system",
                confidence="high",
                disclaimer=False
            )
        
        # Build patient context
        patient_context = f"""
Patient Profile:
- Age: {patient.age} years old
- Sex: {patient.sex.value}
- Kidney Function: {patient.renal_function.value} (eGFR: {patient.egfr})
- Liver Function: {patient.hepatic_function.value}
- Current Medications: {', '.join(patient.medications)}
- Medical Conditions: {', '.join(patient.conditions) or 'None listed'}
- Allergies: {', '.join(patient.allergies) or 'None listed'}
"""
        
        prompt = f"""You are a clinical pharmacist reviewing a patient's medications.

{patient_context}

Patient's Question: {question}

Provide personalized advice considering:
1. Their specific health conditions (age, kidney/liver function)
2. Their current medications
3. Any special precautions for their situation

Be specific to THIS patient. Address them warmly. Keep it to 4-5 sentences.

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return QueryResult(
            query_type=QueryType.PATIENT_SPECIFIC,
            answer=answer.strip(),
            source="hybrid",
            confidence="medium"
        )
    
    def _handle_general_query(self, question: str) -> QueryResult:
        """Handle general medical queries as fallback."""
        prompt = f"""You are a helpful pharmacist assistant answering medication questions.

Question: {question}

Provide a helpful, accurate answer. If this is outside your expertise or requires a doctor's input, say so clearly.

Keep it concise (3-4 sentences).

Answer:"""
        
        answer = self.llm.invoke(prompt)
        
        return QueryResult(
            query_type=QueryType.GENERAL_MEDICAL,
            answer=answer.strip(),
            source="llm",
            confidence="low"
        )


# ==============================================================
# EXAMPLE QUESTIONS FOR UI
# ==============================================================

EXAMPLE_QUESTIONS = {
    "Drug Interactions": [
        "Does Lisinopril interact with Metoprolol?",
        "What drugs interact with Metformin?",
        "Is it safe to take Omeprazole with my blood pressure medications?",
    ],
    "Drug Information": [
        "What is Metformin used for?",
        "How does Lisinopril work?",
        "What kind of medication is Atorvastatin?",
    ],
    "Condition Warnings": [
        "What medications should a diabetic patient avoid?",
        "Which drugs are risky for someone with kidney disease?",
        "What should elderly patients be careful with?",
    ],
    "Timing & Administration": [
        "When should I take Lisinopril?",
        "Should Metformin be taken with food?",
        "What's the best time to take a statin?",
    ],
    "Side Effects": [
        "What are the side effects of Metoprolol?",
        "Does Lisinopril cause a cough?",
        "Can Metformin cause stomach problems?",
    ],
    "Alternatives": [
        "What are alternatives to Lisinopril for blood pressure?",
        "What can I take instead of Metformin?",
    ],
}


# ==============================================================
# STANDALONE TEST
# ==============================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    from src.retrieval.text_to_cypher import TextToCypherEngine
    
    cypher_engine = TextToCypherEngine(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"),
        llm_model="llama3.2:3b"
    )
    
    router = SmartQueryRouter(
        text_to_cypher_engine=cypher_engine,
        llm_model="llama3.2:3b"
    )
    
    test_questions = [
        "Does Lisinopril interact with Metoprolol?",
        "What is Metformin used for?",
        "What medications should a diabetic avoid?",
        "When should I take Lisinopril?",
        "What are the side effects of Metoprolol?",
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = router.route_query(q)
        print(f"Type: {result.query_type.value}")
        print(f"Source: {result.source}")
        print(f"Answer: {result.answer[:200]}...")
    
    cypher_engine.close()