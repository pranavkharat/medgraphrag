"""
Polypharmacy Risk Analyzer with Patient Personalization
========================================================
Analyzes multiple medications for interactions, generates risk scores,
and provides layman's term explanations using LLM.

NEW: Personalizes risk based on patient factors:
- Age (elderly patients = higher risk)
- Kidney function (affects drug clearance)
- Liver function (affects drug metabolism)
- Pregnancy status (contraindications)
- Existing conditions
- Drug allergies

This is the "wow factor" component that differentiates from basic lookup.
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import logging
from neo4j import GraphDatabase
from langchain_ollama import OllamaLLM

if TYPE_CHECKING:
    from src.models.patient import PatientProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DrugInteraction:
    """Single drug-drug interaction."""
    drug1: str
    drug2: str
    severity: str
    description: str
    
    @property
    def severity_score(self) -> int:
        """Numeric severity for calculations."""
        return {"major": 3, "moderate": 2, "minor": 1}.get(self.severity, 0)


@dataclass
class DrugInfo:
    """Information about a single drug with layman explanation."""
    name: str
    found_in_db: bool
    interaction_count: int = 0
    layman_explanation: Optional[str] = None


@dataclass 
class PolypharmacyReport:
    """Complete analysis report for multiple medications."""
    drugs_analyzed: list[DrugInfo]
    interactions: list[DrugInteraction]
    total_interactions: int = 0
    major_count: int = 0
    moderate_count: int = 0
    minor_count: int = 0
    risk_score: float = 0.0
    risk_level: str = "Unknown"
    most_dangerous_pair: Optional[tuple[str, str, str]] = None
    summary: str = ""
    # NEW: Patient-specific fields
    patient_risk_factors: list[str] = field(default_factory=list)
    personalized_warnings: list[str] = field(default_factory=list)
    allergy_alerts: list[str] = field(default_factory=list)
    adjusted_risk_score: Optional[float] = None  # After patient factors
    
    def __post_init__(self):
        self.total_interactions = len(self.interactions)
        self.major_count = sum(1 for i in self.interactions if i.severity == "major")
        self.moderate_count = sum(1 for i in self.interactions if i.severity == "moderate")
        self.minor_count = sum(1 for i in self.interactions if i.severity == "minor")
        self._calculate_risk_score()
        self._find_most_dangerous()
    
    def _calculate_risk_score(self):
        """
        Calculate weighted risk score (0-100).
        Formula: (major*10 + moderate*5 + minor*1) / max_possible * 100
        """
        if not self.interactions:
            self.risk_score = 0.0
            self.risk_level = "Low"
            return
        
        raw_score = (self.major_count * 10) + (self.moderate_count * 5) + (self.minor_count * 1)
        # Normalize: assume max 10 drugs = 45 pairs, all major = 450 points
        max_possible = len(self.drugs_analyzed) * (len(self.drugs_analyzed) - 1) / 2 * 10
        max_possible = max(max_possible, 1)  # Avoid division by zero
        
        self.risk_score = min((raw_score / max_possible) * 100, 100)
        
        # Determine risk level
        if self.major_count >= 2 or self.risk_score >= 60:
            self.risk_level = "High"
        elif self.major_count == 1 or self.moderate_count >= 3 or self.risk_score >= 30:
            self.risk_level = "Moderate"
        else:
            self.risk_level = "Low"
    
    def _find_most_dangerous(self):
        """Find the most dangerous interaction pair."""
        major_interactions = [i for i in self.interactions if i.severity == "major"]
        if major_interactions:
            worst = major_interactions[0]
            self.most_dangerous_pair = (worst.drug1, worst.drug2, worst.description)


class PolypharmacyAnalyzer:
    """
    Analyzes multiple medications for interactions and generates
    human-readable explanations.
    
    Recruiter Talking Point:
    "I built a polypharmacy analyzer that evaluates entire medication
    regimens - not just drug pairs. It calculates a weighted risk score,
    identifies the most dangerous combinations, and uses an LLM to 
    generate plain-English explanations that patients can understand."
    """
    
    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        llm_model: str = "llama3.2:3b",
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.llm = OllamaLLM(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=0.3  # Slightly creative for natural explanations
        )
        self._verify_connection()
    
    def _verify_connection(self):
        """Verify Neo4j connection."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info("Neo4j connection verified")
        except Exception as e:
            logger.error(f"Neo4j connection failed: {e}")
            raise
    
    def find_drug_in_db(self, drug_name: str) -> tuple[bool, str]:
        """
        Check if drug exists and return the exact name from DB.
        Returns (found, exact_name).
        """
        query = """
        MATCH (d:Drug)
        WHERE toLower(d.name) CONTAINS toLower($name)
        RETURN d.name as name
        LIMIT 1
        """
        with self.driver.session() as session:
            result = session.run(query, {"name": drug_name})
            record = result.single()
            if record:
                return True, record["name"]
            return False, drug_name
    
    def get_pairwise_interactions(self, drug_names: list[str]) -> list[DrugInteraction]:
        """
        Get all interactions between the given drugs.
        """
        if len(drug_names) < 2:
            return []
        
        # Build query for all pairs
        query = """
        MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
        WHERE toLower(d1.name) IN $drug_names_lower
          AND toLower(d2.name) IN $drug_names_lower
          AND d1.name < d2.name
        RETURN d1.name AS drug1, d2.name AS drug2, 
               i.severity AS severity, i.description AS description
        """
        
        drug_names_lower = [d.lower() for d in drug_names]
        
        interactions = []
        with self.driver.session() as session:
            result = session.run(query, {"drug_names_lower": drug_names_lower})
            for record in result:
                interactions.append(DrugInteraction(
                    drug1=record["drug1"],
                    drug2=record["drug2"],
                    severity=record["severity"] or "unknown",
                    description=record["description"] or "No description available"
                ))
        
        return interactions
    
    def get_interaction_count(self, drug_name: str) -> int:
        """Get total interaction count for a single drug."""
        query = """
        MATCH (d:Drug)-[i:INTERACTS_WITH]-()
        WHERE toLower(d.name) CONTAINS toLower($name)
        RETURN count(i) as count
        """
        with self.driver.session() as session:
            result = session.run(query, {"name": drug_name})
            record = result.single()
            return record["count"] if record else 0
    
    def generate_drug_explanation(self, drug_name: str) -> str:
        """
        Use LLM to generate a layman's explanation of what a drug does.
        """
        prompt = f"""You are a helpful pharmacist explaining medications to patients in simple terms.

Explain what {drug_name} is in 2-3 sentences. Include:
1. What condition it treats (e.g., "blood pressure", "diabetes", "cholesterol")
2. How it works in simple terms
3. Common usage

Keep it simple - imagine explaining to a grandparent. No medical jargon.
Do NOT include warnings or side effects - just what it does.

Response:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM explanation failed for {drug_name}: {e}")
            return f"{drug_name} is a medication. Please consult your pharmacist for details."
    
    def generate_interaction_explanation(self, drug1: str, drug2: str, severity: str, technical_desc: str) -> str:
        """
        Use LLM to generate a layman's explanation of why two drugs interact.
        """
        prompt = f"""You are a helpful pharmacist explaining drug interactions to patients.

These two medications interact:
- Drug 1: {drug1}
- Drug 2: {drug2}
- Severity: {severity}
- Technical description: {technical_desc}

Explain this interaction in 2-3 simple sentences:
1. What happens when these two drugs are taken together
2. Why this matters for the patient
3. What they should do (talk to doctor, etc.)

Use simple words. No medical jargon. Be reassuring but clear about the risk.

Response:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM interaction explanation failed: {e}")
            return f"Taking {drug1} and {drug2} together may cause issues. Please consult your doctor."
    
    def generate_summary(self, report: PolypharmacyReport) -> str:
        """
        Generate an overall summary of the medication analysis.
        """
        drug_list = ", ".join([d.name for d in report.drugs_analyzed if d.found_in_db])
        
        prompt = f"""You are a helpful pharmacist providing a medication review summary.

Patient's medications: {drug_list}
Total drug interactions found: {report.total_interactions}
- Major (serious) interactions: {report.major_count}
- Moderate interactions: {report.moderate_count}  
- Minor interactions: {report.minor_count}
Overall risk level: {report.risk_level}

Write a 3-4 sentence summary for the patient:
1. Acknowledge what medications they're taking
2. Summarize the interaction findings in plain language
3. Give a clear recommendation (e.g., "looks safe" or "discuss with doctor")

Be warm and reassuring, but honest about risks. Use simple language.

Summary:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM summary generation failed: {e}")
            return f"Found {report.total_interactions} interactions. Risk level: {report.risk_level}. Please consult your healthcare provider."
    
    def analyze(self, medication_list: list[str], generate_explanations: bool = True) -> PolypharmacyReport:
        """
        Main entry point: Analyze a list of medications.
        
        Args:
            medication_list: List of drug names (e.g., ["Lisinopril", "Metformin"])
            generate_explanations: Whether to generate LLM explanations (slower but better)
        
        Returns:
            PolypharmacyReport with complete analysis
        """
        # Clean input
        medications = [m.strip() for m in medication_list if m.strip()]
        
        if not medications:
            return PolypharmacyReport(drugs_analyzed=[], interactions=[])
        
        # Step 1: Verify each drug exists and get info
        drugs_info = []
        verified_names = []
        
        for med in medications:
            found, exact_name = self.find_drug_in_db(med)
            interaction_count = self.get_interaction_count(med) if found else 0
            
            # Generate layman explanation if requested
            explanation = None
            if generate_explanations and found:
                explanation = self.generate_drug_explanation(exact_name)
            
            drug_info = DrugInfo(
                name=exact_name if found else med,
                found_in_db=found,
                interaction_count=interaction_count,
                layman_explanation=explanation
            )
            drugs_info.append(drug_info)
            
            if found:
                verified_names.append(exact_name)
        
        # Step 2: Get all pairwise interactions
        interactions = self.get_pairwise_interactions(verified_names)
        
        # Step 3: Generate layman explanations for interactions
        if generate_explanations:
            for interaction in interactions:
                interaction.layman_explanation = self.generate_interaction_explanation(
                    interaction.drug1,
                    interaction.drug2,
                    interaction.severity,
                    interaction.description
                )
        
        # Step 4: Build report
        report = PolypharmacyReport(
            drugs_analyzed=drugs_info,
            interactions=interactions
        )
        
        # Step 5: Generate overall summary
        if generate_explanations and verified_names:
            report.summary = self.generate_summary(report)
        
        return report
    
    def close(self):
        """Clean up resources."""
        self.driver.close()

    # ==============================================================
    # PATIENT-PERSONALIZED ANALYSIS
    # ==============================================================
    
    def analyze_for_patient(
        self, 
        patient: "PatientProfile",
        generate_explanations: bool = True
    ) -> PolypharmacyReport:
        """
        Analyze a patient's medications with personalized risk assessment.
        
        Takes into account:
        - Patient age (elderly = higher risk)
        - Kidney function (affects drug clearance)
        - Liver function (affects drug metabolism)
        - Pregnancy status
        - Known allergies
        - Existing conditions
        """
        # Get base analysis
        report = self.analyze(patient.medications, generate_explanations=False)
        
        # Add patient risk factors
        report.patient_risk_factors = patient.get_risk_factors()
        
        # Check for allergy conflicts
        report.allergy_alerts = self._check_allergies(patient.medications, patient.allergies)
        
        # Calculate adjusted risk score
        report.adjusted_risk_score = self._calculate_personalized_risk(
            base_score=report.risk_score,
            patient=patient,
            interaction_count=report.total_interactions
        )
        
        # Update risk level based on adjusted score
        if report.adjusted_risk_score >= 60 or len(report.allergy_alerts) > 0:
            report.risk_level = "Critical"
        elif report.adjusted_risk_score >= 45:
            report.risk_level = "High"
        elif report.adjusted_risk_score >= 25:
            report.risk_level = "Moderate"
        else:
            report.risk_level = "Low"
        
        # Generate personalized warnings
        report.personalized_warnings = self._generate_personalized_warnings(patient, report)
        
        # Generate personalized summary if explanations enabled
        if generate_explanations:
            report.summary = self._generate_patient_summary(patient, report)
            
            # Generate explanations for drugs and interactions
            for drug in report.drugs_analyzed:
                if drug.found_in_db:
                    drug.layman_explanation = self.generate_drug_explanation(drug.name)
            
            for interaction in report.interactions:
                interaction.layman_explanation = self.generate_interaction_explanation(
                    interaction.drug1,
                    interaction.drug2,
                    interaction.severity,
                    interaction.description
                )
        
        return report
    
    def _check_allergies(self, medications: list[str], allergies: list[str]) -> list[str]:
        """Check if any medications conflict with known allergies."""
        alerts = []
        allergy_lower = [a.lower() for a in allergies]
        
        for med in medications:
            med_lower = med.lower()
            for allergy in allergy_lower:
                if allergy in med_lower or med_lower in allergy:
                    alerts.append(f"⚠️ ALLERGY ALERT: {med} may conflict with known allergy to {allergy}")
        
        # Check drug class allergies
        class_mappings = {
            "penicillin": ["amoxicillin", "ampicillin", "penicillin"],
            "sulfa": ["sulfamethoxazole", "sulfasalazine"],
            "nsaids": ["ibuprofen", "naproxen", "meloxicam", "celecoxib"],
            "ace inhibitors": ["lisinopril", "enalapril", "ramipril", "benazepril"],
        }
        
        for allergy in allergy_lower:
            if allergy in class_mappings:
                for med in medications:
                    if med.lower() in class_mappings[allergy]:
                        alerts.append(f"⚠️ CLASS ALLERGY: {med} is in the {allergy} class - patient has documented allergy")
        
        return alerts
    
    def _calculate_personalized_risk(
        self, 
        base_score: float, 
        patient: "PatientProfile",
        interaction_count: int
    ) -> float:
        """
        Calculate personalized risk with TWO components:
        1. Interaction risk (from drug-drug interactions) 
        2. Patient vulnerability risk (from patient factors - ADDITIVE)
        
        This ensures high-risk patients get appropriate scores even 
        if their specific medications don't interact.
        """
        # ============================================================
        # COMPONENT 1: Patient Vulnerability Score (ADDITIVE)
        # These points are added regardless of interactions
        # ============================================================
        vulnerability_score = 0.0
        
        # Age risk
        if patient.age >= 85:
            vulnerability_score += 25
        elif patient.age >= 80:
            vulnerability_score += 20
        elif patient.age >= 75:
            vulnerability_score += 15
        elif patient.age >= 65:
            vulnerability_score += 8
        
        # Kidney function risk
        from src.models.patient import RenalFunction
        if patient.renal_function == RenalFunction.KIDNEY_FAILURE:
            vulnerability_score += 30
        elif patient.renal_function == RenalFunction.SEVERE_IMPAIRMENT:
            vulnerability_score += 20
        elif patient.renal_function == RenalFunction.MODERATE_IMPAIRMENT:
            vulnerability_score += 12
        elif patient.renal_function == RenalFunction.MILD_IMPAIRMENT:
            vulnerability_score += 5
        
        # Liver function risk
        from src.models.patient import HepaticFunction
        if patient.hepatic_function == HepaticFunction.SEVERE_IMPAIRMENT:
            vulnerability_score += 20
        elif patient.hepatic_function == HepaticFunction.MODERATE_IMPAIRMENT:
            vulnerability_score += 12
        elif patient.hepatic_function == HepaticFunction.MILD_IMPAIRMENT:
            vulnerability_score += 5
        
        # Pregnancy risk
        from src.models.patient import PregnancyStatus
        if patient.pregnancy_status in [
            PregnancyStatus.PREGNANT_T1, 
            PregnancyStatus.PREGNANT_T2, 
            PregnancyStatus.PREGNANT_T3
        ]:
            vulnerability_score += 15
        
        # Polypharmacy risk (more meds = higher baseline risk)
        med_count = len(patient.medications)
        if med_count >= 10:
            vulnerability_score += 20
        elif med_count >= 8:
            vulnerability_score += 15
        elif med_count >= 6:
            vulnerability_score += 10
        elif med_count >= 5:
            vulnerability_score += 5
        
        # Multiple conditions risk
        condition_count = len(patient.conditions)
        if condition_count >= 5:
            vulnerability_score += 10
        elif condition_count >= 3:
            vulnerability_score += 5
        
        # ============================================================
        # COMPONENT 2: Interaction Risk (MULTIPLICATIVE)
        # Base score from interactions, amplified by vulnerability
        # ============================================================
        if base_score > 0:
            # Vulnerable patients have amplified interaction risk
            vulnerability_multiplier = 1.0 + (vulnerability_score / 100)
            interaction_risk = base_score * vulnerability_multiplier
        else:
            interaction_risk = 0
        
        # ============================================================
        # FINAL SCORE: Vulnerability + Amplified Interactions
        # ============================================================
        final_score = vulnerability_score + interaction_risk
        
        return min(100, round(final_score, 1))
    
    def _generate_personalized_warnings(
        self, 
        patient: "PatientProfile", 
        report: PolypharmacyReport
    ) -> list[str]:
        """Generate patient-specific warnings."""
        warnings = []
        
        from src.models.patient import RenalFunction, PregnancyStatus
        
        # Kidney warnings
        if patient.renal_function in [RenalFunction.MODERATE_IMPAIRMENT, RenalFunction.SEVERE_IMPAIRMENT, RenalFunction.KIDNEY_FAILURE]:
            # Check for renally-cleared drugs
            renal_drugs = ["metformin", "lisinopril", "gabapentin", "pregabalin"]
            for med in patient.medications:
                if med.lower() in renal_drugs:
                    warnings.append(
                        f"⚠️ {med} is cleared by kidneys - dose adjustment may be needed "
                        f"(patient eGFR: {patient.egfr})"
                    )
        
        # Age warnings
        if patient.age >= 75:
            high_risk_elderly = ["benzodiazepines", "lorazepam", "diazepam", "zolpidem", "diphenhydramine"]
            for med in patient.medications:
                if med.lower() in high_risk_elderly:
                    warnings.append(
                        f"⚠️ {med} is on the Beers Criteria list - potentially inappropriate for elderly patients"
                    )
        
        # Pregnancy warnings
        if patient.pregnancy_status in [PregnancyStatus.PREGNANT_T1, PregnancyStatus.PREGNANT_T2, PregnancyStatus.PREGNANT_T3]:
            warnings.append("⚠️ Patient is pregnant - all medications should be reviewed for fetal safety")
            contraindicated = ["methotrexate", "warfarin", "isotretinoin", "statins", "ace inhibitors"]
            for med in patient.medications:
                if any(c in med.lower() for c in contraindicated):
                    warnings.append(f"🚨 CRITICAL: {med} is contraindicated in pregnancy")
        
        return warnings
    
    def _generate_patient_summary(
        self, 
        patient: "PatientProfile", 
        report: PolypharmacyReport
    ) -> str:
        """Generate a personalized summary using LLM."""
        
        risk_factors_str = ", ".join(report.patient_risk_factors) if report.patient_risk_factors else "None identified"
        allergy_str = ", ".join(patient.allergies) if patient.allergies else "None"
        conditions_str = ", ".join(patient.conditions) if patient.conditions else "None"
        
        prompt = f"""You are a helpful pharmacist providing a personalized medication review.

PATIENT PROFILE:
- Name: {patient.full_name}
- Age: {patient.age} years old
- Sex: {patient.sex.value}
- Weight: {patient.weight_kg} kg
- Kidney Function: {patient.renal_function.value} (eGFR: {patient.egfr})
- Liver Function: {patient.hepatic_function.value}
- Allergies: {allergy_str}
- Medical Conditions: {conditions_str}
- Risk Factors: {risk_factors_str}

MEDICATION ANALYSIS:
- Total medications: {len(patient.medications)}
- Drug interactions found: {report.total_interactions}
  - Major: {report.major_count}
  - Moderate: {report.moderate_count}
  - Minor: {report.minor_count}
- Adjusted Risk Score: {report.adjusted_risk_score}/100 ({report.risk_level})
- Allergy Alerts: {len(report.allergy_alerts)}

Write a 4-5 sentence personalized summary for this patient:
1. Address them by name
2. Acknowledge their specific health situation (age, kidney function, etc.)
3. Summarize the medication interaction findings
4. Give specific recommendations based on THEIR situation
5. Be warm but clear about any serious concerns

Use simple language a patient can understand.

Summary:"""
        
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM summary failed: {e}")
            return f"Analysis complete for {patient.full_name}. Found {report.total_interactions} interactions with adjusted risk score of {report.adjusted_risk_score}/100."


# ==============================================================
# STANDALONE TEST
# ==============================================================

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    analyzer = PolypharmacyAnalyzer(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"),
        llm_model="llama3.2:3b"
    )
    
    # Test 1: Basic analysis (no patient context)
    print("\n" + "="*60)
    print("TEST 1: Basic Polypharmacy Analysis")
    print("="*60)
    
    test_meds = ["Lisinopril", "Metformin", "Metoprolol", "Omeprazole"]
    report = analyzer.analyze(test_meds, generate_explanations=False)
    
    print(f"Medications: {', '.join(test_meds)}")
    print(f"Interactions: {report.total_interactions}")
    print(f"Risk Score: {report.risk_score:.1f}/100 ({report.risk_level})")
    
    # Test 2: Patient-personalized analysis
    print("\n" + "="*60)
    print("TEST 2: Patient-Personalized Analysis")
    print("="*60)
    
    from src.models.patient import (
        PatientProfile, Sex, RenalFunction, HepaticFunction, PregnancyStatus
    )
    from datetime import date
    
    test_patient = PatientProfile(
        patient_id="TEST001",
        first_name="Eleanor",
        last_name="Johnson",
        date_of_birth=date(1945, 3, 15),  # 79 years old
        sex=Sex.FEMALE,
        weight_kg=68.5,
        height_cm=162,
        renal_function=RenalFunction.MODERATE_IMPAIRMENT,
        egfr=42.0,
        hepatic_function=HepaticFunction.NORMAL,
        pregnancy_status=PregnancyStatus.NOT_APPLICABLE,
        allergies=["Penicillin", "Sulfa drugs"],
        conditions=["Hypertension", "Type 2 Diabetes", "Osteoarthritis"],
        medications=["Lisinopril", "Metformin", "Metoprolol", "Omeprazole"]
    )
    
    print(f"Patient: {test_patient.full_name}, {test_patient.age} years old")
    print(f"Kidney: {test_patient.renal_function.value} (eGFR: {test_patient.egfr})")
    print(f"Conditions: {', '.join(test_patient.conditions)}")
    print(f"Allergies: {', '.join(test_patient.allergies)}")
    
    patient_report = analyzer.analyze_for_patient(test_patient, generate_explanations=True)
    
    print(f"\n📊 RESULTS:")
    print(f"Base Risk Score: {patient_report.risk_score:.1f}")
    print(f"Adjusted Risk Score: {patient_report.adjusted_risk_score}/100")
    print(f"Risk Level: {patient_report.risk_level}")
    print(f"\n⚠️ Patient Risk Factors:")
    for rf in patient_report.patient_risk_factors:
        print(f"   • {rf}")
    print(f"\n🚨 Personalized Warnings:")
    for w in patient_report.personalized_warnings:
        print(f"   • {w}")
    print(f"\n💬 Summary:\n{patient_report.summary}")
    
    analyzer.close()