"""
MedGraphRAG Evaluation Framework
================================
Comprehensive evaluation of system performance across multiple dimensions:
1. Drug interaction detection accuracy
2. Text-to-Cypher query success rate
3. Self-healing effectiveness
4. Patient risk scoring validation
5. Response time benchmarks

Generates a metrics report for documentation.

Run: python -m src.evaluation.run_evaluation
"""

import json
import time
import os
import sys
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase


# ==============================================================
# METRICS DATA STRUCTURES
# ==============================================================

@dataclass
class TestResult:
    """Result of a single test case."""
    test_id: str
    category: str
    question: str
    expected_interaction: bool
    actual_interaction: bool
    expected_severity: Optional[str]
    actual_severity: Optional[str]
    correct: bool
    severity_correct: Optional[bool]
    response_time_ms: float
    cypher_attempts: int
    error: Optional[str] = None


@dataclass
class EvaluationMetrics:
    """Aggregated evaluation metrics."""
    # Overall
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    overall_accuracy: float = 0.0
    
    # By category
    category_results: dict = field(default_factory=dict)
    
    # Query performance
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    
    # Self-healing
    first_attempt_success: int = 0
    required_retry: int = 0
    self_healing_rate: float = 0.0
    
    # Severity accuracy (for interaction tests)
    severity_tests: int = 0
    severity_correct: int = 0
    severity_accuracy: float = 0.0
    
    # Timestamps
    evaluation_date: str = ""
    evaluation_duration_sec: float = 0.0


# ==============================================================
# TEST CASE GENERATOR (Enhanced)
# ==============================================================

def generate_test_cases(neo4j_driver) -> list[dict]:
    """
    Generate test cases from the actual database.
    Creates a balanced set of positive and negative cases.
    """
    test_cases = []
    
    with neo4j_driver.session() as session:
        # ============================================================
        # CATEGORY 1: Direct Interactions (Should find interaction)
        # ============================================================
        print("Generating direct interaction test cases...")
        
        direct_query = """
        MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
        WHERE i.severity IS NOT NULL
        WITH d1, d2, i, rand() as r
        ORDER BY r
        LIMIT 20
        RETURN d1.name as drug1, d2.name as drug2, 
               i.severity as severity, i.description as description
        """
        result = session.run(direct_query)
        
        for idx, record in enumerate(result):
            test_cases.append({
                "id": f"DIRECT_{idx+1:03d}",
                "category": "direct_interaction",
                "question": f"Does {record['drug1']} interact with {record['drug2']}?",
                "drugs": [record['drug1'], record['drug2']],
                "expected_has_interaction": True,
                "expected_severity": record['severity'],
                "difficulty": "easy"
            })
        
        # ============================================================
        # CATEGORY 2: No Interaction (Should NOT find interaction)
        # ============================================================
        print("Generating no-interaction test cases...")
        
        # Get random drug pairs that DON'T interact
        no_interaction_query = """
        MATCH (d1:Drug), (d2:Drug)
        WHERE d1 <> d2 
          AND NOT (d1)-[:INTERACTS_WITH]-(d2)
          AND rand() < 0.001
        WITH d1, d2, rand() as r
        ORDER BY r
        LIMIT 15
        RETURN d1.name as drug1, d2.name as drug2
        """
        result = session.run(no_interaction_query)
        
        for idx, record in enumerate(result):
            test_cases.append({
                "id": f"NO_INT_{idx+1:03d}",
                "category": "no_interaction",
                "question": f"Does {record['drug1']} interact with {record['drug2']}?",
                "drugs": [record['drug1'], record['drug2']],
                "expected_has_interaction": False,
                "expected_severity": None,
                "difficulty": "easy"
            })
        
        # ============================================================
        # CATEGORY 3: Single Drug Query (List interactions)
        # ============================================================
        print("Generating single drug test cases...")
        
        single_query = """
        MATCH (d:Drug)-[i:INTERACTS_WITH]-()
        WITH d, count(i) as interaction_count
        WHERE interaction_count > 10
        WITH d, interaction_count, rand() as r
        ORDER BY r
        LIMIT 10
        RETURN d.name as drug, interaction_count
        """
        result = session.run(single_query)
        
        for idx, record in enumerate(result):
            test_cases.append({
                "id": f"SINGLE_{idx+1:03d}",
                "category": "single_drug_query",
                "question": f"What drugs interact with {record['drug']}?",
                "drugs": [record['drug']],
                "expected_has_interaction": True,
                "expected_count_min": 5,  # Should find at least 5
                "difficulty": "easy"
            })
        
        # ============================================================
        # CATEGORY 4: Severity Filter (Major interactions only)
        # ============================================================
        print("Generating severity filter test cases...")
        
        severity_query = """
        MATCH (d:Drug)-[i:INTERACTS_WITH {severity: 'major'}]-()
        WITH d, count(i) as major_count
        WHERE major_count > 3
        WITH d, major_count, rand() as r
        ORDER BY r
        LIMIT 5
        RETURN d.name as drug, major_count
        """
        result = session.run(severity_query)
        
        for idx, record in enumerate(result):
            test_cases.append({
                "id": f"SEVERITY_{idx+1:03d}",
                "category": "severity_filter",
                "question": f"What are the major interactions with {record['drug']}?",
                "drugs": [record['drug']],
                "expected_has_interaction": True,
                "expected_severity": "major",
                "difficulty": "moderate"
            })
    
    print(f"Generated {len(test_cases)} test cases")
    return test_cases


# ==============================================================
# EVALUATION RUNNER
# ==============================================================

class MedGraphRAGEvaluator:
    """
    Runs evaluation tests against the MedGraphRAG system.
    """
    
    def __init__(self):
        self.neo4j_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), 
                  os.getenv("NEO4J_PASSWORD", "password123"))
        )
        
        # Import components
        from src.retrieval.text_to_cypher import TextToCypherEngine
        self.cypher_engine = TextToCypherEngine(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"),
            llm_model="llama3.2:3b"
        )
        
        self.results: list[TestResult] = []
    
    def run_direct_query(self, drug1: str, drug2: str) -> tuple[bool, Optional[str], float]:
        """
        Run a direct Cypher query to check interaction (ground truth).
        Returns (has_interaction, severity, time_ms)
        """
        query = """
        MATCH (d1:Drug)-[i:INTERACTS_WITH]-(d2:Drug)
        WHERE toLower(d1.name) CONTAINS toLower($drug1)
          AND toLower(d2.name) CONTAINS toLower($drug2)
        RETURN i.severity as severity
        LIMIT 1
        """
        
        start = time.time()
        with self.neo4j_driver.session() as session:
            result = session.run(query, {"drug1": drug1, "drug2": drug2})
            record = result.single()
        elapsed = (time.time() - start) * 1000
        
        if record:
            return True, record["severity"], elapsed
        return False, None, elapsed
    
    def run_text_to_cypher_test(self, question: str) -> tuple[bool, Optional[str], float, int, Optional[str]]:
        """
        Run Text-to-Cypher query and evaluate results.
        Returns (has_results, severity, time_ms, attempts, error)
        """
        start = time.time()
        result = self.cypher_engine.query(question)
        elapsed = (time.time() - start) * 1000
        
        if not result.success:
            return False, None, elapsed, result.attempts, result.error
        
        if not result.results:
            return False, None, elapsed, result.attempts, None
        
        # Extract severity from results if present
        severity = None
        for r in result.results:
            if 'severity' in r:
                severity = r['severity']
                break
        
        return True, severity, elapsed, result.attempts, None
    
    def evaluate_test_case(self, test_case: dict) -> TestResult:
        """Evaluate a single test case."""
        
        category = test_case["category"]
        question = test_case["question"]
        expected_interaction = test_case.get("expected_has_interaction", True)
        expected_severity = test_case.get("expected_severity")
        
        # Run the test
        has_results, severity, time_ms, attempts, error = self.run_text_to_cypher_test(question)
        
        # Determine correctness
        if category in ["direct_interaction", "no_interaction"]:
            # Binary: did we correctly identify interaction/no-interaction?
            correct = (has_results == expected_interaction)
            severity_correct = None
            if expected_severity and severity:
                severity_correct = (severity == expected_severity)
        elif category == "single_drug_query":
            # Should return results
            correct = has_results
            severity_correct = None
        elif category == "severity_filter":
            # Should return results with correct severity
            correct = has_results
            severity_correct = (severity == expected_severity) if severity else False
        else:
            correct = has_results
            severity_correct = None
        
        return TestResult(
            test_id=test_case["id"],
            category=category,
            question=question,
            expected_interaction=expected_interaction,
            actual_interaction=has_results,
            expected_severity=expected_severity,
            actual_severity=severity,
            correct=correct,
            severity_correct=severity_correct,
            response_time_ms=time_ms,
            cypher_attempts=attempts,
            error=error
        )
    
    def run_evaluation(self, test_cases: list[dict], verbose: bool = True) -> EvaluationMetrics:
        """Run full evaluation and return metrics."""
        
        start_time = time.time()
        self.results = []
        
        print(f"\n{'='*60}")
        print(f"RUNNING EVALUATION - {len(test_cases)} test cases")
        print(f"{'='*60}\n")
        
        for i, test_case in enumerate(test_cases):
            if verbose:
                print(f"[{i+1}/{len(test_cases)}] {test_case['id']}: {test_case['question'][:50]}...")
            
            result = self.evaluate_test_case(test_case)
            self.results.append(result)
            
            if verbose:
                status = "✅ PASS" if result.correct else "❌ FAIL"
                print(f"         {status} ({result.response_time_ms:.0f}ms, {result.cypher_attempts} attempts)")
        
        # Calculate metrics
        metrics = self._calculate_metrics()
        metrics.evaluation_duration_sec = time.time() - start_time
        metrics.evaluation_date = datetime.now().isoformat()
        
        return metrics
    
    def _calculate_metrics(self) -> EvaluationMetrics:
        """Calculate aggregate metrics from results."""
        
        metrics = EvaluationMetrics()
        metrics.total_tests = len(self.results)
        metrics.passed_tests = sum(1 for r in self.results if r.correct)
        metrics.failed_tests = metrics.total_tests - metrics.passed_tests
        metrics.overall_accuracy = (metrics.passed_tests / metrics.total_tests * 100) if metrics.total_tests > 0 else 0
        
        # By category
        category_stats = defaultdict(lambda: {"total": 0, "passed": 0})
        for r in self.results:
            category_stats[r.category]["total"] += 1
            if r.correct:
                category_stats[r.category]["passed"] += 1
        
        metrics.category_results = {
            cat: {
                "total": stats["total"],
                "passed": stats["passed"],
                "accuracy": (stats["passed"] / stats["total"] * 100) if stats["total"] > 0 else 0
            }
            for cat, stats in category_stats.items()
        }
        
        # Response times
        times = [r.response_time_ms for r in self.results]
        metrics.avg_response_time_ms = sum(times) / len(times) if times else 0
        metrics.min_response_time_ms = min(times) if times else 0
        metrics.max_response_time_ms = max(times) if times else 0
        
        # Self-healing stats
        metrics.first_attempt_success = sum(1 for r in self.results if r.cypher_attempts == 1 and r.correct)
        metrics.required_retry = sum(1 for r in self.results if r.cypher_attempts > 1)
        retry_successes = sum(1 for r in self.results if r.cypher_attempts > 1 and r.correct)
        metrics.self_healing_rate = (retry_successes / metrics.required_retry * 100) if metrics.required_retry > 0 else 100
        
        # Severity accuracy
        severity_results = [r for r in self.results if r.severity_correct is not None]
        metrics.severity_tests = len(severity_results)
        metrics.severity_correct = sum(1 for r in severity_results if r.severity_correct)
        metrics.severity_accuracy = (metrics.severity_correct / metrics.severity_tests * 100) if metrics.severity_tests > 0 else 0
        
        return metrics
    
    def close(self):
        """Clean up resources."""
        self.neo4j_driver.close()
        self.cypher_engine.close()


# ==============================================================
# PATIENT RISK SCORING EVALUATION
# ==============================================================

def evaluate_patient_risk_scoring() -> dict:
    """
    Evaluate that patient risk scoring makes clinical sense.
    High-risk patients should get higher scores than low-risk patients.
    """
    print("\n" + "="*60)
    print("EVALUATING PATIENT RISK SCORING")
    print("="*60 + "\n")
    
    from src.models.patient import (
        PatientProfile, Sex, RenalFunction, HepaticFunction, PregnancyStatus
    )
    from src.analysis.polypharmacy_analyzer import PolypharmacyAnalyzer
    from datetime import date
    
    analyzer = PolypharmacyAnalyzer(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password123"),
        llm_model="llama3.2:3b"
    )
    
    # Create test patients with varying risk profiles
    test_patients = [
        # Low risk: Young, healthy
        PatientProfile(
            patient_id="TEST_LOW_1",
            first_name="Test", last_name="LowRisk",
            date_of_birth=date(1990, 1, 1),
            sex=Sex.MALE,
            weight_kg=75, height_cm=175,
            renal_function=RenalFunction.NORMAL,
            egfr=95,
            hepatic_function=HepaticFunction.NORMAL,
            pregnancy_status=PregnancyStatus.NOT_APPLICABLE,
            allergies=[],
            conditions=["Hypertension"],
            medications=["Lisinopril", "Metoprolol"]
        ),
        # Medium risk: Elderly, some issues
        PatientProfile(
            patient_id="TEST_MED_1",
            first_name="Test", last_name="MediumRisk",
            date_of_birth=date(1955, 1, 1),
            sex=Sex.FEMALE,
            weight_kg=68, height_cm=160,
            renal_function=RenalFunction.MILD_IMPAIRMENT,
            egfr=72,
            hepatic_function=HepaticFunction.NORMAL,
            pregnancy_status=PregnancyStatus.NOT_APPLICABLE,
            allergies=["Penicillin"],
            conditions=["Hypertension", "Type 2 Diabetes"],
            medications=["Lisinopril", "Metformin", "Metoprolol", "Omeprazole"]
        ),
        # High risk: Very elderly, kidney failure, many meds
        PatientProfile(
            patient_id="TEST_HIGH_1",
            first_name="Test", last_name="HighRisk",
            date_of_birth=date(1935, 1, 1),
            sex=Sex.MALE,
            weight_kg=60, height_cm=165,
            renal_function=RenalFunction.SEVERE_IMPAIRMENT,
            egfr=22,
            hepatic_function=HepaticFunction.MILD_IMPAIRMENT,
            pregnancy_status=PregnancyStatus.NOT_APPLICABLE,
            allergies=["Penicillin", "Sulfa"],
            conditions=["Heart Failure", "COPD", "Type 2 Diabetes", "CKD", "Atrial Fibrillation"],
            medications=["Lisinopril", "Metformin", "Metoprolol", "Furosemide", "Digoxin", "Omeprazole"]
        ),
    ]
    
    results = []
    
    for patient in test_patients:
        print(f"Testing {patient.patient_id} ({patient.age}yo, {patient.renal_function.value} kidney)...")
        report = analyzer.analyze_for_patient(patient, generate_explanations=False)
        
        results.append({
            "patient_id": patient.patient_id,
            "age": patient.age,
            "kidney": patient.renal_function.value,
            "med_count": len(patient.medications),
            "base_score": report.risk_score,
            "adjusted_score": report.adjusted_risk_score,
            "risk_level": report.risk_level,
            "interactions_found": report.total_interactions
        })
        
        print(f"   → Base: {report.risk_score:.1f}, Adjusted: {report.adjusted_risk_score}, Level: {report.risk_level}")
    
    # Validate: High risk patient should have higher score than low risk
    low_score = results[0]["adjusted_score"]
    med_score = results[1]["adjusted_score"]
    high_score = results[2]["adjusted_score"]
    
    risk_ordering_correct = (low_score <= med_score <= high_score)
    high_risk_identified = results[2]["risk_level"] in ["High", "Critical"]
    
    print(f"\n📊 Risk Scoring Validation:")
    print(f"   Low-risk score: {low_score}")
    print(f"   Medium-risk score: {med_score}")
    print(f"   High-risk score: {high_score}")
    print(f"   Risk ordering correct: {'✅' if risk_ordering_correct else '❌'}")
    print(f"   High-risk patient identified: {'✅' if high_risk_identified else '❌'}")
    
    analyzer.close()
    
    return {
        "patients_tested": len(test_patients),
        "risk_ordering_correct": risk_ordering_correct,
        "high_risk_identified": high_risk_identified,
        "scores": results
    }


# ==============================================================
# REPORT GENERATOR
# ==============================================================

def generate_report(metrics: EvaluationMetrics, patient_results: dict) -> str:
    """Generate a formatted evaluation report."""
    
    report = f"""
================================================================================
                    MEDGRAPHRAG EVALUATION REPORT
================================================================================

Evaluation Date: {metrics.evaluation_date}
Evaluation Duration: {metrics.evaluation_duration_sec:.1f} seconds

================================================================================
                         OVERALL PERFORMANCE
================================================================================

Total Test Cases:     {metrics.total_tests}
Passed:               {metrics.passed_tests}
Failed:               {metrics.failed_tests}
Overall Accuracy:     {metrics.overall_accuracy:.1f}%

================================================================================
                       ACCURACY BY CATEGORY
================================================================================
"""
    
    for category, stats in metrics.category_results.items():
        report += f"\n{category}:\n"
        report += f"   Tests: {stats['total']} | Passed: {stats['passed']} | Accuracy: {stats['accuracy']:.1f}%\n"
    
    report += f"""
================================================================================
                       RESPONSE TIME METRICS
================================================================================

Average Response Time:  {metrics.avg_response_time_ms:.0f} ms
Minimum Response Time:  {metrics.min_response_time_ms:.0f} ms
Maximum Response Time:  {metrics.max_response_time_ms:.0f} ms

================================================================================
                     TEXT-TO-CYPHER PERFORMANCE
================================================================================

First Attempt Success:  {metrics.first_attempt_success}/{metrics.total_tests} ({metrics.first_attempt_success/metrics.total_tests*100:.1f}%)
Required Retry:         {metrics.required_retry}
Self-Healing Rate:      {metrics.self_healing_rate:.1f}%

================================================================================
                     SEVERITY CLASSIFICATION
================================================================================

Severity Tests:         {metrics.severity_tests}
Correct Classifications: {metrics.severity_correct}
Severity Accuracy:      {metrics.severity_accuracy:.1f}%

================================================================================
                     PATIENT RISK SCORING
================================================================================

Patients Tested:        {patient_results['patients_tested']}
Risk Ordering Correct:  {'Yes ✅' if patient_results['risk_ordering_correct'] else 'No ❌'}
High-Risk Identified:   {'Yes ✅' if patient_results['high_risk_identified'] else 'No ❌'}

Risk Score Distribution:
"""
    
    for score_info in patient_results['scores']:
        report += f"   {score_info['patient_id']}: {score_info['adjusted_score']}/100 ({score_info['risk_level']})\n"
    
    report += """
================================================================================
                           SUMMARY
================================================================================
"""
    
    # Generate summary
    if metrics.overall_accuracy >= 80:
        report += "✅ System performing well with high accuracy.\n"
    elif metrics.overall_accuracy >= 60:
        report += "⚠️ System performing adequately but has room for improvement.\n"
    else:
        report += "❌ System needs improvement in query accuracy.\n"
    
    if metrics.self_healing_rate >= 70:
        report += "✅ Self-healing mechanism is effective.\n"
    else:
        report += "⚠️ Self-healing mechanism could be improved.\n"
    
    if patient_results['risk_ordering_correct'] and patient_results['high_risk_identified']:
        report += "✅ Patient risk scoring is clinically appropriate.\n"
    else:
        report += "⚠️ Patient risk scoring needs calibration.\n"
    
    report += "\n================================================================================\n"
    
    return report


# ==============================================================
# MAIN
# ==============================================================

def main():
    print("\n" + "="*60)
    print("      MEDGRAPHRAG EVALUATION FRAMEWORK")
    print("="*60)
    
    # Initialize evaluator
    evaluator = MedGraphRAGEvaluator()
    
    # Generate test cases
    print("\n📝 Generating test cases from database...")
    test_cases = generate_test_cases(evaluator.neo4j_driver)
    
    # Save test cases
    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/test_cases.json", "w") as f:
        json.dump(test_cases, f, indent=2)
    print(f"   Saved to data/evaluation/test_cases.json")
    
    # Run evaluation
    print("\n🧪 Running Text-to-Cypher evaluation...")
    metrics = evaluator.run_evaluation(test_cases, verbose=True)
    
    # Evaluate patient risk scoring
    patient_results = evaluate_patient_risk_scoring()
    
    # Generate report
    report = generate_report(metrics, patient_results)
    print(report)
    
    # Save report
    report_path = "data/evaluation/evaluation_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n📄 Report saved to {report_path}")
    
    # Save metrics as JSON
    metrics_dict = {
        "total_tests": metrics.total_tests,
        "passed_tests": metrics.passed_tests,
        "failed_tests": metrics.failed_tests,
        "overall_accuracy": metrics.overall_accuracy,
        "category_results": metrics.category_results,
        "avg_response_time_ms": metrics.avg_response_time_ms,
        "min_response_time_ms": metrics.min_response_time_ms,
        "max_response_time_ms": metrics.max_response_time_ms,
        "first_attempt_success": metrics.first_attempt_success,
        "required_retry": metrics.required_retry,
        "self_healing_rate": metrics.self_healing_rate,
        "severity_accuracy": metrics.severity_accuracy,
        "evaluation_date": metrics.evaluation_date,
        "evaluation_duration_sec": metrics.evaluation_duration_sec,
        "patient_risk_scoring": patient_results
    }
    
    with open("data/evaluation/metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"📊 Metrics saved to data/evaluation/metrics.json")
    
    evaluator.close()
    
    print("\n✅ Evaluation complete!")
    return metrics, patient_results


if __name__ == "__main__":
    main()