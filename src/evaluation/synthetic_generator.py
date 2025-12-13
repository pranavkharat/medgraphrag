"""
Synthetic Test Case Generator for MedGraphRAG
==============================================
Generates test cases for evaluating the drug interaction system.

This fulfills the "Synthetic Data Generation" assignment requirement.

Run: python -m src.evaluation.synthetic_generator
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
from dotenv import load_dotenv
from neo4j import GraphDatabase
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
load_dotenv()


@dataclass
class TestCase:
    """A single test case for evaluation."""
    id: str
    category: str
    question: str
    drugs_involved: list
    expected_has_interaction: bool
    expected_severity: Optional[str]
    difficulty: str
    notes: str


class SyntheticTestGenerator:
    """
    Generates synthetic test cases from the actual database.
    
    Categories:
    1. Direct interactions (known pairs from DB)
    2. No interaction (random pairs unlikely to interact)
    3. Multi-drug (3+ drugs)
    4. Adversarial (typos, variations)
    """
    
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        self.driver.close()
    
    def get_known_interactions(self, limit: int = 100) -> list:
        """Get known drug interactions from the database."""
        query = """
        MATCH (d1:Drug)-[i:INTERACTS_WITH]->(d2:Drug)
        RETURN d1.name as drug1, d2.name as drug2, 
               i.severity as severity, i.description as description
        LIMIT $limit
        """
        with self.driver.session() as session:
            result = session.run(query, limit=limit)
            return [dict(r) for r in result]
    
    def get_all_drugs(self) -> list:
        """Get all drug names."""
        query = "MATCH (d:Drug) RETURN d.name as name"
        with self.driver.session() as session:
            result = session.run(query)
            return [r['name'] for r in result]
    
    def generate_direct_interaction_tests(self, count: int = 20) -> list:
        """Generate test cases for known interactions."""
        interactions = self.get_known_interactions(count * 2)
        random.shuffle(interactions)
        
        test_cases = []
        question_templates = [
            "Does {drug1} interact with {drug2}?",
            "Is it safe to take {drug1} with {drug2}?",
            "What happens if I take {drug1} and {drug2} together?",
            "Can I combine {drug1} and {drug2}?",
            "Are there any interactions between {drug1} and {drug2}?",
        ]
        
        for i, interaction in enumerate(interactions[:count]):
            template = random.choice(question_templates)
            question = template.format(
                drug1=interaction['drug1'],
                drug2=interaction['drug2']
            )
            
            test_cases.append(TestCase(
                id=f"DIRECT_{i+1:03d}",
                category="direct_interaction",
                question=question,
                drugs_involved=[interaction['drug1'], interaction['drug2']],
                expected_has_interaction=True,
                expected_severity=interaction['severity'],
                difficulty="easy",
                notes=f"Known interaction from database"
            ))
        
        return test_cases
    
    def generate_no_interaction_tests(self, count: int = 10) -> list:
        """Generate test cases for drug pairs with no known interaction."""
        all_drugs = self.get_all_drugs()
        
        # Get drugs with few interactions (more likely to have no interaction with random pair)
        query = """
        MATCH (d:Drug)
        OPTIONAL MATCH (d)-[i:INTERACTS_WITH]-()
        WITH d.name as name, count(i) as interaction_count
        WHERE interaction_count < 10
        RETURN name
        LIMIT 50
        """
        with self.driver.session() as session:
            result = session.run(query)
            low_interaction_drugs = [r['name'] for r in result]
        
        test_cases = []
        used_pairs = set()
        
        for i in range(count):
            # Pick two random drugs from low-interaction list
            if len(low_interaction_drugs) >= 2:
                drug1, drug2 = random.sample(low_interaction_drugs, 2)
            else:
                drug1, drug2 = random.sample(all_drugs, 2)
            
            pair_key = tuple(sorted([drug1, drug2]))
            if pair_key in used_pairs:
                continue
            used_pairs.add(pair_key)
            
            # Verify no interaction exists
            check_query = """
            MATCH (d1:Drug {name: $drug1})-[i:INTERACTS_WITH]-(d2:Drug {name: $drug2})
            RETURN count(i) as count
            """
            with self.driver.session() as session:
                result = session.run(check_query, drug1=drug1, drug2=drug2)
                has_interaction = result.single()['count'] > 0
            
            if not has_interaction:
                test_cases.append(TestCase(
                    id=f"NO_INT_{i+1:03d}",
                    category="no_interaction",
                    question=f"Is there any interaction between {drug1} and {drug2}?",
                    drugs_involved=[drug1, drug2],
                    expected_has_interaction=False,
                    expected_severity=None,
                    difficulty="easy",
                    notes="No known interaction in database"
                ))
        
        return test_cases
    
    def generate_multi_drug_tests(self, count: int = 10) -> list:
        """Generate test cases involving 3+ drugs."""
        all_drugs = self.get_all_drugs()
        interactions = self.get_known_interactions(200)
        
        # Find drugs that appear in many interactions
        drug_counts = {}
        for inter in interactions:
            drug_counts[inter['drug1']] = drug_counts.get(inter['drug1'], 0) + 1
            drug_counts[inter['drug2']] = drug_counts.get(inter['drug2'], 0) + 1
        
        popular_drugs = sorted(drug_counts.keys(), key=lambda x: drug_counts[x], reverse=True)[:30]
        
        test_cases = []
        
        for i in range(count):
            num_drugs = random.randint(3, 5)
            selected_drugs = random.sample(popular_drugs, min(num_drugs, len(popular_drugs)))
            
            question_templates = [
                f"I'm taking {', '.join(selected_drugs[:-1])} and {selected_drugs[-1]}. Are there any interactions?",
                f"Check for interactions between: {', '.join(selected_drugs)}",
                f"Is it safe to combine {', '.join(selected_drugs)}?",
            ]
            
            test_cases.append(TestCase(
                id=f"MULTI_{i+1:03d}",
                category="polypharmacy",
                question=random.choice(question_templates),
                drugs_involved=selected_drugs,
                expected_has_interaction=True,  # Likely with popular drugs
                expected_severity="varies",
                difficulty="hard",
                notes=f"Multi-drug query with {num_drugs} drugs"
            ))
        
        return test_cases
    
    def generate_adversarial_tests(self, count: int = 10) -> list:
        """Generate adversarial test cases with typos and variations."""
        interactions = self.get_known_interactions(50)
        
        def add_typo(drug_name: str) -> str:
            """Add a realistic typo to drug name."""
            if len(drug_name) < 4:
                return drug_name
            
            typo_types = [
                lambda s: s[0].lower() + s[1:],  # lowercase first letter
                lambda s: s.replace('i', 'e', 1),  # common vowel swap
                lambda s: s.replace('a', 'e', 1),
                lambda s: s[:-1],  # missing last letter
                lambda s: s + 'e',  # extra letter
                lambda s: s[:len(s)//2] + s[len(s)//2:].lower(),  # partial lowercase
            ]
            
            return random.choice(typo_types)(drug_name)
        
        test_cases = []
        
        for i, interaction in enumerate(interactions[:count]):
            # Add typo to one drug
            drug1_typo = add_typo(interaction['drug1'])
            drug2 = interaction['drug2']
            
            test_cases.append(TestCase(
                id=f"ADV_{i+1:03d}",
                category="adversarial",
                question=f"Does {drug1_typo} interact with {drug2}?",
                drugs_involved=[interaction['drug1'], drug2],  # Store correct names
                expected_has_interaction=True,
                expected_severity=interaction['severity'],
                difficulty="hard",
                notes=f"Typo: '{drug1_typo}' should be '{interaction['drug1']}'"
            ))
        
        return test_cases
    
    def generate_all_tests(self) -> dict:
        """Generate complete test dataset."""
        print("Generating synthetic test cases...")
        
        direct = self.generate_direct_interaction_tests(20)
        print(f"  ✅ Generated {len(direct)} direct interaction tests")
        
        no_int = self.generate_no_interaction_tests(10)
        print(f"  ✅ Generated {len(no_int)} no-interaction tests")
        
        multi = self.generate_multi_drug_tests(10)
        print(f"  ✅ Generated {len(multi)} multi-drug tests")
        
        adversarial = self.generate_adversarial_tests(10)
        print(f"  ✅ Generated {len(adversarial)} adversarial tests")
        
        all_tests = direct + no_int + multi + adversarial
        
        return {
            "metadata": {
                "total_tests": len(all_tests),
                "categories": {
                    "direct_interaction": len(direct),
                    "no_interaction": len(no_int),
                    "polypharmacy": len(multi),
                    "adversarial": len(adversarial)
                }
            },
            "test_cases": [asdict(tc) for tc in all_tests]
        }


def main():
    """Generate and save synthetic test cases."""
    print("\n" + "="*60)
    print("MEDGRAPHRAG - SYNTHETIC TEST DATA GENERATOR")
    print("="*60 + "\n")
    
    # Create output directory
    output_dir = Path("data/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate tests
    generator = SyntheticTestGenerator()
    
    try:
        test_data = generator.generate_all_tests()
        
        # Save to JSON
        output_file = output_dir / "test_cases.json"
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print("✅ SYNTHETIC TEST DATA GENERATED!")
        print(f"{'='*60}")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 Total test cases: {test_data['metadata']['total_tests']}")
        print(f"\nBreakdown:")
        for category, count in test_data['metadata']['categories'].items():
            print(f"  • {category}: {count}")
        
        # Show sample
        print(f"\n📝 Sample test case:")
        sample = test_data['test_cases'][0]
        print(f"  Question: {sample['question']}")
        print(f"  Expected: interaction={sample['expected_has_interaction']}, severity={sample['expected_severity']}")
        
    finally:
        generator.close()


if __name__ == "__main__":
    main()