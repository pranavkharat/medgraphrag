"""
Synthetic Patient Data Generator
================================
Generates realistic patient profiles with medications for testing
the personalized risk assessment system.

Generates 500-1000 patients with:
- Realistic demographic distributions
- Age-appropriate conditions and medications
- Correlated risk factors (elderly → more conditions → more meds)
"""

import random
import json
import os
from datetime import date, datetime, timedelta
from typing import Optional
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.patient import (
    PatientProfile, Sex, RenalFunction, HepaticFunction, 
    PregnancyStatus, get_renal_category
)


# ==============================================================
# REALISTIC DATA POOLS
# ==============================================================

FIRST_NAMES_MALE = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Christopher", "Daniel", "Matthew", "Anthony", "Mark",
    "Donald", "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian",
    "George", "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan",
    "Jacob", "Gary", "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin",
    "Scott", "Brandon", "Benjamin", "Samuel", "Raymond", "Gregory", "Frank",
    "Alexander", "Patrick", "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Jose",
    "Adam", "Nathan", "Henry", "Douglas", "Zachary", "Peter", "Kyle"
]

FIRST_NAMES_FEMALE = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
    "Ashley", "Kimberly", "Emily", "Donna", "Michelle", "Dorothy", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura",
    "Cynthia", "Kathleen", "Amy", "Angela", "Shirley", "Anna", "Brenda",
    "Pamela", "Emma", "Nicole", "Helen", "Samantha", "Katherine", "Christine",
    "Debra", "Rachel", "Carolyn", "Janet", "Catherine", "Maria", "Heather",
    "Diane", "Ruth", "Julie", "Olivia", "Joyce", "Virginia", "Victoria"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Scott", "Torres", "Nguyen",
    "Hill", "Flores", "Green", "Adams", "Nelson", "Baker", "Hall", "Rivera",
    "Campbell", "Mitchell", "Carter", "Roberts", "Patel", "Shah", "Kim", "Chen",
    "Wong", "Singh", "Kumar", "Cohen", "Murphy", "O'Brien", "Sullivan"
]

# Conditions with age correlations
CONDITIONS_BY_AGE = {
    "any": ["Anxiety", "Depression", "GERD", "Allergic Rhinitis", "Asthma", "Migraine"],
    "adult": ["Hypertension", "Type 2 Diabetes", "Hyperlipidemia", "Obesity", "Sleep Apnea"],
    "elderly": ["Osteoarthritis", "Atrial Fibrillation", "Heart Failure", "COPD", 
                "Chronic Kidney Disease", "Osteoporosis", "Benign Prostatic Hyperplasia",
                "Hypothyroidism", "Coronary Artery Disease", "Dementia"]
}

# Drug allergies (common ones)
COMMON_ALLERGIES = [
    "Penicillin", "Sulfa drugs", "Aspirin", "NSAIDs", "Codeine", 
    "Morphine", "Latex", "Contrast dye", "ACE inhibitors"
]

# Medications mapped to conditions (using drugs that exist in your database)
CONDITION_MEDICATIONS = {
    "Hypertension": ["Lisinopril", "Amlodipine", "Metoprolol", "Losartan", "Hydrochlorothiazide"],
    "Type 2 Diabetes": ["Metformin", "Glipizide", "Sitagliptin", "Pioglitazone"],
    "Hyperlipidemia": ["Atorvastatin", "Simvastatin", "Rosuvastatin", "Pravastatin"],
    "GERD": ["Omeprazole", "Pantoprazole", "Famotidine", "Esomeprazole"],
    "Depression": ["Sertraline", "Fluoxetine", "Escitalopram", "Bupropion", "Venlafaxine"],
    "Anxiety": ["Sertraline", "Escitalopram", "Buspirone", "Lorazepam"],
    "Atrial Fibrillation": ["Metoprolol", "Diltiazem", "Digoxin", "Amiodarone"],
    "Heart Failure": ["Lisinopril", "Carvedilol", "Metoprolol", "Furosemide", "Spironolactone"],
    "COPD": ["Tiotropium", "Fluticasone", "Albuterol", "Budesonide"],
    "Asthma": ["Albuterol", "Fluticasone", "Montelukast", "Budesonide"],
    "Osteoarthritis": ["Acetaminophen", "Meloxicam", "Celecoxib", "Tramadol"],
    "Hypothyroidism": ["Levothyroxine"],
    "Osteoporosis": ["Alendronate", "Calcium", "Vitamin D"],
    "Chronic Kidney Disease": ["Lisinopril", "Losartan", "Sodium Bicarbonate"],
    "Benign Prostatic Hyperplasia": ["Tamsulosin", "Finasteride"],
    "Migraine": ["Sumatriptan", "Topiramate", "Propranolol"],
    "Sleep Apnea": [],  # Usually treated with CPAP, not meds
    "Obesity": ["Metformin", "Phentermine"],
    "Coronary Artery Disease": ["Atorvastatin", "Metoprolol", "Lisinopril", "Clopidogrel"],
    "Allergic Rhinitis": ["Cetirizine", "Loratadine", "Fluticasone nasal"],
    "Dementia": ["Donepezil", "Memantine"],
}


# ==============================================================
# GENERATOR CLASS
# ==============================================================

class SyntheticPatientGenerator:
    """
    Generates realistic synthetic patient data with correlated attributes.
    
    Key correlations:
    - Older patients → more conditions → more medications
    - Conditions → appropriate medications
    - Renal function declines with age
    - Pregnancy only in females of childbearing age
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed:
            random.seed(seed)
    
    def generate_patient(self, patient_id: str) -> PatientProfile:
        """Generate a single realistic patient profile."""
        
        # Sex (roughly 50/50)
        sex = random.choice([Sex.MALE, Sex.FEMALE])
        
        # Age distribution (weighted toward elderly for healthcare setting)
        age = self._generate_age()
        dob = date.today() - timedelta(days=age * 365 + random.randint(0, 364))
        
        # Name based on sex
        if sex == Sex.MALE:
            first_name = random.choice(FIRST_NAMES_MALE)
        else:
            first_name = random.choice(FIRST_NAMES_FEMALE)
        last_name = random.choice(LAST_NAMES)
        
        # Physical attributes (correlated with age/sex)
        weight, height = self._generate_body_stats(age, sex)
        
        # Organ function (correlated with age)
        egfr = self._generate_egfr(age)
        renal_function = get_renal_category(egfr)
        hepatic_function = self._generate_hepatic_function(age)
        
        # Pregnancy (only females 18-45)
        pregnancy_status = self._generate_pregnancy_status(sex, age)
        
        # Conditions (number increases with age)
        conditions = self._generate_conditions(age)
        
        # Medications (based on conditions)
        medications = self._generate_medications(conditions)
        
        # Allergies (10% chance of having any)
        allergies = self._generate_allergies()
        
        return PatientProfile(
            patient_id=patient_id,
            first_name=first_name,
            last_name=last_name,
            date_of_birth=dob,
            sex=sex,
            weight_kg=weight,
            height_cm=height,
            renal_function=renal_function,
            egfr=egfr,
            hepatic_function=hepatic_function,
            pregnancy_status=pregnancy_status,
            allergies=allergies,
            conditions=conditions,
            medications=medications,
            notes=f"Synthetic patient generated for testing"
        )
    
    def _generate_age(self) -> int:
        """Generate age with realistic healthcare population distribution."""
        # Weighted distribution - more elderly patients in healthcare
        roll = random.random()
        if roll < 0.05:  # 5% pediatric
            return random.randint(1, 17)
        elif roll < 0.25:  # 20% young adult
            return random.randint(18, 40)
        elif roll < 0.50:  # 25% middle-aged
            return random.randint(41, 64)
        elif roll < 0.80:  # 30% elderly
            return random.randint(65, 79)
        else:  # 20% very elderly
            return random.randint(80, 95)
    
    def _generate_body_stats(self, age: int, sex: Sex) -> tuple[float, float]:
        """Generate correlated weight and height."""
        if sex == Sex.MALE:
            height = random.gauss(175, 8)  # cm
            bmi = random.gauss(27, 5)  # Slightly overweight average
        else:
            height = random.gauss(162, 7)  # cm
            bmi = random.gauss(26, 5)
        
        # Adjust for age (elderly tend to be shorter, variable weight)
        if age > 70:
            height -= random.uniform(0, 5)
        
        height = max(140, min(200, height))  # Clamp
        weight = bmi * ((height / 100) ** 2)
        weight = max(40, min(150, weight))  # Clamp
        
        return round(weight, 1), round(height, 1)
    
    def _generate_egfr(self, age: int) -> float:
        """Generate eGFR that declines with age."""
        # Base eGFR decreases with age
        if age < 40:
            base_egfr = random.gauss(105, 15)
        elif age < 60:
            base_egfr = random.gauss(90, 15)
        elif age < 75:
            base_egfr = random.gauss(70, 20)
        else:
            base_egfr = random.gauss(55, 20)
        
        # 15% chance of significant kidney disease
        if random.random() < 0.15:
            base_egfr *= random.uniform(0.3, 0.7)
        
        return max(5, min(130, round(base_egfr, 1)))
    
    def _generate_hepatic_function(self, age: int) -> HepaticFunction:
        """Generate hepatic function - mostly normal."""
        roll = random.random()
        if roll < 0.90:
            return HepaticFunction.NORMAL
        elif roll < 0.96:
            return HepaticFunction.MILD_IMPAIRMENT
        elif roll < 0.99:
            return HepaticFunction.MODERATE_IMPAIRMENT
        else:
            return HepaticFunction.SEVERE_IMPAIRMENT
    
    def _generate_pregnancy_status(self, sex: Sex, age: int) -> PregnancyStatus:
        """Generate pregnancy status for females of childbearing age."""
        if sex == Sex.MALE:
            return PregnancyStatus.NOT_APPLICABLE
        
        if age < 18 or age > 50:
            return PregnancyStatus.NOT_APPLICABLE
        
        # 8% of women in childbearing age are pregnant/lactating in our sample
        roll = random.random()
        if roll < 0.02:
            return PregnancyStatus.PREGNANT_T1
        elif roll < 0.04:
            return PregnancyStatus.PREGNANT_T2
        elif roll < 0.06:
            return PregnancyStatus.PREGNANT_T3
        elif roll < 0.08:
            return PregnancyStatus.LACTATING
        else:
            return PregnancyStatus.NOT_PREGNANT
    
    def _generate_conditions(self, age: int) -> list[str]:
        """Generate conditions based on age."""
        conditions = []
        
        # Number of conditions increases with age
        if age < 30:
            num_conditions = random.choices([0, 1, 2], weights=[0.5, 0.35, 0.15])[0]
        elif age < 50:
            num_conditions = random.choices([0, 1, 2, 3], weights=[0.2, 0.4, 0.3, 0.1])[0]
        elif age < 70:
            num_conditions = random.choices([1, 2, 3, 4, 5], weights=[0.1, 0.25, 0.35, 0.2, 0.1])[0]
        else:
            num_conditions = random.choices([2, 3, 4, 5, 6, 7], weights=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05])[0]
        
        # Pool of conditions based on age
        available = CONDITIONS_BY_AGE["any"].copy()
        if age >= 40:
            available.extend(CONDITIONS_BY_AGE["adult"])
        if age >= 60:
            available.extend(CONDITIONS_BY_AGE["elderly"])
        
        # Sample without replacement
        num_conditions = min(num_conditions, len(available))
        conditions = random.sample(available, num_conditions)
        
        return conditions
    
    def _generate_medications(self, conditions: list[str]) -> list[str]:
        """Generate medications based on conditions."""
        medications = set()
        
        for condition in conditions:
            if condition in CONDITION_MEDICATIONS:
                available_meds = CONDITION_MEDICATIONS[condition]
                if available_meds:
                    # Usually 1-2 meds per condition
                    num_meds = random.choices([1, 2], weights=[0.7, 0.3])[0]
                    num_meds = min(num_meds, len(available_meds))
                    selected = random.sample(available_meds, num_meds)
                    medications.update(selected)
        
        return list(medications)
    
    def _generate_allergies(self) -> list[str]:
        """Generate drug allergies (10% of patients have any)."""
        if random.random() > 0.10:
            return []
        
        num_allergies = random.choices([1, 2, 3], weights=[0.7, 0.2, 0.1])[0]
        return random.sample(COMMON_ALLERGIES, min(num_allergies, len(COMMON_ALLERGIES)))
    
    def generate_batch(self, count: int, start_id: int = 1) -> list[PatientProfile]:
        """Generate multiple patients."""
        patients = []
        for i in range(count):
            patient_id = f"PAT{start_id + i:06d}"
            patients.append(self.generate_patient(patient_id))
        return patients


# ==============================================================
# FILE I/O
# ==============================================================

def save_patients(patients: list[PatientProfile], filepath: str):
    """Save patients to JSON file."""
    data = [p.to_dict() for p in patients]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"✅ Saved {len(patients)} patients to {filepath}")


def load_patients(filepath: str) -> list[PatientProfile]:
    """Load patients from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [PatientProfile.from_dict(p) for p in data]


# ==============================================================
# MAIN: Generate 750 patients
# ==============================================================

if __name__ == "__main__":
    print("="*60)
    print("SYNTHETIC PATIENT GENERATOR")
    print("="*60)
    
    generator = SyntheticPatientGenerator(seed=42)  # Reproducible
    
    # Generate 750 patients
    patients = generator.generate_batch(750)
    
    # Save to file
    output_path = "data/synthetic/patients.json"
    save_patients(patients, output_path)
    
    # Statistics
    print(f"\n📊 STATISTICS:")
    print(f"   Total patients: {len(patients)}")
    
    ages = [p.age for p in patients]
    print(f"   Age range: {min(ages)} - {max(ages)} years")
    print(f"   Average age: {sum(ages)/len(ages):.1f} years")
    
    high_risk = sum(1 for p in patients if p.is_high_risk)
    print(f"   High-risk patients: {high_risk} ({high_risk/len(patients)*100:.1f}%)")
    
    avg_meds = sum(len(p.medications) for p in patients) / len(patients)
    print(f"   Average medications: {avg_meds:.1f}")
    
    avg_conditions = sum(len(p.conditions) for p in patients) / len(patients)
    print(f"   Average conditions: {avg_conditions:.1f}")
    
    # Sample patient
    print(f"\n📋 SAMPLE PATIENT:")
    sample = patients[0]
    print(f"   Name: {sample.full_name}")
    print(f"   Age: {sample.age} | Sex: {sample.sex.value}")
    print(f"   Weight: {sample.weight_kg}kg | Height: {sample.height_cm}cm | BMI: {sample.bmi}")
    print(f"   Kidney: {sample.renal_function.value} (eGFR: {sample.egfr})")
    print(f"   Conditions: {', '.join(sample.conditions) or 'None'}")
    print(f"   Medications: {', '.join(sample.medications) or 'None'}")
    print(f"   Risk factors: {sample.get_risk_factors() or 'None'}")