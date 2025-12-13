"""
Patient Data Models
===================
Defines patient profile schema with clinical parameters for personalized risk assessment.

Clinical Parameters Included:
- Demographics (age, sex, weight, height)
- Renal function (eGFR, creatinine)
- Hepatic function (liver status)
- Special populations (pregnancy, lactation)
- Allergies and conditions
- Current medications
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
from datetime import date, datetime
import json


class Sex(Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class RenalFunction(Enum):
    """Kidney function categories based on eGFR (mL/min/1.73m²)"""
    NORMAL = "normal"              # eGFR >= 90
    MILD_IMPAIRMENT = "mild"       # eGFR 60-89
    MODERATE_IMPAIRMENT = "moderate"  # eGFR 30-59
    SEVERE_IMPAIRMENT = "severe"   # eGFR 15-29
    KIDNEY_FAILURE = "failure"     # eGFR < 15


class HepaticFunction(Enum):
    """Liver function categories"""
    NORMAL = "normal"
    MILD_IMPAIRMENT = "mild"       # Child-Pugh A
    MODERATE_IMPAIRMENT = "moderate"  # Child-Pugh B
    SEVERE_IMPAIRMENT = "severe"   # Child-Pugh C


class PregnancyStatus(Enum):
    NOT_PREGNANT = "not_pregnant"
    PREGNANT_T1 = "pregnant_trimester_1"
    PREGNANT_T2 = "pregnant_trimester_2"
    PREGNANT_T3 = "pregnant_trimester_3"
    LACTATING = "lactating"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class PatientProfile:
    """
    Complete patient profile for personalized drug interaction analysis.
    
    Recruiter Talking Point:
    "I modeled patient profiles with clinically relevant parameters - 
    renal function, hepatic status, pregnancy - that directly impact 
    drug metabolism and interaction severity. This enables personalized
    risk assessment, not one-size-fits-all recommendations."
    """
    
    # Identifiers
    patient_id: str
    first_name: str
    last_name: str
    
    # Demographics
    date_of_birth: date
    sex: Sex
    weight_kg: float
    height_cm: float
    
    # Organ Function
    renal_function: RenalFunction
    egfr: Optional[float] = None  # Actual eGFR value if known
    hepatic_function: HepaticFunction = HepaticFunction.NORMAL
    
    # Special Populations
    pregnancy_status: PregnancyStatus = PregnancyStatus.NOT_APPLICABLE
    
    # Allergies (list of drug names/classes)
    allergies: list[str] = field(default_factory=list)
    
    # Medical Conditions
    conditions: list[str] = field(default_factory=list)
    
    # Current Medications
    medications: list[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    @property
    def age(self) -> int:
        """Calculate age from date of birth."""
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )
    
    @property
    def bmi(self) -> float:
        """Calculate BMI."""
        height_m = self.height_cm / 100
        return round(self.weight_kg / (height_m ** 2), 1)
    
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"
    
    @property
    def age_category(self) -> str:
        """Categorize age for risk assessment."""
        if self.age < 18:
            return "pediatric"
        elif self.age < 65:
            return "adult"
        elif self.age < 80:
            return "elderly"
        else:
            return "very_elderly"
    
    @property
    def is_high_risk(self) -> bool:
        """Quick check if patient has high-risk factors."""
        high_risk_factors = [
            self.age >= 75,
            self.renal_function in [RenalFunction.SEVERE_IMPAIRMENT, RenalFunction.KIDNEY_FAILURE],
            self.hepatic_function in [HepaticFunction.MODERATE_IMPAIRMENT, HepaticFunction.SEVERE_IMPAIRMENT],
            self.pregnancy_status in [PregnancyStatus.PREGNANT_T1, PregnancyStatus.PREGNANT_T2, PregnancyStatus.PREGNANT_T3],
            len(self.medications) >= 5,  # Polypharmacy
        ]
        return any(high_risk_factors)
    
    def get_risk_factors(self) -> list[str]:
        """List all risk factors for this patient."""
        factors = []
        
        if self.age >= 75:
            factors.append(f"Advanced age ({self.age} years)")
        elif self.age >= 65:
            factors.append(f"Elderly ({self.age} years)")
        
        if self.renal_function == RenalFunction.SEVERE_IMPAIRMENT:
            factors.append(f"Severe kidney impairment (eGFR: {self.egfr or '<30'})")
        elif self.renal_function == RenalFunction.MODERATE_IMPAIRMENT:
            factors.append(f"Moderate kidney impairment (eGFR: {self.egfr or '30-59'})")
        elif self.renal_function == RenalFunction.KIDNEY_FAILURE:
            factors.append(f"Kidney failure (eGFR: {self.egfr or '<15'})")
        
        if self.hepatic_function != HepaticFunction.NORMAL:
            factors.append(f"Liver impairment ({self.hepatic_function.value})")
        
        if self.pregnancy_status not in [PregnancyStatus.NOT_PREGNANT, PregnancyStatus.NOT_APPLICABLE]:
            factors.append(f"Pregnancy status: {self.pregnancy_status.value}")
        
        if len(self.medications) >= 5:
            factors.append(f"Polypharmacy ({len(self.medications)} medications)")
        
        if len(self.allergies) > 0:
            factors.append(f"Drug allergies: {', '.join(self.allergies[:3])}")
        
        if self.bmi >= 35:
            factors.append(f"Obesity (BMI: {self.bmi})")
        elif self.bmi < 18.5:
            factors.append(f"Underweight (BMI: {self.bmi})")
        
        return factors
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "patient_id": self.patient_id,
            "first_name": self.first_name,
            "last_name": self.last_name,
            "date_of_birth": self.date_of_birth.isoformat(),
            "sex": self.sex.value,
            "weight_kg": self.weight_kg,
            "height_cm": self.height_cm,
            "renal_function": self.renal_function.value,
            "egfr": self.egfr,
            "hepatic_function": self.hepatic_function.value,
            "pregnancy_status": self.pregnancy_status.value,
            "allergies": self.allergies,
            "conditions": self.conditions,
            "medications": self.medications,
            "created_at": self.created_at.isoformat(),
            "notes": self.notes
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "PatientProfile":
        """Create from dictionary."""
        return cls(
            patient_id=data["patient_id"],
            first_name=data["first_name"],
            last_name=data["last_name"],
            date_of_birth=date.fromisoformat(data["date_of_birth"]),
            sex=Sex(data["sex"]),
            weight_kg=data["weight_kg"],
            height_cm=data["height_cm"],
            renal_function=RenalFunction(data["renal_function"]),
            egfr=data.get("egfr"),
            hepatic_function=HepaticFunction(data.get("hepatic_function", "normal")),
            pregnancy_status=PregnancyStatus(data.get("pregnancy_status", "not_applicable")),
            allergies=data.get("allergies", []),
            conditions=data.get("conditions", []),
            medications=data.get("medications", []),
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(),
            notes=data.get("notes", "")
        )


def calculate_egfr(age: int, sex: Sex, creatinine: float, weight_kg: float) -> float:
    """
    Calculate eGFR using Cockcroft-Gault equation.
    
    Args:
        age: Patient age in years
        sex: Patient sex
        creatinine: Serum creatinine in mg/dL
        weight_kg: Body weight in kg
    
    Returns:
        Estimated GFR in mL/min
    """
    egfr = ((140 - age) * weight_kg) / (72 * creatinine)
    if sex == Sex.FEMALE:
        egfr *= 0.85
    return round(egfr, 1)


def get_renal_category(egfr: float) -> RenalFunction:
    """Categorize renal function based on eGFR."""
    if egfr >= 90:
        return RenalFunction.NORMAL
    elif egfr >= 60:
        return RenalFunction.MILD_IMPAIRMENT
    elif egfr >= 30:
        return RenalFunction.MODERATE_IMPAIRMENT
    elif egfr >= 15:
        return RenalFunction.SEVERE_IMPAIRMENT
    else:
        return RenalFunction.KIDNEY_FAILURE