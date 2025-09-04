"""
Medical Text Data Generator for NLP Training
Generates synthetic medical texts including clinical notes, discharge summaries, and patient feedback.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json

class MedicalTextGenerator:
    """Generate synthetic medical text data for NLP training and analysis."""
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with medical vocabulary and templates."""
        random.seed(seed)
        np.random.seed(seed)
        
        # Medical vocabularies
        self.conditions = [
            "hypertension", "diabetes mellitus", "asthma", "chronic obstructive pulmonary disease",
            "coronary artery disease", "atrial fibrillation", "heart failure", "pneumonia",
            "urinary tract infection", "osteoarthritis", "depression", "anxiety disorder",
            "chronic kidney disease", "hypothyroidism", "gastroesophageal reflux disease",
            "migraine", "osteoporosis", "allergic rhinitis", "anemia", "hyperlipidemia"
        ]
        
        self.medications = [
            "lisinopril", "metformin", "albuterol", "atorvastatin", "amlodipine",
            "omeprazole", "levothyroxine", "metoprolol", "losartan", "hydrochlorothiazide",
            "gabapentin", "sertraline", "prednisone", "furosemide", "warfarin",
            "insulin", "aspirin", "ibuprofen", "acetaminophen", "cephalexin"
        ]
        
        self.symptoms = [
            "chest pain", "shortness of breath", "fatigue", "dizziness", "nausea",
            "headache", "fever", "cough", "abdominal pain", "back pain",
            "joint pain", "muscle weakness", "palpitations", "swelling",
            "difficulty breathing", "confusion", "weight loss", "weight gain"
        ]
        
        self.procedures = [
            "echocardiogram", "chest x-ray", "CT scan", "MRI", "blood glucose test",
            "electrocardiogram", "colonoscopy", "endoscopy", "ultrasound",
            "stress test", "pulmonary function test", "bone density scan"
        ]
        
        # Sentiment indicators
        self.positive_indicators = [
            "excellent", "good", "satisfied", "helpful", "professional",
            "caring", "thorough", "understanding", "effective", "improved"
        ]
        
        self.negative_indicators = [
            "poor", "terrible", "dissatisfied", "rude", "unprofessional",
            "rushed", "dismissive", "ineffective", "worse", "disappointed"
        ]
    
    def generate_clinical_note(self) -> Dict:
        """Generate a synthetic clinical note."""
        patient_age = random.randint(18, 95)
        gender = random.choice(["Male", "Female"])
        
        # Chief complaint
        symptom = random.choice(self.symptoms)
        duration = random.choice(["2 days", "1 week", "3 weeks", "2 months", "6 months"])
        chief_complaint = f"Patient presents with {symptom} for {duration}."
        
        # History of present illness
        condition = random.choice(self.conditions)
        severity = random.choice(["mild", "moderate", "severe"])
        hpi = f"{patient_age}-year-old {gender.lower()} with history of {condition} presents with {severity} {symptom}. "
        
        additional_symptoms = random.sample(self.symptoms, random.randint(1, 3))
        hpi += f"Associated symptoms include {', '.join(additional_symptoms)}. "
        
        # Assessment and plan
        medications = random.sample(self.medications, random.randint(1, 3))
        procedures = random.sample(self.procedures, random.randint(0, 2))
        
        assessment = f"Assessment: {condition} with acute exacerbation. "
        plan = f"Plan: Started on {', '.join(medications)}. "
        if procedures:
            plan += f"Ordered {', '.join(procedures)}. "
        plan += "Follow up in 2 weeks."
        
        note_text = f"{chief_complaint} {hpi}{assessment}{plan}"
        
        return {
            'text': note_text,
            'note_type': 'clinical_note',
            'patient_age': patient_age,
            'gender': gender,
            'primary_condition': condition,
            'medications': medications,
            'procedures': procedures,
            'sentiment': 'neutral'
        }
    
    def generate_discharge_summary(self) -> Dict:
        """Generate a synthetic discharge summary."""
        patient_age = random.randint(18, 95)
        gender = random.choice(["Male", "Female"])
        
        # Admission details
        condition = random.choice(self.conditions)
        los = random.randint(1, 14)  # Length of stay
        
        admission = f"ADMISSION DIAGNOSIS: {condition.title()}\n"
        admission += f"DISCHARGE DIAGNOSIS: {condition.title()}\n"
        admission += f"LENGTH OF STAY: {los} days\n\n"
        
        # Hospital course
        medications = random.sample(self.medications, random.randint(2, 5))
        procedures = random.sample(self.procedures, random.randint(1, 3))
        
        course = f"HOSPITAL COURSE: {patient_age}-year-old {gender.lower()} admitted with {condition}. "
        course += f"Patient underwent {', '.join(procedures)}. "
        course += f"Treatment included {', '.join(medications)}. "
        
        outcome = random.choice([
            "Patient responded well to treatment.",
            "Symptoms gradually improved during hospitalization.",
            "Patient showed significant improvement.",
            "Condition stabilized with medical management."
        ])
        course += outcome
        
        # Discharge instructions
        instructions = "\n\nDISCHARGE INSTRUCTIONS:\n"
        instructions += f"Continue {random.choice(medications)} as prescribed. "
        instructions += "Follow up with primary care physician in 1-2 weeks. "
        instructions += "Return to ED if symptoms worsen."
        
        summary_text = admission + course + instructions
        
        return {
            'text': summary_text,
            'note_type': 'discharge_summary',
            'patient_age': patient_age,
            'gender': gender,
            'primary_condition': condition,
            'length_of_stay': los,
            'medications': medications,
            'procedures': procedures,
            'sentiment': 'neutral'
        }
    
    def generate_patient_feedback(self) -> Dict:
        """Generate synthetic patient feedback with sentiment."""
        sentiment = random.choice(['positive', 'negative', 'neutral'])
        
        if sentiment == 'positive':
            descriptors = random.sample(self.positive_indicators, random.randint(2, 4))
            feedback_templates = [
                f"The staff was {descriptors[0]} and {descriptors[1]}. Very {descriptors[2]} experience.",
                f"Dr. Smith was {descriptors[0]} and {descriptors[1]}. I felt {descriptors[2]} throughout my visit.",
                f"Outstanding service. The team was {descriptors[0]} and {descriptors[1]}. Highly recommend.",
                f"Great experience. The nurse was {descriptors[0]} and the doctor was {descriptors[1]}."
            ]
        elif sentiment == 'negative':
            descriptors = random.sample(self.negative_indicators, random.randint(2, 4))
            feedback_templates = [
                f"The service was {descriptors[0]}. Staff seemed {descriptors[1]} and {descriptors[2]}.",
                f"Very {descriptors[0]} experience. The doctor was {descriptors[1]} and made me feel {descriptors[2]}.",
                f"Disappointed with the care. Staff was {descriptors[0]} and {descriptors[1]}.",
                f"Would not recommend. The service was {descriptors[0]} and unprofessional."
            ]
        else:  # neutral
            feedback_templates = [
                "Average experience. Nothing particularly good or bad to report.",
                "Standard care received. Met basic expectations.",
                "Routine visit completed. No major issues or highlights.",
                "Adequate service provided. Room for improvement in some areas."
            ]
            descriptors = ["adequate", "standard"]
        
        feedback_text = random.choice(feedback_templates)
        
        # Add specific details
        department = random.choice([
            "Emergency Department", "Cardiology", "Internal Medicine", 
            "Orthopedics", "Radiology", "Laboratory"
        ])
        
        wait_time = random.randint(15, 180)  # minutes
        feedback_text += f" Visit to {department}. Wait time was approximately {wait_time} minutes."
        
        return {
            'text': feedback_text,
            'note_type': 'patient_feedback',
            'sentiment': sentiment,
            'department': department,
            'wait_time_minutes': wait_time,
            'sentiment_score': 0.8 if sentiment == 'positive' else (-0.6 if sentiment == 'negative' else 0.1)
        }
    
    def generate_progress_note(self) -> Dict:
        """Generate a synthetic progress note."""
        patient_age = random.randint(18, 95)
        gender = random.choice(["Male", "Female"])
        day_of_stay = random.randint(1, 10)
        
        condition = random.choice(self.conditions)
        
        # Subjective
        improvement = random.choice([
            "feeling better", "symptoms improving", "pain decreasing",
            "breathing easier", "energy returning", "appetite improved"
        ])
        subjective = f"Day {day_of_stay}: Patient reports {improvement}. "
        
        # Objective
        vitals = {
            'bp_systolic': random.randint(110, 160),
            'bp_diastolic': random.randint(70, 100),
            'heart_rate': random.randint(60, 100),
            'temp': round(random.uniform(97.0, 101.5), 1),
            'resp_rate': random.randint(12, 24)
        }
        
        objective = f"Vitals: BP {vitals['bp_systolic']}/{vitals['bp_diastolic']}, "
        objective += f"HR {vitals['heart_rate']}, Temp {vitals['temp']}°F, RR {vitals['resp_rate']}. "
        
        # Assessment and plan
        medication = random.choice(self.medications)
        assessment = f"Assessment: {condition} - stable, improving. "
        plan = f"Plan: Continue current {medication}. Monitor symptoms. "
        
        if random.random() > 0.7:
            plan += f"Consider discharge if continues to improve."
        
        note_text = subjective + objective + assessment + plan
        
        return {
            'text': note_text,
            'note_type': 'progress_note',
            'patient_age': patient_age,
            'gender': gender,
            'day_of_stay': day_of_stay,
            'primary_condition': condition,
            'vitals': vitals,
            'sentiment': 'neutral'
        }
    
    def generate_dataset(self, total_records: int = 1000) -> pd.DataFrame:
        """Generate a complete dataset of mixed medical texts."""
        
        # Distribution of note types
        note_distributions = {
            'clinical_note': 0.3,
            'discharge_summary': 0.2,
            'patient_feedback': 0.3,
            'progress_note': 0.2
        }
        
        records = []
        
        for i in range(total_records):
            note_type = np.random.choice(
                list(note_distributions.keys()),
                p=list(note_distributions.values())
            )
            
            if note_type == 'clinical_note':
                record = self.generate_clinical_note()
            elif note_type == 'discharge_summary':
                record = self.generate_discharge_summary()
            elif note_type == 'patient_feedback':
                record = self.generate_patient_feedback()
            else:  # progress_note
                record = self.generate_progress_note()
            
            # Add metadata
            record['record_id'] = f"REC_{i+1:06d}"
            record['created_date'] = (
                datetime.now() - timedelta(days=random.randint(0, 365))
            ).strftime('%Y-%m-%d')
            
            records.append(record)
        
        return pd.DataFrame(records)

def main():
    """Generate and save synthetic medical text data."""
    generator = MedicalTextGenerator(seed=42)
    
    # Generate datasets
    print("Generating synthetic medical text dataset...")
    
    # Training dataset
    train_data = generator.generate_dataset(total_records=800)
    
    # Test dataset
    test_data = generator.generate_dataset(total_records=200)
    
    # Save datasets
    train_data.to_csv('../../data/synthetic/medical_text_train.csv', index=False)
    test_data.to_csv('../../data/synthetic/medical_text_test.csv', index=False)
    
    print(f"Generated {len(train_data)} training records")
    print(f"Generated {len(test_data)} test records")
    print("\nNote type distribution:")
    print(train_data['note_type'].value_counts())
    print("\nSentiment distribution (feedback only):")
    feedback_data = train_data[train_data['note_type'] == 'patient_feedback']
    print(feedback_data['sentiment'].value_counts())
    
    # Sample records
    print("\n" + "="*50)
    print("SAMPLE CLINICAL NOTE:")
    print("="*50)
    sample_clinical = train_data[train_data['note_type'] == 'clinical_note'].iloc[0]
    print(sample_clinical['text'])
    
    print("\n" + "="*50)
    print("SAMPLE PATIENT FEEDBACK:")
    print("="*50)
    sample_feedback = train_data[train_data['note_type'] == 'patient_feedback'].iloc[0]
    print(f"Sentiment: {sample_feedback['sentiment']}")
    print(sample_feedback['text'])

if __name__ == "__main__":
    main()
