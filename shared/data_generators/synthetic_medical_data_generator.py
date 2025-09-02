#!/usr/bin/env python3
"""
Synthetic Medical Dataset Generator for Portfolio
Generates realistic but synthetic patient data for fracture risk analysis
"""

import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class MedicalDataGenerator:
    def __init__(self, n_patients: int = 10000):
        self.n_patients = n_patients
        self.patient_ids = [f"PT{str(i).zfill(6)}" for i in range(1, n_patients + 1)]
        
        # Medical reference data
        self.medications = [
            'Alendronate', 'Risedronate', 'Ibandronate', 'Zoledronic acid',
            'Denosumab', 'Calcium carbonate', 'Vitamin D3', 'Calcitriol',
            'Teriparatide', 'Abaloparatide', 'Romosozumab', 'None'
        ]
        
        self.comorbidities = [
            'Osteoporosis', 'Rheumatoid arthritis', 'Diabetes mellitus type 2',
            'Chronic kidney disease', 'Hyperthyroidism', 'Hyperparathyroidism',
            'Malabsorption syndrome', 'Chronic obstructive pulmonary disease',
            'Cardiovascular disease', 'Depression'
        ]
        
        self.fracture_sites = [
            'Hip', 'Vertebral', 'Wrist', 'Shoulder', 'Ankle', 'Ribs', 'Pelvis'
        ]
        
    def generate_demographics(self) -> pd.DataFrame:
        """Generate patient demographics"""
        data = []
        
        for patient_id in self.patient_ids:
            # Age distribution realistic for fracture risk studies
            age = np.random.normal(68, 12)
            age = max(18, min(95, int(age)))
            
            # Gender (higher female representation in osteoporosis studies)
            gender = np.random.choice(['Female', 'Male'], p=[0.75, 0.25])
            
            # BMI with realistic distribution
            if gender == 'Female':
                bmi = np.random.normal(26.5, 5.2)
            else:
                bmi = np.random.normal(27.8, 4.8)
            bmi = max(16, min(50, round(bmi, 1)))
            
            # Ethnicity
            ethnicity = np.random.choice([
                'Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'
            ], p=[0.65, 0.12, 0.15, 0.06, 0.02])
            
            # Smoking history
            smoking_status = np.random.choice([
                'Never', 'Former', 'Current'
            ], p=[0.45, 0.35, 0.20])
            
            # Alcohol consumption (units per week)
            if gender == 'Female':
                alcohol = max(0, np.random.poisson(3.5))
            else:
                alcohol = max(0, np.random.poisson(5.2))
            
            # Education level
            education = np.random.choice([
                'High School', 'Some College', 'Bachelor\'s', 'Graduate'
            ], p=[0.25, 0.30, 0.30, 0.15])
            
            # Insurance type
            insurance = np.random.choice([
                'Medicare', 'Private', 'Medicaid', 'None'
            ], p=[0.45, 0.40, 0.12, 0.03])
            
            data.append({
                'patient_id': patient_id,
                'age': age,
                'gender': gender,
                'bmi': bmi,
                'ethnicity': ethnicity,
                'smoking_status': smoking_status,
                'alcohol_units_per_week': alcohol,
                'education_level': education,
                'insurance_type': insurance,
                'registration_date': self._random_date(2018, 2024)
            })
        
        return pd.DataFrame(data)
    
    def generate_clinical_measurements(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate clinical measurements and lab results"""
        data = []
        
        for _, patient in demographics_df.iterrows():
            # Bone density measurements (T-scores)
            # T-score: >-1.0 normal, -1.0 to -2.5 osteopenia, <-2.5 osteoporosis
            base_t_score = np.random.normal(-1.5, 0.8)
            
            # Age and gender effects on bone density
            if patient['age'] > 65:
                base_t_score -= 0.5
            if patient['gender'] == 'Female' and patient['age'] > 50:
                base_t_score -= 0.3
                
            spine_t_score = round(base_t_score + np.random.normal(0, 0.2), 2)
            hip_t_score = round(base_t_score + np.random.normal(0, 0.15), 2)
            
            # Laboratory values
            # Vitamin D (normal: 30-100 ng/mL)
            vitamin_d = max(5, np.random.normal(32, 15))
            
            # Calcium (normal: 8.5-10.5 mg/dL)
            calcium = np.random.normal(9.5, 0.4)
            
            # Phosphorus (normal: 2.5-4.5 mg/dL)
            phosphorus = np.random.normal(3.5, 0.3)
            
            # Parathyroid hormone (normal: 10-55 pg/mL)
            pth = max(5, np.random.lognormal(3.2, 0.4))
            
            # Physical measurements
            if patient['gender'] == 'Female':
                height = np.random.normal(162, 7)  # cm
            else:
                height = np.random.normal(175, 8)  # cm
            
            weight = (patient['bmi'] * (height/100)**2)
            
            # Grip strength (kg) - predictor of fracture risk
            if patient['gender'] == 'Female':
                grip_strength = max(10, np.random.normal(25 - (patient['age']-50)*0.3, 5))
            else:
                grip_strength = max(15, np.random.normal(40 - (patient['age']-50)*0.4, 8))
            
            data.append({
                'patient_id': patient['patient_id'],
                'measurement_date': self._random_recent_date(),
                'spine_t_score': spine_t_score,
                'hip_t_score': hip_t_score,
                'height_cm': round(height, 1),
                'weight_kg': round(weight, 1),
                'vitamin_d_ng_ml': round(vitamin_d, 1),
                'calcium_mg_dl': round(calcium, 2),
                'phosphorus_mg_dl': round(phosphorus, 2),
                'pth_pg_ml': round(pth, 1),
                'grip_strength_kg': round(grip_strength, 1)
            })
        
        return pd.DataFrame(data)
    
    def generate_fracture_events(self, demographics_df: pd.DataFrame, 
                               clinical_df: pd.DataFrame) -> pd.DataFrame:
        """Generate fracture events with realistic risk factors"""
        merged_df = demographics_df.merge(clinical_df, on='patient_id')
        fracture_data = []
        
        for _, patient in merged_df.iterrows():
            # Calculate fracture probability based on risk factors
            risk_score = self._calculate_fracture_risk(patient)
            
            # Determine if patient had fracture (25% base rate)
            has_fracture = np.random.random() < (0.15 + risk_score * 0.3)
            
            if has_fracture:
                # Number of fractures (most patients have 1, some have multiple)
                n_fractures = np.random.choice([1, 2, 3], p=[0.75, 0.20, 0.05])
                
                for i in range(n_fractures):
                    fracture_site = np.random.choice(self.fracture_sites, p=[
                        0.35, 0.25, 0.15, 0.08, 0.07, 0.05, 0.05  # Hip most common
                    ])
                    
                    # Fracture severity
                    severity = np.random.choice(['Minor', 'Moderate', 'Severe'], p=[0.4, 0.45, 0.15])
                    
                    # Treatment received
                    treatment = np.random.choice([
                        'Conservative', 'Surgical fixation', 'Joint replacement'
                    ], p=[0.5, 0.35, 0.15])
                    
                    # Recovery time (days)
                    if severity == 'Minor':
                        recovery_days = np.random.poisson(45)
                    elif severity == 'Moderate':
                        recovery_days = np.random.poisson(90)
                    else:
                        recovery_days = np.random.poisson(150)
                    
                    fracture_data.append({
                        'patient_id': patient['patient_id'],
                        'fracture_id': f"FX{len(fracture_data)+1:06d}",
                        'fracture_date': self._random_fracture_date(),
                        'fracture_site': fracture_site,
                        'severity': severity,
                        'treatment': treatment,
                        'recovery_days': recovery_days,
                        'mechanism': np.random.choice([
                            'Fall from standing', 'Fall from height', 'Motor vehicle accident', 
                            'Sports injury', 'Pathological', 'Unknown'
                        ], p=[0.45, 0.15, 0.10, 0.08, 0.12, 0.10])
                    })
            
            # Also add to main demographics whether they had any fracture
            demographics_df.loc[demographics_df['patient_id'] == patient['patient_id'], 'has_fracture'] = has_fracture
        
        return pd.DataFrame(fracture_data)
    
    def generate_medical_history(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate medical history and comorbidities"""
        data = []
        
        for _, patient in demographics_df.iterrows():
            # Number of comorbidities (older patients tend to have more)
            if patient['age'] < 50:
                n_conditions = np.random.poisson(1.2)
            elif patient['age'] < 70:
                n_conditions = np.random.poisson(2.1)
            else:
                n_conditions = np.random.poisson(2.8)
            
            n_conditions = min(n_conditions, len(self.comorbidities))
            
            if n_conditions > 0:
                conditions = np.random.choice(
                    self.comorbidities, size=n_conditions, replace=False
                )
            else:
                conditions = []
            
            # Medications (related to conditions and bone health)
            n_medications = max(0, int(np.random.normal(2.5, 1.5)))
            medications = np.random.choice(
                self.medications, size=min(n_medications, len(self.medications)), 
                replace=False
            )
            
            # Family history of fractures
            family_history = np.random.choice([True, False], p=[0.35, 0.65])
            
            # Previous fractures before current study
            previous_fractures = np.random.choice([0, 1, 2, 3], p=[0.6, 0.25, 0.10, 0.05])
            
            data.append({
                'patient_id': patient['patient_id'],
                'comorbidities': '; '.join(conditions) if hasattr(conditions, 'size') and conditions.size > 0 else 'None',
                'current_medications': '; '.join(medications) if hasattr(medications, 'size') and medications.size > 0 else 'None',
                'family_history_fractures': family_history,
                'previous_fracture_count': previous_fractures,
                'exercise_frequency': np.random.choice([
                    'Never', 'Rarely', 'Weekly', 'Daily'
                ], p=[0.20, 0.30, 0.35, 0.15])
            })
        
        return pd.DataFrame(data)
    
    def generate_patient_surveys(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """Generate patient-reported outcome measures"""
        data = []
        
        for _, patient in demographics_df.iterrows():
            # Quality of life scores (0-100, higher is better)
            qol_physical = max(0, min(100, int(np.random.normal(65, 20))))
            qol_mental = max(0, min(100, int(np.random.normal(70, 18))))
            
            # Pain scores (0-10, higher is worse)
            pain_level = max(0, min(10, int(np.random.normal(3.5, 2.5))))
            
            # Mobility assessment
            mobility = np.random.choice([
                'Independent', 'Uses walking aid', 'Assisted walking', 'Wheelchair'
            ], p=[0.65, 0.20, 0.10, 0.05])
            
            # Fall history in past year
            falls_last_year = np.random.poisson(1.2)
            
            # Fear of falling scale (1-5)
            fear_of_falling = np.random.choice([1, 2, 3, 4, 5], p=[0.15, 0.25, 0.35, 0.20, 0.05])
            
            # Satisfaction with care (1-5)
            care_satisfaction = np.random.choice([1, 2, 3, 4, 5], p=[0.02, 0.08, 0.20, 0.45, 0.25])
            
            data.append({
                'patient_id': patient['patient_id'],
                'survey_date': self._random_recent_date(),
                'quality_of_life_physical': qol_physical,
                'quality_of_life_mental': qol_mental,
                'pain_level_0_10': pain_level,
                'mobility_status': mobility,
                'falls_past_year': falls_last_year,
                'fear_of_falling_1_5': fear_of_falling,
                'care_satisfaction_1_5': care_satisfaction,
                'dietary_calcium_adequate': np.random.choice([True, False], p=[0.6, 0.4]),
                'exercise_minutes_per_week': max(0, int(np.random.normal(120, 80)))
            })
        
        return pd.DataFrame(data)
    
    def generate_clinical_notes(self, demographics_df: pd.DataFrame, 
                              fracture_df: pd.DataFrame) -> List[Dict]:
        """Generate realistic clinical notes for NLP processing"""
        notes = []
        
        # Templates for clinical notes
        visit_templates = [
            "Patient presents for routine follow-up visit.",
            "Patient scheduled for bone density evaluation.",
            "Follow-up visit after recent fracture.",
            "Patient reports concerns about bone health.",
            "Routine osteoporosis management visit."
        ]
        
        assessment_templates = [
            "Bone density results show {t_score_description}.",
            "Patient's fracture risk is assessed as {risk_level}.",
            "Current bone health status is {status}.",
            "Laboratory results indicate {lab_status}."
        ]
        
        plan_templates = [
            "Continue current medication regimen.",
            "Initiated {medication} therapy.",
            "Recommended lifestyle modifications including weight-bearing exercise.",
            "Scheduled follow-up bone density scan in 12 months.",
            "Referred to physical therapy for fall prevention."
        ]
        
        for _, patient in demographics_df.iterrows():
            # Generate 1-3 notes per patient
            n_notes = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            
            for note_num in range(n_notes):
                # Determine note context
                patient_fractures = fracture_df[fracture_df['patient_id'] == patient['patient_id']]
                has_fracture = len(patient_fractures) > 0
                
                # Generate note content
                chief_complaint = random.choice(visit_templates)
                
                # History of present illness
                if has_fracture and np.random.random() < 0.6:
                    hpi = f"Patient sustained {patient_fractures.iloc[0]['fracture_site'].lower()} fracture {np.random.randint(2, 12)} months ago."
                else:
                    hpi = f"Patient reports {random.choice(['no new concerns', 'mild back pain', 'some joint stiffness', 'fatigue'])}."
                
                # Assessment
                t_score = np.random.uniform(-3.5, 1.0)
                if t_score > -1.0:
                    t_score_desc = "normal bone density"
                    risk_level = "low"
                    status = "satisfactory"
                elif t_score > -2.5:
                    t_score_desc = "osteopenia"
                    risk_level = "moderate"
                    status = "concerning"
                else:
                    t_score_desc = "osteoporosis"
                    risk_level = "high"
                    status = "requires intervention"
                
                assessment = random.choice(assessment_templates).format(
                    t_score_description=t_score_desc,
                    risk_level=risk_level,
                    status=status,
                    lab_status=random.choice(["vitamin D deficiency", "normal calcium levels", "elevated PTH"])
                )
                
                # Plan
                plan = random.choice(plan_templates).format(
                    medication=random.choice(self.medications[:-1])  # Exclude 'None'
                )
                
                note_text = f"""
CHIEF COMPLAINT: {chief_complaint}

HISTORY OF PRESENT ILLNESS: {hpi}

PHYSICAL EXAMINATION: 
- Height: {patient['height_cm'] if 'height_cm' in patient else 165} cm
- Weight: {patient['weight_kg'] if 'weight_kg' in patient else 70} kg
- BMI: {patient['bmi']} kg/m²
- General appearance: {random.choice(['Well-appearing', 'Comfortable', 'Stable'])}

ASSESSMENT: {assessment}

PLAN: {plan}

Provider: Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Davis'])}
Date: {self._random_recent_date()}
""".strip()
                
                notes.append({
                    'patient_id': patient['patient_id'],
                    'note_id': f"NOTE{len(notes)+1:06d}",
                    'note_date': self._random_recent_date(),
                    'note_type': random.choice(['Progress Note', 'Consultation', 'Follow-up']),
                    'provider': f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Davis'])}",
                    'note_text': note_text
                })
        
        return notes
    
    def _calculate_fracture_risk(self, patient: pd.Series) -> float:
        """Calculate fracture risk score based on patient characteristics"""
        risk = 0.0
        
        # Age effect
        if patient['age'] > 65:
            risk += 0.3
        if patient['age'] > 75:
            risk += 0.2
        
        # Gender effect
        if patient['gender'] == 'Female':
            risk += 0.2
        
        # BMI effect (both low and high BMI increase risk)
        if patient['bmi'] < 20:
            risk += 0.3
        elif patient['bmi'] > 30:
            risk += 0.1
        
        # Bone density effect
        if 'spine_t_score' in patient:
            if patient['spine_t_score'] < -2.5:
                risk += 0.4
            elif patient['spine_t_score'] < -1.0:
                risk += 0.2
        
        # Smoking effect
        if patient.get('smoking_status') == 'Current':
            risk += 0.2
        elif patient.get('smoking_status') == 'Former':
            risk += 0.1
        
        # Alcohol effect
        if patient.get('alcohol_units_per_week', 0) > 14:
            risk += 0.1
        
        return min(1.0, risk)  # Cap at 1.0
    
    def _random_date(self, start_year: int, end_year: int) -> str:
        """Generate random date in range"""
        start = datetime(start_year, 1, 1)
        end = datetime(end_year, 12, 31)
        delta = end - start
        random_days = np.random.randint(0, delta.days)
        return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')
    
    def _random_recent_date(self) -> str:
        """Generate random recent date (last 2 years)"""
        return self._random_date(2023, 2024)
    
    def _random_fracture_date(self) -> str:
        """Generate random fracture date (last 3 years)"""
        return self._random_date(2022, 2024)
    
    def generate_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets"""
        print("Generating synthetic medical datasets...")
        
        # Generate core datasets
        print("1. Generating patient demographics...")
        demographics = self.generate_demographics()
        
        print("2. Generating clinical measurements...")
        clinical = self.generate_clinical_measurements(demographics)
        
        print("3. Generating medical history...")
        medical_history = self.generate_medical_history(demographics)
        
        print("4. Generating fracture events...")
        fractures = self.generate_fracture_events(demographics, clinical)
        
        print("5. Generating patient surveys...")
        surveys = self.generate_patient_surveys(demographics)
        
        print("6. Generating clinical notes...")
        notes_data = self.generate_clinical_notes(demographics, fractures)
        
        # Create master patient dataset
        master_data = demographics.merge(clinical, on='patient_id')
        master_data = master_data.merge(medical_history, on='patient_id')
        
        # Add summary statistics
        master_data['total_fractures'] = master_data['patient_id'].map(
            fractures['patient_id'].value_counts()
        ).fillna(0).astype(int)
        
        return {
            'master_patient_data': master_data,
            'fracture_events': fractures,
            'patient_surveys': surveys,
            'clinical_notes': notes_data
        }

def main():
    """Generate and save all synthetic datasets"""
    generator = MedicalDataGenerator(n_patients=5000)  # Start with 5000 patients
    datasets = generator.generate_all_datasets()
    
    print("\nDataset Summary:")
    print(f"- Patients: {len(datasets['master_patient_data'])}")
    print(f"- Fracture events: {len(datasets['fracture_events'])}")
    print(f"- Survey responses: {len(datasets['patient_surveys'])}")
    print(f"- Clinical notes: {len(datasets['clinical_notes'])}")
    
    # Save datasets
    print("\nSaving datasets...")
    datasets['master_patient_data'].to_csv('data/synthetic/master_patient_data.csv', index=False)
    datasets['fracture_events'].to_csv('data/synthetic/fracture_events.csv', index=False)
    datasets['patient_surveys'].to_csv('data/synthetic/patient_surveys.csv', index=False)
    
    # Save clinical notes as text file
    with open('data/synthetic/clinical_notes.txt', 'w') as f:
        for note in datasets['clinical_notes']:
            f.write(f"Patient ID: {note['patient_id']}\n")
            f.write(f"Note ID: {note['note_id']}\n")
            f.write(f"Date: {note['note_date']}\n")
            f.write(f"Type: {note['note_type']}\n")
            f.write(f"Provider: {note['provider']}\n")
            f.write(f"{note['note_text']}\n\n")

if __name__ == "__main__":
    main()
