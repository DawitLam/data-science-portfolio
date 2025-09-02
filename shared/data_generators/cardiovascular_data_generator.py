"""
Cardiovascular Disease Risk Assessment - Synthetic Data Generator

This module generates realistic synthetic cardiovascular patient data for 
machine learning model development and testing. The data follows established
medical patterns and risk factor relationships from cardiovascular research.

Author: [Your Name]
Date: September 2025
Purpose: Educational ML portfolio demonstration
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random


class CardiovascularDataGenerator:
    """
    Generates synthetic cardiovascular patient data based on medical literature
    and established risk factor relationships.
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the data generator with reproducible random state."""
        np.random.seed(random_state)
        random.seed(random_state)
        self.random_state = random_state
        
        # Define medical reference ranges and distributions
        self.medical_ranges = {
            'age': {'min': 30, 'max': 80, 'mean': 55, 'std': 12},
            'systolic_bp': {'normal': 120, 'std': 15, 'min': 90, 'max': 200},
            'diastolic_bp': {'normal': 80, 'std': 10, 'min': 60, 'max': 120},
            'total_cholesterol': {'normal': 200, 'std': 40, 'min': 120, 'max': 350},
            'hdl_cholesterol': {'normal': 50, 'std': 15, 'min': 20, 'max': 100},
            'triglycerides': {'normal': 150, 'std': 80, 'min': 50, 'max': 500},
            'glucose': {'normal': 100, 'std': 20, 'min': 70, 'max': 300},
            'bmi': {'normal': 25, 'std': 5, 'min': 18, 'max': 45}
        }
        
        # Risk factor correlations based on medical literature
        self.risk_correlations = {
            'age_cvd': 0.6,        # Strong age correlation
            'smoking_cvd': 0.4,     # Significant smoking impact
            'diabetes_cvd': 0.5,    # Diabetes major risk factor
            'hypertension_cvd': 0.45, # Blood pressure impact
            'cholesterol_cvd': 0.35   # Cholesterol relationship
        }
    
    def generate_dataset(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate a complete synthetic cardiovascular dataset.
        
        Args:
            n_samples: Number of patient records to generate
            
        Returns:
            DataFrame with synthetic patient data and CVD risk labels
        """
        print(f"🫀 Generating {n_samples:,} synthetic cardiovascular patient records...")
        
        # Generate base demographic and clinical data
        data = self._generate_demographics(n_samples)
        data.update(self._generate_vital_signs(n_samples, data))
        data.update(self._generate_laboratory_values(n_samples, data))
        data.update(self._generate_lifestyle_factors(n_samples, data))
        data.update(self._generate_medical_history(n_samples, data))
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Generate CVD risk outcome based on medical risk factors
        df['cvd_risk_10yr'] = self._calculate_cvd_risk(df)
        
        # Add patient metadata
        df['patient_id'] = [f"CVD_{i:06d}" for i in range(1, n_samples + 1)]
        df['assessment_date'] = self._generate_assessment_dates(n_samples)
        
        # Reorder columns for better presentation
        df = self._reorder_columns(df)
        
        print(f"✅ Generated dataset with {len(df)} patients")
        print(f"📊 CVD Risk Distribution: {df['cvd_risk_10yr'].value_counts().to_dict()}")
        
        return df
    
    def _generate_demographics(self, n_samples: int) -> Dict:
        """Generate basic demographic information."""
        
        # Age distribution (slightly skewed toward older patients)
        ages = np.random.gamma(2, 25) + 30
        ages = np.clip(ages, 30, 80).astype(int)
        
        # Gender distribution (approximately balanced)
        genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48])
        
        # Ethnicity distribution (representative sample)
        ethnicities = np.random.choice(
            ['Caucasian', 'African American', 'Hispanic', 'Asian', 'Other'],
            n_samples,
            p=[0.65, 0.15, 0.12, 0.06, 0.02]
        )
        
        return {
            'age': ages,
            'gender': genders,
            'ethnicity': ethnicities
        }
    
    def _generate_vital_signs(self, n_samples: int, demo_data: Dict) -> Dict:
        """Generate vital signs with age and gender correlations."""
        
        ages = demo_data['age']
        genders = demo_data['gender']
        
        # Systolic blood pressure (increases with age)
        base_systolic = 110 + (ages - 30) * 0.8
        systolic_noise = np.random.normal(0, 15, n_samples)
        systolic_bp = np.clip(base_systolic + systolic_noise, 90, 200)
        
        # Diastolic blood pressure (correlated with systolic)
        diastolic_bp = systolic_bp * 0.65 + np.random.normal(0, 8, n_samples)
        diastolic_bp = np.clip(diastolic_bp, 60, 120)
        
        # Heart rate (varies by age and fitness)
        heart_rate = 70 + np.random.normal(0, 12, n_samples)
        heart_rate = np.clip(heart_rate, 50, 120).astype(int)
        
        # BMI (age and lifestyle dependent)
        base_bmi = 24 + (ages - 40) * 0.1
        bmi = base_bmi + np.random.normal(0, 4, n_samples)
        bmi = np.clip(bmi, 18, 45)
        
        return {
            'systolic_bp': systolic_bp.round(0).astype(int),
            'diastolic_bp': diastolic_bp.round(0).astype(int),
            'heart_rate': heart_rate,
            'bmi': bmi.round(1)
        }
    
    def _generate_laboratory_values(self, n_samples: int, demo_data: Dict) -> Dict:
        """Generate laboratory values with medical correlations."""
        
        ages = demo_data['age']
        genders = demo_data['gender']
        
        # Total cholesterol (increases with age)
        total_chol = 180 + (ages - 30) * 1.2 + np.random.normal(0, 30, n_samples)
        total_chol = np.clip(total_chol, 120, 350)
        
        # HDL cholesterol (higher in females)
        hdl_base = np.where(genders == 'Female', 55, 45)
        hdl_chol = hdl_base + np.random.normal(0, 12, n_samples)
        hdl_chol = np.clip(hdl_chol, 20, 100)
        
        # LDL cholesterol (calculated from total and HDL)
        ldl_chol = total_chol - hdl_chol - np.random.uniform(20, 50, n_samples)
        ldl_chol = np.clip(ldl_chol, 50, 250)
        
        # Triglycerides (metabolic correlation)
        triglycerides = 120 + np.random.gamma(2, 30)
        triglycerides = np.clip(triglycerides, 50, 500)
        
        # Glucose (age and weight correlated)
        glucose = 90 + (ages - 40) * 0.5 + np.random.normal(0, 15, n_samples)
        glucose = np.clip(glucose, 70, 300)
        
        # HbA1c (related to glucose)
        hba1c = 5.0 + (glucose - 100) * 0.02 + np.random.normal(0, 0.3, n_samples)
        hba1c = np.clip(hba1c, 4.5, 12.0)
        
        return {
            'total_cholesterol': total_chol.round(0).astype(int),
            'hdl_cholesterol': hdl_chol.round(0).astype(int),
            'ldl_cholesterol': ldl_chol.round(0).astype(int),
            'triglycerides': triglycerides.round(0).astype(int),
            'glucose': glucose.round(0).astype(int),
            'hba1c': hba1c.round(1)
        }
    
    def _generate_lifestyle_factors(self, n_samples: int, demo_data: Dict) -> Dict:
        """Generate lifestyle and behavioral risk factors."""
        
        # Smoking status (age-dependent patterns)
        smoking_prob = np.maximum(0.1, 0.4 - demo_data['age'] * 0.005)
        smoking_status = np.random.binomial(1, smoking_prob, n_samples)
        
        # Physical activity levels
        activity_levels = np.random.choice(
            ['Sedentary', 'Low', 'Moderate', 'High'],
            n_samples,
            p=[0.25, 0.35, 0.30, 0.10]
        )
        
        # Alcohol consumption
        alcohol_levels = np.random.choice(
            ['None', 'Light', 'Moderate', 'Heavy'],
            n_samples,
            p=[0.20, 0.40, 0.30, 0.10]
        )
        
        # Diet quality score (1-10 scale)
        diet_score = np.random.beta(2, 2) * 10
        diet_score = np.clip(diet_score, 1, 10).round(1)
        
        return {
            'smoking_status': smoking_status,
            'physical_activity': activity_levels,
            'alcohol_consumption': alcohol_levels,
            'diet_quality_score': diet_score
        }
    
    def _generate_medical_history(self, n_samples: int, demo_data: Dict) -> Dict:
        """Generate medical history and comorbidities."""
        
        ages = demo_data['age']
        
        # Diabetes (age-dependent)
        diabetes_prob = np.minimum(0.4, (ages - 30) * 0.008)
        diabetes = np.random.binomial(1, diabetes_prob, n_samples)
        
        # Hypertension (age and weight dependent)
        hypertension_prob = np.minimum(0.6, (ages - 30) * 0.012)
        hypertension = np.random.binomial(1, hypertension_prob, n_samples)
        
        # Family history of CVD
        family_history_cvd = np.random.binomial(1, 0.3, n_samples)
        
        # Previous heart conditions
        prev_mi = np.random.binomial(1, 0.05, n_samples)  # Previous MI
        prev_stroke = np.random.binomial(1, 0.03, n_samples)  # Previous stroke
        
        # Medications
        on_statin = np.random.binomial(1, 0.25, n_samples)
        on_bp_meds = np.random.binomial(1, 0.30, n_samples)
        
        return {
            'diabetes': diabetes,
            'hypertension': hypertension,
            'family_history_cvd': family_history_cvd,
            'previous_mi': prev_mi,
            'previous_stroke': prev_stroke,
            'on_statin_therapy': on_statin,
            'on_bp_medication': on_bp_meds
        }
    
    def _calculate_cvd_risk(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate 10-year CVD risk based on established risk factors.
        Uses simplified Framingham-inspired risk calculation.
        """
        
        # Initialize risk score
        risk_score = np.zeros(len(df))
        
        # Age contribution (major factor)
        risk_score += (df['age'] - 30) * 2
        
        # Gender contribution (males higher risk at younger ages)
        male_bonus = np.where(df['gender'] == 'Male', 20, 0)
        risk_score += male_bonus
        
        # Blood pressure contribution
        bp_risk = (df['systolic_bp'] - 120) * 0.5 + (df['diastolic_bp'] - 80) * 0.3
        risk_score += np.maximum(0, bp_risk)
        
        # Cholesterol contribution
        chol_risk = (df['total_cholesterol'] - 200) * 0.2 - (df['hdl_cholesterol'] - 50) * 0.3
        risk_score += chol_risk
        
        # Diabetes major risk factor
        risk_score += df['diabetes'] * 40
        
        # Smoking major risk factor
        risk_score += df['smoking_status'] * 35
        
        # BMI contribution
        bmi_risk = np.maximum(0, (df['bmi'] - 25) * 2)
        risk_score += bmi_risk
        
        # Family history
        risk_score += df['family_history_cvd'] * 15
        
        # Previous cardiovascular events
        risk_score += df['previous_mi'] * 60
        risk_score += df['previous_stroke'] * 50
        
        # Protective factors
        activity_protection = np.where(df['physical_activity'].isin(['Moderate', 'High']), -10, 0)
        risk_score += activity_protection
        
        # Convert to probability using logistic function
        risk_prob = 1 / (1 + np.exp(-(risk_score - 100) / 30))
        
        # Clip to reasonable range (1-40% 10-year risk)
        risk_prob = np.clip(risk_prob, 0.01, 0.40)
        
        # Convert to binary high/low risk (>20% threshold)
        high_risk = (risk_prob > 0.20).astype(int)
        
        return high_risk
    
    def _generate_assessment_dates(self, n_samples: int) -> List[str]:
        """Generate realistic assessment dates."""
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2025, 8, 31)
        
        dates = []
        for _ in range(n_samples):
            random_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days)
            )
            dates.append(random_date.strftime('%Y-%m-%d'))
        
        return dates
    
    def _reorder_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns for logical presentation."""
        
        column_order = [
            'patient_id', 'assessment_date', 'age', 'gender', 'ethnicity',
            'systolic_bp', 'diastolic_bp', 'heart_rate', 'bmi',
            'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol', 
            'triglycerides', 'glucose', 'hba1c',
            'smoking_status', 'physical_activity', 'alcohol_consumption', 
            'diet_quality_score',
            'diabetes', 'hypertension', 'family_history_cvd',
            'previous_mi', 'previous_stroke', 'on_statin_therapy', 
            'on_bp_medication', 'cvd_risk_10yr'
        ]
        
        return df[column_order]


def generate_cardiovascular_data(n_samples: int = 10000, 
                               output_path: str = None) -> pd.DataFrame:
    """
    Main function to generate cardiovascular disease risk dataset.
    
    Args:
        n_samples: Number of patient records to generate
        output_path: Optional path to save the generated data
        
    Returns:
        DataFrame with synthetic cardiovascular patient data
    """
    
    generator = CardiovascularDataGenerator()
    df = generator.generate_dataset(n_samples)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"💾 Dataset saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate sample dataset
    df = generate_cardiovascular_data(
        n_samples=15000,
        output_path="../../data/synthetic/cardiovascular_risk_data.csv"
    )
    
    # Display summary statistics
    print("\n📊 Dataset Summary:")
    print(f"Total patients: {len(df):,}")
    print(f"High CVD risk patients: {df['cvd_risk_10yr'].sum():,} ({df['cvd_risk_10yr'].mean():.1%})")
    print(f"Age range: {df['age'].min()}-{df['age'].max()} years")
    print(f"Mean BMI: {df['bmi'].mean():.1f}")
    print(f"Diabetes prevalence: {df['diabetes'].mean():.1%}")
    print(f"Smoking prevalence: {df['smoking_status'].mean():.1%}")
