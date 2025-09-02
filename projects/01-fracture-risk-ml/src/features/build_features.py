"""
Feature engineering pipeline for fracture risk prediction.

This module contains medical domain-specific feature engineering
functions to create predictive features for fracture risk modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class MedicalFeatureEngineer:
    """Medical domain-specific feature engineering for fracture risk prediction."""
    
    def __init__(self, config: Dict):
        """Initialize feature engineer with configuration."""
        self.config = config
        self.fitted_transformers = {}
        
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create age-related features based on fracture risk patterns."""
        df = df.copy()
        
        # Age groups based on fracture risk
        age_groups = self.config['engineering']['numerical_transformations']['age_groups']
        df['age_group'] = pd.cut(
            df['age'],
            bins=[group[0] for group in age_groups] + [age_groups[-1][1]],
            labels=[group[2] for group in age_groups],
            include_lowest=True
        )
        
        # Age risk factor (exponential increase after 65)
        df['age_risk_factor'] = np.where(
            df['age'] >= 65,
            1 + (df['age'] - 65) * 0.05,  # 5% increase per year after 65
            1.0
        )
        
        # Elderly flag
        df['is_elderly'] = (df['age'] >= 65).astype(int)
        df['is_very_elderly'] = (df['age'] >= 75).astype(int)
        
        logger.info("Created age-related features")
        return df
    
    def create_bmi_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create BMI-related features."""
        df = df.copy()
        
        # BMI categories
        bmi_cats = self.config['engineering']['numerical_transformations']['bmi_categories']
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=[cat[0] for cat in bmi_cats] + [bmi_cats[-1][1]],
            labels=[cat[2] for cat in bmi_cats],
            include_lowest=True
        )
        
        # BMI risk factors
        df['bmi_risk_low'] = (df['bmi'] < 20).astype(int)  # Low BMI increases fracture risk
        df['bmi_risk_high'] = (df['bmi'] >= 30).astype(int)  # High BMI (different mechanism)
        
        # BMI z-score (age and gender adjusted)
        df['bmi_zscore'] = self._calculate_bmi_zscore(df)
        
        logger.info("Created BMI-related features")
        return df
    
    def create_bone_density_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create bone density-related features."""
        df = df.copy()
        
        # WHO T-score categories
        bone_cats = self.config['engineering']['numerical_transformations']['bone_density_categories']
        
        # Categorize spine and hip T-scores
        for site in ['spine', 'hip']:
            col = f'{site}_t_score'
            if col in df.columns:
                df[f'{site}_bone_category'] = pd.cut(
                    df[col],
                    bins=[cat[0] for cat in bone_cats] + [bone_cats[-1][1]],
                    labels=[cat[2] for cat in bone_cats],
                    include_lowest=True
                )
        
        # Worst T-score (most predictive)
        if 'spine_t_score' in df.columns and 'hip_t_score' in df.columns:
            df['worst_t_score'] = df[['spine_t_score', 'hip_t_score']].min(axis=1)
            df['worst_bone_category'] = pd.cut(
                df['worst_t_score'],
                bins=[cat[0] for cat in bone_cats] + [bone_cats[-1][1]],
                labels=[cat[2] for cat in bone_cats],
                include_lowest=True
            )
        
        # Osteoporosis flags
        df['has_osteoporosis'] = (df.get('worst_t_score', -10) < -2.5).astype(int)
        df['has_osteopenia'] = ((df.get('worst_t_score', -10) >= -2.5) & 
                               (df.get('worst_t_score', -10) < -1.0)).astype(int)
        
        logger.info("Created bone density features")
        return df
    
    def create_laboratory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create laboratory-based features."""
        df = df.copy()
        
        # Vitamin D deficiency
        if 'vitamin_d_ng_ml' in df.columns:
            df['vitamin_d_deficient'] = (df['vitamin_d_ng_ml'] < 30).astype(int)
            df['vitamin_d_severely_deficient'] = (df['vitamin_d_ng_ml'] < 20).astype(int)
        
        # Calcium abnormalities
        if 'calcium_mg_dl' in df.columns:
            df['calcium_abnormal'] = ((df['calcium_mg_dl'] < 8.5) | 
                                    (df['calcium_mg_dl'] > 10.5)).astype(int)
            df['calcium_low'] = (df['calcium_mg_dl'] < 8.5).astype(int)
        
        # PTH elevation
        if 'pth_pg_ml' in df.columns:
            df['pth_elevated'] = (df['pth_pg_ml'] > 55).astype(int)
            df['pth_severely_elevated'] = (df['pth_pg_ml'] > 100).astype(int)
        
        # Combined metabolic dysfunction
        metabolic_cols = ['vitamin_d_deficient', 'calcium_abnormal', 'pth_elevated']
        available_cols = [col for col in metabolic_cols if col in df.columns]
        if available_cols:
            df['metabolic_dysfunction_score'] = df[available_cols].sum(axis=1)
        
        logger.info("Created laboratory features")
        return df
    
    def create_physical_function_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create physical function-related features."""
        df = df.copy()
        
        # Grip strength percentiles (age and gender adjusted)
        if 'grip_strength_kg' in df.columns:
            df['grip_strength_low'] = self._calculate_grip_strength_percentile(df)
            df['grip_strength_very_low'] = (df['grip_strength_low'] < 0.25).astype(int)
        
        # Sarcopenia indicators (low muscle mass/strength)
        if all(col in df.columns for col in ['bmi', 'grip_strength_kg']):
            df['sarcopenia_risk'] = ((df['bmi'] < 20) & (df['grip_strength_low'] < 0.5)).astype(int)
        
        logger.info("Created physical function features")
        return df
    
    def create_lifestyle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lifestyle-related features."""
        df = df.copy()
        
        # Smoking risk
        if 'smoking_status' in df.columns:
            df['smoking_risk'] = df['smoking_status'].map({
                'Never': 0,
                'Former': 1,
                'Current': 2
            }).fillna(0)
        
        # Alcohol risk
        if 'alcohol_units_per_week' in df.columns:
            df['alcohol_risk'] = (df['alcohol_units_per_week'] > 14).astype(int)
            df['alcohol_excessive'] = (df['alcohol_units_per_week'] > 21).astype(int)
        
        # Exercise adequacy
        if 'exercise_frequency' in df.columns:
            df['exercise_adequate'] = df['exercise_frequency'].isin(['Weekly', 'Daily']).astype(int)
        
        logger.info("Created lifestyle features")
        return df
    
    def create_medical_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create medical history features."""
        df = df.copy()
        
        # Previous fracture risk
        if 'previous_fracture_count' in df.columns:
            df['has_previous_fracture'] = (df['previous_fracture_count'] > 0).astype(int)
            df['multiple_previous_fractures'] = (df['previous_fracture_count'] > 1).astype(int)
        
        # Family history risk
        if 'family_history_fractures' in df.columns:
            df['family_history_risk'] = df['family_history_fractures'].astype(int)
        
        # Comorbidity count
        if 'comorbidities' in df.columns:
            df['comorbidity_count'] = df['comorbidities'].apply(
                lambda x: len(str(x).split(';')) if pd.notna(x) and str(x) != 'None' else 0
            )
            df['multiple_comorbidities'] = (df['comorbidity_count'] >= 2).astype(int)
        
        logger.info("Created medical history features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables."""
        df = df.copy()
        
        # Age-gender interaction
        if all(col in df.columns for col in ['age', 'gender']):
            df['age_female'] = df['age'] * (df['gender'] == 'Female').astype(int)
            df['postmenopausal'] = ((df['age'] >= 50) & 
                                   (df['gender'] == 'Female')).astype(int)
        
        # BMI-bone density interaction
        if all(col in df.columns for col in ['bmi', 'worst_t_score']):
            df['bmi_bone_interaction'] = df['bmi'] * df['worst_t_score']
        
        # Age-bone density interaction
        if all(col in df.columns for col in ['age', 'worst_t_score']):
            df['age_bone_interaction'] = df['age'] * df['worst_t_score']
        
        # Combined risk score (simplified FRAX-like)
        risk_factors = []
        if 'age_risk_factor' in df.columns:
            risk_factors.append('age_risk_factor')
        if 'bmi_risk_low' in df.columns:
            risk_factors.append('bmi_risk_low')
        if 'has_osteoporosis' in df.columns:
            risk_factors.append('has_osteoporosis')
        if 'smoking_risk' in df.columns:
            risk_factors.append('smoking_risk')
        if 'has_previous_fracture' in df.columns:
            risk_factors.append('has_previous_fracture')
        
        if risk_factors:
            df['frax_like_score'] = df[risk_factors].sum(axis=1)
        
        logger.info("Created interaction features")
        return df
    
    def create_frax_components(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create FRAX-like risk assessment components."""
        df = df.copy()
        
        # Age component (major risk factor)
        if 'age' in df.columns:
            df['frax_age_component'] = np.clip((df['age'] - 40) / 50, 0, 1)
        
        # Gender component
        if 'gender' in df.columns:
            df['frax_gender_component'] = (df['gender'] == 'Female').astype(float)
        
        # BMI component (U-shaped risk)
        if 'bmi' in df.columns:
            optimal_bmi = 25
            df['frax_bmi_component'] = np.abs(df['bmi'] - optimal_bmi) / optimal_bmi
        
        # Previous fracture component
        if 'previous_fracture_count' in df.columns:
            df['frax_previous_fx_component'] = np.clip(df['previous_fracture_count'] / 3, 0, 1)
        
        # Family history component
        if 'family_history_fractures' in df.columns:
            df['frax_family_history_component'] = df['family_history_fractures'].astype(float)
        
        # Smoking component
        if 'smoking_status' in df.columns:
            smoking_weights = {'Never': 0, 'Former': 0.5, 'Current': 1.0}
            df['frax_smoking_component'] = df['smoking_status'].map(smoking_weights).fillna(0)
        
        # Alcohol component
        if 'alcohol_units_per_week' in df.columns:
            df['frax_alcohol_component'] = (df['alcohol_units_per_week'] > 14).astype(float)
        
        logger.info("Created FRAX-like components")
        return df
    
    def _calculate_bmi_zscore(self, df: pd.DataFrame) -> pd.Series:
        """Calculate age and gender-adjusted BMI z-scores."""
        # Simplified z-score calculation
        # In practice, would use age/gender-specific normative data
        mean_bmi = df.groupby(['gender'])['bmi'].transform('mean')
        std_bmi = df.groupby(['gender'])['bmi'].transform('std')
        return (df['bmi'] - mean_bmi) / std_bmi
    
    def _calculate_grip_strength_percentile(self, df: pd.DataFrame) -> pd.Series:
        """Calculate age and gender-adjusted grip strength percentiles."""
        # Simplified percentile calculation
        return df.groupby(['gender'])['grip_strength_kg'].rank(pct=True)
    
    def apply_all_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations."""
        logger.info("Starting comprehensive feature engineering")
        
        df = self.create_age_features(df)
        df = self.create_bmi_features(df)
        df = self.create_bone_density_features(df)
        df = self.create_laboratory_features(df)
        df = self.create_physical_function_features(df)
        df = self.create_lifestyle_features(df)
        df = self.create_medical_history_features(df)
        df = self.create_frax_components(df)
        df = self.create_interaction_features(df)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df


def engineer_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Convenience function to apply feature engineering.
    
    Args:
        df: Input dataframe
        config: Feature configuration
        
    Returns:
        pd.DataFrame: Dataframe with engineered features
    """
    engineer = MedicalFeatureEngineer(config)
    return engineer.apply_all_transformations(df)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load config
    with open('config/feature_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load sample data
    try:
        from src.data.load_data import DataLoader
    except ImportError:
        import sys
        sys.path.append('.')
        from src.data.load_data import DataLoader
        
    loader = DataLoader()
    X, y = loader.load_training_data()
    
    # Apply feature engineering
    engineer = MedicalFeatureEngineer(config)
    X_engineered = engineer.apply_all_transformations(X)
    
    print(f"Original features: {X.shape[1]}")
    print(f"Engineered features: {X_engineered.shape[1]}")
    print(f"New features: {X_engineered.shape[1] - X.shape[1]}")
