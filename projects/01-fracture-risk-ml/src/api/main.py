"""
FastAPI application for fracture risk prediction API.

This module provides a RESTful API for serving the trained fracture risk
prediction model with comprehensive input validation and response formatting.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
import yaml
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load API configuration
with open("config/api_config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title=config['api']['title'],
    description=config['api']['description'],
    version=config['api']['version'],
    contact=config['api']['contact'],
    license_info=config['api']['license']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['cors']['allow_origins'],
    allow_credentials=config['cors']['allow_credentials'],
    allow_methods=config['cors']['allow_methods'],
    allow_headers=config['cors']['allow_headers'],
)


# Pydantic models for request/response validation
class PatientData(BaseModel):
    """Patient data input model for fracture risk prediction."""
    
    # Required fields
    age: int = Field(..., ge=18, le=95, description="Patient age in years")
    gender: str = Field(..., regex="^(Male|Female)$", description="Patient gender")
    bmi: float = Field(..., ge=16.0, le=50.0, description="Body Mass Index")
    spine_t_score: float = Field(..., ge=-4.0, le=2.0, description="Spine bone density T-score")
    hip_t_score: float = Field(..., ge=-4.0, le=2.0, description="Hip bone density T-score")
    
    # Optional fields with defaults
    vitamin_d_ng_ml: Optional[float] = Field(None, ge=5.0, le=100.0, description="Vitamin D level (ng/mL)")
    calcium_mg_dl: Optional[float] = Field(None, ge=7.0, le=12.0, description="Calcium level (mg/dL)")
    phosphorus_mg_dl: Optional[float] = Field(None, ge=1.5, le=6.0, description="Phosphorus level (mg/dL)")
    pth_pg_ml: Optional[float] = Field(None, ge=5.0, le=200.0, description="PTH level (pg/mL)")
    grip_strength_kg: Optional[float] = Field(None, ge=10.0, le=60.0, description="Grip strength (kg)")
    smoking_status: Optional[str] = Field("Never", regex="^(Never|Former|Current)$", description="Smoking status")
    alcohol_units_per_week: Optional[int] = Field(0, ge=0, le=30, description="Weekly alcohol units")
    previous_fracture_count: Optional[int] = Field(0, ge=0, le=10, description="Previous fractures count")
    family_history_fractures: Optional[bool] = Field(False, description="Family history of fractures")
    exercise_frequency: Optional[str] = Field("Weekly", regex="^(Never|Rarely|Weekly|Daily)$", description="Exercise frequency")
    
    class Config:
        schema_extra = {
            "example": {
                "age": 72,
                "gender": "Female",
                "bmi": 22.1,
                "spine_t_score": -2.8,
                "hip_t_score": -2.1,
                "vitamin_d_ng_ml": 18.5,
                "calcium_mg_dl": 9.2,
                "phosphorus_mg_dl": 3.4,
                "pth_pg_ml": 45.0,
                "grip_strength_kg": 18.2,
                "smoking_status": "Former",
                "alcohol_units_per_week": 2,
                "previous_fracture_count": 1,
                "family_history_fractures": True,
                "exercise_frequency": "Rarely"
            }
        }


class PredictionResponse(BaseModel):
    """Fracture risk prediction response model."""
    
    fracture_risk_score: float = Field(..., description="Fracture risk probability (0-1)")
    risk_category: str = Field(..., description="Risk category (Low/Moderate/High/Very High)")
    confidence_interval: Dict[str, float] = Field(..., description="95% confidence interval")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    recommendations: List[str] = Field(..., description="Clinical recommendations")
    model_version: str = Field(..., description="Model version used")
    prediction_timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="API status")
    model_status: str = Field(..., description="Model loading status")
    timestamp: str = Field(..., description="Health check timestamp")
    version: str = Field(..., description="API version")


# Global variables for model and artifacts
model = None
scaler = None
feature_names = []


def load_model_artifacts():
    """Load trained model and preprocessing artifacts."""
    global model, scaler, feature_names
    
    try:
        model_path = Path(config['model']['model_path'])
        scaler_path = Path(config['model']['scaler_path'])
        feature_names_path = Path(config['model']['feature_names_path'])
        
        # Load model
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
            
        # Load scaler
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        else:
            logger.warning(f"Scaler file not found: {scaler_path}")
            
        # Load feature names
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
            logger.info(f"Feature names loaded: {len(feature_names)} features")
        else:
            logger.warning(f"Feature names file not found: {feature_names_path}")
            
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")


def prepare_features(patient_data: PatientData) -> pd.DataFrame:
    """Prepare features for model prediction."""
    # Convert to dictionary and handle None values
    data_dict = patient_data.dict()
    
    # Fill missing values with defaults based on medical knowledge
    defaults = {
        'vitamin_d_ng_ml': 30.0,  # Assume adequate if not provided
        'calcium_mg_dl': 9.5,    # Normal range middle
        'phosphorus_mg_dl': 3.5,  # Normal range middle
        'pth_pg_ml': 32.5,       # Normal range middle
        'grip_strength_kg': 25.0 if data_dict['gender'] == 'Female' else 35.0,
    }
    
    for key, default_value in defaults.items():
        if data_dict[key] is None:
            data_dict[key] = default_value
    
    # Create DataFrame
    df = pd.DataFrame([data_dict])
    
    # Apply basic feature engineering (simplified version)
    # Note: In production, this should match exactly the training pipeline
    
    # Age groups
    df['age_group'] = pd.cut(df['age'], bins=[0, 50, 65, 75, 100], 
                            labels=['young_adult', 'middle_age', 'elderly', 'very_elderly'])
    
    # BMI categories
    df['bmi_category'] = pd.cut(df['bmi'], bins=[0, 18.5, 25, 30, 100],
                               labels=['underweight', 'normal', 'overweight', 'obese'])
    
    # Bone density categories
    df['worst_t_score'] = df[['spine_t_score', 'hip_t_score']].min(axis=1)
    df['worst_bone_category'] = pd.cut(df['worst_t_score'], bins=[-10, -2.5, -1.0, 10],
                                      labels=['osteoporosis', 'osteopenia', 'normal'])
    
    # Risk factors
    df['has_osteoporosis'] = (df['worst_t_score'] < -2.5).astype(int)
    df['vitamin_d_deficient'] = (df['vitamin_d_ng_ml'] < 30).astype(int)
    df['is_elderly'] = (df['age'] >= 65).astype(int)
    df['bmi_risk_low'] = (df['bmi'] < 20).astype(int)
    df['has_previous_fracture'] = (df['previous_fracture_count'] > 0).astype(int)
    
    # Encode categorical variables
    categorical_mappings = {
        'gender': {'Male': 0, 'Female': 1},
        'smoking_status': {'Never': 0, 'Former': 1, 'Current': 2},
        'exercise_frequency': {'Never': 0, 'Rarely': 1, 'Weekly': 2, 'Daily': 3},
        'age_group': {'young_adult': 0, 'middle_age': 1, 'elderly': 2, 'very_elderly': 3},
        'bmi_category': {'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3},
        'worst_bone_category': {'osteoporosis': 0, 'osteopenia': 1, 'normal': 2}
    }
    
    for col, mapping in categorical_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)
    
    return df


def identify_risk_factors(patient_data: PatientData, risk_score: float) -> List[str]:
    """Identify major risk factors for the patient."""
    risk_factors = []
    
    # Age risk
    if patient_data.age >= 75:
        risk_factors.append("Advanced age (≥75 years)")
    elif patient_data.age >= 65:
        risk_factors.append("Elderly age (≥65 years)")
    
    # Gender risk
    if patient_data.gender == "Female" and patient_data.age >= 50:
        risk_factors.append("Postmenopausal female")
    
    # Bone density
    worst_t_score = min(patient_data.spine_t_score, patient_data.hip_t_score)
    if worst_t_score < -2.5:
        risk_factors.append("Osteoporosis (T-score < -2.5)")
    elif worst_t_score < -1.0:
        risk_factors.append("Osteopenia (T-score -1.0 to -2.5)")
    
    # BMI
    if patient_data.bmi < 20:
        risk_factors.append("Low BMI (<20 kg/m²)")
    
    # Previous fractures
    if patient_data.previous_fracture_count > 0:
        risk_factors.append(f"Previous fractures ({patient_data.previous_fracture_count})")
    
    # Family history
    if patient_data.family_history_fractures:
        risk_factors.append("Family history of fractures")
    
    # Lifestyle factors
    if patient_data.smoking_status == "Current":
        risk_factors.append("Current smoking")
    elif patient_data.smoking_status == "Former":
        risk_factors.append("Former smoking")
    
    if patient_data.alcohol_units_per_week > 14:
        risk_factors.append("Excessive alcohol consumption")
    
    # Laboratory values
    if patient_data.vitamin_d_ng_ml and patient_data.vitamin_d_ng_ml < 30:
        risk_factors.append("Vitamin D deficiency")
    
    if patient_data.grip_strength_kg and patient_data.grip_strength_kg < 20:
        risk_factors.append("Low grip strength")
    
    return risk_factors


def generate_recommendations(patient_data: PatientData, risk_score: float, risk_factors: List[str]) -> List[str]:
    """Generate clinical recommendations based on risk assessment."""
    recommendations = []
    
    # Based on risk level
    if risk_score >= 0.8:
        recommendations.append("Immediate specialist referral recommended")
        recommendations.append("Consider pharmacological intervention")
    elif risk_score >= 0.6:
        recommendations.append("Endocrinology consultation recommended")
        recommendations.append("Comprehensive bone health assessment")
    elif risk_score >= 0.3:
        recommendations.append("Regular monitoring and lifestyle interventions")
    else:
        recommendations.append("Continue routine preventive care")
    
    # Specific recommendations based on risk factors
    if "Vitamin D deficiency" in risk_factors:
        recommendations.append("Vitamin D supplementation recommended")
    
    if "Low BMI" in risk_factors:
        recommendations.append("Nutritional assessment and weight management")
    
    if "Current smoking" in risk_factors:
        recommendations.append("Smoking cessation counseling essential")
    
    if "Low grip strength" in risk_factors:
        recommendations.append("Physical therapy and strength training")
    
    if any("Osteo" in factor for factor in risk_factors):
        recommendations.append("Bone-specific exercise program")
        recommendations.append("Fall prevention assessment")
    
    # General recommendations
    recommendations.append("Adequate calcium and vitamin D intake")
    recommendations.append("Weight-bearing and resistance exercises")
    recommendations.append("Fall hazard assessment at home")
    
    return list(set(recommendations))  # Remove duplicates


@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup."""
    load_model_artifacts()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_status = "loaded" if model is not None else "not_loaded"
    
    return HealthResponse(
        status="healthy",
        model_status=model_status,
        timestamp=datetime.now().isoformat(),
        version=config['api']['version']
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fracture_risk(patient_data: PatientData):
    """Predict fracture risk for a patient."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check server configuration."
        )
    
    try:
        # Prepare features
        features_df = prepare_features(patient_data)
        
        # Make prediction
        risk_probability = model.predict_proba(features_df)[0, 1]
        
        # Determine risk category
        thresholds = config['prediction']['risk_thresholds']
        if risk_probability < thresholds['low']:
            risk_category = "Low Risk"
        elif risk_probability < thresholds['moderate']:
            risk_category = "Moderate Risk"
        elif risk_probability < thresholds['high']:
            risk_category = "High Risk"
        else:
            risk_category = "Very High Risk"
        
        # Calculate confidence interval (simplified)
        ci_width = 0.1  # ±10% for demonstration
        confidence_interval = {
            "lower": max(0.0, risk_probability - ci_width),
            "upper": min(1.0, risk_probability + ci_width)
        }
        
        # Identify risk factors
        risk_factors = identify_risk_factors(patient_data, risk_probability)
        
        # Generate recommendations
        recommendations = generate_recommendations(patient_data, risk_probability, risk_factors)
        
        return PredictionResponse(
            fracture_risk_score=round(risk_probability, 4),
            risk_category=risk_category,
            confidence_interval=confidence_interval,
            risk_factors=risk_factors,
            recommendations=recommendations,
            model_version="1.0.0",
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": type(model).__name__,
        "feature_count": len(feature_names),
        "features": feature_names[:10],  # First 10 features
        "model_parameters": getattr(model, 'get_params', lambda: {})(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Load model artifacts
    load_model_artifacts()
    
    # Run the API
    uvicorn.run(
        app, 
        host=config['server']['host'],
        port=config['server']['port'],
        debug=config['server']['debug']
    )
